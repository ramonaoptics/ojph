#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <limits>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <new>

#if defined(_WIN32)
  #include <malloc.h>
  #include <io.h>
  #include <fcntl.h>
#else
  #include <unistd.h>
#endif

#include <ojph/ojph_file.h>
#include <ojph/ojph_codestream.h>
#include <ojph/ojph_mem.h>
#include <ojph/ojph_params.h>

namespace nb = nanobind;
using namespace ojph;

namespace {

// A NumPy-compatible CPU array of any dtype/shape/strides. dlpack reports
// strides in elements; every consumer below works in bytes, so the helpers
// convert once at the boundary.
using any_array = nb::ndarray<nb::device::cpu>;

inline size_t item_size(const any_array& a) { return a.dtype().bits / 8; }

inline bool is_unsigned_dtype(const any_array& a) {
    return a.dtype().code == (uint8_t)nb::dlpack::dtype_code::UInt;
}

inline size_t byte_stride(const any_array& a, size_t dim) {
    return (size_t)a.stride(dim) * item_size(a);
}

// C-contiguity for a 2D array, in the NumPy sense (dimensions of extent <= 1
// impose no stride constraint).
inline bool is_c_contig_2d(const any_array& a) {
    if (a.ndim() != 2) return false;
    if (a.shape(1) > 1 && a.stride(1) != 1) return false;
    if (a.shape(0) > 1 && (size_t)a.stride(0) != (size_t)a.shape(1)) return false;
    return true;
}

// Alignment used for read buffers. 4096 satisfies O_DIRECT sector alignment on
// every platform we target and keeps buffers page/cache-line friendly for SIMD
// decode paths.
constexpr size_t OJPH_READ_ALIGN = 4096;

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Copy one decoded si32 line into the caller's output row, narrowing to T
// and optionally clipping. The contiguous case (col_stride == sizeof(T)) is
// kept as tight per-type loops that the compiler auto-vectorizes into
// packed narrowing stores; the strided fallback handles anything else.
template <typename T>
inline void line_to_out(const si32* line_data, size_t n, char* out,
                        size_t col_stride, bool do_clip,
                        si32 min_val, si32 max_val)
{
    if (col_stride == sizeof(T)) {
        T* dp = reinterpret_cast<T*>(out);
        if (do_clip) {
            for (size_t i = 0; i < n; ++i) {
                si32 val = line_data[i];
                val = val < min_val ? min_val : val;
                val = val > max_val ? max_val : val;
                dp[i] = static_cast<T>(val);
            }
        } else {
            for (size_t i = 0; i < n; ++i)
                dp[i] = static_cast<T>(line_data[i]);
        }
    } else {
        if (do_clip) {
            for (size_t i = 0; i < n; ++i) {
                si32 val = line_data[i];
                val = val < min_val ? min_val : val;
                val = val > max_val ? max_val : val;
                *reinterpret_cast<T*>(out + i * col_stride) =
                    static_cast<T>(val);
            }
        } else {
            for (size_t i = 0; i < n; ++i)
                *reinterpret_cast<T*>(out + i * col_stride) =
                    static_cast<T>(line_data[i]);
        }
    }
}

// Portable aligned allocation. The returned pointer must be released with
// aligned_free(). Sized up to a multiple of ``align`` so O_DIRECT reads that
// round their length up stay inside the allocation.
inline void* aligned_read_alloc(size_t size, size_t align = OJPH_READ_ALIGN) {
  size_t padded = round_up(size ? size : 1, align);
#if defined(_WIN32)
  return _aligned_malloc(padded, align);
#else
  void* p = nullptr;
  if (posix_memalign(&p, align, padded) != 0)
    return nullptr;
  return p;
#endif
}

inline void aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// pread the full [offset, offset+count) region (looping over short reads for a
// regular fd). For an O_DIRECT fd the caller passes an aligned buf/count/offset
// and we issue a single call, tolerating a short tail at EOF.
inline int64_t pread_region(int fd, void* buf, size_t count, int64_t offset,
                            bool o_direct) {
#if defined(_WIN32)
  // Windows has no pread and no O_DIRECT; seek + read. Each read_j2c_fd_into
  // call uses its own fd (opened per page by the caller), so the two sequential
  // reads within a call do not race. Force binary mode so a caller who opened
  // the fd in the CRT default (text) mode does not get CRLF translation or an
  // early stop at a 0x1A byte in the binary codestream.
  (void)o_direct;
  _setmode(fd, _O_BINARY);
  if (_lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  int r = _read(fd, buf, (unsigned int)count);
  return (int64_t)r;
#else
  if (o_direct)
    return (int64_t)::pread(fd, buf, count, (off_t)offset);
  size_t done = 0;
  while (done < count) {
    ssize_t r = ::pread(fd, (char*)buf + done, count - done, (off_t)(offset + done));
    if (r < 0) return (int64_t)r;
    if (r == 0) break;  // EOF
    done += (size_t)r;
  }
  return (int64_t)done;
#endif
}

// Parse the TLM marker segment out of an already-read main header to find how
// many bytes of the codestream are needed to reconstruct ``level`` (number of
// finest resolutions skipped). Returns 0 if the TLM is absent or inconsistent,
// in which case the caller reads the whole tile. Assumes RLCP with one tile-part
// per resolution.
inline size_t tlm_bytes_to_read(const ui8* buf, size_t header_size,
                                ui32 num_decompositions, int level) {
  size_t search = header_size < 200 ? header_size : 200;
  size_t idx = SIZE_MAX;
  for (size_t i = 0; i + 1 < search; ++i)
    if (buf[i] == 0xFF && buf[i + 1] == 0x55) { idx = i; break; }
  if (idx == SIZE_MAX) return 0;

  const ui8* p = buf + idx + 2;                 // past the 0xFF55 marker
  ui32 Ltlm = ((ui32)p[0] << 8) | p[1]; p += 2; // segment length
  p += 1;                                       // Ztlm
  ui8 Stlm = p[0]; p += 1;
  int ST = (Stlm & 0x30) >> 4;
  int Ttlm_bytes = (ST == 0) ? 0 : (ST == 1 ? 1 : 2);
  int SP = (Stlm & 0x40) >> 6;
  int Ptlm_bytes = (SP == 0) ? 2 : 4;
  int per = Ttlm_bytes + Ptlm_bytes;
  if (per == 0) return 0;
  int num_tile_parts = (int)(Ltlm - 4) / per;
  if (num_tile_parts != (int)num_decompositions + 1) return 0;

  if (level < 0) level = 0;
  if (level > (int)num_decompositions) level = (int)num_decompositions;
  int target = num_tile_parts - 1 - level;      // cumulative index for this level

  // Guard against reading past what we actually have in the header buffer.
  const ui8* end = buf + header_size;
  size_t cum = 0;
  for (int i = 0; i <= target; ++i) {
    if (p + per > end) return 0;
    p += Ttlm_bytes;
    size_t ptlm = 0;
    for (int b = 0; b < Ptlm_bytes; ++b) ptlm = (ptlm << 8) | p[b];
    p += Ptlm_bytes;
    cum += ptlm;
  }
  return header_size + cum;
}

// Pull every decoded line of a single component into a 2D output buffer, with
// optional clipping. Shared by read_j2c_into and read_j2c_fd_into. Must be
// called with the GIL released.
inline void pull_single_component_into(
    codestream& cs, char* out_ptr, size_t rows, size_t cols,
    size_t row_stride, size_t element_size, bool is_unsigned,
    bool do_clip, si32 min_val, si32 max_val) {
  ui32 comp = 0;
  for (size_t r = 0; r < rows; ++r) {
    line_buf* line = cs.pull(comp);
    const si32* ld = line->i32;
    char* dst = out_ptr + r * row_stride;
    if (element_size == 1) {
      if (is_unsigned)
        line_to_out<ui8>(ld, cols, dst, 1, do_clip, min_val, max_val);
      else
        line_to_out<si8>(ld, cols, dst, 1, do_clip, min_val, max_val);
    } else if (element_size == 2) {
      if (is_unsigned)
        line_to_out<ui16>(ld, cols, dst, 2, do_clip, min_val, max_val);
      else
        line_to_out<si16>(ld, cols, dst, 2, do_clip, min_val, max_val);
    } else {
      if (is_unsigned)
        line_to_out<ui32>(ld, cols, dst, 4, do_clip, min_val, max_val);
      else
        line_to_out<si32>(ld, cols, dst, 4, do_clip, min_val, max_val);
    }
  }
}

}  // anonymous namespace

// This module runs without the GIL on free-threaded CPython (3.13t/3.14t).
// nanobind declares Py_MOD_GIL_NOT_USED automatically when the extension is
// compiled with NB_FREE_THREADED (set by setup.py when building for a
// free-threaded interpreter). This is safe here because the module holds no
// mutable global state: every binding works on per-instance or stack-local
// C++ objects, and the codec's lazily built lookup tables are guarded by
// std::call_once. As everywhere else in Python, a *single* Codestream /
// infile / outfile object must still not be shared across threads without
// external synchronisation -- the guarantee is that independent objects in
// different threads do not interfere.
NB_MODULE(ojph_bindings, m) {
    nb::class_<infile_base>(m, "InfileBase")
        .def("read", [](infile_base& self,
                        nb::ndarray<ui8, nb::ndim<1>, nb::device::cpu> buffer,
                        size_t size) {
            if (buffer.size() < size)
                throw nb::value_error("Buffer size is smaller than requested read size");
            nb::gil_scoped_release release;
            return self.read(buffer.data(), size);
        }, nb::arg("buffer"), nb::arg("size"))
        .def("seek", &infile_base::seek)
        .def("tell", &infile_base::tell)
        .def("eof", &infile_base::eof)
        .def("close", &infile_base::close);

    nb::class_<j2c_infile, infile_base>(m, "J2CInfile")
        .def(nb::init<>())
        .def("open", &j2c_infile::open)
        .def("seek", [](infile_base& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum infile_base::seek>(origin));
        })
        .def("tell", &j2c_infile::tell)
        .def("eof", &j2c_infile::eof)
        .def("close", &j2c_infile::close);

    nb::class_<mem_infile, infile_base>(m, "MemInfile")
        .def(nb::init<>())
        .def("open", [](mem_infile& self,
                        nb::ndarray<const ui8, nb::ndim<1>, nb::device::cpu> data) {
            self.open(data.data(), data.size());
        }, nb::arg("data"), nb::keep_alive<1, 2>())
        .def("read", [](mem_infile& self,
                        nb::ndarray<ui8, nb::ndim<1>, nb::device::cpu> buffer,
                        size_t size) {
            if (buffer.size() < size)
                throw nb::value_error("Buffer size is smaller than requested read size");
            nb::gil_scoped_release release;
            return self.read(buffer.data(), size);
        }, nb::arg("buffer"), nb::arg("size"))
        .def("seek", [](mem_infile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum infile_base::seek>(origin));
        })
        .def("tell", &mem_infile::tell)
        .def("eof", &mem_infile::eof)
        .def("close", &mem_infile::close);


    nb::class_<outfile_base>(m, "outfileBase")
        .def("write", [](outfile_base& self, nb::bytes data) {
            nb::gil_scoped_release release;
            return self.write(data.c_str(), data.size());
        }, nb::arg("data"))
        .def("seek", &outfile_base::seek)
        .def("tell", &outfile_base::tell)
        .def("close", &outfile_base::close);

    nb::class_<j2c_outfile, outfile_base>(m, "J2COutfile")
        .def(nb::init<>())
        .def("open", &j2c_outfile::open)
        .def("write", [](j2c_outfile& self, nb::bytes data) {
            nb::gil_scoped_release release;
            return self.write(data.c_str(), data.size());
        }, nb::arg("data"))
        .def("tell", &j2c_outfile::tell)
        .def("close", &j2c_outfile::close);

    nb::class_<mem_outfile, outfile_base>(m, "MemOutfile")
        .def(nb::init<>())
        .def("open", &mem_outfile::open, nb::arg("initial_size") = 65536, nb::arg("clear_mem") = false)
        .def("write", [](mem_outfile& self, nb::bytes data) {
            nb::gil_scoped_release release;
            return self.write(data.c_str(), data.size());
        }, nb::arg("data"))
        .def("tell", &mem_outfile::tell)
        .def("get_used_size", &mem_outfile::get_used_size)
        .def("get_buf_size", &mem_outfile::get_buf_size)
        .def("seek", [](mem_outfile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum outfile_base::seek>(origin));
        })
        .def("close", &mem_outfile::close)
        .def("write_to_file", &mem_outfile::write_to_file,
             nb::call_guard<nb::gil_scoped_release>())
        .def("get_data", [](mem_outfile& self) {
            const ui8* data = self.get_data();
            si64 size = self.tell();
            PyObject* view = PyMemoryView_FromMemory(
                (char*)data, (Py_ssize_t)size, PyBUF_READ);
            if (!view)
                throw nb::python_error();
            return nb::steal(view);
        }, nb::keep_alive<0, 1>());

    // Bindings for codestream class
    nb::class_<codestream>(m, "Codestream")
        .def(nb::init<>())
        .def("set_planar", &codestream::set_planar)
        .def("set_profile", &codestream::set_profile)
        .def("set_tilepart_divisions", &codestream::set_tilepart_divisions)
        .def("is_tilepart_division_at_resolutions", &codestream::is_tilepart_division_at_resolutions)
        .def("is_tilepart_division_at_components", &codestream::is_tilepart_division_at_components)
        .def("request_tlm_marker", &codestream::request_tlm_marker)
        .def("is_tlm_requested", &codestream::is_tlm_requested)
        .def("write_headers",
             [](codestream &self, outfile_base *file, nb::object comments, ui32 num_comments) {
                 // Check if the comments argument is None and convert it to nullptr if so
                 const comment_exchange* comments_ptr = comments.is_none()
                     ? nullptr : nb::cast<const comment_exchange*>(comments);
                 nb::gil_scoped_release release;
                 self.write_headers(file, comments_ptr, num_comments);
             },
             nb::arg("file"), nb::arg("comments") = nb::none(), nb::arg("num_comments") = 0)
        .def("exchange",
             [](codestream &self, nb::object line_buf_obj, ui32 next_component) -> line_buf* {
                 line_buf* buf = nullptr;
                 if (!line_buf_obj.is_none()) {
                     buf = nb::cast<line_buf*>(line_buf_obj);
                 }
                 nb::gil_scoped_release release;
                 return self.exchange(buf, next_component);
             },
             nb::arg("line_buf_obj") = nb::none(), nb::arg("next_component") = 0,
             nb::rv_policy::reference)
        .def("push_all_components",
             [](codestream &self, any_array image, ui32 num_components, const std::string& channel_order) {
                 size_t element_size = item_size(image);
                 bool src_unsigned = is_unsigned_dtype(image);

                 size_t height, width;
                 size_t component_stride;

                 if (num_components == 1) {
                     if (image.ndim() == 2) {
                         height = image.shape(0);
                         width = image.shape(1);
                         component_stride = 0;
                     } else if (image.ndim() == 3 && image.shape(2) == 1) {
                         height = image.shape(0);
                         width = image.shape(1);
                         component_stride = 0;
                     } else {
                         throw nb::value_error("Image must be 2-dimensional or 3-dimensional with last dimension of 1 for single component");
                     }
                 } else {
                     if (image.ndim() != 3) {
                         throw nb::value_error("Image must be 3-dimensional for multiple components");
                     }
                     if (channel_order == "CHW") {
                         height = image.shape(1);
                         width = image.shape(2);
                         component_stride = byte_stride(image, 0);
                     } else {
                         height = image.shape(0);
                         width = image.shape(1);
                         component_stride = byte_stride(image, 2);
                     }
                 }

                 size_t row_stride, col_stride;
                 if (num_components == 1) {
                     row_stride = byte_stride(image, 0);
                     col_stride = byte_stride(image, 1);
                 } else if (channel_order == "CHW") {
                     row_stride = byte_stride(image, 1);
                     col_stride = byte_stride(image, 2);
                 } else {
                     row_stride = byte_stride(image, 0);
                     col_stride = byte_stride(image, 1);
                 }

                 const char* err = nullptr;
                 {
                     nb::gil_scoped_release release;
                     ui32 next_comp = 0;
                     ui32& next_comp_ref = next_comp;
                     line_buf* line = self.exchange(nullptr, next_comp_ref);

                     for (ui32 c = 0; c < num_components && !err; ++c) {
                         char* component_base = static_cast<char*>(image.data());
                         if (num_components > 1)
                             component_base += c * component_stride;

                         for (size_t h = 0; h < height; ++h) {
                             char* row_start = component_base + h * row_stride;
                             si32* line_data = line->i32;
                             size_t line_size = line->size;

                             if (line_size != width) {
                                 err = "Line size mismatch";
                                 break;
                             }

                             if (col_stride == element_size) {
                                 // Contiguous row: with the stride known to
                                 // equal the element size, these tight loops
                                 // auto-vectorize into packed widening loads,
                                 // which the generic strided loops below
                                 // cannot.
                                 if (element_size == 1) {
                                     if (src_unsigned) {
                                         const ui8* sp = reinterpret_cast<const ui8*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = static_cast<si32>(sp[i]);
                                     } else {
                                         const si8* sp = reinterpret_cast<const si8*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = static_cast<si32>(sp[i]);
                                     }
                                 } else if (element_size == 2) {
                                     if (src_unsigned) {
                                         const ui16* sp = reinterpret_cast<const ui16*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = static_cast<si32>(sp[i]);
                                     } else {
                                         const si16* sp = reinterpret_cast<const si16*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = static_cast<si32>(sp[i]);
                                     }
                                 } else {
                                     if (src_unsigned) {
                                         const ui32* sp = reinterpret_cast<const ui32*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = static_cast<si32>(sp[i]);
                                     } else {
                                         const si32* sp = reinterpret_cast<const si32*>(row_start);
                                         for (size_t i = 0; i < line_size; ++i)
                                             line_data[i] = sp[i];
                                     }
                                 }
                             } else if (element_size == 1) {
                                 if (src_unsigned) {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const ui8*>(row_start + i * col_stride));
                                     }
                                 } else {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const si8*>(row_start + i * col_stride));
                                     }
                                 }
                             } else if (element_size == 2) {
                                 if (src_unsigned) {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const ui16*>(row_start + i * col_stride));
                                     }
                                 } else {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const si16*>(row_start + i * col_stride));
                                     }
                                 }
                             } else {
                                 if (src_unsigned) {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const ui32*>(row_start + i * col_stride));
                                     }
                                 } else {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = *reinterpret_cast<const si32*>(row_start + i * col_stride);
                                     }
                                 }
                             }

                             next_comp = (h == height - 1 && c < num_components - 1) ? c + 1 : c;
                             line = self.exchange(line, next_comp_ref);
                         }
                     }
                 }
                 if (err)
                     throw nb::value_error(err);
             },
             nb::arg("image"), nb::arg("num_components"), nb::arg("channel_order"))
        .def("flush", &codestream::flush, nb::call_guard<nb::gil_scoped_release>())
        .def("enable_resilience", &codestream::enable_resilience)
        .def("read_headers", &codestream::read_headers, nb::call_guard<nb::gil_scoped_release>())
        .def("restrict_input_resolution", &codestream::restrict_input_resolution)
        .def("create", &codestream::create, nb::call_guard<nb::gil_scoped_release>())
        .def("pull", &codestream::pull, nb::call_guard<nb::gil_scoped_release>(),
             nb::rv_policy::reference)
        .def("pull_all_components",
             [](codestream &self, any_array output, ui32 num_components, const std::string& channel_order, nb::object min_val_obj, nb::object max_val_obj) {
                 bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
                 si32 min_val = 0;
                 si32 max_val = 0;
                 if (do_clip) {
                     min_val = nb::cast<si32>(min_val_obj);
                     max_val = nb::cast<si32>(max_val_obj);
                 }

                 bool is_unsigned = is_unsigned_dtype(output);
                 size_t element_size = item_size(output);

                 size_t height, width;
                 size_t component_stride;

                 if (num_components == 1) {
                     if (output.ndim() != 2) {
                         throw nb::value_error("Output must be 2-dimensional for single component");
                     }
                     height = output.shape(0);
                     width = output.shape(1);
                     component_stride = 0;
                 } else {
                     if (output.ndim() != 3) {
                         throw nb::value_error("Output must be 3-dimensional for multiple components");
                     }
                     if (channel_order == "CHW") {
                         height = output.shape(1);
                         width = output.shape(2);
                         component_stride = byte_stride(output, 0);
                     } else {
                         height = output.shape(0);
                         width = output.shape(1);
                         component_stride = byte_stride(output, 2);
                     }
                 }

                 size_t row_stride = (num_components == 1 || channel_order == "CHW")
                     ? byte_stride(output, output.ndim() - 2) : byte_stride(output, 0);
                 size_t col_stride = (num_components == 1 || channel_order == "CHW")
                     ? byte_stride(output, output.ndim() - 1) : byte_stride(output, 1);

                 const char* err = nullptr;
                 {
                     nb::gil_scoped_release release;
                     for (ui32 c = 0; c < num_components && !err; ++c) {
                         char* component_base = static_cast<char*>(output.data());
                         if (num_components > 1)
                             component_base += c * component_stride;

                         line_buf* first_line = self.pull(c);
                         size_t line_size = first_line->size;
                         if (line_size != width) {
                             err = "Line size mismatch";
                             break;
                         }

                         for (size_t h = 0; h < height; ++h) {
                             line_buf* line = (h == 0) ? first_line : self.pull(c);
                             si32* line_data = line->i32;
                             char* out_row_start = component_base + h * row_stride;

                             if (element_size == 1) {
                                 if (is_unsigned)
                                     line_to_out<ui8>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                                 else
                                     line_to_out<si8>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                             } else if (element_size == 2) {
                                 if (is_unsigned)
                                     line_to_out<ui16>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                                 else
                                     line_to_out<si16>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                             } else {
                                 if (is_unsigned)
                                     line_to_out<ui32>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                                 else
                                     line_to_out<si32>(line_data, line_size,
                                         out_row_start, col_stride,
                                         do_clip, min_val, max_val);
                             }
                         }
                     }
                 }
                 if (err)
                     throw nb::value_error(err);
             },
             nb::arg("output"), nb::arg("num_components"), nb::arg("channel_order"), nb::arg("min_val") = nb::none(), nb::arg("max_val") = nb::none())
        .def("close", &codestream::close)
        .def("access_siz", &codestream::access_siz)
        .def("access_cod", &codestream::access_cod)
        .def("access_qcd", &codestream::access_qcd)
        .def("access_nlt", &codestream::access_nlt)
        .def("is_planar", &codestream::is_planar);

    // -----------------------------------------------------------------------
    // read_j2c_into: single-call, GIL-free reduced-resolution decode.
    //
    // Performs the ENTIRE read pipeline (open memory infile, parse headers,
    // restrict to ``level`` skipped resolutions, create, and pull every line
    // into ``out``) under one ``nb::gil_scoped_release``. This is the key to
    // letting a Python ThreadPoolExecutor actually parallelise many small
    // reduced-resolution decodes: with the orchestration living in Python the
    // threads serialise on the GIL, whereas here the GIL is released for the
    // whole decode. Single-component (grayscale) uint8/uint16 only for now.
    //
    // ``data``  : compressed codestream bytes (a full or partial read; if
    //             partial, the caller must have written an EOF marker).
    // ``out``   : pre-allocated, C-contiguous 2D array at the level shape,
    //             using the caller's own (aligned) allocator.
    // ``level`` : number of finest resolutions to skip (0 == full res).
    // Returns (height, width) actually decoded so the caller can slice.
    // -----------------------------------------------------------------------
    m.def("read_j2c_into",
        [](nb::ndarray<const ui8, nb::ndim<1>, nb::device::cpu> data,
           any_array out, int level,
           nb::object min_val_obj, nb::object max_val_obj)
            -> std::pair<ui32, ui32> {
            if (out.ndim() != 2)
                throw nb::value_error("out must be 2-dimensional (single component)");
            if (!is_c_contig_2d(out))
                throw nb::value_error("out must be C-contiguous");

            const ui8* data_ptr = data.data();
            size_t data_size = data.size();
            char* out_ptr = static_cast<char*>(out.data());
            size_t out_rows = out.shape(0);
            size_t out_cols = out.shape(1);
            size_t row_stride = byte_stride(out, 0);
            size_t element_size = item_size(out);
            bool is_unsigned = is_unsigned_dtype(out);

            bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
            si32 min_val = do_clip ? nb::cast<si32>(min_val_obj) : 0;
            si32 max_val = do_clip ? nb::cast<si32>(max_val_obj) : 0;

            ui32 h = 0, w = 0;
            bool shape_ok = false;
            {
                nb::gil_scoped_release release;

                mem_infile infile;
                infile.open(data_ptr, data_size);
                codestream cs;
                cs.read_headers(&infile);
                cs.restrict_input_resolution((ui32)level, (ui32)level);
                param_siz siz = cs.access_siz();
                h = siz.get_recon_height(0);
                w = siz.get_recon_width(0);
                shape_ok = ((size_t)h == out_rows && (size_t)w == out_cols);
                if (shape_ok) {
                    cs.create();
                    pull_single_component_into(
                        cs, out_ptr, out_rows, out_cols, row_stride,
                        element_size, is_unsigned, do_clip, min_val, max_val);
                }
                cs.close();
            }
            if (!shape_ok)
                throw nb::value_error("out shape does not match decoded level shape");
            return std::make_pair(h, w);
        },
        nb::arg("data"), nb::arg("out"), nb::arg("level"),
        nb::arg("min_val") = nb::none(), nb::arg("max_val") = nb::none());

    // -----------------------------------------------------------------------
    // peek_j2c_fd: read just the main header of a codestream stored at
    // [offset, offset+nbytes) in an open file descriptor and return the
    // information a caller needs to size an output buffer:
    //   (num_decompositions, height, width)  -- full-resolution extent.
    // GIL-free apart from argument marshalling. Callers should cache the result
    // per (device, inode, offset); the codestream header never changes.
    // -----------------------------------------------------------------------
    m.def("peek_j2c_fd",
        [](int fd, int64_t offset, int64_t nbytes, bool o_direct)
            -> nb::tuple {
            constexpr size_t PEEK = 65536;
            ui32 nd = 0, h = 0, w = 0;
            bool ok = false;
            {
                nb::gil_scoped_release release;
                size_t want = PEEK;
                if ((size_t)nbytes < want)
                    want = o_direct ? round_up((size_t)nbytes, OJPH_READ_ALIGN)
                                    : (size_t)nbytes;
                ui8* buf = (ui8*)aligned_read_alloc(want);
                if (buf) {
                    int64_t got = pread_region(fd, buf, want, offset, o_direct);
                    if (got > 0) {
                        mem_infile mf;
                        mf.open(buf, (size_t)got);
                        codestream cs;
                        cs.read_headers(&mf);
                        nd = cs.access_cod().get_num_decompositions();
                        param_siz siz = cs.access_siz();
                        h = siz.get_recon_height(0);
                        w = siz.get_recon_width(0);
                        cs.close();
                        ok = true;
                    }
                    aligned_free(buf);
                }
            }
            if (!ok)
                throw nb::value_error("peek_j2c_fd: failed to read codestream header");
            return nb::make_tuple(nd, h, w);
        },
        nb::arg("fd"), nb::arg("offset"), nb::arg("nbytes"),
        nb::arg("o_direct") = false);

    // -----------------------------------------------------------------------
    // read_j2c_fd_into: the whole reduced-resolution read for one codestream,
    // GIL-free, straight from a file descriptor.
    //
    // Under a single nb::gil_scoped_release it: aligned-reads the header, parses
    // the TLM to learn how many bytes ``level`` needs, aligned-reads exactly
    // those bytes into an aligned buffer (falling back to the whole tile if the
    // TLM is absent), writes the EOC marker, then decodes into ``out``. All I/O
    // uses O_DIRECT-compatible aligned buffers/lengths when ``o_direct`` is set,
    // so the caller's O_DIRECT fast path is preserved.
    //
    // ``out`` must be a pre-allocated C-contiguous 2D array at the level shape
    // (use peek_j2c_fd + get_level_shape math, cached per file). Returns the
    // decoded (height, width). Single-component uint8/uint16/int8/int16.
    // -----------------------------------------------------------------------
    m.def("read_j2c_fd_into",
        [](int fd, int64_t offset, int64_t nbytes, any_array out, int level,
           nb::object min_val_obj, nb::object max_val_obj, bool o_direct)
            -> std::pair<ui32, ui32> {
            if (out.ndim() != 2)
                throw nb::value_error("out must be 2-dimensional (single component)");
            if (!is_c_contig_2d(out))
                throw nb::value_error("out must be C-contiguous");

            char* out_ptr = static_cast<char*>(out.data());
            size_t out_rows = out.shape(0);
            size_t out_cols = out.shape(1);
            size_t row_stride = byte_stride(out, 0);
            size_t element_size = item_size(out);
            bool is_unsigned = is_unsigned_dtype(out);
            bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
            si32 min_val = do_clip ? nb::cast<si32>(min_val_obj) : 0;
            si32 max_val = do_clip ? nb::cast<si32>(max_val_obj) : 0;

            ui32 h = 0, w = 0;
            const char* err = nullptr;
            {
                nb::gil_scoped_release release;

                // First read: enough to cover the main header plus, for a deep
                // zoom-out, all the bytes that level actually needs -- so the
                // common case is a SINGLE read and a SINGLE header parse. The
                // codestream parsed here is reused to decode when the data fits.
                constexpr size_t INITIAL = 65536;
                size_t cap = (size_t)nbytes;
                size_t first_len = INITIAL < cap ? INITIAL : cap;
                if (o_direct) first_len = round_up(first_len, OJPH_READ_ALIGN);
                ui8* buf = (ui8*)aligned_read_alloc(first_len + OJPH_READ_ALIGN);
                if (!buf) {
                    err = "read_j2c_fd_into: allocation failed";
                } else {
                    int64_t got = pread_region(fd, buf, first_len, offset, o_direct);
                    if (got <= 0) {
                        err = "read_j2c_fd_into: short read";
                        aligned_free(buf);
                    } else {
                        size_t have = (size_t)got;
                        mem_infile mf;
                        mf.open(buf, have);
                        codestream cs;
                        cs.read_headers(&mf);
                        ui32 nd = cs.access_cod().get_num_decompositions();
                        size_t header_size = (size_t)mf.tell();
                        size_t bytes_to_read =
                            tlm_bytes_to_read(buf, header_size, nd, level);
                        if (bytes_to_read == 0 || bytes_to_read > cap)
                            bytes_to_read = cap;  // no/!TLM -> whole tile

                        if (bytes_to_read <= have) {
                            // Fast path: needed bytes are already in buf. Reuse
                            // the codestream we just parsed -- no second read,
                            // no second read_headers.
                            buf[bytes_to_read - 2] = 0xFF;
                            buf[bytes_to_read - 1] = 0xD9;
                            cs.restrict_input_resolution((ui32)level, (ui32)level);
                            param_siz siz = cs.access_siz();
                            h = siz.get_recon_height(0);
                            w = siz.get_recon_width(0);
                            if ((size_t)h != out_rows || (size_t)w != out_cols)
                                err = "out shape does not match decoded level shape";
                            else {
                                cs.create();
                                pull_single_component_into(
                                    cs, out_ptr, out_rows, out_cols, row_stride,
                                    element_size, is_unsigned, do_clip,
                                    min_val, max_val);
                            }
                            cs.close();
                            aligned_free(buf);
                        } else {
                            // Shallow level: read the full extent this level
                            // needs into a fresh buffer, then decode.
                            cs.close();
                            aligned_free(buf);
                            size_t read_len = o_direct
                                ? round_up(bytes_to_read, OJPH_READ_ALIGN)
                                : bytes_to_read;
                            ui8* dbuf =
                                (ui8*)aligned_read_alloc(read_len + OJPH_READ_ALIGN);
                            if (!dbuf) {
                                err = "read_j2c_fd_into: allocation failed";
                            } else {
                                int64_t g2 = pread_region(fd, dbuf, read_len,
                                                          offset, o_direct);
                                if (g2 < (int64_t)bytes_to_read) {
                                    err = "read_j2c_fd_into: short read";
                                } else {
                                    dbuf[bytes_to_read - 2] = 0xFF;
                                    dbuf[bytes_to_read - 1] = 0xD9;
                                    mem_infile mf2;
                                    mf2.open(dbuf, bytes_to_read);
                                    codestream cs2;
                                    cs2.read_headers(&mf2);
                                    cs2.restrict_input_resolution((ui32)level,
                                                                  (ui32)level);
                                    param_siz siz = cs2.access_siz();
                                    h = siz.get_recon_height(0);
                                    w = siz.get_recon_width(0);
                                    if ((size_t)h != out_rows || (size_t)w != out_cols)
                                        err = "out shape does not match decoded level shape";
                                    else {
                                        cs2.create();
                                        pull_single_component_into(
                                            cs2, out_ptr, out_rows, out_cols,
                                            row_stride, element_size, is_unsigned,
                                            do_clip, min_val, max_val);
                                    }
                                    cs2.close();
                                }
                                aligned_free(dbuf);
                            }
                        }
                    }
                }
            }
            if (err)
                throw nb::value_error(err);
            return std::make_pair(h, w);
        },
        nb::arg("fd"), nb::arg("offset"), nb::arg("nbytes"), nb::arg("out"),
        nb::arg("level"), nb::arg("min_val") = nb::none(),
        nb::arg("max_val") = nb::none(), nb::arg("o_direct") = false);

    nb::class_<point>(m, "Point")
        .def(nb::init<ui32, ui32>(), nb::arg("x") = 0, nb::arg("y") = 0)  // Constructor with default args
        .def_rw("x", &point::x)
        .def_rw("y", &point::y);

    nb::class_<ojph::size>(m, "Size")
        .def(nb::init<ui32, ui32>(), nb::arg("w") = 0, nb::arg("h") = 0)  // Constructor with default args
        .def_rw("w", &ojph::size::w)  // width
        .def_rw("h", &ojph::size::h)  // height
        .def("area", &ojph::size::area);     // Expose the area function

    nb::class_<param_siz>(m, "ParamSiz")
        .def("is_signed", &param_siz::is_signed)

        .def("set_image_extent", &param_siz::set_image_extent)
        .def("set_tile_size", &param_siz::set_tile_size)
        .def("set_image_offset", &param_siz::set_image_offset)
        .def("set_tile_offset", &param_siz::set_tile_offset)
        .def("set_num_components", &param_siz::set_num_components)
        .def("set_component", &param_siz::set_component, nb::arg("comp_num"), nb::arg("downsampling"), nb::arg("bit_depth"), nb::arg("is_signed"))

        .def("get_image_extent", &param_siz::get_image_extent)
        .def("get_image_offset", &param_siz::get_image_offset)
        .def("get_tile_size", &param_siz::get_tile_size)
        .def("get_tile_offset", &param_siz::get_tile_offset)
        .def("get_num_components", &param_siz::get_num_components)
        .def("get_bit_depth", &param_siz::get_bit_depth)
        .def("get_downsampling", &param_siz::get_downsampling)
        .def("get_recon_width", &param_siz::get_recon_width)
        .def("get_recon_height", &param_siz::get_recon_height);

    nb::class_<param_cod>(m, "ParamCod")
        // OpenJPH >= 0.30.1 (post-release) added COC-segment overloads that take
        // a leading comp_idx argument, so these COD-segment (no comp_idx) methods
        // must be disambiguated with an explicit member-function-pointer cast.
        .def("set_num_decomposition", static_cast<void (param_cod::*)(ui32)>(&param_cod::set_num_decomposition), nb::arg("num_decompositions"))
        .def("set_block_dims", static_cast<void (param_cod::*)(ui32, ui32)>(&param_cod::set_block_dims), nb::arg("width"), nb::arg("height"))
        .def("set_precinct_size", static_cast<void (param_cod::*)(int, ojph::size*)>(&param_cod::set_precinct_size), nb::arg("num_levels"), nb::arg("precinct_size"))
        .def("set_progression_order", &param_cod::set_progression_order, nb::arg("name"))
        .def("set_color_transform", &param_cod::set_color_transform, nb::arg("color_transform"))
        .def("set_reversible", static_cast<void (param_cod::*)(bool)>(&param_cod::set_reversible), nb::arg("reversible"))
        .def("set_wavelet_kern", &param_cod::set_wavelet_kern, nb::arg("kernel"))
        .def("get_wavelet_kern", &param_cod::get_wavelet_kern)
        .def("is_predict_only", &param_cod::is_predict_only)

        .def("get_num_decompositions", static_cast<ui32 (param_cod::*)() const>(&param_cod::get_num_decompositions))
        .def("get_block_dims", static_cast<ojph::size (param_cod::*)() const>(&param_cod::get_block_dims))
        .def("get_log_block_dims", static_cast<ojph::size (param_cod::*)() const>(&param_cod::get_log_block_dims))
        .def("is_reversible", static_cast<bool (param_cod::*)() const>(&param_cod::is_reversible))
        .def("get_precinct_size", static_cast<ojph::size (param_cod::*)(ui32) const>(&param_cod::get_precinct_size), nb::arg("level_num"))
        .def("get_log_precinct_size", static_cast<ojph::size (param_cod::*)(ui32) const>(&param_cod::get_log_precinct_size), nb::arg("level_num"))
        .def("get_progression_order", &param_cod::get_progression_order)
        .def("get_progression_order_as_string", &param_cod::get_progression_order_as_string)
        .def("get_num_layers", &param_cod::get_num_layers)
        .def("is_using_color_transform", &param_cod::is_using_color_transform)
        .def("packets_may_use_sop", &param_cod::packets_may_use_sop)
        .def("packets_use_eph", &param_cod::packets_use_eph)
        .def("get_block_vertical_causality", static_cast<bool (param_cod::*)() const>(&param_cod::get_block_vertical_causality));

    nb::class_<param_qcd>(m, "ParamQcd")
        .def("set_irrev_quant", static_cast<void (param_qcd::*)(float)>(&param_qcd::set_irrev_quant), nb::arg("delta"))
        .def("set_irrev_quant", static_cast<void (param_qcd::*)(ui32, float)>(&param_qcd::set_irrev_quant), nb::arg("comp_idx"), nb::arg("delta"));

    // line_buf pointers handed out by Codestream.pull/exchange are owned by
    // the codestream; rv_policy::reference on those methods keeps Python from
    // deleting them (the job the py::nodelete holder did under pybind11).
    nb::class_<line_buf>(m, "LineBuf")
        .def(nb::init<>())

        .def_rw("size", &line_buf::size)
        .def_rw("pre_size", &line_buf::pre_size)

        // Wrapping i32 and f32 members as addresses (they live in a union)
        .def_prop_rw("i32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.i32); },  // Cast to uintptr_t to pass as integer
            [](line_buf &self, uintptr_t ptr) { self.i32 = reinterpret_cast<si32*>(ptr); }
        )  // Assign pointer back to i32
        .def_prop_rw("f32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.f32); },  // Same for float pointer
            [](line_buf &self, uintptr_t ptr) { self.f32 = reinterpret_cast<float*>(ptr); }
        )
    ;

}
