#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include <stdexcept>
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

#include <openjph/ojph_file.h>
#include <openjph/ojph_codestream.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_params.h>

namespace py = pybind11;
using namespace ojph;

namespace {

// Alignment used for read buffers. 4096 satisfies O_DIRECT sector alignment on
// every platform we target and keeps buffers page/cache-line friendly for SIMD
// decode paths.
constexpr size_t OJPH_READ_ALIGN = 4096;

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

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
      for (size_t i = 0; i < cols; ++i) {
        si32 v = ld[i];
        if (do_clip) { if (v < min_val) v = min_val; if (v > max_val) v = max_val; }
        if (is_unsigned) *reinterpret_cast<ui8*>(dst + i) = (ui8)v;
        else             *reinterpret_cast<si8*>(dst + i) = (si8)v;
      }
    } else if (element_size == 2) {
      for (size_t i = 0; i < cols; ++i) {
        si32 v = ld[i];
        if (do_clip) { if (v < min_val) v = min_val; if (v > max_val) v = max_val; }
        if (is_unsigned) *reinterpret_cast<ui16*>(dst + i * 2) = (ui16)v;
        else             *reinterpret_cast<si16*>(dst + i * 2) = (si16)v;
      }
    } else {
      for (size_t i = 0; i < cols; ++i) {
        si32 v = ld[i];
        if (do_clip) { if (v < min_val) v = min_val; if (v > max_val) v = max_val; }
        if (is_unsigned) *reinterpret_cast<ui32*>(dst + i * 4) = (ui32)v;
        else             *reinterpret_cast<si32*>(dst + i * 4) = v;
      }
    }
  }
}

}  // anonymous namespace

// ===========================================================================
// rev13 -- the reversible predict-only ("1/3") wavelet kernel
// ===========================================================================
// rev13 is the 5/3 kernel with its update step nulled, so the low-pass subband
// of every decomposition is *exactly* the even-indexed samples of the previous
// resolution.  Decoding r skipped resolutions therefore returns exactly
// ``image[::2**r, ::2**r]`` -- no interpolation, no overshoot, and no sample
// values that were absent from the original image -- while a full-resolution
// decode stays lossless.  That is what label/mask images need.
//
// Being outside Part 1, the kernel is signalled with a Part 2 ATK marker
// segment of index 2.  OpenJPH *decodes* such codestreams already (its lifting
// machinery is driven entirely by the ATK coefficients), but it cannot yet
// encode with one: there is no public way to select a kernel index above 1 and
// no ATK writer.  Adding both upstream is tracked in
// https://github.com/aous72/OpenJPH/issues/261; until that lands and ships, the
// encoder-side pieces live here so a stock OpenJPH is enough.
//
// The trick is that everything OpenJPH derives from "which kernel?" while
// writing the main header -- the reversible quantization exponents in QCD, the
// reversibility bit in CAP -- is identical for rev53 and rev13, because rev13
// is reversible and has the same subband gains.  So the codestream is built as
// an ordinary rev53 one, and only two things are changed afterwards:
//
//   1. install_rev13_kernel() below nulls the update step of the live ATK
//      object that OpenJPH drives the forward transform with, which turns the
//      5/3 analysis into rev13 analysis.  This must happen after
//      write_headers() (which is what links COD to that ATK object) and before
//      any samples are pushed.
//   2. ojph/_rev13.py rewrites the finished codestream's main header: it flips
//      SIZ-Rsiz to signal Part 2, changes COD-SPcod.wavelet_trans from 1 to 2,
//      and splices in the ATK marker segment returned by
//      rev13_atk_marker_segment() -- landing the same bytes, in the same
//      order, that a patched OpenJPH would have written.
//
// Step 1 needs the ATK object, which OpenJPH keeps private.  Rather than
// depend on a patched build, the private layouts are mirrored below and the
// pointer OpenJPH hands us is reinterpret_cast to them.  Every cast is gated on
// check_cod_layout() / the rev53 signature check in install_rev13_kernel(),
// which cross-check enough fields through the public API (and against the
// known rev53 lifting coefficients) that a mismatched OpenJPH raises instead of
// silently writing a corrupt file.  Delete this whole section once upstream
// OpenJPH exposes param_cod::set_wavelet_kern().
namespace ojph_rev13 {

// Mirrors of the private layouts in OpenJPH's
// src/core/codestream/ojph_params_local.h.  Copied verbatim (modulo names) so
// the compiler computes identical offsets, rather than hard-coding any.
struct spcod_layout {          // local::cod_SPcod
  ui8 num_decomp;
  ui8 block_width;
  ui8 block_height;
  ui8 block_style;
  ui8 wavelet_trans;
  ui8 precinct_size[33];
};

struct sgcod_layout {          // local::cod_SGcod
  ui8 prog_order;
  ui16 num_layers;
  ui8 mc_trans;
};

union lifting_step_layout {    // local::lifting_step
  struct irv_data { float Aatk; };
  struct rev_data { ui8 Eatk; si16 Batk; si16 Aatk; };
  irv_data irv;
  rev_data rev;
};

struct atk_layout {            // local::param_atk
  ui16 Latk;
  ui16 Satk;
  float Katk;
  ui8 Natk;
  lifting_step_layout* d;
  ui32 max_steps;
  lifting_step_layout d_store[6];
  atk_layout* next;
  atk_layout* top_atk;
  atk_layout* avail;
};

struct cod_layout {            // local::param_cod
  ui8 type;                    // cod_type, declared as `enum : ui8`
  ui16 Lcod;
  ui8 Scod;
  sgcod_layout SGCod;
  spcod_layout SPcod;
  cod_layout* next;
  const atk_layout* atk;
  cod_layout* top_cod;
  ui16 comp_idx;
  cod_layout* avail;
};

// ojph::param_cod is only a handle around a local::param_cod*, but that pointer
// is private.  Reach it with the standard explicit-instantiation friend trick,
// which is ordinary C++ rather than a cast through the object representation.
template <typename Tag, typename Tag::type Member>
struct private_member {
  friend typename Tag::type get_private_member(Tag) { return Member; }
};

struct cod_state_tag {
  typedef ojph::local::param_cod* ojph::param_cod::*type;
  friend type get_private_member(cod_state_tag);
};
template struct private_member<cod_state_tag, &ojph::param_cod::state>;

// Bits of param_atk::Satk, which OpenJPH reads through accessors we cannot
// call on our mirror.
enum : ui16 {
  SATK_INDEX_MASK = 0x00FF,  // the ATK index, in the low byte
  SATK_REVERSIBLE = 0x1000,
  SATK_M_INIT1    = 0x2000,  // set when synthesis starts on odd-indexed samples
};

// Everything the ATK marker segment for rev13 is made of, in one place, so the
// bytes written to the codestream and the coefficients installed into the live
// kernel cannot drift apart.
enum : ui16 {
  // ATK index; 0 and 1 are reserved for the Part 1 kernels, and OpenJPH's ATK
  // reader rejects a marker segment that claims either of them.
  KERNEL_INDEX = 2,
  // whole-sample symmetric extension | reversible | whole-sample filter |
  // 8-bit coefficients | m_init = 0, plus the index in the low byte.
  SATK = (ui16)(0x5800 | KERNEL_INDEX),
};

// The 5/3 kernel exactly as OpenJPH's param_atk::init_rev53() builds it.  Used
// as a signature: if the object we are about to modify does not look like this,
// our layout mirrors are wrong (or OpenJPH changed) and we must not touch it.
struct rev_step { si16 Aatk; si16 Batk; ui8 Eatk; };
constexpr rev_step REV53_STEPS[2] = { { 1, 2, 2 }, { -1, 1, 1 } };
constexpr ui16 REV53_SATK = 0x5801;
constexpr ui16 REV53_LATK = 13;

// rev13: the same prediction step, with the update step nulled.  The nulled
// step is kept rather than dropped because OpenJPH (and T.801) reconstruct
// starting from the even-indexed subsequence when m_init = 0, which requires an
// even number of lifting steps; `A = B = E = 0` makes the step a no-op, so this
// is mathematically a one-step predict-only kernel.
constexpr rev_step REV13_STEPS[2] = { { 0, 0, 0 }, { -1, 1, 1 } };

// The ATK marker segment for rev13, ready to splice into a main header:
//   ATK | Latk | Satk | Natk | (Eatk, Batk, LCatk, Aatk) per lifting step
// with one 8-bit coefficient per step (LCatk = 1).
inline std::string atk_marker_segment() {
  std::string s;
  auto u8 = [&s](int v) { s.push_back((char)(ui8)v); };
  auto u16 = [&u8](int v) { u8((v >> 8) & 0xFF); u8(v & 0xFF); };
  const int num_steps = 2;
  // Latk counts itself: Latk(2) + Satk(2) + Natk(1) + 5 bytes per step.
  const int Latk = 5 + 5 * num_steps;
  u16(0xFF79);                 // ATK marker
  u16(Latk);
  u16(SATK);
  u8(num_steps);               // Natk
  for (int i = 0; i < num_steps; ++i) {
    u8(REV13_STEPS[i].Eatk);
    u16((ui16)REV13_STEPS[i].Batk);
    u8(1);                     // LCatk: one coefficient per lifting step
    u8((ui8)(si8)REV13_STEPS[i].Aatk);
  }
  return s;
}

// Sanity-check the mirrored COD layout against what the public API reports.
// Anything that disagrees means the offsets below are not OpenJPH's.
inline bool check_cod_layout(const ojph::param_cod& pub, const cod_layout* m) {
  if (m == nullptr)
    return false;
  if (m->type != 1)                                     // COD_MAIN
    return false;
  if (m->SPcod.num_decomp != pub.get_num_decompositions())
    return false;
  ojph::size log_block = pub.get_log_block_dims();
  if ((ui32)m->SPcod.block_width + 2u != log_block.w)
    return false;
  if ((ui32)m->SPcod.block_height + 2u != log_block.h)
    return false;
  if (m->SGCod.prog_order != pub.get_progression_order())
    return false;
  if (m->SGCod.num_layers != pub.get_num_layers())
    return false;
  if (m->SGCod.mc_trans != (pub.is_using_color_transform() ? 1 : 0))
    return false;

  const atk_layout* atk = m->atk;
  if (atk == nullptr) {
    // No kernel linked yet, so only a Part 1 kernel can have been selected and
    // is_reversible() pins this byte to 1 (5/3) or 0 (9/7).
    if (m->SPcod.wavelet_trans != (pub.is_reversible() ? 1 : 0))
      return false;
  } else {
    // update_atk() links COD to the ATK object whose index it names, and
    // is_reversible() reads its reversibility bit. Both cross-check the
    // mirrored ATK layout too, whatever kernel is in use.
    if ((atk->Satk & SATK_INDEX_MASK) != m->SPcod.wavelet_trans)
      return false;
    if (((atk->Satk & SATK_REVERSIBLE) != 0) != pub.is_reversible())
      return false;
    if (atk->Natk == 0 || atk->d == nullptr)
      return false;
  }
  return true;
}

// Reach the mirrored COD behind a param_cod handle, or throw if it does not
// look like one. Shared by every entry point below.
inline cod_layout* access_cod_layout(const ojph::param_cod& pub) {
  ojph::local::param_cod* state = pub.*get_private_member(cod_state_tag());
  cod_layout* cod = reinterpret_cast<cod_layout*>(state);
  if (!check_cod_layout(pub, cod))
    throw std::runtime_error(
      "rev13: this OpenJPH build does not match the COD layout the rev13 "
      "support was written against; encode with wavelet='rev53' instead, or "
      "rebuild against a supported OpenJPH.");
  return cod;
}

// True when the kernel never modifies the low-pass (even-indexed) subsequence.
// With m_init = 0 the even-indexed lifting steps are the update steps, so the
// kernel is predict-only when every one of them is a no-op: a reversible step
// adds (Batk + Aatk * (x1 + x2)) >> Eatk, which is identically zero when
// Aatk == 0 and Batk >> Eatk == 0.
inline bool atk_is_predict_only(const atk_layout* atk) {
  if ((atk->Satk & SATK_REVERSIBLE) == 0 || (atk->Satk & SATK_M_INIT1) != 0)
    return false;
  for (ui32 s = 0; s < atk->Natk; s += 2)
    if (atk->d[s].rev.Aatk != 0 ||
        (atk->d[s].rev.Batk >> atk->d[s].rev.Eatk) != 0)
      return false;
  return true;
}

// Whether the codestream's kernel leaves the low-pass subband of every
// decomposition equal to the even-indexed samples of the previous resolution,
// so that resolution level r decodes to the image subsampled by 2^r.
//
// Like upstream's param_cod::is_predict_only(), this inspects the lifting steps
// signalled in the ATK marker segment, never the kernel index: a Part 2 index
// is file-local and says nothing on its own. OpenJPH's own test corpus has a
// codestream using index 2 for an *irreversible* 5/3, which must report false.
inline bool is_predict_only(const ojph::param_cod& pub) {
  const cod_layout* cod = access_cod_layout(pub);
  if (cod->atk == nullptr)
    // Encoding, before write_headers() associates an ATK object with this COD.
    // Nothing describes the kernel yet but the index, and rev13 is the only
    // kernel beyond Part 1 this package can encode with.
    return cod->SPcod.wavelet_trans == KERNEL_INDEX;
  // Upstream short-circuits a Part 1 index to false here. Asking the lifting
  // steps instead gives the same answer for both Part 1 kernels -- 5/3's update
  // step has Aatk = 1, and 9/7 is not reversible -- and it stays right for a
  // codestream mid-encode, whose COD still names 5/3 while
  // install_rev13_kernel() has already nulled that update step.
  return atk_is_predict_only(cod->atk);
}

// Turn the live 5/3 analysis kernel into rev13 by nulling its update step.
// Call after write_headers() and before pushing any samples.
inline void install_rev13_kernel(ojph::codestream& cs) {
  ojph::param_cod pub = cs.access_cod();
  cod_layout* cod = access_cod_layout(pub);

  // COD is linked to a kernel by write_headers(), so a null pointer here means
  // the caller has the ordering wrong rather than that anything is broken.
  atk_layout* atk = const_cast<atk_layout*>(cod->atk);
  if (atk == nullptr)
    throw std::runtime_error(
      "rev13: no wavelet kernel is linked yet; write_headers() must be "
      "called before install_rev13_wavelet().");
  if (atk->Satk != REV53_SATK || atk->Latk != REV53_LATK || atk->Natk != 2 ||
      atk->d == nullptr)
    throw std::runtime_error(
      "rev13: the codestream is not using the reversible 5/3 kernel that "
      "rev13 is derived from.");
  for (int i = 0; i < 2; ++i)
    if (atk->d[i].rev.Aatk != REV53_STEPS[i].Aatk ||
        atk->d[i].rev.Batk != REV53_STEPS[i].Batk ||
        atk->d[i].rev.Eatk != REV53_STEPS[i].Eatk)
      throw std::runtime_error(
        "rev13: the 5/3 lifting coefficients in this OpenJPH build are not "
        "the ones the rev13 encoder was written against.");

  for (int i = 0; i < 2; ++i) {
    atk->d[i].rev.Aatk = REV13_STEPS[i].Aatk;
    atk->d[i].rev.Batk = REV13_STEPS[i].Batk;
    atk->d[i].rev.Eatk = REV13_STEPS[i].Eatk;
  }
}

}  // namespace ojph_rev13

// Declare that this module runs without the GIL, so a free-threaded CPython
// (3.13t/3.14t) keeps the GIL disabled instead of silently re-enabling it at
// import time with a RuntimeWarning. This is safe here because the module holds
// no mutable global state: every binding works on per-instance or stack-local
// C++ objects, and OpenJPH's lazily built lookup tables are guarded by
// std::call_once. As everywhere else in Python, a *single* Codestream / infile /
// outfile object must still not be shared across threads without external
// synchronisation -- the guarantee is that independent objects in different
// threads do not interfere.
//
// py::mod_gil_not_used() needs pybind11 >= 2.13. Building with an older
// pybind11 still works; the module just falls back to re-enabling the GIL.
#if defined(PYBIND11_VERSION_HEX) && PYBIND11_VERSION_HEX >= 0x020D0000
PYBIND11_MODULE(ojph_bindings, m, py::mod_gil_not_used()) {
#else
PYBIND11_MODULE(ojph_bindings, m) {
#endif
    py::class_<infile_base>(m, "InfileBase")
        .def("read", &infile_base::read, py::call_guard<py::gil_scoped_release>())
        .def("seek", &infile_base::seek)
        .def("tell", &infile_base::tell)
        .def("eof", &infile_base::eof)
        .def("close", &infile_base::close);

    py::class_<j2c_infile, infile_base>(m, "J2CInfile")
        .def(py::init<>())
        .def("open", &j2c_infile::open)
        .def("read", &j2c_infile::read, py::call_guard<py::gil_scoped_release>())
        .def("seek", [](infile_base& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum infile_base::seek>(origin));
        })
        .def("tell", &j2c_infile::tell)
        .def("eof", &j2c_infile::eof)
        .def("close", &j2c_infile::close);

    py::class_<mem_infile, infile_base>(m, "MemInfile")
        .def(py::init<>())
        .def("open", [](mem_infile& self, py::array_t<ui8> data) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw py::value_error("Data must be a 1-dimensional array");
            }
            self.open(static_cast<const ui8*>(buf.ptr), buf.size);
        }, py::arg("data"))
        .def("open", [](mem_infile& self, const ui8* data, size_t size) {
            self.open(data, size);
        }, py::arg("data"), py::arg("size"))
        .def("read", [](mem_infile& self, py::array_t<ui8> buffer, size_t size) {
            py::buffer_info buf = buffer.request();
            if (buf.ndim != 1) {
                throw py::value_error("Buffer must be a 1-dimensional array");
            }
            if (buf.size < size) {
                throw py::value_error("Buffer size is smaller than requested read size");
            }
            py::gil_scoped_release release;
            return self.read(static_cast<void*>(buf.ptr), size);
        }, py::arg("buffer"), py::arg("size"))
        .def("seek", [](mem_infile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum infile_base::seek>(origin));
        })
        .def("tell", &mem_infile::tell)
        .def("eof", &mem_infile::eof)
        .def("close", &mem_infile::close);


    py::class_<outfile_base>(m, "outfileBase")
        .def("write", &outfile_base::write, py::call_guard<py::gil_scoped_release>())
        .def("seek", &outfile_base::seek)
        .def("tell", &outfile_base::tell)
        .def("close", &outfile_base::close);

    py::class_<j2c_outfile, outfile_base>(m, "J2COutfile")
        .def(py::init<>())
        .def("open", &j2c_outfile::open)
        .def("write", &j2c_outfile::write, py::call_guard<py::gil_scoped_release>())
        .def("tell", &j2c_outfile::tell)
        .def("close", &j2c_outfile::close);

    py::class_<mem_outfile, outfile_base>(m, "MemOutfile")
        .def(py::init<>())
        .def("open", &mem_outfile::open, py::arg("initial_size") = 65536, py::arg("clear_mem") = false)
        .def("write", &mem_outfile::write, py::call_guard<py::gil_scoped_release>())
        .def("tell", &mem_outfile::tell)
        .def("get_used_size", &mem_outfile::get_used_size)
        .def("get_buf_size", &mem_outfile::get_buf_size)
        .def("seek", [](mem_outfile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum outfile_base::seek>(origin));
        })
        .def("close", &mem_outfile::close)
        .def("write_to_file", &mem_outfile::write_to_file, py::call_guard<py::gil_scoped_release>())
        .def("get_data", [](mem_outfile& self) {
            const ui8* data = self.get_data();
            si64 size = self.tell();
            return py::memoryview::from_memory(data, size);
        }, py::keep_alive<0, 1>());

    // Bindings for codestream class
    py::class_<codestream>(m, "Codestream")
        .def(py::init<>())
        .def("set_planar", &codestream::set_planar)
        .def("set_profile", &codestream::set_profile)
        .def("set_tilepart_divisions", &codestream::set_tilepart_divisions)
        .def("is_tilepart_division_at_resolutions", &codestream::is_tilepart_division_at_resolutions)
        .def("is_tilepart_division_at_components", &codestream::is_tilepart_division_at_components)
        .def("request_tlm_marker", &codestream::request_tlm_marker)
        .def("is_tlm_requested", &codestream::is_tlm_requested)
        .def("write_headers",
             [](codestream &self, outfile_base *file, py::object comments, ui32 num_comments) {
                 // Check if the comments argument is None and convert it to nullptr if so
                 const comment_exchange* comments_ptr = comments.is_none() ? nullptr : comments.cast<const comment_exchange*>();
                 py::gil_scoped_release release;
                 self.write_headers(file, comments_ptr, num_comments);
             },
             py::arg("file"), py::arg("comments") = py::none(), py::arg("num_comments") = 0)
        // Switch the forward transform to the rev13 (reversible predict-only)
        // kernel. Must be called after write_headers() and before any samples
        // are pushed; the resulting codestream still needs the main-header
        // fixups in ojph/_rev13.py to declare the kernel it actually used.
        .def("install_rev13_wavelet",
             [](codestream &self) { ojph_rev13::install_rev13_kernel(self); })
        .def("exchange",
             [](codestream &self, py::object line_buf_obj, ui32 &next_component) -> line_buf* {
                 line_buf* buf = nullptr;
                 if (!line_buf_obj.is_none()) {
                     buf = line_buf_obj.cast<line_buf*>();
                 }
                 return self.exchange(buf, next_component);
             },
             py::arg("line_buf_obj") = py::none(), py::arg("next_component") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("push_all_components",
             [](codestream &self, py::array image, ui32 num_components, const std::string& channel_order) {
                 py::buffer_info buf = image.request();

                 const std::string& fmt = buf.format;
                 char format_char = fmt.empty() ? '\0' : (fmt.size() > 1 && (fmt[0] == '<' || fmt[0] == '>' || fmt[0] == '=' || fmt[0] == '|')) ? fmt[1] : fmt[0];
                 size_t element_size = buf.itemsize;

                 size_t height, width;
                 size_t component_stride;

                 if (num_components == 1) {
                     if (buf.ndim == 2) {
                         height = buf.shape[0];
                         width = buf.shape[1];
                         component_stride = 0;
                     } else if (buf.ndim == 3 && buf.shape[2] == 1) {
                         height = buf.shape[0];
                         width = buf.shape[1];
                         component_stride = 0;
                     } else {
                         throw py::value_error("Image must be 2-dimensional or 3-dimensional with last dimension of 1 for single component");
                     }
                 } else {
                     if (buf.ndim != 3) {
                         throw py::value_error("Image must be 3-dimensional for multiple components");
                     }
                     if (channel_order == "CHW") {
                         height = buf.shape[1];
                         width = buf.shape[2];
                         component_stride = buf.strides[0];
                     } else {
                         height = buf.shape[0];
                         width = buf.shape[1];
                         component_stride = buf.strides[2];
                     }
                 }

                 size_t row_stride, col_stride;
                 if (num_components == 1) {
                     row_stride = buf.strides[0];
                     col_stride = buf.strides[1];
                 } else if (channel_order == "CHW") {
                     row_stride = buf.strides[1];
                     col_stride = buf.strides[2];
                 } else {
                     row_stride = buf.strides[0];
                     col_stride = buf.strides[1];
                 }

                 {
                     py::gil_scoped_release release;
                     ui32 next_comp = 0;
                     ui32& next_comp_ref = next_comp;
                     line_buf* line = self.exchange(nullptr, next_comp_ref);

                     for (ui32 c = 0; c < num_components; ++c) {
                         char* component_base = static_cast<char*>(buf.ptr);
                         if (num_components > 1) {
                             if (channel_order == "CHW") {
                                 component_base += c * component_stride;
                             } else {
                                 component_base += c * component_stride;
                             }
                         }

                         for (size_t h = 0; h < height; ++h) {
                             char* row_start = component_base + h * row_stride;
                             si32* line_data = line->i32;
                             size_t line_size = line->size;

                             if (line_size != width) {
                                 throw py::value_error("Line size mismatch");
                             }

                             if (element_size == 1) {
                                 if (format_char == 'B') {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const ui8*>(row_start + i * col_stride));
                                     }
                                 } else {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const si8*>(row_start + i * col_stride));
                                     }
                                 }
                             } else if (element_size == 2) {
                                 if (format_char == 'H') {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const ui16*>(row_start + i * col_stride));
                                     }
                                 } else {
                                     for (size_t i = 0; i < line_size; ++i) {
                                         line_data[i] = static_cast<si32>(*reinterpret_cast<const si16*>(row_start + i * col_stride));
                                     }
                                 }
                             } else {
                                 if (format_char == 'I' || format_char == 'L') {
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
             },
             py::arg("image"), py::arg("num_components"), py::arg("channel_order"))
        .def("flush", &codestream::flush, py::call_guard<py::gil_scoped_release>())
        .def("enable_resilience", &codestream::enable_resilience)
        .def("read_headers", &codestream::read_headers, py::call_guard<py::gil_scoped_release>())
        .def("restrict_input_resolution", &codestream::restrict_input_resolution)
        .def("create", &codestream::create, py::call_guard<py::gil_scoped_release>())
        .def("pull", &codestream::pull, py::call_guard<py::gil_scoped_release>())
        .def("pull_all_components",
             [](codestream &self, py::array output, ui32 num_components, const std::string& channel_order, py::object min_val_obj, py::object max_val_obj) {
                py::buffer_info buf = output.request();

                bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
                si32 min_val = 0;
                si32 max_val = 0;
                if (do_clip) {
                    min_val = min_val_obj.cast<si32>();
                    max_val = max_val_obj.cast<si32>();
                }

                const std::string& fmt = buf.format;
                char format_char = fmt.empty() ? '\0' : (fmt.size() > 1 && (fmt[0] == '<' || fmt[0] == '>' || fmt[0] == '=' || fmt[0] == '|')) ? fmt[1] : fmt[0];
                bool is_unsigned = (format_char == 'B' || format_char == 'H' || format_char == 'I' || format_char == 'L');
                 size_t element_size = buf.itemsize;

                 size_t height, width;
                 size_t component_stride;

                 if (num_components == 1) {
                     if (buf.ndim != 2) {
                         throw py::value_error("Output must be 2-dimensional for single component");
                     }
                     height = buf.shape[0];
                     width = buf.shape[1];
                     component_stride = 0;
                 } else {
                     if (buf.ndim != 3) {
                         throw py::value_error("Output must be 3-dimensional for multiple components");
                     }
                     if (channel_order == "CHW") {
                         height = buf.shape[1];
                         width = buf.shape[2];
                         component_stride = buf.strides[0];
                     } else {
                         height = buf.shape[0];
                         width = buf.shape[1];
                         component_stride = buf.strides[2];
                     }
                 }

                 size_t row_stride = (num_components == 1 || channel_order == "CHW") ? buf.strides[buf.ndim - 2] : buf.strides[0];
                 size_t col_stride = (num_components == 1 || channel_order == "CHW") ? buf.strides[buf.ndim - 1] : buf.strides[1];

                 {
                     py::gil_scoped_release release;
                     for (ui32 c = 0; c < num_components; ++c) {
                         char* component_base = static_cast<char*>(buf.ptr);
                         if (num_components > 1) {
                             if (channel_order == "CHW") {
                                 component_base += c * component_stride;
                             } else {
                                 component_base += c * component_stride;
                             }
                         }

                         line_buf* first_line = self.pull(c);
                         size_t line_size = first_line->size;
                         if (line_size != width) {
                             throw py::value_error("Line size mismatch");
                         }

                         for (size_t h = 0; h < height; ++h) {
                             line_buf* line = (h == 0) ? first_line : self.pull(c);
                             si32* line_data = line->i32;
                             char* out_row_start = component_base + h * row_stride;

                             if (do_clip) {
                                 if (element_size == 1) {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<ui8*>(out_row_start + i * col_stride) = static_cast<ui8>(val);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<si8*>(out_row_start + i * col_stride) = static_cast<si8>(val);
                                         }
                                     }
                                 } else if (element_size == 2) {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<ui16*>(out_row_start + i * col_stride) = static_cast<ui16>(val);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<si16*>(out_row_start + i * col_stride) = static_cast<si16>(val);
                                         }
                                     }
                                 } else {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<ui32*>(out_row_start + i * col_stride) = static_cast<ui32>(val);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             si32 val = line_data[i];
                                             if (val < min_val) val = min_val;
                                             if (val > max_val) val = max_val;
                                             *reinterpret_cast<si32*>(out_row_start + i * col_stride) = val;
                                         }
                                     }
                                 }
                             } else {
                                 if (element_size == 1) {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<ui8*>(out_row_start + i * col_stride) = static_cast<ui8>(line_data[i]);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<si8*>(out_row_start + i * col_stride) = static_cast<si8>(line_data[i]);
                                         }
                                     }
                                 } else if (element_size == 2) {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<ui16*>(out_row_start + i * col_stride) = static_cast<ui16>(line_data[i]);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<si16*>(out_row_start + i * col_stride) = static_cast<si16>(line_data[i]);
                                         }
                                     }
                                 } else {
                                     if (is_unsigned) {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<ui32*>(out_row_start + i * col_stride) = static_cast<ui32>(line_data[i]);
                                         }
                                     } else {
                                         for (size_t i = 0; i < line_size; ++i) {
                                             *reinterpret_cast<si32*>(out_row_start + i * col_stride) = line_data[i];
                                         }
                                     }
                                 }
                             }
                         }
                     }
                 }
             },
             py::arg("output"), py::arg("num_components"), py::arg("channel_order"), py::arg("min_val") = py::none(), py::arg("max_val") = py::none())
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
    // into ``out``) under one ``py::gil_scoped_release``. This is the key to
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
        [](py::array_t<ui8> data, py::array out, int level,
           py::object min_val_obj, py::object max_val_obj)
            -> std::pair<ui32, ui32> {
            py::buffer_info in = data.request();
            py::buffer_info ob = out.request();
            if (ob.ndim != 2)
                throw py::value_error("out must be 2-dimensional (single component)");
            if (!(out.flags() & py::array::c_style))
                throw py::value_error("out must be C-contiguous");

            const ui8* data_ptr = static_cast<const ui8*>(in.ptr);
            size_t data_size = (size_t)in.shape[0];
            char* out_ptr = static_cast<char*>(ob.ptr);
            size_t out_rows = (size_t)ob.shape[0];
            size_t out_cols = (size_t)ob.shape[1];
            size_t row_stride = (size_t)ob.strides[0];
            size_t element_size = (size_t)ob.itemsize;

            const std::string& fmt = ob.format;
            char fc = fmt.empty() ? '\0'
                : (fmt.size() > 1 && (fmt[0]=='<'||fmt[0]=='>'||fmt[0]=='='||fmt[0]=='|'))
                    ? fmt[1] : fmt[0];
            bool is_unsigned = (fc=='B'||fc=='H'||fc=='I'||fc=='L');

            bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
            si32 min_val = do_clip ? min_val_obj.cast<si32>() : 0;
            si32 max_val = do_clip ? max_val_obj.cast<si32>() : 0;

            ui32 h = 0, w = 0;
            {
                py::gil_scoped_release release;

                mem_infile infile;
                infile.open(data_ptr, data_size);
                codestream cs;
                cs.read_headers(&infile);
                cs.restrict_input_resolution((ui32)level, (ui32)level);
                param_siz siz = cs.access_siz();
                h = siz.get_recon_height(0);
                w = siz.get_recon_width(0);
                bool shape_ok = ((size_t)h == out_rows && (size_t)w == out_cols);
                if (shape_ok) {
                    cs.create();
                    pull_single_component_into(
                        cs, out_ptr, out_rows, out_cols, row_stride,
                        element_size, is_unsigned, do_clip, min_val, max_val);
                }
                cs.close();
                if (!shape_ok)
                    throw py::value_error("out shape does not match decoded level shape");
            }
            return std::make_pair(h, w);
        },
        py::arg("data"), py::arg("out"), py::arg("level"),
        py::arg("min_val") = py::none(), py::arg("max_val") = py::none());

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
            -> py::tuple {
            constexpr size_t PEEK = 65536;
            ui32 nd = 0, h = 0, w = 0;
            bool ok = false;
            {
                py::gil_scoped_release release;
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
                throw py::value_error("peek_j2c_fd: failed to read codestream header");
            return py::make_tuple(nd, h, w);
        },
        py::arg("fd"), py::arg("offset"), py::arg("nbytes"),
        py::arg("o_direct") = false);

    // -----------------------------------------------------------------------
    // read_j2c_fd_into: the whole reduced-resolution read for one codestream,
    // GIL-free, straight from a file descriptor.
    //
    // Under a single py::gil_scoped_release it: aligned-reads the header, parses
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
        [](int fd, int64_t offset, int64_t nbytes, py::array out, int level,
           py::object min_val_obj, py::object max_val_obj, bool o_direct)
            -> std::pair<ui32, ui32> {
            py::buffer_info ob = out.request();
            if (ob.ndim != 2)
                throw py::value_error("out must be 2-dimensional (single component)");
            if (!(out.flags() & py::array::c_style))
                throw py::value_error("out must be C-contiguous");

            char* out_ptr = static_cast<char*>(ob.ptr);
            size_t out_rows = (size_t)ob.shape[0];
            size_t out_cols = (size_t)ob.shape[1];
            size_t row_stride = (size_t)ob.strides[0];
            size_t element_size = (size_t)ob.itemsize;
            const std::string& fmt = ob.format;
            char fc = fmt.empty() ? '\0'
                : (fmt.size() > 1 && (fmt[0]=='<'||fmt[0]=='>'||fmt[0]=='='||fmt[0]=='|'))
                    ? fmt[1] : fmt[0];
            bool is_unsigned = (fc=='B'||fc=='H'||fc=='I'||fc=='L');
            bool do_clip = !min_val_obj.is_none() && !max_val_obj.is_none();
            si32 min_val = do_clip ? min_val_obj.cast<si32>() : 0;
            si32 max_val = do_clip ? max_val_obj.cast<si32>() : 0;

            ui32 h = 0, w = 0;
            const char* err = nullptr;
            {
                py::gil_scoped_release release;

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
                throw py::value_error(err);
            return std::make_pair(h, w);
        },
        py::arg("fd"), py::arg("offset"), py::arg("nbytes"), py::arg("out"),
        py::arg("level"), py::arg("min_val") = py::none(),
        py::arg("max_val") = py::none(), py::arg("o_direct") = false);

    py::class_<point>(m, "Point")
        .def(py::init<ui32, ui32>(), py::arg("x") = 0, py::arg("y") = 0)  // Constructor with default args
        .def_readwrite("x", &point::x)
        .def_readwrite("y", &point::y);

    py::class_<size>(m, "Size")
        .def(py::init<ui32, ui32>(), py::arg("w") = 0, py::arg("h") = 0)  // Constructor with default args
        .def_readwrite("w", &size::w)  // width
        .def_readwrite("h", &size::h)  // height
        .def("area", &size::area);     // Expose the area function

    py::class_<param_siz>(m, "ParamSiz")
        // .def(py::init<local::param_siz*>())  // Constructor with local::param_siz* argument
        .def("is_signed", &param_siz::is_signed)

        .def("set_image_extent", &param_siz::set_image_extent)
        .def("set_tile_size", &param_siz::set_tile_size)
        .def("set_image_offset", &param_siz::set_image_offset)
        .def("set_tile_offset", &param_siz::set_tile_offset)
        .def("set_num_components", &param_siz::set_num_components)
        .def("set_component", &param_siz::set_component, py::arg("comp_num"), py::arg("downsampling"), py::arg("bit_depth"), py::arg("is_signed"))

        .def("get_image_extent", &param_siz::get_image_extent)
        .def("get_image_offset", &param_siz::get_image_offset)
        .def("get_tile_size", &param_siz::get_tile_size)
        .def("get_tile_offset", &param_siz::get_tile_offset)
        .def("get_num_components", &param_siz::get_num_components)
        .def("get_bit_depth", &param_siz::get_bit_depth)
        .def("get_downsampling", &param_siz::get_downsampling)
        .def("get_recon_width", &param_siz::get_recon_width)
        .def("get_recon_height", &param_siz::get_recon_height);

    py::class_<param_cod>(m, "ParamCod")
        // .def(py::init<local::param_cod*>())

        // OpenJPH >= 0.30.1 (post-release) added COC-segment overloads that take
        // a leading comp_idx argument, so these COD-segment (no comp_idx) methods
        // must be disambiguated with an explicit member-function-pointer cast.
        .def("set_num_decomposition", static_cast<void (param_cod::*)(ui32)>(&param_cod::set_num_decomposition), py::arg("num_decompositions"))
        .def("set_block_dims", static_cast<void (param_cod::*)(ui32, ui32)>(&param_cod::set_block_dims), py::arg("width"), py::arg("height"))
        .def("set_precinct_size", static_cast<void (param_cod::*)(int, size*)>(&param_cod::set_precinct_size), py::arg("num_levels"), py::arg("precinct_size"))
        .def("set_progression_order", &param_cod::set_progression_order, py::arg("name"))
        .def("set_color_transform", &param_cod::set_color_transform, py::arg("color_transform"))
        .def("set_reversible", static_cast<void (param_cod::*)(bool)>(&param_cod::set_reversible), py::arg("reversible"))

        .def("get_num_decompositions", static_cast<ui32 (param_cod::*)() const>(&param_cod::get_num_decompositions))
        .def("get_block_dims", static_cast<size (param_cod::*)() const>(&param_cod::get_block_dims))
        .def("get_log_block_dims", static_cast<size (param_cod::*)() const>(&param_cod::get_log_block_dims))
        .def("is_reversible", static_cast<bool (param_cod::*)() const>(&param_cod::is_reversible))
        // True when the employed kernel has no effective update steps, so the
        // low-pass subband of each decomposition holds the even-indexed samples
        // of the previous resolution untouched; with a reversible kernel and
        // lossless coding, resolution level r decodes to the image subsampled
        // by 2^r. Decided from the lifting steps signalled in the ATK marker
        // segment, so it is reliable for codestreams from other encoders too.
        .def("is_predict_only",
             [](const param_cod& self) {
                 return ojph_rev13::is_predict_only(self);
             })
        .def("get_precinct_size", static_cast<size (param_cod::*)(ui32) const>(&param_cod::get_precinct_size), py::arg("level_num"))
        .def("get_log_precinct_size", static_cast<size (param_cod::*)(ui32) const>(&param_cod::get_log_precinct_size), py::arg("level_num"))
        .def("get_progression_order", &param_cod::get_progression_order)
        .def("get_progression_order_as_string", &param_cod::get_progression_order_as_string)
        .def("get_num_layers", &param_cod::get_num_layers)
        .def("is_using_color_transform", &param_cod::is_using_color_transform)
        .def("packets_may_use_sop", &param_cod::packets_may_use_sop)
        .def("packets_use_eph", &param_cod::packets_use_eph)
        .def("get_block_vertical_causality", static_cast<bool (param_cod::*)() const>(&param_cod::get_block_vertical_causality));

    // rev13 main-header pieces, exported so ojph/_rev13.py splices exactly the
    // bytes that describe the kernel install_rev13_wavelet() actually applied.
    m.attr("REV13_WAVELET_INDEX") = (ui32)ojph_rev13::KERNEL_INDEX;
    m.def("rev13_atk_marker_segment",
          []() { return py::bytes(ojph_rev13::atk_marker_segment()); },
          "The Part 2 ATK marker segment (marker included) describing the "
          "rev13 reversible predict-only wavelet kernel.");

    py::class_<param_qcd>(m, "ParamQcd")
        .def("set_irrev_quant", static_cast<void (param_qcd::*)(float)>(&param_qcd::set_irrev_quant), py::arg("delta"))
        .def("set_irrev_quant", static_cast<void (param_qcd::*)(ui32, float)>(&param_qcd::set_irrev_quant), py::arg("comp_idx"), py::arg("delta"));

    py::class_<line_buf, std::unique_ptr<line_buf, py::nodelete>>(m, "LineBuf")
        .def(py::init<>())

        .def_readwrite("size", &line_buf::size)
        .def_readwrite("pre_size", &line_buf::pre_size)

        // Wrapping i32 and f32 members as properties (since they are in a union)
        // .def_property("i32", [](line_buf &self) { return self.i32; }, [](line_buf &self, si32* ptr) { self.i32 = ptr; })
        // .def_property("f32", [](line_buf &self) { return self.f32; }, [](line_buf &self, float* ptr) { self.f32 = ptr; })
        // Wrapping i32 and f32 members as addresses (pointers)
        .def_property("i32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.i32); },  // Cast to uintptr_t to pass as integer
            [](line_buf &self, uintptr_t ptr) { self.i32 = reinterpret_cast<si32*>(ptr); }
        )  // Assign pointer back to i32
        .def_property("f32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.f32); },  // Same for float pointer
            [](line_buf &self, uintptr_t ptr) { self.f32 = reinterpret_cast<float*>(ptr); }
        )

        // Explicit instantiations for pre_alloc, finalize_alloc, and wrap for int and float
        // .def("pre_alloc_int", &line_buf::pre_alloc<int>, py::arg("allocator"), py::arg("num_ele"), py::arg("pre_size"))
        // .def("pre_alloc_float", &line_buf::pre_alloc<float>, py::arg("allocator"), py::arg("num_ele"), py::arg("pre_size"))

        // .def("finalize_alloc_int", &line_buf::finalize_alloc<int>, py::arg("allocator"))
        // .def("finalize_alloc_float", &line_buf::finalize_alloc<float>, py::arg("allocator"))

        // .def("wrap_int", &line_buf::wrap<int>, py::arg("buffer"), py::arg("num_ele"), py::arg("pre_size"))
        // .def("wrap_float", &line_buf::wrap<float>, py::arg("buffer"), py::arg("num_ele"), py::arg("pre_size"))
    ;

}
