#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include <string>

#include <openjph/ojph_file.h>
#include <openjph/ojph_codestream.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_params.h>

namespace py = pybind11;
using namespace ojph;

PYBIND11_MODULE(ojph_bindings, m) {
    py::class_<infile_base>(m, "InfileBase")
        .def("read", &infile_base::read)
        .def("seek", &infile_base::seek)
        .def("tell", &infile_base::tell)
        .def("eof", &infile_base::eof)
        .def("close", &infile_base::close);

    py::class_<j2c_infile, infile_base>(m, "J2CInfile")
        .def(py::init<>())
        .def("open", &j2c_infile::open)
        .def("read", &j2c_infile::read)
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
            return self.read(static_cast<void*>(buf.ptr), size);
        }, py::arg("buffer"), py::arg("size"))
        .def("seek", [](mem_infile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum infile_base::seek>(origin));
        })
        .def("tell", &mem_infile::tell)
        .def("eof", &mem_infile::eof)
        .def("close", &mem_infile::close);


    py::class_<outfile_base>(m, "outfileBase")
        .def("write", &outfile_base::write)
        .def("seek", &outfile_base::seek)
        .def("tell", &outfile_base::tell)
        .def("close", &outfile_base::close);

    py::class_<j2c_outfile, outfile_base>(m, "J2COutfile")
        .def(py::init<>())
        .def("open", &j2c_outfile::open)
        .def("write", &j2c_outfile::write)
        .def("tell", &j2c_outfile::tell)
        .def("close", &j2c_outfile::close);

    py::class_<mem_outfile, outfile_base>(m, "MemOutfile")
        .def(py::init<>())
        .def("open", &mem_outfile::open, py::arg("initial_size") = 65536, py::arg("clear_mem") = false)
        .def("write", &mem_outfile::write)
        .def("tell", &mem_outfile::tell)
        .def("get_used_size", &mem_outfile::get_used_size)
        .def("get_buf_size", &mem_outfile::get_buf_size)
        .def("seek", [](mem_outfile& self, si64 offset, int origin) {
            return self.seek(offset, static_cast<enum outfile_base::seek>(origin));
        })
        .def("close", &mem_outfile::close)
        .def("write_to_file", &mem_outfile::write_to_file)
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
                 self.write_headers(file, comments_ptr, num_comments);
             },
             py::arg("file"), py::arg("comments") = py::none(), py::arg("num_comments") = 0)
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

                 char format_char = buf.format[0];
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
        .def("flush", &codestream::flush)
        .def("enable_resilience", &codestream::enable_resilience)
        .def("read_headers", &codestream::read_headers)
        .def("restrict_input_resolution", &codestream::restrict_input_resolution)
        .def("create", &codestream::create)
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

                 char format_char = buf.format[0];
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

        .def("set_num_decomposition", &param_cod::set_num_decomposition, py::arg("num_decompositions"))
        .def("set_block_dims", &param_cod::set_block_dims, py::arg("width"), py::arg("height"))
        .def("set_precinct_size", &param_cod::set_precinct_size, py::arg("num_levels"), py::arg("precinct_size"))
        .def("set_progression_order", &param_cod::set_progression_order, py::arg("name"))
        .def("set_color_transform", &param_cod::set_color_transform, py::arg("color_transform"))
        .def("set_reversible", &param_cod::set_reversible, py::arg("reversible"))

        .def("get_num_decompositions", &param_cod::get_num_decompositions)
        .def("get_block_dims", &param_cod::get_block_dims)
        .def("get_log_block_dims", &param_cod::get_log_block_dims)
        .def("is_reversible", &param_cod::is_reversible)
        .def("get_precinct_size", &param_cod::get_precinct_size, py::arg("level_num"))
        .def("get_log_precinct_size", &param_cod::get_log_precinct_size, py::arg("level_num"))
        .def("get_progression_order", &param_cod::get_progression_order)
        .def("get_progression_order_as_string", &param_cod::get_progression_order_as_string)
        .def("get_num_layers", &param_cod::get_num_layers)
        .def("is_using_color_transform", &param_cod::is_using_color_transform)
        .def("packets_may_use_sop", &param_cod::packets_may_use_sop)
        .def("packets_use_eph", &param_cod::packets_use_eph)
        .def("get_block_vertical_causality", &param_cod::get_block_vertical_causality);

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
