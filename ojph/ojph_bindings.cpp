#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


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
        .def("write_headers", &codestream::write_headers, py::arg("file"), py::arg("comments") = nullptr, py::arg("num_comments") = 0)
        .def("exchange", &codestream::exchange)
        .def("flush", &codestream::flush)
        .def("enable_resilience", &codestream::enable_resilience)
        .def("read_headers", &codestream::read_headers)
        .def("restrict_input_resolution", &codestream::restrict_input_resolution)
        .def("create", &codestream::create)
        .def("pull", &codestream::pull)
        .def("close", &codestream::close)
        .def("access_siz", &codestream::access_siz)
        .def("access_cod", &codestream::access_cod)
        .def("access_qcd", &codestream::access_qcd)
        .def("access_nlt", &codestream::access_nlt)
        .def("is_planar", &codestream::is_planar);

    // Exposing point struct
    py::class_<point>(m, "Point")
        .def(py::init<ui32, ui32>(), py::arg("x") = 0, py::arg("y") = 0)  // Constructor with default args
        .def_readwrite("x", &point::x)
        .def_readwrite("y", &point::y);

    // Exposing size struct
    py::class_<size>(m, "Size")
        .def(py::init<ui32, ui32>(), py::arg("w") = 0, py::arg("h") = 0)  // Constructor with default args
        .def_readwrite("w", &size::w)  // width
        .def_readwrite("h", &size::h)  // height
        .def("area", &size::area);     // Expose the area function

    // Wrapping param_siz
    py::class_<param_siz>(m, "ParamSiz")
        // .def(py::init<local::param_siz*>())  // Constructor with local::param_siz* argument
        // Setters
        .def("set_image_extent", &param_siz::set_image_extent)
        .def("set_tile_size", &param_siz::set_tile_size)
        .def("set_image_offset", &param_siz::set_image_offset)
        .def("set_tile_offset", &param_siz::set_tile_offset)
        .def("set_num_components", &param_siz::set_num_components)
        .def("set_component", &param_siz::set_component, py::arg("comp_num"), py::arg("downsampling"), py::arg("bit_depth"), py::arg("is_signed"))
        // Getters
        .def("get_image_extent", &param_siz::get_image_extent)
        .def("get_image_offset", &param_siz::get_image_offset)
        .def("get_tile_size", &param_siz::get_tile_size)
        .def("get_tile_offset", &param_siz::get_tile_offset)
        .def("get_num_components", &param_siz::get_num_components)
        .def("get_bit_depth", &param_siz::get_bit_depth)
        .def("is_signed", &param_siz::is_signed)
        .def("get_downsampling", &param_siz::get_downsampling)
        // Deeper getters
        .def("get_recon_width", &param_siz::get_recon_width)
        .def("get_recon_height", &param_siz::get_recon_height);

    py::class_<line_buf>(m, "LineBuf")
        .def(py::init<>())  // Default constructor

        // Wrapping size and pre_size members
        .def_readwrite("size", &line_buf::size)
        .def_readwrite("pre_size", &line_buf::pre_size)

        // Wrapping i32 and f32 members as properties (since they are in a union)
        // .def_property("i32", [](line_buf &self) { return self.i32; }, [](line_buf &self, si32* ptr) { self.i32 = ptr; })
        // .def_property("f32", [](line_buf &self) { return self.f32; }, [](line_buf &self, float* ptr) { self.f32 = ptr; })
        // Wrapping i32 and f32 members as addresses (pointers)
        .def_property("i32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.i32); },  // Cast to uintptr_t to pass as integer
            [](line_buf &self, uintptr_t ptr) { self.i32 = reinterpret_cast<si32*>(ptr); })  // Assign pointer back to i32
        .def_property("f32_address",
            [](line_buf &self) { return reinterpret_cast<uintptr_t>(self.f32); },  // Same for float pointer
            [](line_buf &self, uintptr_t ptr) { self.f32 = reinterpret_cast<float*>(ptr); })

        // Explicit instantiations for pre_alloc, finalize_alloc, and wrap for int and float
        .def("pre_alloc_int", &line_buf::pre_alloc<int>, py::arg("allocator"), py::arg("num_ele"), py::arg("pre_size"))
        .def("pre_alloc_float", &line_buf::pre_alloc<float>, py::arg("allocator"), py::arg("num_ele"), py::arg("pre_size"))

        .def("finalize_alloc_int", &line_buf::finalize_alloc<int>, py::arg("allocator"))
        .def("finalize_alloc_float", &line_buf::finalize_alloc<float>, py::arg("allocator"))

        .def("wrap_int", &line_buf::wrap<int>, py::arg("buffer"), py::arg("num_ele"), py::arg("pre_size"))
        .def("wrap_float", &line_buf::wrap<float>, py::arg("buffer"), py::arg("num_ele"), py::arg("pre_size"));

}

