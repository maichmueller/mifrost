from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps


class MifrostRecipe(ConanFile):
    name = "mifrost"
    version = "0.0.1"
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    # Match Mimir's boost version logic
    BOOST_COMPS = (
        "atomic",
        "charconv",
        "chrono",
        "cobalt",
        "container",
        "context",
        "contract",
        "coroutine",
        "date_time",
        "exception",
        "fiber",
        "filesystem",
        "graph",
        "graph_parallel",
        "iostreams",
        "json",
        "locale",
        "log",
        "math",
        "mpi",
        "nowide",
        "process",
        "program_options",
        "python",
        "random",
        "regex",
        "serialization",
        "stacktrace",
        "system",
        "test",
        "thread",
        "timer",
        "type_erasure",
        "url",
        "wave",
    )

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_benchmarks": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_benchmarks": False,
        "cista/*:with_fmt": True,
        "hwloc/*:shared": True,
    }

    # Configure Boost options to match Mimir
    default_options.update({f"boost/*:without_{comp}": True for comp in BOOST_COMPS})
    for comp in ("iostreams", "random", "regex", "system"):
        default_options.update({f"boost/*:without_{comp}": False})

    def export_sources(self):
        self.copy("conandata.yml")

    def requirements(self):
        # Public dependencies from Mimir
        self.requires("boost/[>=1.74.0]")
        self.requires("fmt/11.2.0")
        self.requires("range-v3/0.12.0")
        self.requires("nanobind/2.9.2")
        self.requires("strong_type/v10")
        if self.options.with_benchmarks:
            self.requires("argparse/3.2")
            self.requires("benchmark/1.8.3")

        # Custom dependencies from conandata.yml
        # conandata.yml format matches requirements list: ["pkg/version", ...]
        # We need to parse "pkg/version" to get the package name and version if needed,
        # or just pass the full string if it matches conan syntax.
        # However, conandata.yml structure is `requirements: [ "pkg/v", ... ]`
        # self.conan_data might be None if not loaded properly or file missing.

        if self.conan_data and "requirements" in self.conan_data:
            for req in self.conan_data["requirements"]:
                self.requires(req)
        else:
            # Fallback if conandata not loaded (should not happen if exported correctly)
            self.output.warning(
                "conandata.yml not specified or empty, using defaults/failsafes"
            )
            self.requires("loki/f86e5e10f685a77897269742c84c9d780237084a")
            self.requires("nauty/2.8.8")
            self.requires("gtest/1.14.0")
            self.requires("unordered_dense/4.8.1")
            self.requires("parallel-hashmap/1.3.11")
            self.requires("valla/f487d3fcba00b88bba1e88d9e111747f53679da1")

    def validate(self):
        check_min_cppstd(self, "20")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        # Nanobind's Conan recipe relies on its own CMake config; ensure CMakeDeps
        # still generates a config file so cmake-conan can find it without a toolchain.
        deps.set_property("nanobind", "cmake_find_mode", "both")
        deps.generate()
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def package(self):
        cmake = CMake(self)
        cmake.install()
