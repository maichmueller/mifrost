import re
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps


def _read_project_version() -> str:
    pyproject = Path(__file__).with_name("pyproject.toml")
    if not pyproject.is_file():
        raise RuntimeError("pyproject.toml is required to determine the Conan version")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text())
    if match is None:
        raise RuntimeError("pyproject.toml does not define project.version")
    return match.group(1)


class MifrostRecipe(ConanFile):
    name = "mifrost"
    version = _read_project_version()
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    exports = "conandata.yml", "pyproject.toml"
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
    for comp in ("container", "iostreams", "json", "random", "regex", "system"):
        default_options.update({f"boost/*:without_{comp}": False})

    def export_sources(self):
        self.copy("conandata.yml")

    def requirements(self):
        # Public dependencies from Mimir
        self.requires("boost/[>=1.74.0]")
        self.requires("abseil/20240116.2", override=True)
        self.requires("fmt/11.2.0")
        self.requires("range-v3/0.12.0")
        # nanobind deliberately does NOT come from conan: this module joins
        # pymimir's nanobind type registry and must share that wheel's nanobind
        # internals generation (pymimir >= 0.14.3 pins 2.15.x = generation 21).
        # conancenter stops at 2.13.0, one generation short, so src/CMakeLists.txt
        # resolves nanobind from pip -- see cmake/NanobindAbi.cmake.
        if self.options.with_benchmarks:
            self.requires("argparse/3.2")
            self.requires("benchmark/1.8.3")

        custom_requirements = []
        if isinstance(self.conan_data, dict):
            custom_requirements = self.conan_data.get("requirements") or []
        if not custom_requirements:
            raise ConanInvalidConfiguration(
                "conandata.yml must define a non-empty 'requirements' list"
            )
        for req in custom_requirements:
            self.requires(req)

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
        deps.generate()
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def package(self):
        cmake = CMake(self)
        cmake.install()
