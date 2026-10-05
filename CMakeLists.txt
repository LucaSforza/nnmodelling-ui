cmake_minimum_required(VERSION 3.20)
project(NNModelling VERSION 0.1.0 LANGUAGES C)

option(NN_BUILD_GUI "Build the Qt 6 Widgets frontend" ON)
include(CTest)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(LUA_NAMES lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject
    lopcodes lparser lstate lstring ltable ltm lundump lvm lzio lauxlib lbaselib
    lcorolib ldblib liolib lmathlib loadlib loslib lstrlib ltablib lutf8lib linit)
set(LUA_SOURCES)
foreach(name IN LISTS LUA_NAMES)
    list(APPEND LUA_SOURCES third_party/lua/src/${name}.c)
endforeach()
add_library(nn_lua STATIC ${LUA_SOURCES})
target_include_directories(nn_lua PUBLIC third_party/lua/src)
if(UNIX)
    target_compile_definitions(nn_lua PRIVATE LUA_USE_LINUX)
    target_link_libraries(nn_lua PUBLIC ${CMAKE_DL_LIBS} m)
endif()

add_library(nn_json STATIC third_party/yyjson/yyjson.c)
target_include_directories(nn_json PUBLIC third_party/yyjson)
add_library(nn_utils STATIC src/utils/utils.c)
target_include_directories(nn_utils PUBLIC src)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nn_utils PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

set(NN_MODEL_SOURCES
    src/model/model.c src/model/nodes.c src/model/edges.c
    src/model/parameters.c src/model/value_json.c src/model/values.c src/model/model_utils.c)
set(NN_CATALOG_SOURCES
    src/catalog/catalog.c src/catalog/catalog_utils.c src/catalog/definition.c
    src/catalog/manifest.c src/catalog/dependencies.c src/catalog/references.c)
set(NN_PROJECT_SOURCES
    src/project/project.c src/project/open.c src/project/graph_json.c
    src/project/save.c src/project/datasets.c src/project/templates.c
    src/project/resources.c src/project/project_utils.c)
set(NN_INFERENCE_SOURCES
    src/inference/results.c src/inference/evaluation_utils.c src/inference/scope.c src/inference/lua_rule.c
    src/inference/lua_runtime.c src/inference/lua_sandbox.c
    src/inference/lua_tensor_ops.c src/inference/owned_tensors.c
    src/inference/result_extraction.c src/inference/report.c src/inference/status.c)
set(NN_APPLICATION_SOURCES
    src/application/application.c src/application/resources.c
    src/application/history.c
    src/application/graph_commands.c src/application/parameters.c src/application/ports.c)
set(NN_AUTOMATION_SOURCES
    src/automation/dispatch.c src/automation/commands.c src/automation/snapshot.c
    src/automation/diagnostics.c src/automation/transport.c src/automation/automation_utils.c)
add_library(nnmodelling_core STATIC
    ${NN_MODEL_SOURCES} ${NN_CATALOG_SOURCES} ${NN_PROJECT_SOURCES}
    ${NN_INFERENCE_SOURCES} ${NN_APPLICATION_SOURCES} ${NN_AUTOMATION_SOURCES})
target_include_directories(nnmodelling_core PUBLIC src)
target_link_libraries(nnmodelling_core PRIVATE nn_json nn_lua nn_utils)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nnmodelling_core PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

add_executable(nnmodelctl src/nnmodelctl/main.c src/nnmodelctl/request.c src/nnmodelctl/transport.c)
target_link_libraries(nnmodelctl PRIVATE nn_json)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nnmodelctl PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

if(BUILD_TESTING)
    foreach(name IN ITEMS model catalog inference application resources automation history stereotype_reference)
        add_executable(${name}_test tests/${name}_test.c)
        target_link_libraries(${name}_test PRIVATE nnmodelling_core)
        add_test(NAME ${name} COMMAND ${name}_test)
        set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS core)
    endforeach()
    add_executable(utils_test tests/utils_test.c)
    target_link_libraries(utils_test PRIVATE nn_utils)
    add_test(NAME utils COMMAND utils_test)
    set_tests_properties(utils PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS core)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        add_executable(allocation_failure_test tests/allocation_failure_test.c)
        target_link_options(allocation_failure_test PRIVATE
            -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc)
        target_link_libraries(allocation_failure_test PRIVATE nnmodelling_core nn_json)
        target_compile_options(allocation_failure_test PRIVATE
            -Wall -Wextra -Werror -pedantic -UNDEBUG)
        add_test(NAME allocation_failure COMMAND allocation_failure_test)
        set_tests_properties(allocation_failure PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS core)
    endif()
    # This existing fixture supplies its own catalog, isolating persistence.
    add_executable(project_test tests/project_test.c ${NN_PROJECT_SOURCES} ${NN_MODEL_SOURCES})
    target_include_directories(project_test PRIVATE src)
    target_link_libraries(project_test PRIVATE nn_json nn_utils m)
    add_test(NAME project COMMAND project_test)
    set_tests_properties(project PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS core)
    add_executable(lua_smoke tests/lua_smoke.c)
    target_link_libraries(lua_smoke PRIVATE nn_lua)
    add_test(NAME lua COMMAND lua_smoke)
    set_tests_properties(lua PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS core)
    # Existing tests use assert for their checks, including in Release builds.
    foreach(name IN ITEMS model_test catalog_test inference_test application_test resources_test automation_test history_test project_test lua_smoke utils_test stereotype_reference_test)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${name} PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
        endif()
    endforeach()
endif()

if(NN_BUILD_GUI)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    set(QT_CANVAS_SOURCES
        src/gui/qt/GraphScene.cpp src/gui/qt/GraphScene.hpp
        src/gui/qt/GraphView.cpp src/gui/qt/GraphView.hpp
        src/gui/qt/NodeItem.cpp src/gui/qt/NodeItem.hpp
        src/gui/qt/PortItem.cpp src/gui/qt/PortItem.hpp
        src/gui/qt/EdgeItem.cpp src/gui/qt/EdgeItem.hpp
        src/gui/qt/OrthogonalRouter.cpp src/gui/qt/OrthogonalRouter.hpp
        src/gui/qt/DataflowLayout.cpp src/gui/qt/DataflowLayout.hpp
        src/gui/qt/SubflowItem.cpp src/gui/qt/SubflowItem.hpp)
    set(QT_WINDOW_SOURCES
        src/gui/qt/MainWindow.cpp src/gui/qt/MainWindow.hpp
        src/gui/qt/MainWindowProject.cpp src/gui/qt/MainWindowAutomation.cpp
        src/gui/qt/MainWindowStereotypeAuthoring.cpp src/gui/qt/MainWindowDatasetAuthoring.cpp
        src/gui/qt/MainWindowRefresh.cpp src/gui/qt/MainWindowInspectorResources.cpp
        src/gui/qt/MainWindowDiagnostics.cpp src/gui/qt/MainWindowGraphActions.cpp
        src/gui/qt/MainWindowUtils.cpp src/gui/qt/MainWindowUtils.hpp)
    add_library(nnmodelling_canvas STATIC ${QT_CANVAS_SOURCES})
    set_target_properties(nnmodelling_canvas PROPERTIES AUTOMOC ON)
    target_include_directories(nnmodelling_canvas PUBLIC src/gui/qt)
    target_link_libraries(nnmodelling_canvas PUBLIC Qt6::Widgets nnmodelling_core)
    add_executable(nnmodelling-ui src/gui/qt/main.cpp ${QT_WINDOW_SOURCES})
    set_target_properties(nnmodelling-ui PROPERTIES AUTOMOC ON)
    target_compile_definitions(nnmodelling-ui PRIVATE NN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(nnmodelling-ui PRIVATE nnmodelling_canvas)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(nnmodelling_canvas PRIVATE -Wall -Wextra -Werror -pedantic)
        target_compile_options(nnmodelling-ui PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
    if(BUILD_TESTING)
        find_package(Qt6 REQUIRED COMPONENTS Test)
        add_executable(qt_canvas_test tests/qt_canvas_test.cpp)
        set_target_properties(qt_canvas_test PROPERTIES AUTOMOC ON)
        target_link_libraries(qt_canvas_test PRIVATE nnmodelling_canvas Qt6::Test)
        target_compile_definitions(qt_canvas_test PRIVATE NN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        add_executable(qt_window_test tests/qt_window_test.cpp ${QT_WINDOW_SOURCES})
        set_target_properties(qt_window_test PROPERTIES AUTOMOC ON)
        target_link_libraries(qt_window_test PRIVATE nnmodelling_canvas Qt6::Test)
        target_compile_definitions(qt_window_test PRIVATE NN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        add_executable(qt_routing_test tests/qt_routing_test.cpp)
        set_target_properties(qt_routing_test PROPERTIES AUTOMOC ON)
        target_link_libraries(qt_routing_test PRIVATE nnmodelling_canvas Qt6::Test)
        add_test(NAME qt_routing COMMAND qt_routing_test)
        set_tests_properties(qt_routing PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen" LABELS gui TIMEOUT 45)
        add_executable(qt_layout_test tests/qt_layout_test.cpp)
        set_target_properties(qt_layout_test PROPERTIES AUTOMOC ON)
        target_link_libraries(qt_layout_test PRIVATE nnmodelling_canvas Qt6::Test)
        add_test(NAME qt_layout COMMAND qt_layout_test)
        set_tests_properties(qt_layout PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen" LABELS gui TIMEOUT 45)
        add_test(NAME qt_window COMMAND qt_window_test)
        set_tests_properties(qt_window PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen" LABELS gui TIMEOUT 45)
        add_test(NAME qt_canvas COMMAND qt_canvas_test)
        set_tests_properties(qt_canvas PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen" LABELS gui TIMEOUT 45)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME cli_ui COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/cli_test.py" "$<TARGET_FILE:nnmodelling-ui>" "$<TARGET_FILE:nnmodelctl>")
        set_tests_properties(cli_ui PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LABELS gui TIMEOUT 60)
    endif()
endif()

# Audit explicit target source coverage in both C-only and optional Qt builds.
if(BUILD_TESTING)
    set(NN_LAYOUT_SOURCES)
    set(NN_LAYOUT_TARGETS nn_utils nnmodelling_core nnmodelctl)
    if(NN_BUILD_GUI)
        list(APPEND NN_LAYOUT_TARGETS nnmodelling_canvas nnmodelling-ui)
    endif()
    foreach(target IN LISTS NN_LAYOUT_TARGETS)
        get_target_property(sources ${target} SOURCES)
        list(APPEND NN_LAYOUT_SOURCES ${sources})
    endforeach()
    string(JOIN "\n" NN_LAYOUT_MANIFEST ${NN_LAYOUT_SOURCES})
    file(WRITE "${CMAKE_BINARY_DIR}/source-layout-manifest.txt" "${NN_LAYOUT_MANIFEST}\n")
    add_test(NAME source_layout COMMAND ${CMAKE_COMMAND}
        -DROOT=${CMAKE_SOURCE_DIR} -DGUI=${NN_BUILD_GUI}
        -DMANIFEST=${CMAKE_BINARY_DIR}/source-layout-manifest.txt
        -P ${CMAKE_SOURCE_DIR}/tests/source_layout_test.cmake)
    set_tests_properties(source_layout PROPERTIES LABELS core)
endif()
