# Create test executable
function(add_unit_test test_name test_file)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} 
        inner_lib
        unity
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add your test files here
add_unit_test(test_hhp_matrix test_cpu.c)