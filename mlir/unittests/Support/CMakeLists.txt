add_mlir_unittest(MLIRSupportTests
  DebugActionTest.cpp
  DebugCounterTest.cpp
  DebugExecutionContextTest.cpp
  FileLineColLocBreakpointManagerTest.cpp
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
  StorageUniquerTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport
  MLIRIR)
