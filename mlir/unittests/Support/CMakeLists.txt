add_mlir_unittest(MLIRSupportTests
  DebugActionTest.cpp
  DebugClientTest.cpp
  DebugCounterTest.cpp
  IndentedOstreamTest.cpp
  MathExtrasTest.cpp
  StorageUniquerTest.cpp
)

target_link_libraries(MLIRSupportTests
  PRIVATE MLIRSupport)
