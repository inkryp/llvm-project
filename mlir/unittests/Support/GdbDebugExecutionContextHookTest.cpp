//===- GdbDebugExecutionContextHookTest.cpp - Gdb integration first impl --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/GdbDebugExecutionContextHook.h"
#include "gmock/gmock.h"
#include <signal.h>

using namespace mlir;

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

namespace {
struct DebuggerAction : public DebugAction<DebuggerAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebuggerAction)
  static StringRef getTag() { return "debugger-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};

ActionResult noOp() { return {IRUnit(), false, success()}; }

TEST(GdbDebugExecutionContextHook, Demo) {
  DebugActionManager manager;
  DebugExecutionContext *dbg;
  auto onBreakpoint = [&](ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags, StringRef tag,
                          StringRef desc, const int &depth,
                          const DebugActionInformation *backtrace) {
    return GdbOnBreakpoint(dbg);
  };

  auto ptr = std::make_unique<DebugExecutionContext>(onBreakpoint);
  dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));
  SimpleBreakpointManager::getGlobalInstance().addBreakpoint(
      DebuggerAction::getTag());

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(GDB_RETURN, 1);
}
} // namespace

#endif
