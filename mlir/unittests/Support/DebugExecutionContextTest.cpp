//===- DebugClientTest.cpp - Debug Client initial behavior ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
#include "gmock/gmock.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

// DebugActionManager is only enabled in DEBUG mode.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS

namespace {
struct DebuggerAction : public DebugAction<> {
  static StringRef getTag() { return "debugger-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};
struct OtherAction : public DebugAction<> {
  static StringRef getTag() { return "other-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};

ActionResult noOp() { return { nullptr, false, success() }; }

struct SimpleBreakpoint {
  StringRef tag;
  bool enabled;
  SimpleBreakpoint(StringRef &_tag) : tag(_tag), enabled(true) {}
};

struct SimpleBreakpointManager {
  llvm::Optional<SimpleBreakpoint*> match(StringRef &tag) {
    for (auto &[breakpoint, status] : breakpoints) {
      if (status && ((breakpoint->tag) == tag)) {
        return breakpoint;
      }
    }
    return {};
  }
  SimpleBreakpoint* addBreakpoint(StringRef tag) {
    auto breakpoint = new SimpleBreakpoint(tag);
    breakpoints[breakpoint] = true;
    return breakpoint;
  }
  void enableBreakpoint(SimpleBreakpoint* breakpoint) {
    breakpoints[breakpoint] = true;
  }
 void disableBreakpoint(SimpleBreakpoint* breakpoint) {
    breakpoints[breakpoint] = false;
  }
  static SimpleBreakpointManager& getGlobalSbm() {
    static SimpleBreakpointManager* sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  llvm::MapVector<SimpleBreakpoint*, bool> breakpoints;
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext() : sbm(sbm.getGlobalSbm()) {
  }
  FailureOr<bool> execute(StringRef tag, StringRef desc) final {
    llvm::Optional<SimpleBreakpoint*> breakpoint = sbm.match(tag);
    if (breakpoint) {
      return Callback();
    }
    return true;
  }
  int getTimesMatched() { return match; }
  SimpleBreakpoint* addSimpleBreakpoint(StringRef tag) {
    return sbm.addBreakpoint(tag);
  }
  void disableSimpleBreakpoint(SimpleBreakpoint* breakpoint) {
    sbm.disableBreakpoint(breakpoint);
  }
  void enableSimpleBreakpoint(SimpleBreakpoint* breakpoint) {
    sbm.enableBreakpoint(breakpoint);
  }

private:
  bool Callback() {
    match++;
    return false;
  }
  SimpleBreakpointManager& sbm;
  static int match;
};

int DebugExecutionContext::match = 0;

TEST(DebugExecutionContext, DebuggerTest) {
  DebugActionManager manager;
  manager.registerActionHandler<DebugExecutionContext>();

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  DebugExecutionContext dbg;
  EXPECT_EQ(dbg.getTimesMatched(), 0);

  auto dbgBreakpoint = dbg.addSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 1);

  dbg.disableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 1);

  dbg.enableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 2);
}

} // namespace

#endif
