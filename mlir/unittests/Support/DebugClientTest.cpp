//===- DebugClientTest.cpp - Debug Client initial behavior ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugAction.h"
#include "gmock/gmock.h"
#include <vector>

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
  llvm::Optional<SimpleBreakpoint> match(StringRef tag) {
    for (SimpleBreakpoint sbp : breakpoints) {
      if (sbp.enabled && (sbp.tag == tag)) {
        return sbp;
      }
    }
    return {};
  }
  void addBreakpoint(StringRef tag) {
    breakpoints.push_back(SimpleBreakpoint(tag));
  }
  void enableBreakpoint(StringRef tag) {
    for (SimpleBreakpoint &sbp : breakpoints) {
      if (sbp.tag == tag) {
         sbp.enabled = true;
      }
    }
  }
  void disableBreakpoint(StringRef tag) {
    for (SimpleBreakpoint &sbp : breakpoints) {
      if (sbp.tag == tag) {
        sbp.enabled = false;
      }
    }
  }
  SimpleBreakpointManager& getGlobalSbm() {
    static SimpleBreakpointManager* sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  std::vector<SimpleBreakpoint> breakpoints;
};

class DebugClient : public DebugActionManager::GenericHandler {
public:
  DebugClient() : sbm(sbm.getGlobalSbm()) {
  }
  FailureOr<bool> execute(StringRef tag, StringRef desc) final {
    llvm::Optional<SimpleBreakpoint> breakpoint = sbm.match(tag);
    if (breakpoint) {
      return Callback();
    }
    return true;
  }
  int getTimesMatched() { return match; }
  void addSimpleBreakpoint(StringRef tag) {
    sbm.addBreakpoint(tag);
  }
  void disableSimpleBreakpoint(StringRef tag) {
    sbm.disableBreakpoint(tag);
  }
  void enableSimpleBreakpoint(StringRef tag) {
    sbm.enableBreakpoint(tag);
  }

private:
  bool Callback() {
    match++;
    return false;
  }
  SimpleBreakpointManager& sbm;
  static int match;
};

int DebugClient::match = 0;

TEST(DebugClientTest, DebuggerTest) {
  DebugActionManager manager;
  manager.registerActionHandler<DebugClient>();

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  DebugClient dbg;
  EXPECT_EQ(dbg.getTimesMatched(), 0);

  dbg.addSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 1);

  dbg.disableSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 1);

  dbg.enableSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(dbg.getTimesMatched(), 2);
}

} // namespace

#endif
