//===- DebugExecutionContextTest.cpp - Debug Execution Context first impl -===//
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
  std::string tag;
  bool enabled;
  SimpleBreakpoint(const std::string &_tag) : tag(_tag), enabled(true) {}
};

struct SimpleBreakpointManager {
  llvm::Optional<SimpleBreakpoint*> match(StringRef &tag) {
    auto it = breakpoints.find(tag);
    if (it != breakpoints.end() && it->second->enabled) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint* addBreakpoint(StringRef tag) {
    // TODO: Avoid doing two lookups by using insert()
    // Insert nullptr and only if it gets inserted do the make_unique and overwrite it
    breakpoints[tag] = std::make_unique<SimpleBreakpoint>(tag.str());
    return breakpoints[tag].get();
  }
  void enableBreakpoint(SimpleBreakpoint* breakpoint) {
    breakpoint->enabled = true;
  }
  void disableBreakpoint(SimpleBreakpoint* breakpoint) {
    breakpoint->enabled = false;
  }
  void deleteBreakpoint(SimpleBreakpoint* breakpoint) {
    breakpoints.erase(breakpoint->tag);
  }
  static SimpleBreakpointManager& getGlobalSbm() {
    static SimpleBreakpointManager* sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  llvm::StringMap<std::unique_ptr<SimpleBreakpoint>> breakpoints;
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext() : sbm(sbm.getGlobalSbm()) {
  }
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                                  ArrayRef<StringRef> instanceTags,
                                  llvm::function_ref<ActionResult()> transform,
                                  StringRef tag, StringRef desc) final {
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
  void deleteSimpleBreakpoint(SimpleBreakpoint* breakpoint) {
    sbm.deleteBreakpoint(breakpoint);
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
  auto ptr = std::make_unique<DebugExecutionContext>();
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  EXPECT_EQ(dbg->getTimesMatched(), 0);

  auto dbgBreakpoint = dbg->addSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg->getTimesMatched(), 1);

  dbg->disableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg->getTimesMatched(), 1);

  dbg->enableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_FALSE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg->getTimesMatched(), 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(dbg->getTimesMatched(), 2);

  dbg->deleteSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(dbg->getTimesMatched(), 2);
}

} // namespace

#endif
