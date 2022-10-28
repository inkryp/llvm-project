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
struct DebuggerAction : public DebugAction<DebuggerAction> {
  static StringRef getTag() { return "debugger-action"; }
  static StringRef getDescription() { return "Test action for debug client"; }
};
struct OtherAction : public DebugAction<OtherAction> {
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
  llvm::Optional<SimpleBreakpoint*> match(const StringRef &tag) {
    auto it = breakpoints.find(tag);
    if (it != breakpoints.end() && it->second->enabled) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint* addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    assert(result.second);
    auto &it = result.first;
    it->second = std::make_unique<SimpleBreakpoint>(tag.str());
    return it->second.get();
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

enum DebugExecutionControl {
  Apply = 1,
  Skip = 2,
  Step = 3,
  Next = 4,
  VirtualRun = 5
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext(
    llvm::function_ref<DebugExecutionControl(ArrayRef<IRUnit>,
                          ArrayRef<StringRef>, StringRef, StringRef)> callback)
      : OnBreakpoint(callback), sbm(sbm.getGlobalSbm()) {}
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                                  ArrayRef<StringRef> instanceTags,
                                  llvm::function_ref<ActionResult()> transform,
                                  const DebugActionBase& action) final {
    auto breakpoint = sbm.match(action.tag);
    if (breakpoint) {
      auto todoNext = OnBreakpoint(units, instanceTags, action.tag, action.desc);
      switch (todoNext) {
        case DebugExecutionControl::Apply:
          transform();
          return true;
        case DebugExecutionControl::Skip:
          return false;
        // TODO: Support the other DebugExecutionControl
        // Implement finate state machine that represent debugger flow
        default:
          return false;
      }
    }
    transform();
    return true;
  }
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
  llvm::function_ref<DebugExecutionControl(ArrayRef<IRUnit>,
    ArrayRef<StringRef>, StringRef, StringRef)> OnBreakpoint;
  SimpleBreakpointManager& sbm;
};

TEST(DebugExecutionContext, DebuggerTest) {
  DebugActionManager manager;
  int match = 0;
  auto callback = [&match](ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
      StringRef tag, StringRef desc){
    match++;
    return DebugExecutionControl::Skip;
  };
  auto ptr = std::make_unique<DebugExecutionContext>(callback);
  auto dbg = ptr.get();
  manager.registerActionHandler(std::move(ptr));

  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));

  EXPECT_EQ(match, 0);

  auto dbgBreakpoint = dbg->addSimpleBreakpoint(DebuggerAction::getTag());
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  dbg->disableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 1);

  dbg->enableSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(failed(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  EXPECT_TRUE(succeeded(manager.execute<OtherAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);

  dbg->deleteSimpleBreakpoint(dbgBreakpoint);
  EXPECT_TRUE(succeeded(manager.execute<DebuggerAction>({}, {}, noOp)));
  EXPECT_EQ(match, 2);
}

} // namespace

#endif
