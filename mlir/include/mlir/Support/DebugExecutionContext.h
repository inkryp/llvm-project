//===- DebugExecutionContext.h - Debug Execution Context Support *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Write a proper description for the service
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
#define MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H

#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

enum DebugExecutionControl {
  Apply = 1,
  Skip = 2,
  Step = 3,
  Next = 4,
  Finish = 5
};

struct SimpleBreakpoint {
  std::string tag;
  bool enabled;
  SimpleBreakpoint(const std::string &_tag) : tag(_tag), enabled(true) {}
};

struct SimpleBreakpointManager {
  llvm::Optional<SimpleBreakpoint *> match(const StringRef &tag) {
    auto it = breakpoints.find(tag);
    if (it != breakpoints.end() && it->second->enabled) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second = std::make_unique<SimpleBreakpoint>(tag.str());
    }
    return it->second.get();
  }
  void enableBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoint->enabled = true;
  }
  void disableBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoint->enabled = false;
  }
  void deleteBreakpoint(SimpleBreakpoint *breakpoint) {
    breakpoints.erase(breakpoint->tag);
  }
  static SimpleBreakpointManager &getGlobalSbm() {
    static SimpleBreakpointManager *sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  llvm::StringMap<std::unique_ptr<SimpleBreakpoint>> breakpoints;
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext(
      llvm::function_ref<DebugExecutionControl(
          ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef)>
          callback)
      : OnBreakpoint(callback), sbm(SimpleBreakpointManager::getGlobalSbm()) {}
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &action) final {
    ++depth;
    auto handleUserInput = [&]() -> bool {
      auto todoNext =
          OnBreakpoint(units, instanceTags, action.tag, action.desc);
      while (depth == 1 && todoNext == DebugExecutionControl::Finish) {
        todoNext = OnBreakpoint(units, instanceTags, action.tag, action.desc);
      }
      switch (todoNext) {
      case DebugExecutionControl::Apply:
        depthToBreak = std::nullopt;
        return true;
      case DebugExecutionControl::Skip:
        depthToBreak = std::nullopt;
        return false;
      case DebugExecutionControl::Step:
        depthToBreak = depth + 1;
        return true;
      case DebugExecutionControl::Next:
        depthToBreak = depth;
        return true;
      case DebugExecutionControl::Finish:
        depthToBreak = depth - 1;
        return true;
      }
    };
    auto breakpoint = sbm.match(action.tag);
    bool apply = true;
    if (breakpoint || (depthToBreak && depth <= depthToBreak)) {
      apply = handleUserInput();
    }

    if (apply) {
      transform();
    }

    if (depthToBreak && depth <= depthToBreak) {
      handleUserInput();
    }
    --depth;
    return apply;
  }
  SimpleBreakpoint *addSimpleBreakpoint(StringRef tag) {
    return sbm.addBreakpoint(tag);
  }
  void disableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.disableBreakpoint(breakpoint);
  }
  void enableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.enableBreakpoint(breakpoint);
  }
  void deleteSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.deleteBreakpoint(breakpoint);
  }

private:
  llvm::function_ref<DebugExecutionControl(
      ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef)>
      OnBreakpoint;
  SimpleBreakpointManager &sbm;
  int depth = 0;
  Optional<int> depthToBreak;
};

} // namespace mlir

#endif // MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
