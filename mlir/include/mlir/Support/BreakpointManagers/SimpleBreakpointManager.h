//===- SimpleBreakpointManager.h - Simple breakpoint Support ----*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_BREAKPOINTMANAGERS_SIMPLEBREAKPOINTMANAGER_H
#define MLIR_SUPPORT_BREAKPOINTMANAGERS_SIMPLEBREAKPOINTMANAGER_H

#include "mlir/Support/BreakpointManagers/BreakpointManager.h"
#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

class SimpleBreakpoint : public BreakpointBase {
public:
  SimpleBreakpoint() : BreakpointBase(TypeID::get<SimpleBreakpoint>()) {}

  SimpleBreakpoint(const std::string &_tag)
      : BreakpointBase(TypeID::get<SimpleBreakpoint>()), tag(_tag) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const BreakpointBase *breakpoint) {
    return breakpoint->getBreakpointID() == TypeID::get<SimpleBreakpoint>();
  }

private:
  /// A tag to associate the SimpleBreakpoint with.
  std::string tag;

  /// Allow access to `tag`.
  friend class SimpleBreakpointManager;
};

class SimpleBreakpointManager : public BreakpointManagerBase {
public:
  SimpleBreakpointManager()
      : BreakpointManagerBase(TypeID::get<SimpleBreakpointManager>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManagerBase *breakpointManager) {
    return breakpointManager->getBreakpointManagerID() ==
           TypeID::get<SimpleBreakpointManager>();
  }

  llvm::Optional<BreakpointBase *> match(const DebugActionBase &action,
                                         ArrayRef<StringRef> instanceTags,
                                         ArrayRef<IRUnit> unit) override {
    auto it = breakpoints.find(action.tag);
    if (it != breakpoints.end() && it->second->getEnableStatus()) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint *addBreakpoint(StringRef tag) override {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second = std::make_unique<SimpleBreakpoint>(tag.str());
    }
    return it->second.get();
  }
  void deleteBreakpoint(BreakpointBase *breakpointBase) override {
    auto *breakpoint = dyn_cast<SimpleBreakpoint>(breakpointBase);
    breakpoints.erase(breakpoint->tag);
  }
  static SimpleBreakpointManager &getGlobalSbm() {
    static SimpleBreakpointManager *sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  llvm::StringMap<std::unique_ptr<SimpleBreakpoint>> breakpoints;
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_SIMPLEBREAKPOINTMANAGER_H
