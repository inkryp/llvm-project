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
#include "mlir/Support/DebugExecutionContext.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

class SimpleBreakpoint : public Breakpoint {
public:
  SimpleBreakpoint() : Breakpoint(TypeID::get<SimpleBreakpoint>()) {}

  SimpleBreakpoint(const std::string &_tag)
      : Breakpoint(TypeID::get<SimpleBreakpoint>()), tag(_tag) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getBreakpointTypeID() == TypeID::get<SimpleBreakpoint>();
  }

  void print(raw_ostream &os) const override { os << "Tag: `" << tag << '`'; }

private:
  /// A tag to associate the SimpleBreakpoint with.
  std::string tag;

  /// Allow access to `tag`.
  friend class SimpleBreakpointManager;
};

class SimpleBreakpointManager : public BreakpointManager {
public:
  SimpleBreakpointManager()
      : BreakpointManager(TypeID::get<SimpleBreakpointManager>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManager *breakpointManager) {
    return breakpointManager->getBreakpointManagerID() ==
           TypeID::get<SimpleBreakpointManager>();
  }

  llvm::Optional<Breakpoint *> match(const DebugActionBase &action,
                                     ArrayRef<StringRef> instanceTags,
                                     ArrayRef<IRUnit> units) override {
    auto it = breakpoints.find(action.tag);
    if (it != breakpoints.end() && it->second->getEnableStatus()) {
      return it->second.get();
    }
    return {};
  }
  SimpleBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second = std::make_unique<SimpleBreakpoint>(tag.str());
      SimpleBreakpoint *justAddedBreakpoint = it->second.get();
      auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
      mp.insert({justAddedBreakpoint->getBreakpointID(),
                 std::tuple<mlir::Breakpoint *, mlir::BreakpointManager &>(
                     justAddedBreakpoint, *this)});
    }
    return it->second.get();
  }
  bool deleteBreakpoint(Breakpoint *breakpoint) override {
    if (auto *simpleBreakpoint = dyn_cast<SimpleBreakpoint>(breakpoint)) {
      auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
      auto successfullyDeleted = mp.erase(breakpoint->getBreakpointID());
      return successfullyDeleted && breakpoints.erase(simpleBreakpoint->tag);
    }
    return false;
  }
  static SimpleBreakpointManager &getGlobalInstance() {
    static SimpleBreakpointManager *sbm = new SimpleBreakpointManager();
    return *sbm;
  }
  // TODO: Change implementation so that tags don't get duplicated
  llvm::StringMap<std::unique_ptr<SimpleBreakpoint>> breakpoints;
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_SIMPLEBREAKPOINTMANAGER_H
