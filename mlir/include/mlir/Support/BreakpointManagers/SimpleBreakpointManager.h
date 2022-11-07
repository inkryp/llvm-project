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

#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

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

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_SIMPLEBREAKPOINTMANAGER_H
