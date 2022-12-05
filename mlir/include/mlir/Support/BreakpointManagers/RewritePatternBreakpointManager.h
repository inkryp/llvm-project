//===- RewritePatternBreakpointManager.h - TODO: add message ----*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_BREAKPOINTMANAGERS_REWRITEPATTERNBREAKPOINTMANAGER_H
#define MLIR_SUPPORT_BREAKPOINTMANAGERS_REWRITEPATTERNBREAKPOINTMANAGER_H

#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/BreakpointManagers/BreakpointManager.h"
#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

class RewritePatternBreakpoint : public Breakpoint {
public:
  RewritePatternBreakpoint()
      : Breakpoint(TypeID::get<RewritePatternBreakpoint>()) {}

  RewritePatternBreakpoint(const std::string &_patternNameInfo)
      : Breakpoint(TypeID::get<RewritePatternBreakpoint>()),
        patternNameInfo(_patternNameInfo) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getBreakpointID() ==
           TypeID::get<RewritePatternBreakpoint>();
  }

private:
  /// A tag to associate the RewritePatternBreakpoint with.
  std::string patternNameInfo;

  /// Allow access to `patternNameInfo`.
  friend class RewritePatternBreakpointManager;
};

class RewritePatternBreakpointManager : public BreakpointManager {
public:
  RewritePatternBreakpointManager()
      : BreakpointManager(TypeID::get<RewritePatternBreakpointManager>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManager *breakpointManager) {
    return breakpointManager->getBreakpointManagerID() ==
           TypeID::get<RewritePatternBreakpointManager>();
  }

  llvm::Optional<Breakpoint *> match(const DebugActionBase &action,
                                     ArrayRef<StringRef> instanceTags,
                                     ArrayRef<IRUnit> units) override {
    if (llvm::isa<ApplyPatternAction>(action)) {
      const auto &applyPatternAction = llvm::cast<ApplyPatternAction>(action);
      auto it = breakpoints.find(applyPatternAction.pattern.getDebugName());
      if (it != breakpoints.end() && it->second->getEnableStatus()) {
        return it->second.get();
      }
      for (auto &debugLabel : applyPatternAction.pattern.getDebugLabels()) {
        it = breakpoints.find(debugLabel);
        if (it != breakpoints.end() && it->second->getEnableStatus()) {
          return it->second.get();
        }
      }
    }
    return {};
  }

  RewritePatternBreakpoint *addBreakpoint(StringRef tag) {
    auto result = breakpoints.insert({tag, nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second = std::make_unique<RewritePatternBreakpoint>(tag.str());
    }
    return it->second.get();
  }

  void deleteBreakpoint(Breakpoint *breakpoint) {
    auto *rewritePatternBreakpoint =
        dyn_cast<RewritePatternBreakpoint>(breakpoint);
    breakpoints.erase(rewritePatternBreakpoint->patternNameInfo);
  }

  static RewritePatternBreakpointManager &getGlobalInstance() {
    static RewritePatternBreakpointManager *manager =
        new RewritePatternBreakpointManager();
    return *manager;
  }

  llvm::StringMap<std::unique_ptr<RewritePatternBreakpoint>> breakpoints;
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_REWRITEPATTERNBREAKPOINTMANAGER_H
