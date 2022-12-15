//===- BreakpointManager.h - Breakpoint Manager Support ----*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H
#define MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H

#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

/// This class represents the base class of a breakpoint.
class Breakpoint {
public:
  virtual ~Breakpoint() = default;

  /// Return the unique breakpoint id of this breakpoint, use for casting
  /// functionality.
  TypeID getBreakpointTypeID() const { return breakpointTypeID; }

  unsigned getBreakpointID() { return breakpointID; }

protected:
  Breakpoint(TypeID breakpointTypeID)
      : breakpointTypeID(breakpointTypeID), enableStatus(true),
        breakpointID(++breakpointCounterID) {}

  /// The type of the derived breakpoint class. This allows for detecting the
  /// specific handler of a given breakpoint type.
  TypeID breakpointTypeID;

  bool getEnableStatus() const { return enableStatus; }

  void setEnableStatusTrue() { enableStatus = true; }

  void setEnableStatusFalse() { enableStatus = false; }

private:
  /// The current state of the breakpoint. A breakpoint can be either enabled
  /// or disabled.
  bool enableStatus;

  unsigned breakpointID;

  static inline unsigned breakpointCounterID{0};

  /// Allow access to `enableStatus` operations.
  friend class BreakpointManager;
};

/// This class represents the base class of a breakpoint manager.
class BreakpointManager {
public:
  virtual ~BreakpointManager() = default;

  /// Return the unique breakpoint manager id of this breakpoint manager, use
  /// for casting functionality.
  TypeID getBreakpointManagerID() const { return breakpointManagerID; }

  virtual llvm::Optional<Breakpoint *> match(const DebugActionBase &action,
                                             ArrayRef<StringRef> instanceTags,
                                             ArrayRef<IRUnit> units) = 0;

  void enableBreakpoint(Breakpoint *breakpoint) {
    breakpoint->setEnableStatusTrue();
  }

  void disableBreakpoint(Breakpoint *breakpoint) {
    breakpoint->setEnableStatusFalse();
  }

protected:
  BreakpointManager(TypeID breakpointManagerID)
      : breakpointManagerID(breakpointManagerID) {}

  /// The type of the derived breakpoint manager class. This allows for
  /// detecting the specific handler of a given breakpoint type.
  TypeID breakpointManagerID;
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_BREAKPOINTMANAGER_H
