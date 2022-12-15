//===- FileLineColLocBreakpointManager.h - TODO: add message ----*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
#define MLIR_SUPPORT_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H

#include "mlir/IR/Location.h"
#include "mlir/Support/BreakpointManagers/BreakpointManager.h"
#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/DenseMap.h"
#include <variant>

namespace mlir {

class FileLineColLocBreakpoint : public Breakpoint {
public:
  FileLineColLocBreakpoint()
      : Breakpoint(TypeID::get<FileLineColLocBreakpoint>()) {}

  FileLineColLocBreakpoint(std::string file, unsigned line, unsigned col)
      : Breakpoint(TypeID::get<FileLineColLocBreakpoint>()), file(file),
        line(line), col(col) {}

  /// Provide classof to allow casting between breakpoint types.
  static bool classof(const Breakpoint *breakpoint) {
    return breakpoint->getBreakpointTypeID() ==
           TypeID::get<FileLineColLocBreakpoint>();
  }

private:
  /// A FileLineColLoc information associate the FileLineColLocBreakpoint with.
  // TODO: I already started with this, but should be changed to be a simple
  // tuple.
  std::string file;

  unsigned line;

  unsigned col;

  /// Allow access to `FileLineColLocBreakpoint`'s identifiers.
  friend class FileLineColLocBreakpointManager;
};

class FileLineColLocBreakpointManager : public BreakpointManager {
public:
  FileLineColLocBreakpointManager()
      : BreakpointManager(TypeID::get<FileLineColLocBreakpointManager>()) {}

  /// Provide classof to allow casting between breakpoint manager types.
  static bool classof(const BreakpointManager *breakpointManager) {
    return breakpointManager->getBreakpointManagerID() ==
           TypeID::get<FileLineColLocBreakpointManager>();
  }

  llvm::Optional<Breakpoint *> match(const DebugActionBase &action,
                                     ArrayRef<StringRef> instanceTags,
                                     ArrayRef<IRUnit> units) override {
    for (auto &unit : units) {
      if (std::holds_alternative<Operation *>(unit)) {
        auto *op = std::get<Operation *>(unit);
        if (auto match = matchFromLocation(op->getLoc())) {
          return match;
        }
      } else if (std::holds_alternative<Block *>(unit)) {
        auto *block = std::get<Block *>(unit);
        for (auto &op : block->getOperations()) {
          if (auto match = matchFromLocation(op.getLoc())) {
            return match;
          }
        }
      } else {
        auto *region = std::get<Region *>(unit);
        if (auto match = matchFromLocation(region->getLoc())) {
          return match;
        }
      }
    }
    return {};
  }

  FileLineColLocBreakpoint *addBreakpoint(StringRef file, unsigned line,
                                          unsigned col = -1) {
    auto result =
        breakpoints.insert({std::make_tuple(file, line, col), nullptr});
    auto &it = result.first;
    if (result.second) {
      it->second =
          std::make_unique<FileLineColLocBreakpoint>(file.str(), line, col);
    }
    return it->second.get();
  }

  void deleteBreakpoint(Breakpoint *breakpoint) {
    if (auto *fileLineColLocBreakpoint =
            dyn_cast<FileLineColLocBreakpoint>(breakpoint)) {
      deleteBreakpoint(fileLineColLocBreakpoint->file,
                       fileLineColLocBreakpoint->line,
                       fileLineColLocBreakpoint->col);
    }
  }

  void deleteBreakpoint(std::string file, unsigned line, unsigned col) {
    breakpoints.erase(std::make_tuple(StringRef(file), line, col));
  }

  static FileLineColLocBreakpointManager &getGlobalInstance() {
    static FileLineColLocBreakpointManager *manager =
        new FileLineColLocBreakpointManager();
    return *manager;
  }

  DenseMap<std::tuple<StringRef, unsigned, unsigned>,
           std::unique_ptr<FileLineColLocBreakpoint>>
      breakpoints;

private:
  llvm::Optional<Breakpoint *> matchFromLocation(Location loc) {
    if (auto fileLoc = loc.dyn_cast<FileLineColLoc>()) {
      auto it = breakpoints.find(
          std::make_tuple(fileLoc.getFilename().getValue(), fileLoc.getLine(),
                          fileLoc.getColumn()));
      if (it != breakpoints.end() && it->second->getEnableStatus()) {
        return it->second.get();
      }
    }
    return {};
  }
};

} // namespace mlir

#endif // MLIR_SUPPORT_BREAKPOINTMANAGERS_FILELINECOLLOCBREAKPOINTMANAGER_H
