//===- GdbDebugExecutionContextHook.cpp - GDB for Debugger Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/GdbDebugExecutionContextHook.h"
#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include "mlir/Support/DebugExecutionContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <signal.h>

mlir::DebugExecutionControl GDB_RETURN = mlir::DebugExecutionControl::Apply;

extern "C" {
void mlirDebuggerSetControl(int controlOption) {
  GDB_RETURN = static_cast<mlir::DebugExecutionControl>(controlOption);
}

void mlirDebuggerAddSimpleBreakpoint(const char *tag) {
  auto &simpleBreakpointManager =
      mlir::SimpleBreakpointManager::getGlobalInstance();
  simpleBreakpointManager.addBreakpoint(mlir::StringRef(tag, strlen(tag)));
}

void mlirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo) {
  auto &rewritePatternBreakpointManager =
      mlir::RewritePatternBreakpointManager::getGlobalInstance();
  rewritePatternBreakpointManager.addBreakpoint(
      mlir::StringRef(patternNameInfo, strlen(patternNameInfo)));
}

void mlirDebuggerAddFileLineColLocBreakpoint(const char *file, unsigned line,
                                             unsigned col) {
  auto &fileLineColLocBreakpointManager =
      mlir::FileLineColLocBreakpointManager::getGlobalInstance();
  fileLineColLocBreakpointManager.addBreakpoint(
      mlir::StringRef(file, strlen(file)), line, col);
}

bool mlirDebuggerDeleteBreakpoint(unsigned breakpointID) {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  auto it = mp.find(breakpointID);
  if (it != mp.end()) {
    auto &breakpointInstanceById = it->second;
    auto *breakpoint = std::get<0>(breakpointInstanceById);
    auto &breakpointManager = std::get<1>(breakpointInstanceById);
    return breakpointManager.deleteBreakpoint(breakpoint);
  }
  return false;
}

bool mlirDebuggerChangeStatusOfBreakpoint(unsigned breakpointID, bool status) {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  auto it = mp.find(breakpointID);
  if (it != mp.end()) {
    auto *breakpoint = std::get<0>(it->second);
    if (status) {
      breakpoint->setEnableStatusTrue();
    } else {
      breakpoint->setEnableStatusFalse();
    }
    return true;
  }
  return false;
}

bool mlirDebuggerListBreakpoints() {
  auto &mp = getGlobalInstanceOfBreakpoindIdsMap();
  if (mp.empty()) {
    return false;
  }
  llvm::dbgs() << llvm::formatv("{0,-8}", "ID")
               << llvm::formatv("{0,-15}", "Type")
               << llvm::formatv("{0,-4}", "Enb") << "Info\n";
  for (auto &[id, tuple] : mp) {
    auto *breakpoint = std::get<0>(tuple);
    std::string type;
    if (llvm::isa<mlir::SimpleBreakpoint>(*breakpoint)) {
      type = "Tag";
    } else if (llvm::isa<mlir::RewritePatternBreakpoint>(*breakpoint)) {
      type = "Pattern";
    } else if (llvm::isa<mlir::FileLineColLocBreakpoint>(*breakpoint)) {
      type = "Location";
    } else {
      type = "Unknown";
    }
    llvm::dbgs() << llvm::formatv("{0,-8}", id)
                 << llvm::formatv("{0,-15}", type)
                 << llvm::formatv("{0,-4}",
                                  breakpoint->getEnableStatus() ? 'y' : 'n')
                 << *breakpoint << '\n';
  }
  return true;
}

bool mlirDebuggerPrintAction() {
  if (auto *daiHead =
          mlir::GdbDebugExecutionContextInformation::getGlobalInstance()
              .getDebugActionInformation()) {
    daiHead->action.print(llvm::dbgs());
    return true;
  }
  return false;
}

void mlirDebuggerPrintIRUnit(const void *irUnitPtr) {
  static int indent = 3;
  static auto printIndent = []() -> llvm::raw_ostream & {
    for (int i = 0; i < indent; ++i) {
      llvm::dbgs() << "  ";
    }
    return llvm::dbgs();
  };
  static auto printBlock = [](mlir::Block *block) {
    printIndent() << "Block with " << block->getNumArguments() << " arguments, "
                  << block->getNumSuccessors() << " successors, and "
                  << block->getOperations().size() << " operations\n";
    ++indent;
    for (auto &op : block->getOperations()) {
      printIndent() << op << '\n';
    }
    --indent;
  };
  static auto printRegion = [](mlir::Region *region) {
    printIndent() << "Region with " << region->getBlocks().size()
                  << " blocks:\n";
    ++indent;
    for (auto &block : region->getBlocks()) {
      printBlock(&block);
    }
    --indent;
  };
  auto &unit = *reinterpret_cast<const mlir::IRUnit *>(irUnitPtr);
  if (std::holds_alternative<mlir::Operation *>(unit)) {
    auto *op = std::get<mlir::Operation *>(unit);
    llvm::dbgs() << "Operation:\n";
    printIndent() << *op << '\n';
  } else if (std::holds_alternative<mlir::Block *>(unit)) {
    auto *block = std::get<mlir::Block *>(unit);
    llvm::dbgs() << "Block:\n";
    printBlock(block);
  } else if (std::holds_alternative<mlir::Region *>(unit)) {
    auto *region = std::get<mlir::Region *>(unit);
    llvm::dbgs() << "Region:\n";
    printRegion(region);
  } else {
    llvm::dbgs() << "Invalid pointer to IRUnit.\n";
  }
}

bool mlirDebuggerShowContext() {
  auto &units = mlir::GdbDebugExecutionContextInformation::getGlobalInstance()
                    .getArrayOfIRUnits();
  if (units.empty()) {
    return false;
  }
  llvm::dbgs() << "Available IRUnits: " << units.size() << '\n';
  for (int i = 0; i < (int)units.size(); ++i) {
    auto &unit = units[i];
    llvm::dbgs() << llvm::formatv("#{0,3}: ", i);
    mlirDebuggerPrintIRUnit(&unit);
  }
  return true;
}

const void *mlirDebuggerRetrieveIRUnit(unsigned id) {
  auto &units = mlir::GdbDebugExecutionContextInformation::getGlobalInstance()
                    .getArrayOfIRUnits();
  if (units.empty() || id >= units.size()) {
    return NULL;
  }
  return &units[id];
}

bool mlirDebuggerIRUnitIndexIsAvailable(unsigned id) {
  auto &units = mlir::GdbDebugExecutionContextInformation::getGlobalInstance()
                    .getArrayOfIRUnits();
  if (units.empty() || id >= units.size()) {
    return false;
  }
  return true;
}

// TODO(inkryp): Make this safe. Study behavior of when reaching Module Op.
void *mlirDebuggerSelectParentIRUnit(void *irUnitPtr) {
  auto &unit = *reinterpret_cast<const mlir::IRUnit *>(irUnitPtr);
  if (std::holds_alternative<mlir::Operation *>(unit)) {
    auto *op = std::get<mlir::Operation *>(unit);
    return new mlir::IRUnit(op->getBlock());
  } else if (std::holds_alternative<mlir::Block *>(unit)) {
    auto *block = std::get<mlir::Block *>(unit);
    return new mlir::IRUnit(block->getParent());
  } else if (std::holds_alternative<mlir::Region *>(unit)) {
    auto *region = std::get<mlir::Region *>(unit);
    return new mlir::IRUnit(region->getParentOp());
  }
  return NULL;
}

// TODO(inkryp): Make this safe. Study behavior of when reaching the end of IR.
void *mlirDebuggerSelectChildIRUnit(void *irUnitPtr) {
  auto &unit = *reinterpret_cast<const mlir::IRUnit *>(irUnitPtr);
  if (std::holds_alternative<mlir::Operation *>(unit)) {
    auto *op = std::get<mlir::Operation *>(unit);
    return new mlir::IRUnit(&op->getRegion(0));
  } else if (std::holds_alternative<mlir::Block *>(unit)) {
    auto *block = std::get<mlir::Block *>(unit);
    return new mlir::IRUnit(&block->front());
  } else if (std::holds_alternative<mlir::Region *>(unit)) {
    auto *region = std::get<mlir::Region *>(unit);
    return new mlir::IRUnit(&region->front());
  }
  return NULL;
}
}

namespace mlir {

GdbDebugExecutionContextInformation &
GdbDebugExecutionContextInformation::getGlobalInstance() {
  static GdbDebugExecutionContextInformation *ctx =
      new GdbDebugExecutionContextInformation();
  return *ctx;
}

void GdbDebugExecutionContextInformation::updateContents(
    ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags, StringRef tag,
    StringRef desc, const int &depth, const DebugActionInformation *daiHead) {
  this->units = units;
  this->instanceTags = instanceTags;
  this->tag = tag;
  this->desc = desc;
  this->depth = depth;
  this->daiHead = daiHead;
  idxActiveUnit = 0;
  activeUnit = this->units[idxActiveUnit];
}

static void *volatile sink;

DebugExecutionControl
GdbCallBackFunction(ArrayRef<IRUnit> units, ArrayRef<StringRef> instanceTags,
                    StringRef tag, StringRef desc, const int &depth,
                    const DebugActionInformation *daiHead) {
  static bool initialized = [&]() {
    sink = (void *)mlirDebuggerSetControl;
    sink = (void *)mlirDebuggerAddSimpleBreakpoint;
    sink = (void *)mlirDebuggerAddRewritePatternBreakpoint;
    sink = (void *)mlirDebuggerAddFileLineColLocBreakpoint;
    sink = (void *)mlirDebuggerDeleteBreakpoint;
    sink = (void *)mlirDebuggerChangeStatusOfBreakpoint;
    sink = (void *)mlirDebuggerListBreakpoints;
    sink = (void *)mlirDebuggerPrintAction;
    sink = (void *)mlirDebuggerPrintIRUnit;
    sink = (void *)mlirDebuggerShowContext;
    sink = (void *)mlirDebuggerRetrieveIRUnit;
    sink = (void *)mlirDebuggerIRUnitIndexIsAvailable;
    sink = (void *)mlirDebuggerSelectParentIRUnit;
    sink = (void *)mlirDebuggerSelectChildIRUnit;
    return true;
  }();
  (void)initialized;
  auto &ctx = GdbDebugExecutionContextInformation::getGlobalInstance();
  ctx.updateContents(units, instanceTags, tag, desc, depth, daiHead);
  raise(SIGTRAP);
  return GDB_RETURN;
}

} // namespace mlir
