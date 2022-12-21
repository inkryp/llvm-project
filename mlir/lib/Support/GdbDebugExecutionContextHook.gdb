define-prefix mlir

set $mlirDebuggerCurrentlyActivatedIRUnit = ((void *) 0)

define mlirDebuggerInitializeCurrentlyActivatedIRUnit
  set $mlirDebuggerCurrentlyActivatedIRUnit = \
          ((void *(*)(unsigned))mlirDebuggerRetrieveIRUnit)(0)
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    printf "Currently there is no available IRUnit with that ID.\n"
  end
end

define mlir apply
  call ((void (*)(int))mlirDebuggerSetControl)(1)
  continue
end

define mlir skip
  call ((void (*)(int))mlirDebuggerSetControl)(2)
  continue
end

define mlir step
  call ((void (*)(int))mlirDebuggerSetControl)(3)
  continue
end

define mlir next
  call ((void (*)(int))mlirDebuggerSetControl)(4)
  continue
end

define mlir finish
  call ((void (*)(int))mlirDebuggerSetControl)(5)
  continue
end

define mlir simpleBreakpoint
  set $i = 0
  while $i < $argc
    eval "call ((void (*)(const char *))mlirDebuggerAddSimpleBreakpoint)( \
              $arg%d)", $i
    set $i = $i + 1
  end
end

define mlir patternBreakpoint
  set $i = 0
  while $i < $argc
    eval "call ((void (*)(const char *)) \
                    mlirDebuggerAddRewritePatternBreakpoint)($arg%d)", $i
    set $i = $i + 1
  end
end

# TODO: Find a way to enforce user to have the correct types on the arguments
define mlir locationBreakpoint
  if $argc == 2
    call ((void (*)(const char *, unsigned, unsigned)) \
              mlirDebuggerAddFileLineColLocBreakpoint)($arg0, $arg1, 0)
  else
    if $argc == 3
      call ((void (*)(const char *, unsigned, unsigned)) \
                mlirDebuggerAddFileLineColLocBreakpoint)($arg0, $arg1, $arg2)
    end
  end
end

# TODO: Support parsing to a string to represent a range.
define mlir deleteBreakpoint
  if $argc == 1
    # Casting `_Bool` as bool is not a C type.
    # TODO: It is not obvious if SIGTRAP is always going to put GDB's language
    # as C. Find a way to store the GDB `language` variable, to temporarily set
    # it to C, and then revert back to whatever the user had in the beginning.
    set $mlirDebuggerDeleteBreakpointResult = \
        ((_Bool (*)(unsigned))mlirDebuggerDeleteBreakpoint)($arg0)
    if !$mlirDebuggerDeleteBreakpointResult
      printf "Could not find Breakpoint with ID %d\n", $arg0
    end
  end
end

define mlir disable
  if $argc == 1
    set $mlirDebuggerChangeStatusOfBreakpointResult = \
        ((_Bool(*)(unsigned, _Bool))mlirDebuggerChangeStatusOfBreakpoint)( \
            $arg0, 0)
    if !$mlirDebuggerChangeStatusOfBreakpointResult
      printf "Could not find Breakpoint with ID %d\n", $arg0
    end
  end
end

define mlir enable
  if $argc == 1
    set $mlirDebuggerChangeStatusOfBreakpointResult = \
        ((_Bool(*)(unsigned, _Bool))mlirDebuggerChangeStatusOfBreakpoint)( \
            $arg0, 1)
    if !$mlirDebuggerChangeStatusOfBreakpointResult
      printf "Could not find Breakpoint with ID %d\n", $arg0
    end
  end
end

define mlir listBreakpoints
  set $mlirDebuggerListBreakpointsResult = \
      ((_Bool (*)())mlirDebuggerListBreakpoints)()
  if !$mlirDebuggerListBreakpointsResult
    printf "No breakpoints.\n"
  end
end

define mlir print-action
  set $mlirDebuggerPrintAction = ((_Bool (*)())mlirDebuggerPrintAction)()
  if !$mlirDebuggerPrintAction
    printf "No action has been selected\n"
  end
end

define mlir action-backtrace
  set $mlirDebuggerPrintActionBacktraceResult = \
          ((_Bool(*)())mlirDebuggerPrintActionBacktrace)()
  if !$mlirDebuggerPrintActionBacktraceResult
    printf "No action has been selected\n"
  end
end

define mlir show-context
  set $mlirDebuggerShowContextResult = ((_Bool (*)())mlirDebuggerShowContext)()
  if !$mlirDebuggerShowContextResult
    printf "Currently there are no available IRUnits.\n"
  end
end

define mlir print
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    mlirDebuggerInitializeCurrentlyActivatedIRUnit
  end
  if $mlirDebuggerCurrentlyActivatedIRUnit
    call ((void (*)(const void *))mlirDebuggerPrintIRUnit)( \
        $mlirDebuggerCurrentlyActivatedIRUnit)
  end
end

define mlir select
  if $argc == 1
    if ((_Bool (*)(unsigned))mlirDebuggerIRUnitIndexIsAvailable)($arg0)
      set $mlirDebuggerCurrentlyActivatedIRUnit = \
              ((void *(*)(unsigned))mlirDebuggerRetrieveIRUnit)($arg0)
    else
      printf "Currently there is no available IRUnit with that ID. "
      printf "Active unit remains the same.\n"
    end
  end
end

define mlir select-parent
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    mlirDebuggerInitializeCurrentlyActivatedIRUnit
  end
  if $mlirDebuggerCurrentlyActivatedIRUnit
    set $mlirDebuggerCurrentlyActivatedIRUnit = \
            ((void *(*)(void *))mlirDebuggerSelectParentIRUnit)( \
                $mlirDebuggerCurrentlyActivatedIRUnit)
  else
    printf "No IRUnit is activated right now.\n"
  end
end

# TODO(inkryp): Is it worth it to retrieve #id child?
define mlir select-child
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    mlirDebuggerInitializeCurrentlyActivatedIRUnit
  end
  if $mlirDebuggerCurrentlyActivatedIRUnit
    set $mlirDebuggerCurrentlyActivatedIRUnit = \
            ((void *(*)(void *))mlirDebuggerSelectChildIRUnit)( \
                $mlirDebuggerCurrentlyActivatedIRUnit)
  else
    printf "No IRUnit is activated right now.\n"
  end
end

define mlir prev-operation
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    mlirDebuggerInitializeCurrentlyActivatedIRUnit
  end
  if $mlirDebuggerCurrentlyActivatedIRUnit
    set $mlirDebuggerCurrentlyActivatedIRUnit = \
            ((void *(*)(void *))mlirDebuggerPreviousOperation)( \
                $mlirDebuggerCurrentlyActivatedIRUnit)
  else
    printf "No IRUnit is activated right now.\n"
  end
end

define mlir next-operation
  if !$mlirDebuggerCurrentlyActivatedIRUnit
    mlirDebuggerInitializeCurrentlyActivatedIRUnit
  end
  if $mlirDebuggerCurrentlyActivatedIRUnit
    set $mlirDebuggerCurrentlyActivatedIRUnit = \
            ((void *(*)(void *))mlirDebuggerNextOperation)( \
                $mlirDebuggerCurrentlyActivatedIRUnit)
  else
    printf "No IRUnit is activated right now.\n"
  end
end

define hook-continue
  set $mlirDebuggerCurrentlyActivatedIRUnit = ((void *) 0)
end
