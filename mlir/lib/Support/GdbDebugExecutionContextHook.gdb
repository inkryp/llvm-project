define-prefix mlir

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
