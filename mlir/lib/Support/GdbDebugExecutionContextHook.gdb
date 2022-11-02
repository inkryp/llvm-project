define-prefix mlir

define mlir apply
  call mlirDebuggerSetControl(1)
  continue
end

define mlir skip
  call mlirDebuggerSetControl(2)
  continue
end

define mlir step
  call mlirDebuggerSetControl(3)
  continue
end

define mlir next
  call mlirDebuggerSetControl(4)
  continue
end

define mlir finish
  call mlirDebuggerSetControl(5)
  continue
end
