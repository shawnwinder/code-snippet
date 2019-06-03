	.file	"bar.cpp"
	.text
	.globl	_Z3foov
	.type	_Z3foov, @function
_Z3foov:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	_Z3foov, .-_Z3foov
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	$10, -28(%rbp)
	call	_Z3foov
	call	_Z3foov
	movl	$10, -4(%rbp)
	call	_Z3foov
	leaq	-28(%rbp), %rax
	movq	%rax, -16(%rbp)
	call	_Z3foov
	leaq	-28(%rbp), %rax
	movq	%rax, -24(%rbp)
	call	_Z3foov
	movq	-16(%rbp), %rax
	movl	$8, (%rax)
	call	_Z3foov
	movq	-24(%rbp), %rax
	movl	$0, (%rax)
	call	_Z3foov
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (GNU) 4.8.5 20150623 (Red Hat 4.8.5-11)"
	.section	.note.GNU-stack,"",@progbits
