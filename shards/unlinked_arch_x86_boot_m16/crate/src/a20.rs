
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! -*- linux-c -*- ------------------------------------------------------- *
//!
//! Copyright (C) 1991, 1992 Linus Torvalds
//! Copyright 2007-2008 rPath, Inc. - All Rights Reserved
//! Copyright 2009 Intel Corporation; author H. Peter Anvin
//! number of edd_info structs starting at EDDBUF
//! This returns a constant expression while determining if an argument is
//! a constant expression, most importantly without evaluating the argument.
//! Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//!
//! Details:
//! - sizeof() return an integer constant expression, and does not evaluate
//!   the value of its operand; it only examines the type of its operand.
//! - The results of comparing two integer constant expressions is also
//!   an integer constant expression.
//! - The first literal "8" isn't important. It could be any literal value.
//! - The second literal "8" is to avoid warnings about unaligned pointers;
//!   this could otherwise just be "1".
//! - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//!   architectures.
//! - The C Standard defines "null pointer constant", "(void *)0", as
//!   distinct from other void pointers.
//! - If (x) is an integer constant expression, then the "* 0l" resolves
//!   it into an integer constant expression of value 0. Since it is cast to
//!   "void *", this makes the second operand a null pointer constant.
//! - If (x) is not an integer constant expression, then the second operand
//!   resolves to a void pointer (but not a null pointer constant: the value
//!   is not an integer constant 0).
//! - The conditional operator's third operand, "(int *)8", is an object
//!   pointer (to type "int").
//! - The behavior (including the return type) of the conditional operator
//!   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//!   given for the second and third operands. This is the central mechanism
//!   of the macro:
//!   - When one operand is a null pointer constant (i.e. when x is an integer
//!     constant expression) and the other is an object pointer (i.e. our
//!     third operand), the conditional operator returns the type of the
//!     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//!     would then get:
//!       sizeof(*((int *)(...))  == sizeof(int)  == 4
//!   - When one operand is a void pointer (i.e. when x is not an integer
//!     constant expression) and the other is an object pointer (i.e. our
//!     third operand), the conditional operator returns a "void *" type.
//!     Here, within the sizeof(), we would then get:
//!       sizeof(*((void *)(...)) == sizeof(void) == 1
//! - The equality comparison to "sizeof(int)" therefore depends on (x):
//!     sizeof(int) == sizeof(int)     (x) was a constant expression
//!     sizeof(int) != sizeof(void)    (x) was not a constant expression
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Monochrome Text Display
//! We need to compute the minimum and maximum values representable in a given
//! type. These macros may also be useful elsewhere. It would seem more obvious
//! to do something like:
//!
//! #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
//! #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
//!
//! Unfortunately, the middle expressions, strictly speaking, have
//! undefined behaviour, and at least some versions of gcc warn about
//! the type_max expression (but not if -fsanitize=undefined is in
//! effect; in that case, the warning is deferred to runtime...).
//!
//! The slightly excessive casting in type_min is to make sure the
//! macros also produce sensible values for the exotic type _Bool. [The
//! overflow checkers only almost work for _Bool, but that's
//! a-feature-not-a-bug, since people shouldn't be doing arithmetic on
//! _Bools. Besides, the gcc builtins don't allow _Bool* as third
//! argument.]
//!
//! Idea stolen from
//! https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
//! credit to Christian Biere.
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Defines x86 CPU feature bits
// This header exists to force full rebuild when the compiler is upgraded.
//
// When fixdep scans this, it will find this string "CONFIG_CC_VERSION_TEXT"
// and add dependency on include/config/CC_VERSION_TEXT, which is touched
// by Kconfig when the version string from the compiler changes.
// Additional tree-wide dependencies start here.
// If any of the GCC plugins change, we need to rebuild everything that
// was built with them, as they may have changed their behavior and those
// behaviors may need to be synchronized across all translation units.
// If the randstruct seed itself changes (whether for GCC plugins or
// Clang), the entire tree needs to be rebuilt since the randomization of
// structures may change between compilation units if not.
// If any external changes affect Clang's integer wrapping sanitizer
// behavior, a full rebuild is needed as the coverage for wrapping types
// may have changed, which may impact the expected behaviors that should
// not differ between compilation units.
// The use of "&&" / "||" is limited in certain expressions.
// The following enable to calculate "and" / "or" with macro expansion only.
// Helper macros to use CONFIG_ options in C/CPP expressions. Note that
// these only work with boolean and tristate options.
// Getting something that works in C and CPP for an arg that may or may
// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
// we match on the placeholder define, insert the "0," for arg1 and generate
// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
// the last step cherry picks the 2nd arg, we get a zero.
// IS_BUILTIN(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y', 0
// otherwise. For boolean options, this is equivalent to
// IS_ENABLED(CONFIG_FOO).
// IS_MODULE(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'm', 0
// otherwise.  CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1" in
// autoconf.h.
// IS_REACHABLE(CONFIG_FOO) evaluates to 1 if the currently compiled
// code can call a function defined in code compiled based on CONFIG_FOO.
// This is similar to IS_ENABLED(), but returns false when invoked from
// built-in code when CONFIG_FOO is set to 'm'.
// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
// __has_builtin is supported on gcc >= 10, clang >= 3 and icc >= 21.
// In the meantime, to support gcc < 10, we implement __has_builtin
// by hand.
// Indirect macros required for expanded argument pasting, eg. __LINE__.
// C23 introduces "auto" as a standard way to define type-inferred
// variables, but "auto" has been a (useless) keyword even since K&R C,
// so it has always been "namespace reserved."
//
// Until at some future time we require C23 support, we need the gcc
// extension __auto_type, but there is no reason to put that elsewhere
// in the source code.
// nothing
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-section-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-section-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-warn_005funused_005fresult-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Label-Attributes.html#index-cold-label-attribute
//
// When -falign-functions=N is in use, we must avoid the cold attribute as
// GCC drops the alignment for cold functions. Worse, GCC can implicitly mark
// callees of cold functions as cold themselves, so it's not sufficient to add
// __function_aligned here as that will not ensure that callees are correctly
// aligned.
//
// See:
//
//   https://lore.kernel.org/lkml/Y77%2FqVgvaJidFpYt@FVFF77S0Q05N
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88345#c9
// On x86-64 and arm64 targets, __preserve_most changes the calling convention
// of a function to make the code in the caller as unintrusive as possible. This
// convention behaves identically to the C calling convention on how arguments
// and return values are passed, but uses a different set of caller- and callee-
// saved registers.
//
// The purpose is to alleviates the burden of saving and recovering a large
// register set before and after the call in the caller.  This is beneficial for
// rarely taken slow paths, such as error-reporting functions that may be called
// from hot paths.
//
// Note: This may conflict with instrumentation inserted on function entry which
// does not use __preserve_most or equivalent convention (if in assembly). Since
// function tracing assumes the normal C calling convention, where the attribute
// is supported, __preserve_most implies notrace.  It is recommended to restrict
// use of the attribute to functions that should or already disable tracing.
//
// Optional: not supported by gcc.
//
// clang: https://clang.llvm.org/docs/AttributeReference.html#preserve-most
// Compiler specific macros.
// Some architectures need to provide custom definitions of macros provided
// by linux/compiler-*.h, and can do so using asm/compiler.h. We include that
// conditionally rather than using an asm-generic wrapper in order to avoid
// build failures if any C compilation, which will include this file via an
// -include argument in c_flags, occurs prior to the asm-generic wrappers being
// generated.
// it doesn't make sense on ARM (currently the only user of __naked)
// to trace naked functions because then mcount is called without
// stack and frame pointer being set up and there is no chance to
// restore the lr register to the value before mcount was called.
// Prefer gnu_inline, so that extern inline functions do not emit an
// externally visible function. This makes extern inline behave as per gnu89
// semantics rather than c99. This prevents multiple symbol definition errors
// of extern inline functions at link time.
// A lot of inline functions can cause havoc with function tracing.
// gcc provides both __inline__ and __inline as alternate spellings of
// the inline keyword, though the latter is undocumented. New kernel
// code should only use the inline spelling, but some existing code
// uses __inline__. Since we #define inline above, to ensure
// __inline__ has the same semantics, we need this #define.
//
// However, the spelling __inline is strictly reserved for referring
// to the bare keyword.
// GCC does not warn about unused static inline functions for -Wunused-function.
// Suppress the warning in clang as well by using __maybe_unused, but enable it
// for W=2 build. This will allow clang to find unused functions.
// Rather then using noinline to prevent stack consumption, use
// noinline_for_stack instead.  For documentation reasons.
// Use noinline_for_tracing for functions that should not be inlined.
// For tracing reasons.
// Sanitizer helper attributes: Because using __always_inline and
// __no_sanitize_* conflict, provide helper attributes that will either expand
// to __no_sanitize_* in compilation units where instrumentation is enabled
// (__SANITIZE_*__), or __always_inline in compilation units without
// instrumentation (__SANITIZE_*__ undefined).
//
// The assume attribute is used to indicate that a certain condition is
// assumed to be true. If this condition is violated at runtime, the behavior
// is undefined. Compilers may or may not use this indication to generate
// optimized code.
//
// Note that the clang documentation states that optimizers may react
// differently to this attribute, and this may even have a negative
// performance impact. Therefore this attribute should be used with care.
//
// Optional: only supported since gcc >= 13
// Optional: only supported since clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Statement-Attributes.html#index-assume-statement-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#id13
// Optional: only supported since gcc >= 15
// Optional: only supported since clang >= 18
//
//   gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108896
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
//
// __bdos on clang < 19.1.2 can erroneously return 0:
// https://github.com/llvm/llvm-project/pull/110497
//
// __bdos on clang < 19.1.3 can be off by 4:
// https://github.com/llvm/llvm-project/pull/112636
// Runtime track number of objects pointed to by a pointer member for use by
// CONFIG_FORTIFY_SOURCE and CONFIG_UBSAN_BOUNDS.
//
// Optional: only supported since gcc >= 16
// Optional: only supported since clang >= 22
//
//   gcc: https://gcc.gnu.org/pipermail/gcc-patches/2025-April/681727.html
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
// Optional: only supported since gcc >= 15
// Optional: not supported by Clang
//
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117178
// Apply __counted_by() when the Endianness matches to increase test coverage.
// This designates the minimum number of elements a passed array parameter must
// have. For example:
//
//     void some_function(u8 param[at_least 7]);
//
// If a caller passes an array with fewer than 7 elements, the compiler will
// emit a warning.
// Section for code which can't be instrumented at all
// The __cpuidle section is used twofold:
//
//  1) the original use -- identifying if a CPU is 'stuck' in idle state based
//     on it's instruction pointer. See cpu_in_idle().
//
//  2) supressing instrumentation around where cpuidle disables RCU; where the
//     function isn't strictly required for #1, this is interchangeable with
//     noinstr.
// The below symbols may be defined for one or more, but not ALL, of the above
// compilers. We don't consider that to be an error, so set them to nothing.
// For example, some of them are for compiler specific plugins.
// Any place that could be marked with the "alloc_size" attribute is also
// a place to be marked with the "malloc" attribute, except those that may
// be performing a _reallocation_, as that may alias the existing pointer.
// For these, use __realloc_size().
// When the size of an allocated object is needed, use the best available
// mechanism to find it. (For cases where sizeof() cannot be used.)
//
// Optional: only supported since gcc >= 12
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#evaluating-object-size
// Determine if an attribute has been applied to a variable.
// Using __annotated needs to check for __annotated being available,
// or negative tests may fail when annotation cannot be checked. For
// example, see the definition of __is_cstr().
// __flex_counter() - Get pointer to counter member for the given
//                    flexible array, if it was annotated with __counted_by()
// @FAM: Pointer to flexible array member of an addressable struct instance
//
// For example, with:
//
// 	struct foo {
// 		int counter;
// 		short array[] __counted_by(counter);
// 	} *p;
//
// __flex_counter(p->array) will resolve to &p->counter.
//
// Note that Clang may not allow this to be assigned to a separate
// variable; it must be used directly.
//
// If p->array is unannotated, this returns (void *)NULL.
// Some versions of gcc do not mark 'asm goto' volatile:
//
//  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103979
//
// We do it here by hand, because it doesn't hurt.
// Clang has trouble with constraints with multiple
// alternative behaviors ("g" , "rm" and "=rm").
// Use __typeof_unqual__() when available.
// Are two types/vars the same type (ignoring qualifiers)?
// __unqual_scalar_typeof(x) - Declare an unqualified scalar type, leaving
// 			       non-scalar types unchanged.
// __signed_scalar_typeof(x) - Declare a signed scalar type, leaving
// 			       non-scalar types unchanged.
// Is this type a native word size -- useful for atomic operations
// #ifdef __OPTIMIZE__ is only a good approximation; for instance "make
// CFLAGS_foo.o=-Og" defines __OPTIMIZE__, does not elide the conditional code
// and can break compilation with wrong error message(s). Combine with
// -U__OPTIMIZE__ when needed.
// 									 * __noreturn is needed to give the compiler enough			 * information to avoid certain possibly-uninitialized			 * warnings (regardless of the build failing).
// compiletime_assert - break build and emit msg if condition is false
// @condition: a compile-time constant condition to check
// @msg:       a message to emit if condition is false
//
// In tradition of POSIX assert, this macro will break the build if the
// supplied condition is *false*, emitting the supplied error message if the
// compiler has support to do so.
// Helpers for emitting diagnostics in pragmas.
// max number of signatures to store
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
// Similar to statically_true() but produces a constant expression
//
// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
// which require their input to be a constant expression and for which
// statically_true() would otherwise fail.
//
// This is a trade-off: const_true() requires all its operands to be
// compile time constants. Else, it would always returns false even on
// the most trivial cases like:
//
//   true || non_const_var
//
// On the opposite, statically_true() is able to fold more complex
// tautologies and will return true on expressions such as:
//
//   !(non_const_var * 8 % 4)
//
// For the general case, statically_true() is better.
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
// check_add_overflow() - Calculate addition with overflow checking
// @a: first addend
// @b: second addend
// @d: pointer to store sum
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted addition, regardless of whether
// wrap-around occurred.
// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// N 32-bit words worth of info
// Skipped when running bindgen due to a libclang issue;
// see https://github.com/rust-lang/rust-bindgen/issues/2244.
// Annotating a function/variable with __retain tells the compiler to place
// the object in its own section and set the flag SHF_GNU_RETAIN. This flag
// instructs the linker to retain the object during garbage-cleanup or LTO
// phases.
//
// Note that the __used macro is also used to prevent functions or data
// being optimized out, but operates at the compiler/IR-level and may still
// allow unintended removal of objects during linking.
//
// Optional: only supported since gcc >= 11, clang >= 13
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-retain-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#retain
macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u32).wrapping_add(($off) as u32) as *mut ()
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

unsafe extern "C" {
    static mut pio_ops: port_io_ops;
    fn intcall(arg0: u8, arg1: *const biosregs, arg2: *mut biosregs);
    fn initregs(arg0: *mut biosregs);
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
#[repr(C)]
#[derive(Clone, Copy)]
struct port_io_ops {
    f_inb: Option<unsafe extern "C" fn(u16) -> u8>,
    f_outb: Option<unsafe extern "C" fn(u8, u16)>,
    f_outw: Option<unsafe extern "C" fn(u16, u16)>,
}


// Redirect port I/O operations via pio_ops callbacks.
// TDX guests override these callbacks with TDX-specific helpers.
fn io_delay() {
let DELAY_PORT: u16 = 128;
unsafe { unsafe { pio_ops.f_outb }.unwrap()(0u8, DELAY_PORT) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_boot_h_205 {
    __slate_anon_0: __SlateRecord_boot_h_206,
    __slate_anon_1: __SlateRecord_boot_h_219,
    __slate_anon_2: __SlateRecord_boot_h_232,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_boot_h_206 {
    edi: u32,
    esi: u32,
    ebp: u32,
    _esp: u32,
    ebx: u32,
    edx: u32,
    ecx: u32,
    eax: u32,
    _fsgs: u32,
    _dses: u32,
    eflags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_boot_h_219 {
    di: u16,
    hdi: u16,
    si: u16,
    hsi: u16,
    bp: u16,
    hbp: u16,
    _sp: u16,
    _hsp: u16,
    bx: u16,
    hbx: u16,
    dx: u16,
    hdx: u16,
    cx: u16,
    hcx: u16,
    ax: u16,
    hax: u16,
    gs: u16,
    fs: u16,
    es: u16,
    ds: u16,
    flags: u16,
    hflags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_boot_h_232 {
    dil: u8,
    dih: u8,
    edi2: u8,
    edi3: u8,
    sil: u8,
    sih: u8,
    esi2: u8,
    esi3: u8,
    bpl: u8,
    bph: u8,
    ebp2: u8,
    ebp3: u8,
    _spl: u8,
    _sph: u8,
    _esp2: u8,
    _esp3: u8,
    bl: u8,
    bh: u8,
    ebx2: u8,
    ebx3: u8,
    dl: u8,
    dh: u8,
    edx2: u8,
    edx3: u8,
    cl: u8,
    ch: u8,
    ecx2: u8,
    ecx3: u8,
    al: u8,
    ah: u8,
    eax2: u8,
    eax3: u8,
}


fn set_fs(mut seg: u16) {
unsafe {
core::arch::asm!("movw {0:x},%fs", in(reg) seg as u16, options(nostack, att_syntax));
}
}

fn set_gs(mut seg: u16) {
unsafe {
core::arch::asm!("movw {0:x},%gs", in(reg) seg as u16, options(nostack, att_syntax));
}
}

fn rdfs32(addr: u32) -> u32 {
let mut ptr: *mut u32 = absolute_pointer!(addr) as *mut u32;
let mut v: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl %fs:({1:e}),{0:e}", lateout(reg) __t0, in(reg) ptr, options(nostack, att_syntax));
v = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
v
}

fn wrfs32(mut v: u32, addr: u32) {
let mut ptr: *mut u32 = absolute_pointer!(addr) as *mut u32;
unsafe {
core::arch::asm!("movl {1:e},%fs:({0:e})", in(reg) ptr, in(reg) v as u32, options(nostack, att_syntax));
}
}

fn rdgs32(addr: u32) -> u32 {
let mut ptr: *mut u32 = absolute_pointer!(addr) as *mut u32;
let mut v: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl %gs:({1:e}),{0:e}", lateout(reg) __t0, in(reg) ptr, options(nostack, att_syntax));
v = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
v
}

#[repr(C)]
#[derive(Clone, Copy)]
struct biosregs {
    __slate_anon_0: __SlateRecord_boot_h_205,
}


// Enable A20 gate (return -1 on failure)
fn empty_8042() -> i32 {
let mut status: u8 = 0;
let mut loops: i32 = 100000;
let mut ffs: i32 = 32;
loop {
let __v2: i32 = loops;
loops = __v2 - 1;
if __v2 == 0 {
break;
}
io_delay();
status = unsafe { unsafe { pio_ops.f_inb }.unwrap()(100u16) };
if (status as i32) == 255 {
// FF is a plausible, but very unlikely status
let __v1: i32 = ffs - 1;
ffs = __v1;
if __v1 == 0 {
return -1;
}
// Assume no KBC present
}
if (status as i32) & 1 != 0 {
// Read and discard input data
io_delay();
unsafe { unsafe { pio_ops.f_inb }.unwrap()(96u16) };
} else {
if (status as i32) & 2 == 0 {
// Buffers empty, finished!
return 0;
}
}
}
-1
}

// Returns nonzero if the A20 line is enabled.  The memory address
// used as a test is the int $0x80 vector, which should be safe.
// 2^21
fn a20_test(mut loops: i32) -> i32 {
let mut ok: i32 = 0;
let mut saved: i32 = 0;
let mut ctr: i32 = 0;
set_fs(0u16);
set_gs(65535u16);
let __v0: i32 = rdfs32(512u32) as i32;
ctr = __v0;
saved = __v0;
loop {
let __v4: i32 = loops;
loops = __v4 - 1;
if __v4 == 0 {
break;
}
let __v2: i32 = ctr + 1;
ctr = __v2;
wrfs32(__v2 as u32, 512u32);
io_delay(); // Serialize and make delay constant
ok = (rdgs32(528u32) ^ (ctr as u32)) as i32;
if ok != 0 {
break;
}
}
wrfs32(saved as u32, 512u32);
ok
}

/// Quick test to see if A20 is already enabled
fn a20_test_short() -> i32 {
a20_test(32i32)
}

/// Longer test that actually waits for A20 to come on line; this
/// is useful when dealing with the KBC or other slow external circuitry.
fn a20_test_long() -> i32 {
a20_test(2097152i32)
}

fn enable_a20_bios() {
let mut ireg: biosregs = unsafe { core::mem::zeroed() };
unsafe { initregs(core::ptr::addr_of_mut!(ireg)) };
unsafe {
ireg.__slate_anon_0.__slate_anon_1.ax = 9217;
}
unsafe { intcall(21u8, core::ptr::addr_of_mut!(ireg) as *const biosregs, core::ptr::null_mut::<biosregs>()) };
}

fn enable_a20_kbc() {
empty_8042();
unsafe { unsafe { pio_ops.f_outb }.unwrap()(209u8, 100u16) }; // Command write
empty_8042();
unsafe { unsafe { pio_ops.f_outb }.unwrap()(223u8, 96u16) }; // A20 on
empty_8042();
unsafe { unsafe { pio_ops.f_outb }.unwrap()(255u8, 100u16) }; // Null command, but UHCI wants it
empty_8042();
}

fn enable_a20_fast() {
// Configuration port A
let mut port_a: u8 = unsafe { unsafe { pio_ops.f_inb }.unwrap()(146u16) };
port_a = ((port_a as i32) | 2) as u8; // Enable A20
let __v3: u8 = ((port_a as i32) & !1) as u8;
port_a = __v3; // Do not reset machine
unsafe { unsafe { pio_ops.f_outb }.unwrap()(port_a, 146u16) };
}

// Actual routine to enable A20; return 0 on ok, -1 on failure
// Number of times to try
#[unsafe(no_mangle)]
extern "C" fn enable_a20() -> i32 {
let mut loops: i32 = 255;
let mut kbc_err: i32 = 0;
loop {
let __v0: i32 = loops;
loops = __v0 - 1;
if __v0 == 0 {
break;
}
// First, check to see if A20 is already enabled
// (legacy free, etc.)
if a20_test_short() != 0 {
return 0;
}
// Next, try the BIOS (INT 0x15, AX=0x2401)
enable_a20_bios();
if a20_test_short() != 0 {
return 0;
}
// Try enabling A20 through the keyboard controller
kbc_err = empty_8042();
if a20_test_short() != 0 {
return 0;
}
// BIOS worked, but with delayed reaction
if kbc_err == 0 {
enable_a20_kbc();
if a20_test_long() != 0 {
return 0;
}
}
// Finally, try enabling the "fast A20 gate"
enable_a20_fast();
if a20_test_long() != 0 {
return 0;
}
}
-1
}
