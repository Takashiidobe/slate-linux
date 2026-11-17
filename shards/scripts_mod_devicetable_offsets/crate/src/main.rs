#![no_std]
#![feature(linkage)]
#![no_main]

//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//! This version is i.e. to prevent dead stores elimination on @ptr
//! where gcc and llvm may behave differently when otherwise using
//! normal barrier(): while gcc behavior gets along with a normal
//! barrier(), llvm needs an explicit input variable to be assumed
//! clobbered. The issue is as follows: while the inline asm might
//! access any memory it wants, the compiler could have fit all of
//! @ptr into memory registers instead, and since @ptr never escaped
//! from that, it proved that the inline asm wasn't touching any of
//! it. This version works well with both compilers, i.e. we're telling
//! the compiler that the inline asm absolutely may see the contents
//! of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Value too large for defined data type
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! check_add_overflow() - Calculate addition with overflow checking
//! @a: first addend
//! @b: second addend
//! @d: pointer to store sum
//!
//! Returns true on wrap-around, false otherwise.
//!
//! *@d holds the results of the attempted addition, regardless of whether
//! wrap-around occurred.
//! 64 bit
//! Input
mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


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
// Note: do not use this directly. Instead, use __alloc_size() since it is conditionally
// available and includes other attributes. For GCC < 9.1, __alloc_size__ gets undefined
// in compiler-gcc.h, due to misbehaviors.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alloc_005fsize-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#alloc-size
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
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
// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn __slate_main() -> i32 {
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_usb_device_id ${0} sizeof(struct usb_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_match_flags ${0} offsetof(struct usb_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_idVendor ${0} offsetof(struct usb_device_id, idVendor)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_idProduct ${0} offsetof(struct usb_device_id, idProduct)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bcdDevice_lo ${0} offsetof(struct usb_device_id, bcdDevice_lo)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bcdDevice_hi ${0} offsetof(struct usb_device_id, bcdDevice_hi)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bDeviceClass ${0} offsetof(struct usb_device_id, bDeviceClass)\"", const 10 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bDeviceSubClass ${0} offsetof(struct usb_device_id, bDeviceSubClass)\"", const 11 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bDeviceProtocol ${0} offsetof(struct usb_device_id, bDeviceProtocol)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bInterfaceClass ${0} offsetof(struct usb_device_id, bInterfaceClass)\"", const 13 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bInterfaceSubClass ${0} offsetof(struct usb_device_id, bInterfaceSubClass)\"", const 14 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bInterfaceProtocol ${0} offsetof(struct usb_device_id, bInterfaceProtocol)\"", const 15 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_usb_device_id_bInterfaceNumber ${0} offsetof(struct usb_device_id, bInterfaceNumber)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_hid_device_id ${0} sizeof(struct hid_device_id)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hid_device_id_bus ${0} offsetof(struct hid_device_id, bus)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hid_device_id_group ${0} offsetof(struct hid_device_id, group)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hid_device_id_vendor ${0} offsetof(struct hid_device_id, vendor)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hid_device_id_product ${0} offsetof(struct hid_device_id, product)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ieee1394_device_id ${0} sizeof(struct ieee1394_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ieee1394_device_id_match_flags ${0} offsetof(struct ieee1394_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ieee1394_device_id_vendor_id ${0} offsetof(struct ieee1394_device_id, vendor_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ieee1394_device_id_model_id ${0} offsetof(struct ieee1394_device_id, model_id)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ieee1394_device_id_specifier_id ${0} offsetof(struct ieee1394_device_id, specifier_id)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ieee1394_device_id_version ${0} offsetof(struct ieee1394_device_id, version)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_pci_device_id ${0} sizeof(struct pci_device_id)\"", const 40 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_vendor ${0} offsetof(struct pci_device_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_device ${0} offsetof(struct pci_device_id, device)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_subvendor ${0} offsetof(struct pci_device_id, subvendor)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_subdevice ${0} offsetof(struct pci_device_id, subdevice)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_class ${0} offsetof(struct pci_device_id, class)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_class_mask ${0} offsetof(struct pci_device_id, class_mask)\"", const 20 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pci_device_id_override_only ${0} offsetof(struct pci_device_id, override_only)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ccw_device_id ${0} sizeof(struct ccw_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ccw_device_id_match_flags ${0} offsetof(struct ccw_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ccw_device_id_cu_type ${0} offsetof(struct ccw_device_id, cu_type)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ccw_device_id_cu_model ${0} offsetof(struct ccw_device_id, cu_model)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ccw_device_id_dev_type ${0} offsetof(struct ccw_device_id, dev_type)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ccw_device_id_dev_model ${0} offsetof(struct ccw_device_id, dev_model)\"", const 7 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ap_device_id ${0} sizeof(struct ap_device_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ap_device_id_dev_type ${0} offsetof(struct ap_device_id, dev_type)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_css_device_id ${0} sizeof(struct css_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_css_device_id_type ${0} offsetof(struct css_device_id, type)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_serio_device_id ${0} sizeof(struct serio_device_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_serio_device_id_type ${0} offsetof(struct serio_device_id, type)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_serio_device_id_proto ${0} offsetof(struct serio_device_id, proto)\"", const 3 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_serio_device_id_id ${0} offsetof(struct serio_device_id, id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_serio_device_id_extra ${0} offsetof(struct serio_device_id, extra)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_acpi_device_id ${0} sizeof(struct acpi_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_acpi_device_id_id ${0} offsetof(struct acpi_device_id, id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_acpi_device_id_cls ${0} offsetof(struct acpi_device_id, cls)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_acpi_device_id_cls_msk ${0} offsetof(struct acpi_device_id, cls_msk)\"", const 28 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_pnp_device_id ${0} sizeof(struct pnp_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pnp_device_id_id ${0} offsetof(struct pnp_device_id, id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_pnp_card_device_id ${0} sizeof(struct pnp_card_device_id)\"", const 80 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pnp_card_device_id_devs ${0} offsetof(struct pnp_card_device_id, devs)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_pcmcia_device_id ${0} sizeof(struct pcmcia_device_id)\"", const 80 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_match_flags ${0} offsetof(struct pcmcia_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_manf_id ${0} offsetof(struct pcmcia_device_id, manf_id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_card_id ${0} offsetof(struct pcmcia_device_id, card_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_func_id ${0} offsetof(struct pcmcia_device_id, func_id)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_function ${0} offsetof(struct pcmcia_device_id, function)\"", const 7 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_device_no ${0} offsetof(struct pcmcia_device_id, device_no)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_pcmcia_device_id_prod_id_hash ${0} offsetof(struct pcmcia_device_id, prod_id_hash)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_of_device_id ${0} sizeof(struct of_device_id)\"", const 200 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_of_device_id_name ${0} offsetof(struct of_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_of_device_id_type ${0} offsetof(struct of_device_id, type)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_of_device_id_compatible ${0} offsetof(struct of_device_id, compatible)\"", const 64 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_vio_device_id ${0} sizeof(struct vio_device_id)\"", const 64 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_vio_device_id_type ${0} offsetof(struct vio_device_id, type)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_vio_device_id_compat ${0} offsetof(struct vio_device_id, compat)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_input_device_id ${0} sizeof(struct input_device_id)\"", const 200 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_flags ${0} offsetof(struct input_device_id, flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_bustype ${0} offsetof(struct input_device_id, bustype)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_vendor ${0} offsetof(struct input_device_id, vendor)\"", const 10 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_product ${0} offsetof(struct input_device_id, product)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_version ${0} offsetof(struct input_device_id, version)\"", const 14 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_evbit ${0} offsetof(struct input_device_id, evbit)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_keybit ${0} offsetof(struct input_device_id, keybit)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_relbit ${0} offsetof(struct input_device_id, relbit)\"", const 120 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_absbit ${0} offsetof(struct input_device_id, absbit)\"", const 128 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_mscbit ${0} offsetof(struct input_device_id, mscbit)\"", const 136 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_ledbit ${0} offsetof(struct input_device_id, ledbit)\"", const 144 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_sndbit ${0} offsetof(struct input_device_id, sndbit)\"", const 152 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_ffbit ${0} offsetof(struct input_device_id, ffbit)\"", const 160 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_input_device_id_swbit ${0} offsetof(struct input_device_id, swbit)\"", const 176 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_eisa_device_id ${0} sizeof(struct eisa_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_eisa_device_id_sig ${0} offsetof(struct eisa_device_id, sig)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_parisc_device_id ${0} sizeof(struct parisc_device_id)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_parisc_device_id_hw_type ${0} offsetof(struct parisc_device_id, hw_type)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_parisc_device_id_hversion ${0} offsetof(struct parisc_device_id, hversion)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_parisc_device_id_hversion_rev ${0} offsetof(struct parisc_device_id, hversion_rev)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_parisc_device_id_sversion ${0} offsetof(struct parisc_device_id, sversion)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_sdio_device_id ${0} sizeof(struct sdio_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdio_device_id_class ${0} offsetof(struct sdio_device_id, class)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdio_device_id_vendor ${0} offsetof(struct sdio_device_id, vendor)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdio_device_id_device ${0} offsetof(struct sdio_device_id, device)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ssb_device_id ${0} sizeof(struct ssb_device_id)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssb_device_id_vendor ${0} offsetof(struct ssb_device_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssb_device_id_coreid ${0} offsetof(struct ssb_device_id, coreid)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssb_device_id_revision ${0} offsetof(struct ssb_device_id, revision)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_bcma_device_id ${0} sizeof(struct bcma_device_id)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_bcma_device_id_manuf ${0} offsetof(struct bcma_device_id, manuf)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_bcma_device_id_id ${0} offsetof(struct bcma_device_id, id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_bcma_device_id_rev ${0} offsetof(struct bcma_device_id, rev)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_bcma_device_id_class ${0} offsetof(struct bcma_device_id, class)\"", const 5 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_virtio_device_id ${0} sizeof(struct virtio_device_id)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_virtio_device_id_device ${0} offsetof(struct virtio_device_id, device)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_virtio_device_id_vendor ${0} offsetof(struct virtio_device_id, vendor)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_hv_vmbus_device_id ${0} sizeof(struct hv_vmbus_device_id)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hv_vmbus_device_id_guid ${0} offsetof(struct hv_vmbus_device_id, guid)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_rpmsg_device_id ${0} sizeof(struct rpmsg_device_id)\"", const 40 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_rpmsg_device_id_name ${0} offsetof(struct rpmsg_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_i2c_device_id ${0} sizeof(struct i2c_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i2c_device_id_name ${0} offsetof(struct i2c_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_i3c_device_id ${0} sizeof(struct i3c_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i3c_device_id_match_flags ${0} offsetof(struct i3c_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i3c_device_id_dcr ${0} offsetof(struct i3c_device_id, dcr)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i3c_device_id_manuf_id ${0} offsetof(struct i3c_device_id, manuf_id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i3c_device_id_part_id ${0} offsetof(struct i3c_device_id, part_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_i3c_device_id_extra_info ${0} offsetof(struct i3c_device_id, extra_info)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_slim_device_id ${0} sizeof(struct slim_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_slim_device_id_manf_id ${0} offsetof(struct slim_device_id, manf_id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_slim_device_id_prod_code ${0} offsetof(struct slim_device_id, prod_code)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_spi_device_id ${0} sizeof(struct spi_device_id)\"", const 40 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_spi_device_id_name ${0} offsetof(struct spi_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_dmi_system_id ${0} sizeof(struct dmi_system_id)\"", const 344 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_dmi_system_id_matches ${0} offsetof(struct dmi_system_id, matches)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_platform_device_id ${0} sizeof(struct platform_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_platform_device_id_name ${0} offsetof(struct platform_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_mdio_device_id ${0} sizeof(struct mdio_device_id)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mdio_device_id_phy_id ${0} offsetof(struct mdio_device_id, phy_id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mdio_device_id_phy_id_mask ${0} offsetof(struct mdio_device_id, phy_id_mask)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_zorro_device_id ${0} sizeof(struct zorro_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_zorro_device_id_id ${0} offsetof(struct zorro_device_id, id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_isapnp_device_id ${0} sizeof(struct isapnp_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_isapnp_device_id_vendor ${0} offsetof(struct isapnp_device_id, vendor)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_isapnp_device_id_function ${0} offsetof(struct isapnp_device_id, function)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ipack_device_id ${0} sizeof(struct ipack_device_id)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ipack_device_id_format ${0} offsetof(struct ipack_device_id, format)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ipack_device_id_vendor ${0} offsetof(struct ipack_device_id, vendor)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ipack_device_id_device ${0} offsetof(struct ipack_device_id, device)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_amba_id ${0} sizeof(struct amba_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_amba_id_id ${0} offsetof(struct amba_id, id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_amba_id_mask ${0} offsetof(struct amba_id, mask)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_mips_cdmm_device_id ${0} sizeof(struct mips_cdmm_device_id)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mips_cdmm_device_id_type ${0} offsetof(struct mips_cdmm_device_id, type)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_x86_cpu_id ${0} sizeof(struct x86_cpu_id)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_x86_cpu_id_feature ${0} offsetof(struct x86_cpu_id, feature)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_x86_cpu_id_family ${0} offsetof(struct x86_cpu_id, family)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_x86_cpu_id_model ${0} offsetof(struct x86_cpu_id, model)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_x86_cpu_id_vendor ${0} offsetof(struct x86_cpu_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_cpu_feature ${0} sizeof(struct cpu_feature)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cpu_feature_feature ${0} offsetof(struct cpu_feature, feature)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_mcb_device_id ${0} sizeof(struct mcb_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mcb_device_id_device ${0} offsetof(struct mcb_device_id, device)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_mei_cl_device_id ${0} sizeof(struct mei_cl_device_id)\"", const 64 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mei_cl_device_id_name ${0} offsetof(struct mei_cl_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mei_cl_device_id_uuid ${0} offsetof(struct mei_cl_device_id, uuid)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mei_cl_device_id_version ${0} offsetof(struct mei_cl_device_id, version)\"", const 48 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_rio_device_id ${0} sizeof(struct rio_device_id)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_rio_device_id_did ${0} offsetof(struct rio_device_id, did)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_rio_device_id_vid ${0} offsetof(struct rio_device_id, vid)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_rio_device_id_asm_did ${0} offsetof(struct rio_device_id, asm_did)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_rio_device_id_asm_vid ${0} offsetof(struct rio_device_id, asm_vid)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ulpi_device_id ${0} sizeof(struct ulpi_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ulpi_device_id_vendor ${0} offsetof(struct ulpi_device_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ulpi_device_id_product ${0} offsetof(struct ulpi_device_id, product)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_hda_device_id ${0} sizeof(struct hda_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hda_device_id_vendor_id ${0} offsetof(struct hda_device_id, vendor_id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hda_device_id_rev_id ${0} offsetof(struct hda_device_id, rev_id)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_hda_device_id_api_version ${0} offsetof(struct hda_device_id, api_version)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_sdw_device_id ${0} sizeof(struct sdw_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdw_device_id_mfg_id ${0} offsetof(struct sdw_device_id, mfg_id)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdw_device_id_part_id ${0} offsetof(struct sdw_device_id, part_id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdw_device_id_sdw_version ${0} offsetof(struct sdw_device_id, sdw_version)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_sdw_device_id_class_id ${0} offsetof(struct sdw_device_id, class_id)\"", const 5 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_fsl_mc_device_id ${0} sizeof(struct fsl_mc_device_id)\"", const 18 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_fsl_mc_device_id_vendor ${0} offsetof(struct fsl_mc_device_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_fsl_mc_device_id_obj_type ${0} offsetof(struct fsl_mc_device_id, obj_type)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_tb_service_id ${0} sizeof(struct tb_service_id)\"", const 40 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tb_service_id_match_flags ${0} offsetof(struct tb_service_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tb_service_id_protocol_key ${0} offsetof(struct tb_service_id, protocol_key)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tb_service_id_protocol_id ${0} offsetof(struct tb_service_id, protocol_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tb_service_id_protocol_version ${0} offsetof(struct tb_service_id, protocol_version)\"", const 20 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tb_service_id_protocol_revision ${0} offsetof(struct tb_service_id, protocol_revision)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_typec_device_id ${0} sizeof(struct typec_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_typec_device_id_svid ${0} offsetof(struct typec_device_id, svid)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_tee_client_device_id ${0} sizeof(struct tee_client_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_tee_client_device_id_uuid ${0} offsetof(struct tee_client_device_id, uuid)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_wmi_device_id ${0} sizeof(struct wmi_device_id)\"", const 48 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_wmi_device_id_guid_string ${0} offsetof(struct wmi_device_id, guid_string)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_mhi_device_id ${0} sizeof(struct mhi_device_id)\"", const 40 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_mhi_device_id_chan ${0} offsetof(struct mhi_device_id, chan)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_auxiliary_device_id ${0} sizeof(struct auxiliary_device_id)\"", const 48 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_auxiliary_device_id_name ${0} offsetof(struct auxiliary_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ssam_device_id ${0} sizeof(struct ssam_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_match_flags ${0} offsetof(struct ssam_device_id, match_flags)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_domain ${0} offsetof(struct ssam_device_id, domain)\"", const 1 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_category ${0} offsetof(struct ssam_device_id, category)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_target ${0} offsetof(struct ssam_device_id, target)\"", const 3 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_instance ${0} offsetof(struct ssam_device_id, instance)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ssam_device_id_function ${0} offsetof(struct ssam_device_id, function)\"", const 5 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_dfl_device_id ${0} sizeof(struct dfl_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_dfl_device_id_type ${0} offsetof(struct dfl_device_id, type)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_dfl_device_id_feature_id ${0} offsetof(struct dfl_device_id, feature_id)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_ishtp_device_id ${0} sizeof(struct ishtp_device_id)\"", const 24 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_ishtp_device_id_guid ${0} offsetof(struct ishtp_device_id, guid)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_cdx_device_id ${0} sizeof(struct cdx_device_id)\"", const 20 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_vendor ${0} offsetof(struct cdx_device_id, vendor)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_device ${0} offsetof(struct cdx_device_id, device)\"", const 2 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_subvendor ${0} offsetof(struct cdx_device_id, subvendor)\"", const 4 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_subdevice ${0} offsetof(struct cdx_device_id, subdevice)\"", const 6 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_class ${0} offsetof(struct cdx_device_id, class)\"", const 8 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_class_mask ${0} offsetof(struct cdx_device_id, class_mask)\"", const 12 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_cdx_device_id_override_only ${0} offsetof(struct cdx_device_id, override_only)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_vchiq_device_id ${0} sizeof(struct vchiq_device_id)\"", const 32 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_vchiq_device_id_name ${0} offsetof(struct vchiq_device_id, name)\"", const 0 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->SIZE_coreboot_device_id ${0} sizeof(struct coreboot_device_id)\"", const 16 as u64, options(nostack, att_syntax));
}
unsafe {
core::arch::asm!("\n.ascii \"->OFF_coreboot_device_id_tag ${0} offsetof(struct coreboot_device_id, tag)\"", const 0 as u64, options(nostack, att_syntax));
}
0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn main(argc: i32, argv: *mut *mut core::ffi::c_char, envp: *mut *mut core::ffi::c_char) -> i32 {
unsafe { __slate_main() }
}
