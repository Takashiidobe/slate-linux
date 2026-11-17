
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! context_unsafe() - disable context checking for contained code
//!
//! Disables context checking for contained statements or expression.
//!
//! .. code-block:: c
//!
//! 	struct some_data {
//! 		spinlock_t lock;
//! 		int counter __guarded_by(&lock);
//! 	};
//!
//! 	int foo(struct some_data *d)
//! 	{
//! 		// ...
//! 		// other code that is still checked ...
//! 		// ...
//! 		return context_unsafe(d->counter);
//! 	}
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+
//!
//! XArray implementation
//! Copyright (c) 2017-2018 Microsoft Corporation
//! Copyright (c) 2018-2020 Oracle
//! Author: Matthew Wilcox <willy@infradead.org>
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! The type of an index into the pagecache.
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! Atomic operations are already serializing on x86
//! Many architecture-specific non-atomic bitops contain inline asm code and due
//! to that the compiler can't optimize them to compile-time expressions or
//! constants. In contrary, generic_*() helpers are defined in pure C and
//! compilers optimize them just well.
//! Therefore, to make `unsigned long foo = 0; __set_bit(BAR, &foo)` effectively
//! equal to `unsigned long foo = BIT(BAR)`, pick the generic C alternative when
//! the arguments can be resolved at compile time. That expression itself is a
//! constant and doesn't bring any functional changes to the rest of cases.
//! The casts to `uintptr_t` are needed to mitigate `-Waddress` warnings when
//! passing a bitmap from .bss or .data (-> `!!addr` is always true).
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! No such file or directory
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! To support Context Analysis, we need to allow the compiler to see the
//! acquisition and release of the context lock. However, the "cleanup" helpers
//! wrap the lock in a struct passed through separate helper functions, which
//! hides the lock alias from the compiler (no inter-procedural analysis).
//!
//! To make it work, we introduce an explicit alias to the context lock instance
//! that is "cleaned" up with a separate cleanup helper. This helper is a dummy
//! function that does nothing at runtime, but has the "_unlock" attribute to
//! tell the compiler what happens at the end of the scope.
//!
//! To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
//! to redefine the constructor, which then also creates the alias variable with
//! the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
//! used.
//!
//! Example usage:
//!
//!   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//!   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//!
//! Note: To support the for-loop based scoped helpers, the auxiliary variable
//! must be a pointer to the "class" type because it is defined in the same
//! statement as the guard variable. However, we initialize it with the lock
//! pointer (despite the type mismatch, the compiler's alias analysis still works
//! as expected). The "_unlock" attribute receives a pointer to the auxiliary
//! variable (a double pointer to the class type), and must be cast and
//! dereferenced appropriately.
//! restart by calling sys_restart_syscall
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! CR3's layout varies depending on several things.
//!
//! If CR4.PCIDE is set (64-bit only), then CR3[11:0] is the address space ID.
//! If PAE is enabled, then CR3[11:5] is part of the PDPT address
//! (i.e. it's 32-byte aligned, not page-aligned) and CR3[4:0] is ignored.
//! Otherwise (non-PAE, non-PCID), CR3[3] is PWT, CR3[4] is PCD, and
//! CR3[2:0] and CR3[11:5] are ignored.
//!
//! In all cases, Linux puts zeros in the low ignored bits and in PWT and PCD.
//!
//! CR3[63] is always read as zero.  If CR4.PCIDE is set, then CR3[63] may be
//! written as 1 to prevent the write to CR3 from flushing the TLB.
//!
//! On systems with SME, one bit (in a variable position!) is stolen to indicate
//! that the top-level paging structure is encrypted.
//!
//! On systemms with LAM, bits 61 and 62 are used to indicate LAM mode.
//!
//! All of the remaining bits indicate the physical address of the top-level
//! paging structure.
//!
//! CR3_ADDR_MASK is the mask used by read_cr3_pa().
//!
//! Mask off the address space ID and SME encryption bits.
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! on 4KB pages
//! Use alternative patching for paravirt calls:
//! - For replacing an indirect call with a direct one, use the "normal"
//!   ALTERNATIVE() macro with the indirect call as the initial code sequence,
//!   which will be replaced with the related direct call by using the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! - In case the replacement is either a direct call or a short code sequence
//!   depending on a feature bit, the ALTERNATIVE_2() macro is being used.
//!   The indirect call is the initial code sequence again, while the special
//!   code sequence is selected with the specified feature bit. In case the
//!   feature is not active, the direct call is used as above via the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! abs - return absolute value of an argument
//! @x: the value.
//!
//! If it is unsigned type, @x is converted to signed type first.
//! char is treated as if it was signed (regardless of whether it really is)
//! but the macro's return type is preserved as char.
//!
//! NOTE, for signed type if @x is the minimum, the returned result is undefined
//! as there is not enough bits to represent it as a positive number.
//!
//! Return: an absolute value of @x.
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! rescheduling necessary
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! If a tracepoint needs to be called from a header file, it is not
//! recommended to call it directly, as tracepoints in header files
//! may cause side-effects and bloat the kernel. Instead, use
//! tracepoint_enabled() to test if the tracepoint is enabled, then if
//! it is, call a wrapper function defined in a C file that will then
//! call the tracepoint.
//!
//! For "trace_foo_bar()", you would need to create a wrapper function
//! in a C file to call trace_foo_bar():
//!   void do_trace_foo_bar(args) { trace_foo_bar(args); }
//! Then in the header file, declare the tracepoint:
//!   DECLARE_TRACEPOINT(foo_bar);
//! And call your wrapper:
//!   static inline void some_inlined_function() {
//!            [..]
//!            if (tracepoint_enabled(foo_bar))
//!                    do_trace_foo_bar(args);
//!            [..]
//!   }
//!
//! Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//!   but is safe to have in headers, where trace_foo_bar_enabled() is not.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! wait_queue_entry::flags
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! The seqlock seqcount_t interface does not prescribe a precise sequence of
//! read begin/retry/end. For readers, typically there is a call to
//! read_seqcount_begin() and read_seqcount_retry(), however, there are more
//! esoteric cases which do not follow this pattern.
//!
//! As a consequence, we take the following best-effort approach for raw usage
//! via seqcount_t under KCSAN: upon beginning a seq-reader critical section,
//! pessimistically mark the next KCSAN_SEQLOCK_REGION_MAX memory accesses as
//! atomics; if there is a matching read_seqcount_retry() call, no following
//! memory operations are considered atomic. Usage of the seqlock_t interface
//! is not affected.
//! __MAX_NR_ZONES
//! These constants are defined in the MSan LLVM instrumentation pass.
//! get_task_state():
//! Allocated nodes are mutable until they have been inserted into the tree,
//! at which time they cannot change their type until they have been removed
//! from the tree and an RCU grace period has passed.
//!
//! Removed nodes have their ->parent set to point to themselves.  RCU readers
//! check ->parent before relying on the value that they loaded from the
//! slots array.  This lets us reuse the slots array for the RCU head.
//!
//! Nodes in the tree point to their parent unless bit 0 is set.
//!
//! 64bit sizes
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! Change timeval to jiffies, trying to avoid the
//! most obvious overflows..
//!
//! And some not so obvious.
//!
//! Note that we don't want to return LONG_MAX, because
//! for various timeout reasons we often end up having
//! to wait "jiffies+1" in order to guarantee that we wait
//! at _least_ "jiffies" - so "jiffies+1" had better still
//! be positive.
//! Subtract two ktime_t variables. rem = lhs -rhs:
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! Opaque type representing current VMA (vm_area_struct) flag state. Must be
//! accessed via vma_flags_xxx() helper functions.
//! page_folio - Converts from page to folio.
//! @p: The page.
//!
//! Every page is part of a folio.  This function cannot be called on a
//! NULL pointer.
//!
//! Context: No reference, nor lock is required on @page.  If the caller
//! does not hold a reference, this call may race with a folio split, so
//! it should re-check the folio still contains this page after gaining
//! a reference on the folio.
//! Return: The folio which contains this page.
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! We do arithmetic on the LRU lists in various places in the code,
//! so it is important to keep the active lists LRU_ACTIVE higher in
//! the array than the corresponding inactive lists, and to keep
//! the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//!
//! This has to be kept in sync with the statistics in zone_stat_item
//! above and the descriptions in vmstat_text in mm/vmstat.c
//! This hash multiplies the input by a large odd number and takes the
//! high bits.  Since multiplication propagates changes to the most
//! significant end only, it is essential that the high bits of the
//! product be used for the hash value.
//!
//! Chuck Lever verified the effectiveness of this technique:
//! http://www.citi.umich.edu/techreports/reports/citi-tr-00-1.pdf
//!
//! Although a random odd number will do, it turns out that the golden
//! ratio phi = (sqrt(5)-1)/2, or its negative, has particularly nice
//! properties.  (See Knuth vol 3, section 6.4, exercise 9.)
//!
//! These are the negative, (1 - phi) = phi**2 = (3 - sqrt(5))/2,
//! which is very slightly easier to multiply by and makes no
//! difference to the hash distribution.
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
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
// __context_unsafe() - function attribute, disable context checking
// @comment: comment explaining why opt-out is safe
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Forces adding an inline comment as argument.
// token_context_lock() - declare an abstract global context lock instance
// @name: token context lock name
//
// Helper that declares an abstract global context lock instance @name, but not
// backed by a real data structure (linker error if accidentally referenced).
// The type name is `__ctx_lock_@name`.
// token_context_lock_instance() - declare another instance of a global context lock
// @ctx: token context lock previously declared with token_context_lock()
// @name: name of additional global context lock instance
//
// Helper that declares an additional instance @name of the same token context
// lock class @ctx. This is helpful where multiple related token contexts are
// declared, to allow using the same underlying type (`__ctx_lock_@ctx`) as
// function arguments.
// __must_hold() - function attribute, caller must hold exclusive context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) exclusively.
// __acquires() - function attribute, function acquires context lock exclusively
//
// Function attribute declaring that the function acquires the given context
// lock instance(s) exclusively, but does not release them.
// __cond_acquires() - function attribute, function conditionally
//                     acquires a context lock exclusively
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x exclusively, but does not release it. The
// function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases() - function attribute, function releases a context lock exclusively
//
// Function attribute declaring that the function releases the given context
// lock instance(s) exclusively. The associated context(s) must be active on
// entry.
// __must_hold_shared() - function attribute, caller must hold shared context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) with shared access.
// __acquires_shared() - function attribute, function acquires context lock shared
//
// Function attribute declaring that the function acquires the given
// context lock instance(s) with shared access, but does not release them.
// __cond_acquires_shared() - function attribute, function conditionally
//                            acquires a context lock shared
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x with shared access, but does not release it.
// The function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases_shared() - function attribute, function releases a
//                       context lock shared
//
// Function attribute declaring that the function releases the given context
// lock instance(s) with shared access. The associated context(s) must be
// active on entry.
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
// The second argument is optional (default 0), so we use a variadic macro
// to make the shorthand.
//
// Beware: Do not apply this to functions which may return
// ERR_PTRs. Also, it is probably unwise to apply it to functions
// returning extra information in the low bits (but in that case the
// compiler should see some alignment anyway, when the return value is
// massaged by 'flags = ptr & 3; ptr &= ~3;').
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-assume_005faligned-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#assume-aligned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// Note: deliberately not named '__nonnull', to avoid clashing with glibc's
// __nonnull() when kernel and userspace headers are combined.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html#index-nonnull
// clang: https://clang.llvm.org/docs/AttributeReference.html#nonnull
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noreturn-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#noreturn
// clang: https://clang.llvm.org/docs/AttributeReference.html#id1
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
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
// This version is i.e. to prevent dead stores elimination on @ptr
// where gcc and llvm may behave differently when otherwise using
// normal barrier(): while gcc behavior gets along with a normal
// barrier(), llvm needs an explicit input variable to be assumed
// clobbered. The issue is as follows: while the inline asm might
// access any memory it wants, the compiler could have fit all of
// @ptr into memory registers instead, and since @ptr never escaped
// from that, it proved that the inline asm wasn't touching any of
// it. This version works well with both compilers, i.e. we're telling
// the compiler that the inline asm absolutely may see the contents
// of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// data_race - mark an expression as containing intentional data races
//
// This data_race() macro is useful for situations in which data races
// should be forgiven.  One example is diagnostic code that accesses
// shared variables but is not a part of the core synchronization design.
// For example, if accesses to a given variable are protected by a lock,
// except for diagnostic code, then the accesses under the lock should
// be plain C-language accesses and those in the diagnostic code should
// use data_race().  This way, KCSAN will complain if buggy lockless
// accesses to that variable are introduced, even if the buggy accesses
// are protected by READ_ONCE() or WRITE_ONCE().
//
// This macro *does not* affect normal code generation, but is a hint
// to tooling that data races here are to be ignored.  If the access must
// be atomic *and* KCSAN should ignore the access, use both data_race()
// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// This returns a constant expression while determining if an argument is
// a constant expression, most importantly without evaluating the argument.
// Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//
// Details:
// - sizeof() return an integer constant expression, and does not evaluate
//   the value of its operand; it only examines the type of its operand.
// - The results of comparing two integer constant expressions is also
//   an integer constant expression.
// - The first literal "8" isn't important. It could be any literal value.
// - The second literal "8" is to avoid warnings about unaligned pointers;
//   this could otherwise just be "1".
// - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//   architectures.
// - The C Standard defines "null pointer constant", "(void *)0", as
//   distinct from other void pointers.
// - If (x) is an integer constant expression, then the "* 0l" resolves
//   it into an integer constant expression of value 0. Since it is cast to
//   "void *", this makes the second operand a null pointer constant.
// - If (x) is not an integer constant expression, then the second operand
//   resolves to a void pointer (but not a null pointer constant: the value
//   is not an integer constant 0).
// - The conditional operator's third operand, "(int *)8", is an object
//   pointer (to type "int").
// - The behavior (including the return type) of the conditional operator
//   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//   given for the second and third operands. This is the central mechanism
//   of the macro:
//   - When one operand is a null pointer constant (i.e. when x is an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns the type of the
//     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//     would then get:
//       sizeof(*((int *)(...))  == sizeof(int)  == 4
//   - When one operand is a void pointer (i.e. when x is not an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns a "void *" type.
//     Here, within the sizeof(), we would then get:
//       sizeof(*((void *)(...)) == sizeof(void) == 1
// - The equality comparison to "sizeof(int)" therefore depends on (x):
//     sizeof(int) == sizeof(int)     (x) was a constant expression
//     sizeof(int) != sizeof(void)    (x) was not a constant expression
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
// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
// expression but avoids the generation of any code, even if that expression
// has side-effects.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
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
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
// Because instrumentation_{begin,end}() can nest, objtool validation considers
// _begin() a +1 and _end() a -1 and computes a sum over the instructions.
// When the value is greater than 0, we consider instrumentation allowed.
//
// There is a problem with code like:
//
// noinstr void foo()
// {
// 	instrumentation_begin();
// 	...
// 	if (cond) {
// 		instrumentation_begin();
// 		...
// 		instrumentation_end();
// 	}
// 	bar();
// 	instrumentation_end();
// }
//
// If instrumentation_end() would be an empty label, like all the other
// annotations, the inner _end(), which is at the end of a conditional block,
// would land on the instruction after the block.
//
// If we then consider the sum of the !cond path, we'll see that the call to
// bar() is with a 0-value, even though, we meant it to happen with a positive
// value.
//
// To avoid this, have _end() be a NOP instruction, this ensures it will be
// part of the condition block and does not escape.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// printk - print a kernel message
// @fmt: format string
//
// This is printk(). It can be called from any context. We want it to work.
//
// If printk indexing is enabled, _printk() is called from printk_index_wrap.
// Otherwise, printk is simply #defined to _printk.
//
// We try to grab the console_lock. If we succeed, it's easy - we log the
// output and call the console drivers.  If we fail to get the semaphore, we
// place the output into the log buffer and return. The current holder of
// the console_sem will notice the new output in console_unlock(); and will
// send it to the consoles before releasing the lock.
//
// One effect of this deferred printing is that code which calls printk() and
// then changes console_loglevel may break. This is because console_loglevel
// is inspected when the actual printing occurs.
//
// See also:
// printf(3)
//
// See the vsnprintf() documentation for format string extensions over C99.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// If @feature is set, patch in @newinstr_yes, otherwise @newinstr_no.
// Alternative instructions for different CPU types or capabilities.
//
// This allows to use optimized instructions even on generic binary
// kernels.
//
// length of oldinstr must be longer or equal the length of newinstr
// It can be padded with nops as needed.
//
// For non barrier like inlines please define new variants
// without volatile and memory clobber.
// Alternative inline assembly with input.
//
// Peculiarities:
// No memory clobber here.
// Argument numbers start with 1.
// Leaving an unused argument 0 to keep API compatibility.
// Like alternative_input, but with a single output argument
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// Non-serializing WRMSR
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "div0" AMD DIV0 speculation bug
// Check that the bitops prototypes are sane
// Try again
// Out of memory
// Device or resource busy
// Invalid argument
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// Operation is not supported
// list_first_entry - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note, that list is expected to be not empty.
// list_next_entry - get the next element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_entry_is_head - test if the entry points to the head of the list
// @pos:	the type * to cursor
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
// ZONE_MOVABLE allowed
// DOC: Page mobility and placement hints
//
// Page mobility and placement hints
//
// These flags provide hints about how mobile the page is. Pages with similar
// mobility are placed within the same pageblocks to minimise problems due
// to external fragmentation.
//
// %__GFP_MOVABLE (also a zone modifier) indicates that the page can be
// moved by page migration during memory compaction or can be reclaimed.
//
// %__GFP_RECLAIMABLE is used for slab allocations that specify
// SLAB_RECLAIM_ACCOUNT and whose pages can be freed via shrinkers.
//
// %__GFP_WRITE indicates the caller intends to dirty the page. Where possible,
// these pages will be spread between local zones to avoid all the dirty
// pages being in one zone (fair zone allocation policy).
//
// %__GFP_HARDWALL enforces the cpuset memory allocation policy.
//
// %__GFP_THISNODE forces the allocation to be satisfied from the requested
// node with no fallbacks or placement policy enforcements.
//
// %__GFP_ACCOUNT causes the allocation to be accounted to the active
// cgroup context.
// DOC: Reclaim modifiers
//
// Reclaim modifiers
// Please note that all the following flags are only applicable to sleepable
// allocations (e.g. %GFP_NOWAIT and %GFP_ATOMIC will ignore them).
//
// %__GFP_IO can start physical IO.
//
// %__GFP_FS can call down to the low-level FS. Clearing the flag avoids the
// allocator recursing into the filesystem which might already be holding
// locks.
//
// %__GFP_DIRECT_RECLAIM indicates that the caller may enter direct reclaim.
// This flag can be cleared to avoid unnecessary delays when a fallback
// option is available.
//
// %__GFP_KSWAPD_RECLAIM indicates that the caller wants to wake kswapd when
// the low watermark is reached and have it reclaim pages until the high
// watermark is reached. A caller may wish to clear this flag when fallback
// options are available and the reclaim is likely to disrupt the system. The
// canonical example is THP allocation where a fallback is cheap but
// reclaim/compaction may cause indirect stalls.
//
// %__GFP_RECLAIM is shorthand to allow/forbid both direct and kswapd reclaim.
//
// The default allocator behavior depends on the request size. We have a concept
// of so-called costly allocations (with order > %PAGE_ALLOC_COSTLY_ORDER).
// !costly allocations are too essential to fail so they are implicitly
// non-failing by default (with some exceptions like OOM victims might fail so
// the caller still has to check for failures) while costly requests try to be
// not disruptive and back off even without invoking the OOM killer.
// The following three modifiers might be used to override some of these
// implicit rules. Please note that all of them must be used along with
// %__GFP_DIRECT_RECLAIM flag.
//
// %__GFP_NORETRY: The VM implementation will try only very lightweight
// memory direct reclaim to get some memory under memory pressure (thus
// it can sleep). It will avoid disruptive actions like OOM killer. The
// caller must handle the failure which is quite likely to happen under
// heavy memory pressure. The flag is suitable when failure can easily be
// handled at small cost, such as reduced throughput.
//
// %__GFP_RETRY_MAYFAIL: The VM implementation will retry memory reclaim
// procedures that have previously failed if there is some indication
// that progress has been made elsewhere.  It can wait for other
// tasks to attempt high-level approaches to freeing memory such as
// compaction (which removes fragmentation) and page-out.
// There is still a definite limit to the number of retries, but it is
// a larger limit than with %__GFP_NORETRY.
// Allocations with this flag may fail, but only when there is
// genuinely little unused memory. While these allocations do not
// directly trigger the OOM killer, their failure indicates that
// the system is likely to need to use the OOM killer soon.  The
// caller must handle failure, but can reasonably do so by failing
// a higher-level request, or completing it only in a much less
// efficient manner.
// If the allocation does fail, and the caller is in a position to
// free some non-essential memory, doing so could benefit the system
// as a whole.
//
// %__GFP_NOFAIL: The VM implementation _must_ retry infinitely: the caller
// cannot handle allocation failures. The allocation could block
// indefinitely but will never return with failure. Testing for
// failure is pointless.
// It _must_ be blockable and used together with __GFP_DIRECT_RECLAIM.
// It should _never_ be used in non-sleepable contexts.
// New users should be evaluated carefully (and the flag should be
// used only when there is no reasonable failure policy) but it is
// definitely preferable to use the flag rather than opencode endless
// loop around allocator.
// Allocating pages from the buddy with __GFP_NOFAIL and order > 1 is
// not supported. Please consider using kvmalloc() instead.
// Caller can reclaim
// DOC: Action modifiers
//
// Action modifiers
//
// %__GFP_NOWARN suppresses allocation failure reports.
//
// %__GFP_COMP address compound page metadata.
//
// %__GFP_ZERO returns a zeroed page on success.
//
// %__GFP_ZEROTAGS zeroes memory tags at allocation time. Setting memory tags at
// the same time as zeroing memory (e.g., with __GFP_ZERO) has minimal
// additional performance impact. However, __GFP_ZEROTAGS also zeroes the tags
// even if memory is not getting zeroed at allocation time (e.g.,
// with init_on_free).
//
// %__GFP_SKIP_KASAN makes KASAN skip unpoisoning on page allocation.
// Used for userspace and vmalloc pages; the latter are unpoisoned by
// kasan_unpoison_vmalloc instead. If passed to vmalloc, kasan_unpoison_vmalloc
// is skipped too. For userspace pages, results in poisoning being skipped as
// well, see should_skip_kasan_poison for details. Only effective in HW_TAGS mode.
// Disable lockdep for GFP context tracking
// DOC: Useful GFP flag combinations
//
// Useful GFP flag combinations
//
// Useful GFP flag combinations that are commonly used. It is recommended
// that subsystems start with one of these combinations and then set/clear
// %__GFP_FOO flags as necessary.
//
// %GFP_ATOMIC users can not sleep and need the allocation to succeed. A lower
// watermark is applied to allow access to "atomic reserves".
// The current implementation doesn't support NMI, nor contexts that disable
// preemption under PREEMPT_RT. This includes raw_spin_lock() and plain
// preempt_disable() - see "Memory allocation" in
// Documentation/core-api/real-time/differences.rst for more info.
//
// %GFP_KERNEL is typical for kernel-internal allocations. The caller requires
// %ZONE_NORMAL or a lower zone for direct access but can direct reclaim.
//
// %GFP_KERNEL_ACCOUNT is the same as GFP_KERNEL, except the allocation is
// accounted to the active cgroup context.
//
// %GFP_NOWAIT is for kernel allocations that should not stall for direct
// reclaim, start physical IO or use any filesystem callback.  It is very
// likely to fail to allocate memory, even for very small allocations.
// The same restrictions on calling contexts apply as for %GFP_ATOMIC.
//
// %GFP_NOIO will use direct reclaim to discard clean pages or slab pages
// that do not require the starting of any physical IO.
// Please try to avoid using this flag directly and instead use
// memalloc_noio_{save,restore} to mark the whole scope which cannot
// perform any IO with a short explanation why. All allocation requests
// will inherit GFP_NOIO implicitly.
//
// %GFP_NOFS will use direct reclaim but will not use any filesystem interfaces.
// Please try to avoid using this flag directly and instead use
// memalloc_nofs_{save,restore} to mark the whole scope which cannot/shouldn't
// recurse into the FS layer with a short explanation why. All allocation
// requests will inherit GFP_NOFS implicitly.
//
// %GFP_USER is for userspace allocations that also need to be directly
// accessibly by the kernel or hardware. It is typically used by hardware
// for buffers that are mapped to userspace (e.g. graphics) that hardware
// still must DMA to. cpuset limits are enforced for these allocations.
//
// %GFP_DMA exists for historical reasons and should be avoided where possible.
// The flags indicates that the caller requires that the lowest zone be
// used (%ZONE_DMA or 16M on x86-64). Ideally, this would be removed but
// it would require careful auditing as some users really require it and
// others use the flag to avoid lowmem reserves in %ZONE_DMA and treat the
// lowest zone as a type of emergency reserve.
//
// %GFP_DMA32 is similar to %GFP_DMA except that the caller requires a 32-bit
// address. Note that kmalloc(..., GFP_DMA32) does not return DMA32 memory
// because the DMA32 kmalloc cache array is not implemented.
// (Reason: there is no such user in kernel).
//
// %GFP_HIGHUSER is for userspace allocations that may be mapped to userspace,
// do not need to be directly accessible by the kernel but that cannot
// move once in use. An example may be a hardware allocation that maps
// data directly into userspace but has no addressing limitations.
//
// %GFP_HIGHUSER_MOVABLE is for userspace allocations that the kernel does not
// need direct access to but can use kmap() when access is required. They
// are expected to be movable via page reclaim or page migration. Typically,
// pages on the LRU would also be allocated with %GFP_HIGHUSER_MOVABLE.
//
// %GFP_TRANSHUGE and %GFP_TRANSHUGE_LIGHT are used for THP allocations. They
// are compound allocations that will generally fail quickly if memory is not
// available and will not wake kswapd/kcompactd on failure. The _LIGHT
// version does not attempt reclaim/compaction at all and is by default used
// in page fault path, while the non-light is used by khugepaged.
// Generate a per-CPU add to memory instruction and optimize code
// if one is added or subtracted.
// Add return operation
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
// alpha modules require percpu variables to be defined as
// weak to force the compiler to generate GOT based external
// references for them.  This is necessary because percpu sections
// will be located outside of the usually addressable area.
//
// This definition puts the following two extra restrictions when
// defining percpu variables.
//
// 1. The symbol must be globally unique, even the static ones.
// 2. Static percpu variables cannot be defined inside a function.
//
// Archs which need weak percpu definitions should set
// CONFIG_ARCH_MODULE_NEEDS_WEAK_PER_CPU when necessary.
//
// Normal declaration and definition macros.
// Variant on the per-CPU variable declaration/definition theme used for
// ordinary per-CPU variables.
// Declaration/definition used for per-CPU variables that are frequently
// accessed and should be in a single cacheline.
//
// For use only by architecture and core code.  Only use scalar or pointer
// types to maximize density.
// Declaration/definition used for per-CPU variables that must be cacheline
// aligned under SMP conditions so that, whilst a particular instance of the
// data corresponds to a particular CPU, inefficiencies due to direct access by
// other CPUs are reduced by preventing the data from unnecessarily spanning
// cachelines.
//
// An example of this would be statistical data, where each CPU's set of data
// is updated by that CPU alone, but the data from across all CPUs is collated
// by a CPU processing a read from a proc file.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
// @ptr and is invoked once before a percpu area is accessed by all
// accessors and operations.  This is performed in the generic part of
// percpu and arch overrides don't need to worry about it; however, if an
// arch wants to implement an arch-specific percpu accessor or operation,
// it may use __verify_pcpu_ptr() to verify the parameters.
//
// + 0 is required in order to convert the pointer type from a
// potential array type to a pointer to a single item of the array.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
// Operations for contexts where we do not want to do any checks for
// preemptions.  Unless strictly necessary, always use [__]this_cpu_*()
// instead.
//
// If there is no other protection through preempt disable and/or disabling
// interrupts then one of these RMW operations can show unexpected behavior
// because the execution thread was rescheduled on another processor or an
// interrupt occurred and the same percpu variable was modified from the
// interrupt context.
// The preempt_count offset needed for things like:
//
//  spin_lock_bh()
//
// Which need to disable both preemption (CONFIG_PREEMPT_COUNT) and
// softirqs, such that unlock sequences of:
//
//  spin_unlock();
//  local_bh_enable();
//
// Work as expected.
// early_idt_handler_array is an array of entry points referenced in the
// early IDT.  For simplicity, it's a real array with one entry point
// every nine bytes.  That leaves room for an optional 'push $0' if the
// vector has no error code (two bytes), a 'push $vector_number' (two
// bytes), and a jump to the common entry code (up to five bytes).
// Load a segment. Fall back on loading the zero segment if something goes
// wrong.  This variant assumes that loading zero fully clears the segment.
// This is always the case on Intel CPUs and, even on 64-bit AMD CPUs, any
// failure to fully clear the cached descriptor is only observable for
// FS and GS.
// Save a segment register away:
// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
// Note: no "lock" prefix even on SMP: xchg always implies lock anyway.
// Since this is generally used to protect other memory information, we
// use "asm volatile" and "memory" clobbers to prevent gcc from moving
// information around.
// Atomic compare and exchange.  Compare OLD with MEM, if identical,
// store NEW in MEM.  Return the initial value in MEM.  Success is
// indicated by comparing RETURN with OLD.
// xadd() adds "inc" to "*ptr" and atomically returns the previous
// value of "*ptr".
//
// xadd() is locked when multiple CPUs are online
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// order_base_2 - calculate the (rounded up) base 2 order of the argument
// @n: parameter
//
// The first few values calculated by this routine:
//  ob2(0) = 0
//  ob2(1) = 0
//  ob2(2) = 1
//  ob2(3) = 2
//  ob2(4) = 2
//  ob2(5) = 3
//  ... and so on.
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
// __CPUID_LEAF() - Define a CPUID output and parse info entry
// @_name:	Struct type name of the CPUID leaf/subleaf (e.g. 'leaf_0x7_0'). Such
// 		types are defined at <cpuid/leaf_types.h> and follow the leaf_0xM_N
// 		format, where 0xM is the leaf and N is the subleaf.
// @_count:	Number of storage entries to allocate for this leaf/subleaf.
//
// For a given leaf/subleaf, define an array of CPUID storage entries and an associated
// query info structure.
//
// Use an array of storage entries to accommodate CPUID leaves with multiple subleaves
// having the same output format.  This is common for hierarchical enumeration; e.g.,
// CPUID(0x4), CPUID(0x12), and CPUID(0x8000001d).
// CPUID_LEAF() - Define a 'struct cpuid_leaves' storage entry
// @_leaf:	Leaf number, in compile-time 0xN format
// @_subleaf:	Subleaf number, in compile-time decimal format
//
// Convenience wrapper around __CPUID_LEAF().
// N 32-bit words worth of info
// If !LOCKDEP, the value is meaningless
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_begin() - begin a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_retry() - end a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count, from read_seqcount_begin()
//
// read_seqcount_retry closes the read critical section of given
// seqcount_t.  If the critical section was invalid, it must be ignored
// (and typically retried).
//
// Return: true if a read section retry is required, else false
// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
// set_current_state() includes a barrier so that the write of current->__state
// is correctly serialised wrt the caller's subsequent test of whether to
// actually sleep:
//
//   for (;;) {
// 	set_current_state(TASK_UNINTERRUPTIBLE);
// 	if (CONDITION)
// 	   break;
//
// 	schedule();
//   }
//   __set_current_state(TASK_RUNNING);
//
// If the caller does not need such serialisation (because, for instance, the
// CONDITION test and condition change and wakeup are under the same lock) then
// use __set_current_state().
//
// The above is typically ordered against the wakeup, which does:
//
//   CONDITION = 1;
//   wake_up_state(p, TASK_UNINTERRUPTIBLE);
//
// where wake_up_state()/try_to_wake_up() executes a full memory barrier before
// accessing p->__state.
//
// Wakeup will do: if (@state & p->__state) p->__state = TASK_RUNNING, that is,
// once it observes the TASK_UNINTERRUPTIBLE store the waking CPU can issue a
// TASK_RUNNING store which can collide with __set_current_state(TASK_RUNNING).
//
// However, with slightly different timing the wakeup TASK_RUNNING store can
// also collide with the TASK_UNINTERRUPTIBLE store. Losing that store is not
// a problem either because that will result in one extra go around the loop
// and our @cond test will save the day.
//
// Also see the comments of try_to_wake_up().
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
// Page flags policies wrt compound pages
//
// PF_POISONED_CHECK
//     check if this struct page poisoned/uninitialized
//
// PF_ANY:
//     the page flag is relevant for small, head and tail pages.
//
// PF_HEAD:
//     for compound page all operations related to the page flag applied to
//     head page.
//
// PF_NO_TAIL:
//     modifications of the page flag must be done on small or head pages,
//     checks can be done on tail pages too.
//
// PF_NO_COMPOUND:
//     the page flag is not relevant for compound pages.
//
// PF_SECOND:
//     the page flag is stored in the first tail page.
// Which page is the flag stored in
// Macros to create function definitions for page flags
// On an anonymous folio mapped into a user virtual memory area,
// folio->mapping points to its anon_vma, not to a struct address_space;
// with the FOLIO_MAPPING_ANON bit set to distinguish it.  See rmap.h.
//
// On an anonymous folio in a VM_MERGEABLE area, if CONFIG_KSM is enabled,
// the FOLIO_MAPPING_ANON_KSM bit may be set along with the FOLIO_MAPPING_ANON
// bit; and then folio->mapping points, not to an anon_vma, but to a private
// structure which KSM associates with that merged folio.  See ksm.h.
//
// Please note that, confusingly, "folio_mapping" refers to the inode
// address_space which maps the folio from disk; whereas "folio_mapped"
// refers to user virtual address space into which the folio is mapped.
//
// For slab pages, since slab reuses the bits in struct page to store its
// internal states, the folio->mapping does not exist as such, nor do
// these flags below.  So in order to avoid testing non-existent bits,
// please make sure that folio_test_slab(folio) actually evaluates to
// false before calling the following functions (e.g., folio_test_anon).
// See mm/slab.h.
// local_lock - Acquire a per CPU local lock
// @lock:	The lock variable
// local_lock_irq - Acquire a per CPU local lock and disable interrupts
// @lock:	The lock variable
// local_lock_irqsave - Acquire a per CPU local lock, save and disable
// 			 interrupts
// @lock:	The lock variable
// @flags:	Storage for interrupt flags
// local_unlock - Release a per CPU local lock
// @lock:	The lock variable
// local_unlock_irq - Release a per CPU local lock and enable interrupts
// @lock:	The lock variable
// local_unlock_irqrestore - Release a per CPU local lock and restore
// 			      interrupt flags
// @lock:	The lock variable
// @flags:      Interrupt flags to restore
// local_trylock_init - Runtime initialize a lock instance
// @lock:	The lock variable
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

macro_rules! BITMAP_LAST_WORD_MASK {
($nbits:expr) => {
18446744073709551615u64 >> ($nbits.wrapping_neg() & 63u32)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT_WORD__2 {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_bitmap_h_555>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_207>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__2 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__6 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__4!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as u32).wrapping_sub(1) == 0
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
/// @v: The value to statically initialize with.
macro_rules! RCU_INITIALIZER {
($v:expr) => {
$v
};
}

macro_rules! RCU_LOCKDEP_WARN {
($c:expr) => {
loop {
let __v: bool = false && $c;
if !__v {
break;
}
}
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

/// We encode errnos in the xas->xa_node.  If an error has happened, we need to
/// drop the lock to fix it, and once we've done so the xa_state is invalid.
macro_rules! XA_ERROR {
($errno:expr) => {
((($errno) as u64) << 2i32 | 2) as *mut xa_node
};
}

macro_rules! XA_FLAGS_MARK {
($mark:expr) => {
16777216u32 << (($mark) as u32)
};
}

macro_rules! XA_NODE_BUG_ON {
() => {
while false {
}
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
};
}

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut xa_node, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut (), $val) };
if true {
break;
}
}
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! ____or {
() => {
1i32
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
() => {
__acquire_ctx_lock!();
};
}

macro_rules! __acquire_ctx_lock {
() => {
while false {
}
};
}

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared {
() => {
__acquire_shared_ctx_lock!();
};
}

macro_rules! __acquire_shared_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
};
}

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __rcu_dereference_protected__8 {
($p:expr, $c:expr) => {
{
let mut __v: *mut xa_node;
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
}
};
}

macro_rules! __rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
{
let mut __v: *mut ();
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
}
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
() => {
__release_ctx_lock!();
};
}

macro_rules! __release_ctx_lock {
() => {
while false {
}
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
() => {
__release_shared_ctx_lock!();
};
}

macro_rules! __release_shared_ctx_lock {
() => {
while false {
}
};
}

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit__3 {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_70b71f185ebb8c4a($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __test_and_clear_bit {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
generic___test_and_clear_bit(($nr) as u64, ($addr) as *mut u64) as i32
} else {
___test_and_clear_bit(($nr) as u64, ($addr) as *mut u64) as i32
};
__v2
}
};
}

macro_rules! __test_and_set_bit {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
generic___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
} else {
___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
};
__v2
}
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks {
($_do_alloc:expr) => {
{
let mut __v: *mut ();
let mut __v2: *mut ();
let mut __v3: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! bitmap_size {
($nbits:expr) => {
ALIGN__2!($nbits, 64i32) / 8
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

/// kcsan_check_read - check regular read access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read {
($ptr:expr, $size:expr) => {
kcsan_check_access(($ptr) as *const (), $size, 0i32);
};
}

/// kcsan_check_read_write - check regular read-write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 3i32);
};
}

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
};
}

macro_rules! kcsan_release {
() => {
while false {
}
};
}

macro_rules! mark_inc {
($mark:expr) => {
loop {
$mark = (($mark) as u32).wrapping_add(1);
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_bh {
($lock:expr) => {
unsafe { _raw_spin_lock_bh($lock) };
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
$flags = unsafe { _raw_spin_lock_irqsave($lock) };
if true {
break;
}
}
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! raw_spin_unlock_bh {
($lock:expr) => {
unsafe { _raw_spin_unlock_bh($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected {
($p:expr, $c:expr) => {
__rcu_dereference_protected__2!($p, $c)
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected__10 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__8!($p, $c)
};
}

macro_rules! rcu_lock_acquire {
() => {
while false {
}
};
}

macro_rules! rcu_lock_release {
() => {
while false {
}
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
}
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
};
}

macro_rules! test_bit__9 {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! xa_lock {
($xa:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_lock_bh {
($xa:expr) => {
spin_lock_bh(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_lock_irq {
($xa:expr) => {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_lock_irqsave {
($xa:expr, $flags:expr) => {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) }, $flags);
};
}

macro_rules! xa_unlock {
($xa:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_unlock_bh {
($xa:expr) => {
spin_unlock_bh(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_unlock_irq {
($xa:expr) => {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_unlock_irqrestore {
($xa:expr, $flags:expr) => {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) }, $flags);
};
}

macro_rules! xas_lock {
($xas:expr) => {
xa_lock!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_lock_bh {
($xas:expr) => {
xa_lock_bh!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_lock_irq {
($xas:expr) => {
xa_lock_irq!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_lock_irqsave {
($xas:expr, $flags:expr) => {
xa_lock_irqsave!(unsafe { (*$xas).xa }, $flags);
};
}

macro_rules! xas_unlock {
($xas:expr) => {
xa_unlock!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_unlock_bh {
($xas:expr) => {
xa_unlock_bh!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_unlock_irq {
($xas:expr) => {
xa_unlock_irq!(unsafe { (*$xas).xa });
};
}

macro_rules! xas_unlock_irqrestore {
($xas:expr, $flags:expr) => {
xa_unlock_irqrestore!(unsafe { (*$xas).xa }, $flags);
};
}

unsafe extern "C" {
    static mut radix_tree_node_cachep: *mut kmem_cache;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __bitmap_clear(arg0: *mut u64, arg1: u32, arg2: i32);
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_bh(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_bh(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn kmem_cache_alloc_lru_noprof(arg0: *mut kmem_cache, arg1: *mut list_lru, arg2: u32) -> *mut ();
    fn __compiletime_assert_217();
    fn __compiletime_assert_219();
    fn __compiletime_assert_221();
    fn radix_tree_node_rcu_free(arg0: *mut callback_head);
    fn __compiletime_assert_224();
    fn __compiletime_assert_227();
    fn __compiletime_assert_228();
    fn __compiletime_assert_229();
    fn __compiletime_assert_230();
    fn __compiletime_assert_231();
    fn __compiletime_assert_232();
    fn __compiletime_assert_233();
    fn __compiletime_assert_234();
    fn __compiletime_assert_235();
    fn __compiletime_assert_236();
    fn __compiletime_assert_237();
    fn __compiletime_assert_238();
    fn __compiletime_assert_239();
    fn __compiletime_assert_240();
    fn __compiletime_assert_242();
    fn __compiletime_assert_243();
    fn __compiletime_assert_244();
    fn __compiletime_assert_250();
    fn __compiletime_assert_251();
    fn __compiletime_assert_254();
    fn __compiletime_assert_257();
    fn __compiletime_assert_258();
    fn __compiletime_assert_259();
    fn __compiletime_assert_260();
    fn __compiletime_assert_261();
    fn __compiletime_assert_262();
    fn __compiletime_assert_263();
    fn __compiletime_assert_264();
    fn __compiletime_assert_265();
    fn __compiletime_assert_266();
    fn __compiletime_assert_267();
    fn __compiletime_assert_270();
    fn __compiletime_assert_272();
    fn __compiletime_assert_273();
    fn __compiletime_assert_274();
    fn __compiletime_assert_275();
    fn __compiletime_assert_276();
    fn __compiletime_assert_277();
    fn __compiletime_assert_278();
    fn __compiletime_assert_279();
    fn __compiletime_assert_280();
    fn __compiletime_assert_281();
    fn __compiletime_assert_282();
    fn __compiletime_assert_313();
    fn __compiletime_assert_324();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qspinlock_types_h_44 {
    val: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qspinlock_types_h_44_2,
    __slate_anon_2: __SlateRecord_qspinlock_types_h_44_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_2 {
    locked: u8,
    pending: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_3 {
    locked_pending: u16,
    tail: u16,
}


fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bitmap_h_555 {
}


fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
//
// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
//
// Assert that there are no concurrent writes to a subset of bits in @var;
// concurrent readers are permitted. This assertion captures more detailed
// bit-level properties, compared to the other (word granularity) assertions.
// Only the bits set in @mask are checked for concurrent modifications, while
// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
// are ignored.
//
// Use this for variables, where some bits must not be modified concurrently,
// yet other bits are expected to be modified concurrently.
//
// For example, variables where, after initialization, some bits are read-only,
// but other bits may still be modified concurrently. A reader may wish to
// assert that this is true as follows:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
// to access the masked bits only, and KCSAN optimistically assumes it is
// therefore safe, even in the presence of data races, and marking it with
// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
// it may still be advisable to do so, since we cannot reason about all compiler
// optimizations when it comes to bit manipulations (on the reader and writer
// side). If you are sure nothing can go wrong, we can write the above simply
// as:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Another example, where this may be used, is when certain bits of @var may
// only be modified when holding the appropriate lock, but other bits may still
// be modified concurrently. Writers, where other bits may change concurrently,
// could use the assertion as follows:
//
// .. code-block:: c
//
// 	spin_lock(&foo_lock);
// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
// 	old_flags = flags;
// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
// 	spin_unlock(&foo_lock);
//
// @var: variable to assert on
// @mask: only check for modifications to bits set in @mask
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qspinlock {
    __slate_anon_0: __SlateRecord_qspinlock_types_h_44,
}


// Bitfields in the atomic value:
//
// When NR_CPUS < 16K
//  0- 7: locked byte
//  8-15: pending byte
// 16-17: tail index
// 18-31: tail cpu (+1)
//
// When NR_CPUS >= 16K
//  0- 7: locked byte
//     8: pending
//  9-10: tail index
// 11-31: tail cpu (+1)
#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
}


#[inline(always)]
fn generic___clear_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let __v2: u64 = (unsafe { *p }) & !mask;
unsafe {
*p = __v2;
}
}

#[inline(always)]
fn generic___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let old: u64 = unsafe { *p };
unsafe {
*p = old | mask;
}
old & mask != 0
}

#[inline(always)]
fn generic___test_and_clear_bit(nr: u64, addr: *mut u64) -> bool {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let old: u64 = unsafe { *p };
unsafe {
*p = old & !mask;
}
old & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1175 {
    private_list: list_head,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1180 {
    tags: [[u64; 1]; 3],
    marks: [[u64; 1]; 3],
}


// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch___clear_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch___test_and_set_bit(mut nr: u64, mut addr: *mut u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btsq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch___test_and_clear_bit(mut nr: u64, mut addr: *mut u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btrq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
}

#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
}

#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_instrument_atomic_check_alignment_3, sym __slate_asm_string_instrument_atomic_check_alignment_4, const 67 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
}

#[inline(always)]
fn __instrument_read_write_bitop(nr: i64, addr: *mut u64) {
if IS_ENABLED!() != 0 {
// We treat non-atomic read-write bitops a little more special.
// Given the operations here only modify a single bit, assuming
// non-atomicity of the writer is sufficient may be reasonable
// for certain usage (and follows the permissible nature of the
// assume-plain-writes-atomic rule):
// 1. report read-modify-write races -> check read;
// 2. do not report races with marked readers, but do report
//    races with unmarked readers -> check "atomic" write.
kcsan_check_read!(addr.wrapping_offset(BIT_WORD__2!(nr) as isize), core::mem::size_of::<i64>() as u64);
// Use generic write instrumentation, in case other sanitizers
// or tools are enabled alongside KCSAN.
instrument_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
} else {
instrument_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
}
}

#[inline(always)]
fn ___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_set_bit(nr, addr)
}

#[inline(always)]
fn ___test_and_clear_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_clear_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__2!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
}

#[inline(always)]
fn bitmap_fill(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits__2!(nbits) {
unsafe {
*dst = !0;
}
} else {
unsafe { memset(dst as *mut (), 255i32, len as u64) };
}
}

#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits__2!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
}

#[inline(always)]
fn bitmap_clear(map: *mut u64, start: u32, nbits: u32) {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits) }) && nbits == 1 {
__clear_bit!(start, map);
} else {
if small_const_nbits__2!(start.wrapping_add(nbits)) {
let __v1: u64 = unsafe { *map };
let __v2: u64 = __v1 & !GENMASK__4!(start.wrapping_add(nbits).wrapping_sub(1), start);
unsafe {
*map = __v2;
}
} else {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(start & 7u32) }) && IS_ALIGNED!(start, 8i32) && unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits & 7u32) } && IS_ALIGNED!(nbits, 8i32) {
unsafe { memset((map as *mut u8).wrapping_offset((start / 8) as isize) as *mut (), 0i32, (nbits / 8) as u64) };
} else {
unsafe { __bitmap_clear(map, start, nbits as i32) };
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_lock_bh(lock: *mut spinlock) {
raw_spin_lock_bh!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_bh(lock: *mut spinlock) {
raw_spin_unlock_bh!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

fn rcu_read_lock_held() -> i32 {
1
}

// Dependency order vs. p above.
// Dependency order vs. p above.
// rcu_assign_pointer() - assign to RCU-protected pointer
// @p: pointer to assign to
// @v: value to assign (publish)
//
// Assigns the specified value to the specified RCU-protected
// pointer, ensuring that any concurrent RCU readers will see
// any prior initialization.
//
// Inserts memory barriers on architectures that require them
// (which is most of them), and also prevents the compiler from
// reordering the code that initializes the structure after the pointer
// assignment.  More importantly, this call documents which pointers
// will be dereferenced by RCU read-side code.
//
// In some special cases, you may use RCU_INIT_POINTER() instead
// of rcu_assign_pointer().  RCU_INIT_POINTER() is a bit faster due
// to the fact that it does not constrain either the CPU or the compiler.
// That said, using RCU_INIT_POINTER() when you should have used
// rcu_assign_pointer() is a very bad thing that results in
// impossible-to-diagnose memory corruption.  So please be careful.
// See the RCU_INIT_POINTER() comment header for details.
//
// Note that rcu_assign_pointer() evaluates each of its arguments only
// once, appearances notwithstanding.  One of the "extra" evaluations
// is in typeof() and the other visible only to sparse (__CHECKER__),
// neither of which actually execute the argument.  As with most cpp
// macros, this execute-arguments-only-once property is important, so
// please be careful when making changes to rcu_assign_pointer() and the
// other macros that it invokes.
// rcu_dereference_check() - rcu_dereference with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Do an rcu_dereference(), but check that the conditions under which the
// dereference will take place are correct.  Typically the conditions
// indicate the various locking conditions that should be held at that
// point.  The check should return true if the conditions are satisfied.
// An implicit check for being in an RCU read-side critical section
// (rcu_read_lock()) is included.
//
// For example:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock));
//
// could be used to indicate to lockdep that foo->bar may only be dereferenced
// if either rcu_read_lock() is held, or that the lock required to replace
// the bar struct at foo->bar is held.
//
// Note that the list of conditions may also include indications of when a lock
// need not be held, for example during initialisation or destruction of the
// target struct:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock) ||
// 					      atomic_read(&foo->usage) == 0);
//
// Inserts memory barriers on architectures that require them
// (currently only the Alpha), prevents the compiler from refetching
// (and from merging fetches), and, more importantly, documents exactly
// which pointers are protected by RCU and checks that the pointer is
// annotated as __rcu.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

// RCU_INIT_POINTER() - initialize an RCU protected pointer
// @p: The pointer to be initialized.
// @v: The value to initialized the pointer to.
//
// Initialize an RCU-protected pointer in special cases where readers
// do not need ordering constraints on the CPU or the compiler.  These
// special cases are:
//
// 1.	This use of RCU_INIT_POINTER() is NULLing out the pointer *or*
// 2.	The caller has taken whatever steps are required to prevent
// 	RCU readers from concurrently accessing this pointer *or*
// 3.	The referenced data structure has already been exposed to
// 	readers either at compile time or via rcu_assign_pointer() *and*
//
// 	a.	You have not made *any* reader-visible changes to
// 		this structure since then *or*
// 	b.	It is OK for readers accessing this structure from its
// 		new location to see the old state of the structure.  (For
// 		example, the changes were to statistical counters or to
// 		other state where exact synchronization is not required.)
//
// Failure to follow these rules governing use of RCU_INIT_POINTER() will
// result in impossible-to-diagnose memory corruption.  As in the structures
// will look OK in crash dumps, but any concurrent RCU readers might
// see pre-initialized values of the referenced data structure.  So
// please be very careful how you use RCU_INIT_POINTER()!!!
//
// If you are creating an RCU-protected linked structure that is accessed
// by a single external-to-structure RCU-protected pointer, then you may
// use RCU_INIT_POINTER() to initialize the internal RCU-protected
// pointers, but you must use rcu_assign_pointer() to initialize the
// external-to-structure pointer *after* you have completely initialized
// the reader-accessible portions of the linked structure.
//
// Note that unlike rcu_assign_pointer(), RCU_INIT_POINTER() provides no
// ordering guarantees for either the CPU or the compiler.
#[repr(C)]
#[derive(Clone, Copy)]
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn xa_is_value(entry: *const ()) -> bool {
(entry as u64) & 1 != 0
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_xas_load: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_destroy: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_nomem: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_create_range: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_store: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_get_mark: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_set_mark: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_clear_mark: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_init_marks: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_split_alloc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_split: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_try_split_min_order: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_try_split: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_pause: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xas_prev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xas_next: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_find: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_find_marked: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_find_conflict: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_load: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_erase: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_erase: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_store: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_store: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_cmpxchg: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_insert: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_store_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xas_get_order: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_get_order: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_alloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_alloc_cyclic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_set_mark: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___xa_clear_mark: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_get_mark: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_set_mark: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_clear_mark: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_find: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_find_after: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_extract: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_delete_node: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xa_destroy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous", sym xas_load, sym xas_destroy, sym xas_nomem, sym xas_create_range, sym xas_store, sym xas_get_mark, sym xas_set_mark, sym xas_clear_mark, sym xas_init_marks, sym xas_split_alloc, sym xas_split, sym xas_try_split_min_order, sym xas_try_split, sym xas_pause, sym __xas_prev, sym __xas_next, sym xas_find, sym xas_find_marked, sym xas_find_conflict, sym xa_load, sym __xa_erase, sym xa_erase, sym __xa_store, sym xa_store, sym __xa_cmpxchg, sym __xa_insert, sym xa_store_range, sym xas_get_order, sym xa_get_order, sym __xa_alloc, sym __xa_alloc_cyclic, sym __xa_set_mark, sym __xa_clear_mark, sym xa_get_mark, sym xa_set_mark, sym xa_clear_mark, sym xa_find, sym xa_find_after, sym xa_extract, sym xa_delete_node, sym xa_destroy, options(att_syntax));

fn xa_mk_internal(v: u64) -> *mut () {
(v << 2i32 | 2) as *mut ()
}

fn xa_to_internal(entry: *const ()) -> u64 {
(entry as u64) >> 2i32
}

fn xa_is_internal(entry: *const ()) -> bool {
(entry as u64) & 3 == 2
}

fn xa_is_zero(entry: *const ()) -> bool {
unlikely!(entry == (xa_mk_internal(257u64) as *const ())) != 0
}

fn xa_is_err(entry: *const ()) -> bool {
let __v0: bool = xa_is_internal(entry) && entry >= (xa_mk_internal(4095i32.wrapping_neg() as u64) as *const ());
__v0
}

fn xa_err(entry: *mut ()) -> i32 {
// xa_to_internal() would not do sign extension.
if xa_is_err(entry as *const ()) {
return ((entry as i64) >> 2i32) as i32;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xa_limit {
    max: u32,
    min: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_destroy")]
extern "C" fn xa_destroy(xa: *mut xarray) {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = 0u64;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut flags: u64 = 0;
xas.xa_node = core::ptr::null_mut::<xa_node>();
xas_lock_irqsave!(core::ptr::addr_of_mut!(xas), flags);
let entry: *mut () = xa_head_locked(xa as *const xarray);
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_324() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*xa).xa_head, RCU_INITIALIZER!(core::ptr::null_mut::<()>()));
if true {
break;
}
}
xas_init_marks(core::ptr::addr_of_mut!(xas) as *const xa_state);
if xa_zero_busy(xa as *const xarray) {
xa_mark_clear(xa, 0u32);
}
// lockdep checks we're still holding the lock in xas_free_nodes()
if xa_is_node(entry as *const ()) {
xas_free_nodes(core::ptr::addr_of_mut!(xas), xa_to_node(entry as *const ()));
}
xas_unlock_irqrestore!(core::ptr::addr_of_mut!(xas), flags);
}

fn xa_marked(xa: *const xarray, mark: u32) -> bool {
(unsafe { (*xa).xa_flags }) & XA_FLAGS_MARK!(mark) != 0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_alloc_cyclic")]
extern "C" fn __xa_alloc_cyclic(xa: *mut xarray, id: *mut u32, entry: *mut (), mut limit: xa_limit, next: *mut u32, gfp: u32) -> i32 {
let min: u32 = limit.min;
let mut ret: i32 = 0;
let mut __v0: u32;
let __UNIQUE_ID_x__311: u32 = min;
let __UNIQUE_ID_y__312: u32 = unsafe { *next };
loop {
if __types_ok__2!(__UNIQUE_ID_x__311, __UNIQUE_ID_y__312) == 0 {
unsafe { __compiletime_assert_313() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__311 > __UNIQUE_ID_y__312 { __UNIQUE_ID_x__311 } else { __UNIQUE_ID_y__312 };
limit.min = __v0;
ret = __xa_alloc(xa, id, entry, limit, gfp);
if (unsafe { (*xa).xa_flags }) & 16 != 0 && ret == 0 {
let __v4: u32 = (unsafe { (*xa).xa_flags }) & !16;
unsafe {
(*xa).xa_flags = __v4;
}
ret = 1;
}
if ret < 0 && limit.min > min {
limit.min = min;
ret = __xa_alloc(xa, id, entry, limit, gfp);
if ret == 0 {
ret = 1;
}
}
if ret >= 0 {
unsafe {
*next = unsafe { *id }.wrapping_add(1);
}
if (unsafe { *next }) == 0 {
unsafe {
(*xa).xa_flags = (*xa).xa_flags | 16;
}
}
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xa_node {
    shift: u8,
    offset: u8,
    count: u8,
    nr_values: u8,
    parent: *mut xa_node,
    array: *mut xarray,
    __slate_anon_6: __SlateRecord_xarray_h_1175,
    slots: [*mut (); 64],
    __slate_anon_8: __SlateRecord_xarray_h_1180,
}


fn xa_head(xa: *const xarray) -> *mut () {
let mut __v0: *mut ();
let mut __v1: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_217() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*xa).xa_head) as *const *mut ()) };
let __UNIQUE_ID_rcu_216: *mut () = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = (unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) }) != 0 || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_216;
__v0
}

fn xa_head_locked(xa: *const xarray) -> *mut () {
rcu_dereference_protected!(unsafe { (*xa).xa_head }, unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) })
}

fn xa_entry(xa: *const xarray, node: *const xa_node, offset: u32) -> *mut () {
XA_NODE_BUG_ON!();
let mut __v1: *mut ();
let mut __v2: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_219() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(((*node).slots.as_ptr() as *const *mut ()).wrapping_offset(offset as isize) as *const *mut ()) };
let __UNIQUE_ID_rcu_218: *mut () = __v2;
loop {
let __v5: bool = if false {
let __v6: bool = (unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) }) != 0 || rcu_read_lock_held() != 0;
!__v6
} else {
false
};
if !__v5 {
break;
}
}
__v1 = __UNIQUE_ID_rcu_218;
__v1
}

fn xa_entry_locked(xa: *const xarray, node: *const xa_node, offset: u32) -> *mut () {
XA_NODE_BUG_ON!();
rcu_dereference_protected!(unsafe { *((*node).slots.as_ptr() as *const *mut ()).wrapping_offset(offset as isize) }, unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) })
}

fn xa_parent(xa: *const xarray, node: *const xa_node) -> *mut xa_node {
let mut __v0: *mut xa_node;
let mut __v1: *mut xa_node;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_221() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*node).parent) as *const *mut xa_node) };
let __UNIQUE_ID_rcu_220: *mut xa_node = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = (unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) }) != 0 || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_220;
__v0
}

fn xa_parent_locked(xa: *const xarray, node: *const xa_node) -> *mut xa_node {
rcu_dereference_protected__10!(unsafe { (*node).parent }, unsafe { lockdep_is_held(core::ptr::addr_of!((*xa).xa_lock) as *const ()) })
}

fn xa_mk_node(node: *const xa_node) -> *mut () {
((node as u64) | 2) as *mut ()
}

fn xa_to_node(entry: *const ()) -> *mut xa_node {
(entry as u64).wrapping_sub(2) as *mut xa_node
}

fn xa_is_node(entry: *const ()) -> bool {
xa_is_internal(entry) && (entry as u64) > 4096
}

fn xa_mk_sibling(offset: u32) -> *mut () {
xa_mk_internal(offset as u64)
}

fn xa_to_sibling(entry: *const ()) -> u64 {
xa_to_internal(entry)
}

fn xa_is_sibling(entry: *const ()) -> bool {
let __v0: bool = IS_ENABLED__2!() != 0 && xa_is_internal(entry);
__v0 && entry < (xa_mk_sibling((1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) as u32) as *const ())
}

fn xa_is_retry(entry: *const ()) -> bool {
unlikely!(entry == (xa_mk_internal(256u64) as *const ())) != 0
}

fn xa_is_advanced(entry: *const ()) -> bool {
xa_is_internal(entry) && entry <= (xa_mk_internal(256u64) as *const ())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xa_state {
    xa: *mut xarray,
    xa_index: u64,
    xa_shift: u8,
    xa_sibs: u8,
    xa_offset: u8,
    xa_pad: u8,
    xa_node: *mut xa_node,
    xa_alloc: *mut xa_node,
    xa_update: Option<unsafe extern "C" fn(*mut xa_node)>,
    xa_lru: *mut list_lru,
}


// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
fn xas_error(xas: *const xa_state) -> i32 {
xa_err((unsafe { (*xas).xa_node }) as *mut ())
}

fn xas_set_err(xas: *mut xa_state, err: i64) {
unsafe {
(*xas).xa_node = XA_ERROR!(err);
}
}

fn xas_invalid(xas: *const xa_state) -> bool {
((unsafe { (*xas).xa_node }) as u64) & 3 != 0
}

fn xas_valid(xas: *const xa_state) -> bool {
!xas_invalid(xas)
}

fn xas_is_node(xas: *const xa_state) -> bool {
xas_valid(xas) && !unsafe { (*xas).xa_node }.is_null()
}

fn xas_not_node(node: *mut xa_node) -> bool {
(node as u64) & 3 != 0 || node.is_null()
}

fn xas_frozen(node: *mut xa_node) -> bool {
(node as u64) & 2 != 0
}

fn xas_top(node: *mut xa_node) -> bool {
node <= (3u64 as *mut xa_node)
}

fn xas_reset(xas: *mut xa_state) {
unsafe {
(*xas).xa_node = 3u64 as *mut xa_node;
}
}

fn xas_retry(xas: *mut xa_state, entry: *const ()) -> bool {
if xa_is_zero(entry) {
return true;
}
if !xa_is_retry(entry) {
return false;
}
xas_reset(xas);
true
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_store")]
extern "C" fn xas_store(xas: *mut xa_state, mut entry: *mut ()) -> *mut () {
let mut slot: *mut *mut () = unsafe { core::ptr::addr_of_mut!((*(*xas).xa).xa_head) };
let mut offset: u32 = 0;
let mut max: u32 = 0;
let mut count: i32 = 0;
let mut values: i32 = 0;
let mut first: *mut () = unsafe { core::mem::zeroed() };
let mut next: *mut () = unsafe { core::mem::zeroed() };
let value: bool = xa_is_value(entry as *const ());
if !entry.is_null() {
let mut allow_root: bool = false;
allow_root = !xa_is_node(entry as *const ()) && !xa_is_zero(entry as *const ());
first = xas_create(xas, allow_root);
} else {
first = xas_load(xas);
}
if xas_invalid(xas as *const xa_state) {
return first;
}
let node: *mut xa_node = unsafe { (*xas).xa_node };
if !node.is_null() && ((unsafe { (*xas).xa_shift }) as i32) < ((unsafe { (*node).shift }) as i32) {
unsafe {
(*xas).xa_sibs = 0;
}
}
if first == entry && (unsafe { (*xas).xa_sibs }) == 0 {
return first;
}
next = first;
offset = (unsafe { (*xas).xa_offset }) as u32;
max = ((unsafe { (*xas).xa_offset }) as i32).wrapping_add((unsafe { (*xas).xa_sibs }) as i32) as u32;
if !node.is_null() {
slot = unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize);
if (unsafe { (*xas).xa_sibs }) != 0 {
xas_squash_marks(xas as *const xa_state);
}
}
if entry.is_null() {
xas_init_marks(xas as *const xa_state);
}
loop {
// Must clear the marks before setting the entry to NULL,
// otherwise xas_for_each_marked may find a NULL entry and
// stop early.  rcu_assign_pointer contains a release barrier
// so the mark clearing will appear to happen before the
// entry is set to NULL.
let _r_a_p__v: u64 = entry as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_242() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slot, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_243() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_244() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slot, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
if xa_is_node(next as *const ()) && (node.is_null() || (unsafe { (*node).shift }) != 0) {
xas_free_nodes(xas, xa_to_node(next as *const ()));
}
if node.is_null() {
break;
}
let __v12: i32 = count;
count = __v12.wrapping_add((next.is_null() as i32).wrapping_sub(entry.is_null() as i32));
let __v14: i32 = (!xa_is_value(first as *const ()) as i32).wrapping_sub(!value as i32);
values = values.wrapping_add(__v14);
if !entry.is_null() {
if offset == max {
break;
}
if !xa_is_sibling(entry as *const ()) {
entry = xa_mk_sibling((unsafe { (*xas).xa_offset }) as u32);
}
} else {
if (offset as u64) == (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
break;
}
}
let __v18: u32 = offset.wrapping_add(1);
offset = __v18;
next = xa_entry_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, __v18);
if !xa_is_sibling(next as *const ()) {
if entry.is_null() && offset > max {
break;
}
first = next;
}
let __v19: *mut *mut () = slot;
slot = __v19.wrapping_offset(1 as isize);
}
update_node(xas, node, count, values);
first
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_destroy")]
extern "C" fn xas_destroy(xas: *mut xa_state) {
let mut node: *mut xa_node = unsafe { (*xas).xa_alloc };
while !node.is_null() {
XA_NODE_BUG_ON!();
let mut __v1: *mut xa_node;
let mut __v2: *mut xa_node;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_224() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*node).parent) as *const *mut xa_node) };
let __UNIQUE_ID_rcu_223: *mut xa_node = __v2;
__v1 = __UNIQUE_ID_rcu_223;
let next: *mut xa_node = __v1;
unsafe { radix_tree_node_rcu_free(core::ptr::addr_of_mut!((*node).__slate_anon_6.callback_head)) };
let __v4: *mut xa_node = next;
node = __v4;
unsafe {
(*xas).xa_alloc = __v4;
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_split")]
extern "C" fn xas_split(xas: *mut xa_state, entry: *mut (), order: u32) {
let sibs: u32 = (1i32 << order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)).wrapping_sub(1) as u32;
let mut offset: u32 = 0;
let mut marks: u32 = 0;
let mut node: *mut xa_node = unsafe { core::mem::zeroed() };
let curr: *mut () = xas_load(xas);
let mut values: i32 = 0;
node = unsafe { (*xas).xa_node };
if xas_top(node) {
return;
}
marks = node_get_marks(node, (unsafe { (*xas).xa_offset }) as u32);
offset = (((unsafe { (*xas).xa_offset }) as i32) as u32).wrapping_add(sibs);
loop {
if ((unsafe { (*xas).xa_shift }) as i32) < ((unsafe { (*node).shift }) as i32) {
let child: *mut xa_node = unsafe { (*xas).xa_alloc };
let mut __v0: *mut xa_node;
let mut __v1: *mut xa_node;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_257() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*child).parent) as *const *mut xa_node) };
let __UNIQUE_ID_rcu_256: *mut xa_node = __v1;
__v0 = __UNIQUE_ID_rcu_256;
unsafe {
(*xas).xa_alloc = __v0;
}
unsafe {
(*child).shift = ((*node).shift as i32).wrapping_sub(if IS_ENABLED!() != 0 { 4 } else { 6 }) as u8;
}
unsafe {
(*child).offset = offset as u8;
}
unsafe {
(*child).count = (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u8;
}
unsafe {
(*child).nr_values = (if xa_is_value(entry as *const ()) { 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } } else { 0u64 }) as u8;
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_258() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!((*child).parent, RCU_INITIALIZER!(node));
if true {
break;
}
}
node_set_marks(node, offset, child, (unsafe { (*xas).xa_sibs }) as u32, marks);
let _r_a_p__v: u64 = xa_mk_node(child as *const xa_node) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_259() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_260() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_261() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
if xa_is_value(curr as *const ()) {
values = values.wrapping_sub(1);
}
xas_update(xas, child);
} else {
let canon: u32 = offset.wrapping_sub(((unsafe { (*xas).xa_sibs }) as i32) as u32);
node_set_marks(node, canon, core::ptr::null_mut::<xa_node>(), 0u32, marks);
{
let _r_a_p__v: u64 = entry as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_262() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(canon as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_263() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_264() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(canon as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}
while offset > canon {
let _r_a_p__v: u64 = xa_mk_sibling(canon) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_265() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let __v29: u32 = offset;
offset = __v29.wrapping_sub(1);
unsafe { core::ptr::write_volatile(((*node).slots.as_mut_ptr() as *mut *mut ()).wrapping_offset(__v29 as isize) as *mut *mut (), _r_a_p__v as *mut ()) };
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_266() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_267() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let __v36: u32 = offset;
offset = __v36.wrapping_sub(1);
unsafe { core::ptr::write_volatile(((*node).slots.as_mut_ptr() as *mut *mut ()).wrapping_offset(__v36 as isize) as *mut *mut (), RCU_INITIALIZER!(_r_a_p__v as *mut ())) };
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}
let __v43: i32 = (xa_is_value(entry as *const ()) as i32).wrapping_sub(xa_is_value(curr as *const ()) as i32).wrapping_mul(((unsafe { (*xas).xa_sibs }) as i32).wrapping_add(1));
values = values.wrapping_add(__v43);
}
let __v47: u32 = offset;
offset = __v47.wrapping_sub(1);
if __v47 <= (((unsafe { (*xas).xa_offset }) as i32) as u32) {
break;
}
}
unsafe {
(*node).nr_values = ((*node).nr_values as i32).wrapping_add(values) as u8;
}
xas_update(xas, node);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_split_alloc")]
extern "C" fn xas_split_alloc(xas: *mut xa_state, entry: *mut (), order: u32, mut gfp: u32) {
let mut node: core::mem::MaybeUninit<*mut xa_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut xa_node = core::ptr::addr_of_mut!(node) as *mut *mut xa_node;
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut sibs: u32 = unsafe { core::mem::zeroed() };
unsafe {
sibs = (1i32 << order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)).wrapping_sub(1) as u32;
__ret_warn_on = ((((*xas).xa_shift as i32).wrapping_add(2i32.wrapping_mul(if IS_ENABLED!() != 0 { 4i32 } else { 6i32 })) as u32) <= order) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("252: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 252b - ., 3; .popsection\n# {0}", const 252 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xas_split_alloc_15, sym __slate_asm_string_xas_split_alloc_16, const 1051 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("253: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 253b - ., 4; .popsection\n# {0}", const 253 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
'__join_0: {
// XXX: no support for splitting really large entries yet
let __v7: i64 = (__ret_warn_on != 0) as i64;
if __v7 == 0 {
if (((*xas).xa_shift as i32).wrapping_add(if IS_ENABLED!() != 0 { 4 } else { 6 }) as u32) > order {
return;
} else {
if (*(*xas).xa).xa_flags & 32 != 0 {
let __v4: u32 = gfp;
gfp = __v4 | (BIT!(22i32) as u32);
}
loop {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = alloc_hooks!(kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp)) as *mut xa_node;
if (*__slate_slot_2).is_null() {
break '__join_0;
} else {
__xas_init_node_for_split(xas, *__slate_slot_2, entry);
loop {
if true {
__WRITE_ONCE__5!((*(*__slate_slot_2)).parent, RCU_INITIALIZER!(unsafe { (*xas).xa_alloc }));
if true {
break;
}
}
}
(*xas).xa_alloc = *__slate_slot_2;
let __v1: u32 = sibs;
sibs = __v1.wrapping_sub(1);
if __v1 <= 0 {
break;
}
}
}
return;
}
}
}
xas_destroy(xas);
xas_set_err(xas, 12i32.wrapping_neg() as i64);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_try_split")]
extern "C" fn xas_try_split(xas: *mut xa_state, entry: *mut (), order: u32) {
let sibs: u32 = (1i32 << order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)).wrapping_sub(1) as u32;
let mut offset: u32 = 0;
let mut marks: u32 = 0;
let mut node: *mut xa_node = unsafe { core::mem::zeroed() };
let curr: *mut () = xas_load(xas);
let mut values: i32 = 0;
let mut gfp: u32 = (BIT!(11i32) as u32) | (BIT!(13i32) as u32);
node = unsafe { (*xas).xa_node };
if xas_top(node) {
return;
}
if (unsafe { (*(*xas).xa).xa_flags }) & 32 != 0 {
let __v0: u32 = gfp;
gfp = __v0 | (BIT!(22i32) as u32);
}
marks = node_get_marks(node, (unsafe { (*xas).xa_offset }) as u32);
offset = (((unsafe { (*xas).xa_offset }) as i32) as u32).wrapping_add(sibs);
if ((unsafe { (*xas).xa_shift }) as i32) < ((unsafe { (*node).shift }) as i32) {
let mut child: *mut xa_node = unsafe { (*xas).xa_alloc };
let expected_sibs: u32 = (1i32 << order.wrapping_sub(1) % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)).wrapping_sub(1) as u32;
// No support for splitting sibling entries
// (horizontally) or cascade split (vertically), which
// requires two or more new xa_nodes.
// Since if one xa_node allocation fails,
// it is hard to free the prior allocations.
if sibs != 0 || (((unsafe { (*xas).xa_sibs }) as i32) as u32) != expected_sibs {
xas_destroy(xas);
xas_set_err(xas, 22i32.wrapping_neg() as i64);
return;
}
if child.is_null() {
child = alloc_hooks!(unsafe { kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp) }) as *mut xa_node;
if child.is_null() {
xas_destroy(xas);
xas_set_err(xas, 12i32.wrapping_neg() as i64);
return;
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_270() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!((*child).parent, RCU_INITIALIZER!(unsafe { (*xas).xa_alloc }));
if true {
break;
}
}
}
__xas_init_node_for_split(xas, child, entry);
let mut __v10: *mut xa_node;
let mut __v11: *mut xa_node;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_272() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*child).parent) as *const *mut xa_node) };
let __UNIQUE_ID_rcu_271: *mut xa_node = __v11;
__v10 = __UNIQUE_ID_rcu_271;
unsafe {
(*xas).xa_alloc = __v10;
}
unsafe {
(*child).shift = ((*node).shift as i32).wrapping_sub(if IS_ENABLED!() != 0 { 4 } else { 6 }) as u8;
}
unsafe {
(*child).offset = offset as u8;
}
unsafe {
(*child).count = (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u8;
}
unsafe {
(*child).nr_values = (if xa_is_value(entry as *const ()) { 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } } else { 0u64 }) as u8;
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_273() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!((*child).parent, RCU_INITIALIZER!(node));
if true {
break;
}
}
node_set_marks(node, offset, child, (unsafe { (*xas).xa_sibs }) as u32, marks);
let _r_a_p__v: u64 = xa_mk_node(child as *const xa_node) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_274() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_275() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_276() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
if xa_is_value(curr as *const ()) {
values = values.wrapping_sub(1);
}
xas_update(xas, child);
} else {
loop {
let canon: u32 = offset.wrapping_sub(((unsafe { (*xas).xa_sibs }) as i32) as u32);
node_set_marks(node, canon, core::ptr::null_mut::<xa_node>(), 0u32, marks);
{
let _r_a_p__v: u64 = entry as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_277() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(canon as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_278() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_279() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(canon as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}
while offset > canon {
let _r_a_p__v: u64 = xa_mk_sibling(canon) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_280() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let __v39: u32 = offset;
offset = __v39.wrapping_sub(1);
unsafe { core::ptr::write_volatile(((*node).slots.as_mut_ptr() as *mut *mut ()).wrapping_offset(__v39 as isize) as *mut *mut (), _r_a_p__v as *mut ()) };
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_281() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_282() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let __v46: u32 = offset;
offset = __v46.wrapping_sub(1);
unsafe { core::ptr::write_volatile(((*node).slots.as_mut_ptr() as *mut *mut ()).wrapping_offset(__v46 as isize) as *mut *mut (), RCU_INITIALIZER!(_r_a_p__v as *mut ())) };
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}
let __v53: i32 = (xa_is_value(entry as *const ()) as i32).wrapping_sub(xa_is_value(curr as *const ()) as i32).wrapping_mul(((unsafe { (*xas).xa_sibs }) as i32).wrapping_add(1));
values = values.wrapping_add(__v53);
let __v57: u32 = offset;
offset = __v57.wrapping_sub(1);
if __v57 <= (((unsafe { (*xas).xa_offset }) as i32) as u32) {
break;
}
}
}
unsafe {
(*node).nr_values = ((*node).nr_values as i32).wrapping_add(values) as u8;
}
xas_update(xas, node);
}

fn xas_reload(xas: *mut xa_state) -> *mut () {
let node: *mut xa_node = unsafe { (*xas).xa_node };
let mut offset: u8 = 0;
if node.is_null() {
return xa_head((unsafe { (*xas).xa }) as *const xarray);
}
if IS_ENABLED__2!() != 0 {
offset = ((unsafe { (*xas).xa_index }) >> ((unsafe { (*node).shift }) as i32) & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) as u8;
let entry: *mut () = xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset as u32);
if !xa_is_sibling(entry as *const ()) {
return entry;
}
offset = xa_to_sibling(entry as *const ()) as u8;
} else {
offset = unsafe { (*xas).xa_offset };
}
xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset as u32)
}

fn xas_set(xas: &mut xa_state, index: u64) {
unsafe {
(*(xas as *mut xa_state)).xa_index = index;
}
unsafe {
(*(xas as *mut xa_state)).xa_node = 3u64 as *mut xa_node;
}
}

fn xas_set_order(xas: &mut xa_state, index: u64, order: u32) {
unsafe {
(*(xas as *mut xa_state)).xa_index = if order < 64 { index >> order << order } else { 0 };
}
unsafe {
(*(xas as *mut xa_state)).xa_shift = order.wrapping_sub(order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)) as u8;
}
unsafe {
(*(xas as *mut xa_state)).xa_sibs = (1i32 << order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)).wrapping_sub(1) as u8;
}
unsafe {
(*(xas as *mut xa_state)).xa_node = 3u64 as *mut xa_node;
}
}

fn xas_next_entry(xas: *mut xa_state, max: u64) -> *mut () {
let node: *mut xa_node = unsafe { (*xas).xa_node };
let mut entry: *mut () = unsafe { core::mem::zeroed() };
if unlikely!(xas_not_node(node) || (unsafe { (*node).shift }) != 0 || (((unsafe { (*xas).xa_offset }) as i32) as u64) != (unsafe { (*xas).xa_index }) & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) != 0 {
return xas_find(xas, max);
}
loop {
if unlikely!((unsafe { (*xas).xa_index }) >= max) != 0 {
return xas_find(xas, max);
}
if unlikely!((((unsafe { (*xas).xa_offset }) as i32) as u64) == (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) != 0 {
return xas_find(xas, max);
}
entry = xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, ((unsafe { (*xas).xa_offset }) as i32).wrapping_add(1) as u32);
if unlikely!(xa_is_internal(entry as *const ())) != 0 {
return xas_find(xas, max);
}
unsafe {
(*xas).xa_offset = ((*xas).xa_offset as i32).wrapping_add(1) as u8;
}
let __v4: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v4.wrapping_add(1);
}
if !entry.is_null() {
break;
}
}
entry
}

fn xas_find_chunk(xas: *mut xa_state, advance: bool, mark: u32) -> u32 {
let addr: *mut u64 = unsafe { (*((*(*xas).xa_node).__slate_anon_8.marks.as_mut_ptr() as *mut [u64; 1]).wrapping_offset(mark as isize)).as_mut_ptr() as *mut u64 };
let mut offset: u32 = (unsafe { (*xas).xa_offset }) as u32;
if advance {
offset = offset.wrapping_add(1);
}
if 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } == 64 {
if (offset as u64) < 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
let data: u64 = (unsafe { *addr }) & 18446744073709551615u64 << offset;
if data != 0 {
return __ffs!(data) as u32;
}
}
return (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32;
}
find_next_bit(addr as *const u64, 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }, offset as u64) as u32
}

fn xas_next_marked(xas: *mut xa_state, max: u64, mark: u32) -> *mut () {
let node: *mut xa_node = unsafe { (*xas).xa_node };
let mut offset: u32 = 0;
if unlikely!(xas_not_node(node) || (unsafe { (*node).shift }) != 0) != 0 {
return xas_find_marked(xas, max, mark);
}
offset = xas_find_chunk(xas, true, mark);
unsafe {
(*xas).xa_offset = offset as u8;
}
unsafe {
(*xas).xa_index = ((*xas).xa_index & !(1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)).wrapping_add(offset as u64);
}
if (unsafe { (*xas).xa_index }) > max {
return core::ptr::null_mut::<()>();
}
if (offset as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
return xas_find_marked(xas, max, mark);
}
let entry: *mut () = xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset);
if entry.is_null() {
return xas_find_marked(xas, max, mark);
}
entry
}

// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
// xas_for_each_marked() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
// @mark: Mark to search for.
//
// The loop body will be executed for each marked entry in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
// Coding conventions in this file:
//
// @xa is used to refer to the entire xarray.
// @xas is the 'xarray operation state'.  It may be either a pointer to
// an xa_state, or an xa_state stored on the stack.  This is an unfortunate
// ambiguity.
// @index is the index of the entry being operated on
// @mark is an xa_mark_t; a small number indicating one of the mark bits.
// @node refers to an xa_node; usually the primary one being operated on by
// this function.
// @offset is the index into the slots array inside an xa_node.
// @parent refers to the @xa_node closer to the head than @node.
// @entry refers to something stored in a slot in the xarray
fn xa_lock_type(xa: *const xarray) -> u32 {
(unsafe { (*xa).xa_flags }) & 3
}

fn xas_lock_type(xas: *mut xa_state, lock_type: u32) {
if lock_type == 1 {
xas_lock_irq!(xas);
} else {
if lock_type == 2 {
xas_lock_bh!(xas);
} else {
xas_lock!(xas);
}
}
}

fn xas_unlock_type(xas: *mut xa_state, lock_type: u32) {
if lock_type == 1 {
xas_unlock_irq!(xas);
} else {
if lock_type == 2 {
xas_unlock_bh!(xas);
} else {
xas_unlock!(xas);
}
}
}

fn xa_track_free(xa: *const xarray) -> bool {
(unsafe { (*xa).xa_flags }) & 4 != 0
}

fn xa_zero_busy(xa: *const xarray) -> bool {
(unsafe { (*xa).xa_flags }) & 8 != 0
}

fn xa_mark_set(xa: *mut xarray, mark: u32) {
if (unsafe { (*xa).xa_flags }) & XA_FLAGS_MARK!(mark) == 0 {
let __v1: u32 = unsafe { (*xa).xa_flags };
let __v2: u32 = __v1 | XA_FLAGS_MARK!(mark);
unsafe {
(*xa).xa_flags = __v2;
}
}
}

fn xa_mark_clear(xa: *mut xarray, mark: u32) {
if (unsafe { (*xa).xa_flags }) & XA_FLAGS_MARK!(mark) != 0 {
let __v1: u32 = unsafe { (*xa).xa_flags };
let __v2: u32 = __v1 & !XA_FLAGS_MARK!(mark);
unsafe {
(*xa).xa_flags = __v2;
}
}
}

fn node_marks(node: *mut xa_node, mark: u32) -> *mut u64 {
unsafe { (*((*node).__slate_anon_8.marks.as_mut_ptr() as *mut [u64; 1]).wrapping_offset(mark as isize)).as_mut_ptr() as *mut u64 }
}

fn node_get_mark(node: *mut xa_node, offset: u32, mark: u32) -> bool {
test_bit__9!(offset, node_marks(node, mark)) != 0
}

/// returns true if the bit was set
fn node_set_mark(node: *mut xa_node, offset: u32, mark: u32) -> bool {
__test_and_set_bit!(offset, node_marks(node, mark)) != 0
}

/// returns true if the bit was set
fn node_clear_mark(node: *mut xa_node, offset: u32, mark: u32) -> bool {
__test_and_clear_bit!(offset, node_marks(node, mark)) != 0
}

fn node_any_mark(node: *mut xa_node, mark: u32) -> bool {
!bitmap_empty(node_marks(node, mark) as *const u64, (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32)
}

fn node_mark_all(node: *mut xa_node, mark: u32) {
bitmap_fill(node_marks(node, mark), (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32);
}

/// xas_squash_marks() - Merge all marks to the first entry
/// @xas: Array operation state.
///
/// Set a mark on the first entry if any entry has it set.  Clear marks on
/// all sibling entries.
fn xas_squash_marks(xas: *const xa_state) {
let mut mark: u32 = 0;
let limit: u32 = ((unsafe { (*xas).xa_offset }) as i32).wrapping_add((unsafe { (*xas).xa_sibs }) as i32).wrapping_add(1) as u32;
loop {
let marks: *mut u64 = node_marks(unsafe { (*xas).xa_node }, mark);
if find_next_bit(marks as *const u64, limit as u64, ((unsafe { (*xas).xa_offset }) as i32).wrapping_add(1) as u64) != (limit as u64) {
__set_bit__3!(unsafe { (*xas).xa_offset }, marks);
bitmap_clear(marks, ((unsafe { (*xas).xa_offset }) as i32).wrapping_add(1) as u32, (unsafe { (*xas).xa_sibs }) as u32);
}
if mark == 2 {
break;
}
mark_inc!(mark);
}
}

/// extracts the offset within this node from the index
fn get_offset(index: u64, node: *mut xa_node) -> u32 {
(index >> ((unsafe { (*node).shift }) as i32) & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) as u32
}

fn xas_set_offset(xas: *mut xa_state) {
unsafe {
(*xas).xa_offset = get_offset((*xas).xa_index, (*xas).xa_node) as u8;
}
}

/// move the index either forwards (find) or backwards (sibling slot)
fn xas_move_index(xas: *mut xa_state, offset: u64) {
let shift: u32 = (unsafe { (*(*xas).xa_node).shift }) as u32;
let __v1: u64 = unsafe { (*xas).xa_index };
let __v2: u64 = __v1 & !(1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) << shift;
unsafe {
(*xas).xa_index = __v2;
}
let __v4: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v4.wrapping_add(offset << shift);
}
}

fn xas_next_offset(xas: *mut xa_state) {
unsafe {
(*xas).xa_offset = ((*xas).xa_offset as i32).wrapping_add(1) as u8;
}
xas_move_index(xas, (unsafe { (*xas).xa_offset }) as u64);
}

fn set_bounds(xas: *mut xa_state) -> *mut () {
unsafe {
(*xas).xa_node = 1u64 as *mut xa_node;
}
core::ptr::null_mut::<()>()
}

/// Starts a walk.  If the @xas is already valid, we assume that it's on
/// the right path and just return where we've got to.  If we're in an
/// error state, return NULL.  If the index is outside the current scope
/// of the xarray, return NULL without changing @xas->xa_node.  Otherwise
/// set @xas->xa_node to NULL and return the current head of the array.
fn xas_start(xas: *mut xa_state) -> *mut () {
if xas_valid(xas as *const xa_state) {
return xas_reload(xas);
}
if xas_error(xas as *const xa_state) != 0 {
return core::ptr::null_mut::<()>();
}
let entry: *mut () = xa_head((unsafe { (*xas).xa }) as *const xarray);
if !xa_is_node(entry as *const ()) {
if (unsafe { (*xas).xa_index }) != 0 {
return set_bounds(xas);
}
} else {
if (unsafe { (*xas).xa_index }) >> ((unsafe { (*xa_to_node(entry as *const ())).shift }) as i32) > (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
return set_bounds(xas);
}
}
unsafe {
(*xas).xa_node = core::ptr::null_mut::<xa_node>();
}
entry
}

#[inline(always)]
fn xas_descend(xas: *mut xa_state, node: *mut xa_node) -> *mut () {
let mut offset: u32 = get_offset(unsafe { (*xas).xa_index }, node);
let mut entry: *mut () = xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset);
unsafe {
(*xas).xa_node = node;
}
while xa_is_sibling(entry as *const ()) {
offset = xa_to_sibling(entry as *const ()) as u32;
entry = xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset);
let __v0: bool = (unsafe { (*node).shift }) != 0 && xa_is_node(entry as *const ());
if __v0 {
entry = xa_mk_internal(256u64);
}
}
unsafe {
(*xas).xa_offset = offset as u8;
}
entry
}

/// xas_load() - Load an entry from the XArray (advanced).
/// @xas: XArray operation state.
///
/// Usually walks the @xas to the appropriate state to load the entry
/// stored at xa_index.  However, it will do nothing and return %NULL if
/// @xas is in an error state.  xas_load() will never expand the tree.
///
/// If the xa_state is set up to operate on a multi-index entry, xas_load()
/// may return %NULL or an internal entry, even if there are entries
/// present within the range specified by @xas.
///
/// Context: Any context.  The caller should hold the xa_lock or the RCU lock.
/// Return: Usually an entry in the XArray, but see description for exceptions.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_load")]
extern "C" fn xas_load(xas: *mut xa_state) -> *mut () {
let mut entry: *mut () = xas_start(xas);
while xa_is_node(entry as *const ()) {
let node: *mut xa_node = xa_to_node(entry as *const ());
if ((unsafe { (*xas).xa_shift }) as i32) > ((unsafe { (*node).shift }) as i32) {
break;
}
entry = xas_descend(xas, node);
if ((unsafe { (*node).shift }) as i32) == 0 {
break;
}
}
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_load_222: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state) -> *mut ()>, *mut ()>(Some(xas_load)) };

fn xa_node_free(node: *mut xa_node) {
XA_NODE_BUG_ON!();
unsafe {
(*node).array = 1i32 as *mut xarray;
}
unsafe { call_rcu(core::ptr::addr_of_mut!((*node).__slate_anon_6.callback_head), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut callback_head)>>(radix_tree_node_rcu_free as *const ()) }) };
}

// xas_destroy() - Free any resources allocated during the XArray operation.
// @xas: XArray operation state.
//
// Most users will not need to call this function; it is called for you
// by xas_nomem().
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_destroy_225: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state)>, *mut ()>(Some(xas_destroy)) };

/// xas_nomem() - Allocate memory if needed.
/// @xas: XArray operation state.
/// @gfp: Memory allocation flags.
///
/// If we need to add new nodes to the XArray, we try to allocate memory
/// with GFP_NOWAIT while holding the lock, which will usually succeed.
/// If it fails, @xas is flagged as needing memory to continue.  The caller
/// should drop the lock and call xas_nomem().  If xas_nomem() succeeds,
/// the caller should retry the operation.
///
/// Forward progress is guaranteed as one node is allocated here and
/// stored in the xa_state where it will be found by xas_alloc().  More
/// nodes will likely be found in the slab allocator, but we do not tie
/// them up here.
///
/// Return: true if memory was needed, and was successfully allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_nomem")]
extern "C" fn xas_nomem(xas: *mut xa_state, mut gfp: u32) -> bool {
if (unsafe { (*xas).xa_node }) != XA_ERROR!(12i32.wrapping_neg()) {
xas_destroy(xas);
return false;
}
if (unsafe { (*(*xas).xa).xa_flags }) & 32 != 0 {
let __v0: u32 = gfp;
gfp = __v0 | (BIT!(22i32) as u32);
}
unsafe {
(*xas).xa_alloc = alloc_hooks!(kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp)) as *mut xa_node;
}
if unsafe { (*xas).xa_alloc }.is_null() {
return false;
}
unsafe {
(*(*xas).xa_alloc).parent = core::ptr::null_mut::<xa_node>();
}
XA_NODE_BUG_ON!();
unsafe {
(*xas).xa_node = 3u64 as *mut xa_node;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_nomem_226: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, u32) -> bool>, *mut ()>(Some(xas_nomem)) };

/// __xas_nomem() - Drop locks and allocate memory if needed.
/// @xas: XArray operation state.
/// @gfp: Memory allocation flags.
///
/// Internal variant of xas_nomem().
///
/// Return: true if memory was needed, and was successfully allocated.
fn __xas_nomem(xas: *mut xa_state, mut gfp: u32) -> bool {
let lock_type: u32 = xa_lock_type((unsafe { (*xas).xa }) as *const xarray);
if (unsafe { (*xas).xa_node }) != XA_ERROR!(12i32.wrapping_neg()) {
xas_destroy(xas);
return false;
}
if (unsafe { (*(*xas).xa).xa_flags }) & 32 != 0 {
let __v0: u32 = gfp;
gfp = __v0 | (BIT!(22i32) as u32);
}
if gfpflags_allow_blocking(gfp) {
xas_unlock_type(xas, lock_type);
unsafe {
(*xas).xa_alloc = alloc_hooks!(kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp)) as *mut xa_node;
}
xas_lock_type(xas, lock_type);
} else {
unsafe {
(*xas).xa_alloc = alloc_hooks!(kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp)) as *mut xa_node;
}
}
if unsafe { (*xas).xa_alloc }.is_null() {
return false;
}
unsafe {
(*(*xas).xa_alloc).parent = core::ptr::null_mut::<xa_node>();
}
XA_NODE_BUG_ON!();
unsafe {
(*xas).xa_node = 3u64 as *mut xa_node;
}
true
}

fn xas_update(xas: *mut xa_state, node: *mut xa_node) {
if (unsafe { (*xas).xa_update }) != None {
unsafe { unsafe { (*xas).xa_update }.unwrap()(node) };
} else {
XA_NODE_BUG_ON!();
}
}

fn xas_alloc(xas: *mut xa_state, shift: u32) -> *mut () {
let parent: *mut xa_node = unsafe { (*xas).xa_node };
let mut node: *mut xa_node = unsafe { (*xas).xa_alloc };
if xas_invalid(xas as *const xa_state) {
return core::ptr::null_mut::<()>();
}
if !node.is_null() {
unsafe {
(*xas).xa_alloc = core::ptr::null_mut::<xa_node>();
}
} else {
let mut gfp: u32 = (BIT!(11i32) as u32) | (BIT!(13i32) as u32);
if (unsafe { (*(*xas).xa).xa_flags }) & 32 != 0 {
let __v0: u32 = gfp;
gfp = __v0 | (BIT!(22i32) as u32);
}
node = alloc_hooks!(unsafe { kmem_cache_alloc_lru_noprof(radix_tree_node_cachep, (*xas).xa_lru, gfp) }) as *mut xa_node;
if node.is_null() {
xas_set_err(xas, 12i32.wrapping_neg() as i64);
return core::ptr::null_mut::<()>();
}
}
if !parent.is_null() {
unsafe {
(*node).offset = (*xas).xa_offset;
}
unsafe {
(*parent).count = ((*parent).count as i32).wrapping_add(1) as u8;
}
XA_NODE_BUG_ON!();
xas_update(xas, parent);
}
XA_NODE_BUG_ON!();
XA_NODE_BUG_ON!();
unsafe {
(*node).shift = shift as u8;
}
unsafe {
(*node).count = 0;
}
unsafe {
(*node).nr_values = 0;
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_227() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!((*node).parent, RCU_INITIALIZER!(unsafe { (*xas).xa_node }));
if true {
break;
}
}
unsafe {
(*node).array = (*xas).xa;
}
node as *mut ()
}

/// Returns the number of indices covered by a given xa_state
fn xas_size(xas: *const xa_state) -> u64 {
(((unsafe { (*xas).xa_sibs }) as i32) as u64).wrapping_add(1) << ((unsafe { (*xas).xa_shift }) as i32)
}

/// Use this to calculate the maximum index that will need to be created
/// in order to add the entry described by @xas.  Because we cannot store a
/// multi-index entry at index 0, the calculation is a little more complex
/// than you might expect.
fn xas_max(xas: *mut xa_state) -> u64 {
let mut max: u64 = unsafe { (*xas).xa_index };
if (unsafe { (*xas).xa_shift }) != 0 || (unsafe { (*xas).xa_sibs }) != 0 {
let mask: u64 = xas_size(xas as *const xa_state).wrapping_sub(1);
max |= mask;
if mask == max {
max = max.wrapping_add(1);
}
}
max
}

/// The maximum index that can be contained in the array without expanding it
fn max_index(entry: *mut ()) -> u64 {
if !xa_is_node(entry as *const ()) {
return 0;
}
(1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } << ((unsafe { (*xa_to_node(entry as *const ())).shift }) as i32)).wrapping_sub(1)
}

fn xa_zero_to_null(entry: *mut ()) -> *mut () {
if xa_is_zero(entry as *const ()) { core::ptr::null_mut::<()>() } else { entry }
}

fn xas_shrink(xas: *mut xa_state) {
let xa: *mut xarray = unsafe { (*xas).xa };
let mut node: *mut xa_node = unsafe { (*xas).xa_node };
loop {
let mut entry: *mut () = unsafe { core::mem::zeroed() };
XA_NODE_BUG_ON!();
if ((unsafe { (*node).count }) as i32) != 1 {
break;
}
entry = xa_entry_locked(xa as *const xarray, node as *const xa_node, 0u32);
if entry.is_null() {
break;
}
if !xa_is_node(entry as *const ()) && (unsafe { (*node).shift }) != 0 {
break;
}
if xa_zero_busy(xa as *const xarray) {
entry = xa_zero_to_null(entry);
}
unsafe {
(*xas).xa_node = 1u64 as *mut xa_node;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_228() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*xa).xa_head, RCU_INITIALIZER!(entry));
if true {
break;
}
}
let __v5: bool = xa_track_free(xa as *const xarray) && !node_get_mark(node, 0u32, 0u32);
if __v5 {
xa_mark_clear(xa, 0u32);
}
unsafe {
(*node).count = 0;
}
unsafe {
(*node).nr_values = 0;
}
if !xa_is_node(entry as *const ()) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_229() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(0 as isize), RCU_INITIALIZER!(xa_mk_internal(256u64)));
if true {
break;
}
}
}
xas_update(xas, node);
xa_node_free(node);
if !xa_is_node(entry as *const ()) {
break;
}
node = xa_to_node(entry as *const ());
unsafe {
(*node).parent = core::ptr::null_mut::<xa_node>();
}
}
}

/// xas_delete_node() - Attempt to delete an xa_node
/// @xas: Array operation state.
///
/// Attempts to delete the @xas->xa_node.  This will fail if xa->node has
/// a non-zero reference count.
fn xas_delete_node(xas: *mut xa_state) {
let mut node: *mut xa_node = unsafe { (*xas).xa_node };
loop {
XA_NODE_BUG_ON!();
if (unsafe { (*node).count }) != 0 {
break;
}
let parent: *mut xa_node = xa_parent_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node);
unsafe {
(*xas).xa_node = parent;
}
unsafe {
(*xas).xa_offset = (*node).offset;
}
xa_node_free(node);
if parent.is_null() {
unsafe {
(*(*xas).xa).xa_head = core::ptr::null_mut::<()>();
}
unsafe {
(*xas).xa_node = 1u64 as *mut xa_node;
}
return;
}
unsafe {
*((*parent).slots.as_mut_ptr() as *mut *mut ()).wrapping_offset(((*xas).xa_offset as i32) as isize) = core::ptr::null_mut::<()>();
}
unsafe {
(*parent).count = ((*parent).count as i32).wrapping_sub(1) as u8;
}
XA_NODE_BUG_ON!();
node = parent;
xas_update(xas, node);
}
if unsafe { (*node).parent }.is_null() {
xas_shrink(xas);
}
}

// xas_free_nodes() - Free this node and all nodes that it references
// @xas: Array operation state.
// @top: Node to free
//
// This node has been removed from the tree.  We must now free it and all
// of its subnodes.  There may be RCU walkers with references into the tree,
// so we must replace all entries with retry markers.
fn xas_free_nodes(xas: *mut xa_state, top: *mut xa_node) {
let mut offset: u32 = 0;
let mut node: *mut xa_node = top;
loop {
let entry: *mut () = xa_entry_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset);
let __v0: bool = (unsafe { (*node).shift }) != 0 && xa_is_node(entry as *const ());
if __v0 {
node = xa_to_node(entry as *const ());
offset = 0;
} else {
if !entry.is_null() {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_230() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize), RCU_INITIALIZER!(xa_mk_internal(256u64)));
if true {
break;
}
}
}
offset = offset.wrapping_add(1);
while (offset as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
let parent: *mut xa_node = xa_parent_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node);
offset = ((unsafe { (*node).offset }) as i32).wrapping_add(1) as u32;
unsafe {
(*node).count = 0;
}
unsafe {
(*node).nr_values = 0;
}
xas_update(xas, node);
xa_node_free(node);
if node == top {
return;
}
node = parent;
}
}
}
}

// xas_expand adds nodes to the head of the tree until it has reached
// sufficient height to be able to contain @xas->xa_index
fn xas_expand(xas: *mut xa_state, mut head: *mut ()) -> i32 {
let xa: *mut xarray = unsafe { (*xas).xa };
let mut node: *mut xa_node = core::ptr::null_mut::<xa_node>();
let mut shift: u32 = 0;
let max: u64 = xas_max(xas);
if head.is_null() {
if max == 0 {
return 0;
}
while max >> shift >= 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
let __v0: u32 = shift;
shift = __v0.wrapping_add((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32);
}
return shift.wrapping_add((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32) as i32;
} else {
if xa_is_node(head as *const ()) {
node = xa_to_node(head as *const ());
shift = ((unsafe { (*node).shift }) as i32).wrapping_add(if IS_ENABLED!() != 0 { 4 } else { 6 }) as u32;
}
}
unsafe {
(*xas).xa_node = core::ptr::null_mut::<xa_node>();
}
while max > max_index(head) {
let mut mark: u32 = 0;
XA_NODE_BUG_ON!();
node = xas_alloc(xas, shift) as *mut xa_node;
if node.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*node).count = 1;
}
if xa_is_value(head as *const ()) {
unsafe {
(*node).nr_values = 1;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_231() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(0 as isize), RCU_INITIALIZER!(head));
if true {
break;
}
}
// Propagate the aggregated mark info to the new child
loop {
if xa_track_free(xa as *const xarray) && mark == 0 {
node_mark_all(node, 0u32);
if !xa_marked(xa as *const xarray, 0u32) {
node_clear_mark(node, 0u32, 0u32);
xa_mark_set(xa, 0u32);
}
} else {
if xa_marked(xa as *const xarray, mark) {
node_set_mark(node, 0u32, mark);
}
}
if mark == 2 {
break;
}
mark_inc!(mark);
}
// Now that the new node is fully initialised, we can add
// it to the tree
if xa_is_node(head as *const ()) {
unsafe {
(*xa_to_node(head as *const ())).offset = 0;
}
let _r_a_p__v: u64 = node as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_232() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!((*xa_to_node(head as *const ())).parent, _r_a_p__v as *mut xa_node);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__8!() {
unsafe { __compiletime_assert_233() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut xa_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_234() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!(*unsafe { core::ptr::addr_of_mut!((*xa_to_node(head as *const ())).parent) }, RCU_INITIALIZER!(_r_a_p__v as *mut xa_node));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}
head = xa_mk_node(node as *const xa_node);
let _r_a_p__v: u64 = head as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_235() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*xa).xa_head, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_236() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_237() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*xa).xa_head) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
xas_update(xas, node);
let __v29: u32 = shift;
shift = __v29.wrapping_add((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32);
}
unsafe {
(*xas).xa_node = node;
}
shift as i32
}

// xas_create() - Create a slot to store an entry in.
// @xas: XArray operation state.
// @allow_root: %true if we can store the entry in the root directly
//
// Most users will not need to call this function directly, as it is called
// by xas_store().  It is useful for doing conditional store operations
// (see the xa_cmpxchg() implementation for an example).
//
// Return: If the slot already existed, returns the contents of this slot.
// If the slot was newly created, returns %NULL.  If it failed to create the
// slot, returns %NULL and indicates the error in @xas.
fn xas_create(xas: *mut xa_state, allow_root: bool) -> *mut () {
let xa: *mut xarray = unsafe { (*xas).xa };
let mut entry: *mut () = unsafe { core::mem::zeroed() };
let mut slot: *mut *mut () = unsafe { core::mem::zeroed() };
let mut node: *mut xa_node = unsafe { (*xas).xa_node };
let mut shift: i32 = 0;
let order: u32 = (unsafe { (*xas).xa_shift }) as u32;
if xas_top(node) {
entry = xa_head_locked(xa as *const xarray);
unsafe {
(*xas).xa_node = core::ptr::null_mut::<xa_node>();
}
let __v0: bool = entry.is_null() && xa_zero_busy(xa as *const xarray);
if __v0 {
entry = xa_mk_internal(257u64);
}
shift = xas_expand(xas, entry);
if shift < 0 {
return core::ptr::null_mut::<()>();
}
if shift == 0 && !allow_root {
shift = if IS_ENABLED!() != 0 { 4 } else { 6 };
}
entry = xa_head_locked(xa as *const xarray);
slot = unsafe { core::ptr::addr_of_mut!((*xa).xa_head) };
} else {
if xas_error(xas as *const xa_state) != 0 {
return core::ptr::null_mut::<()>();
} else {
if !node.is_null() {
let offset: u32 = (unsafe { (*xas).xa_offset }) as u32;
shift = (unsafe { (*node).shift }) as i32;
entry = xa_entry_locked(xa as *const xarray, node as *const xa_node, offset);
slot = unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(offset as isize);
} else {
shift = 0;
entry = xa_head_locked(xa as *const xarray);
slot = unsafe { core::ptr::addr_of_mut!((*xa).xa_head) };
}
}
}
while (shift as u32) > order {
let __v1: i32 = shift;
shift = __v1.wrapping_sub(if IS_ENABLED!() != 0 { 4 } else { 6 });
if entry.is_null() {
node = xas_alloc(xas, shift as u32) as *mut xa_node;
if node.is_null() {
break;
}
if xa_track_free(xa as *const xarray) {
node_mark_all(node, 0u32);
}
let _r_a_p__v: u64 = xa_mk_node(node as *const xa_node) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_238() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slot, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_239() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_240() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slot, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
} else {
if xa_is_node(entry as *const ()) {
node = xa_to_node(entry as *const ());
} else {
break;
}
}
entry = xas_descend(xas, node);
slot = unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(((unsafe { (*xas).xa_offset }) as i32) as isize);
}
entry
}

/// xas_create_range() - Ensure that stores to this range will succeed
/// @xas: XArray operation state.
///
/// Creates all of the slots in the range covered by @xas.  Sets @xas to
/// create single-index entries and positions it at the beginning of the
/// range.  This is for the benefit of users which have not yet been
/// converted to use multi-index entries.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_create_range")]
extern "C" fn xas_create_range(xas: *mut xa_state) {
let mut node: *mut xa_node = unsafe { core::mem::zeroed() };
let mut sibs: u8 = unsafe { core::mem::zeroed() };
let mut shift: u8 = unsafe { core::mem::zeroed() };
let mut index: u64 = unsafe { core::mem::zeroed() };
unsafe {
index = (*xas).xa_index;
shift = (*xas).xa_shift;
sibs = (*xas).xa_sibs;
(*xas).xa_index = (*xas).xa_index | (((sibs as i32) as u64).wrapping_add(1) << (shift as i32)).wrapping_sub(1);
if xas_is_node(xas as *const xa_state) && ((*(*xas).xa_node).shift as i32) == ((*xas).xa_shift as i32) {
(*xas).xa_offset = (((*xas).xa_offset as i32) | (sibs as i32)) as u8;
}
(*xas).xa_shift = 0;
(*xas).xa_sibs = 0;
'__loop_8: loop {
xas_create(xas, true);
if xas_error(xas as *const xa_state) != 0 {
(*xas).xa_shift = shift;
(*xas).xa_sibs = sibs;
(*xas).xa_index = index;
return;
} else {
if (*xas).xa_index <= index | (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
break;
} else {
let __v2: u64 = (*xas).xa_index;
let __v1: u64 = __v2.wrapping_sub(1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 });
(*xas).xa_index = __v1;
loop {
node = (*xas).xa_node;
if ((*node).shift as i32) >= (shift as i32) {
continue '__loop_8;
} else {
(*xas).xa_node = xa_parent_locked((*xas).xa as *const xarray, node as *const xa_node);
(*xas).xa_offset = ((*node).offset as i32).wrapping_sub(1) as u8;
if ((*node).offset as i32) != 0 {
continue '__loop_8;
}
}
}
}
}
}
(*xas).xa_index = index;
if !unsafe { (*xas).xa_node }.is_null() {
xas_set_offset(xas);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_create_range_241: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state)>, *mut ()>(Some(xas_create_range)) };

fn update_node(xas: *mut xa_state, node: *mut xa_node, count: i32, values: i32) {
if node.is_null() || count == 0 && values == 0 {
return;
}
unsafe {
(*node).count = ((*node).count as i32).wrapping_add(count) as u8;
}
unsafe {
(*node).nr_values = ((*node).nr_values as i32).wrapping_add(values) as u8;
}
XA_NODE_BUG_ON!();
XA_NODE_BUG_ON!();
xas_update(xas, node);
if count < 0 {
xas_delete_node(xas);
}
}

// xas_store() - Store this entry in the XArray.
// @xas: XArray operation state.
// @entry: New entry.
//
// If @xas is operating on a multi-index entry, the entry returned by this
// function is essentially meaningless (it may be an internal entry or it
// may be %NULL, even if there are non-NULL entries at some of the indices
// covered by the range).  This is not a problem for any current users,
// and can be changed if needed.
//
// Return: The old entry at this index.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_store_245: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, *mut ()) -> *mut ()>, *mut ()>(Some(xas_store)) };

/// xas_get_mark() - Returns the state of this mark.
/// @xas: XArray operation state.
/// @mark: Mark number.
///
/// Return: true if the mark is set, false if the mark is clear or @xas
/// is in an error state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_get_mark")]
extern "C" fn xas_get_mark(xas: *const xa_state, mark: u32) -> bool {
if xas_invalid(xas) {
return false;
}
if unsafe { (*xas).xa_node }.is_null() {
return xa_marked((unsafe { (*xas).xa }) as *const xarray, mark);
}
node_get_mark(unsafe { (*xas).xa_node }, (unsafe { (*xas).xa_offset }) as u32, mark)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_get_mark_246: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const xa_state, u32) -> bool>, *mut ()>(Some(xas_get_mark)) };

/// xas_set_mark() - Sets the mark on this entry and its parents.
/// @xas: XArray operation state.
/// @mark: Mark number.
///
/// Sets the specified mark on this entry, and walks up the tree setting it
/// on all the ancestor entries.  Does nothing if @xas has not been walked to
/// an entry, or is in an error state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_set_mark")]
extern "C" fn xas_set_mark(xas: *const xa_state, mark: u32) {
let mut node: *mut xa_node = unsafe { (*xas).xa_node };
let mut offset: u32 = (unsafe { (*xas).xa_offset }) as u32;
if xas_invalid(xas) {
return;
}
while !node.is_null() {
if node_set_mark(node, offset, mark) {
return;
}
offset = (unsafe { (*node).offset }) as u32;
node = xa_parent_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node);
}
if !xa_marked((unsafe { (*xas).xa }) as *const xarray, mark) {
xa_mark_set(unsafe { (*xas).xa }, mark);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_set_mark_247: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const xa_state, u32)>, *mut ()>(Some(xas_set_mark)) };

/// xas_clear_mark() - Clears the mark on this entry and its parents.
/// @xas: XArray operation state.
/// @mark: Mark number.
///
/// Clears the specified mark on this entry, and walks back to the head
/// attempting to clear it on all the ancestor entries.  Does nothing if
/// @xas has not been walked to an entry, or is in an error state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_clear_mark")]
extern "C" fn xas_clear_mark(xas: *const xa_state, mark: u32) {
let mut node: *mut xa_node = unsafe { (*xas).xa_node };
let mut offset: u32 = (unsafe { (*xas).xa_offset }) as u32;
if xas_invalid(xas) {
return;
}
while !node.is_null() {
if !node_clear_mark(node, offset, mark) {
return;
}
if node_any_mark(node, mark) {
return;
}
offset = (unsafe { (*node).offset }) as u32;
node = xa_parent_locked((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node);
}
if xa_marked((unsafe { (*xas).xa }) as *const xarray, mark) {
xa_mark_clear(unsafe { (*xas).xa }, mark);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_clear_mark_248: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const xa_state, u32)>, *mut ()>(Some(xas_clear_mark)) };

/// xas_init_marks() - Initialise all marks for the entry
/// @xas: Array operations state.
///
/// Initialise all marks for the entry specified by @xas.  If we're tracking
/// free entries with a mark, we need to set it on all entries.  All other
/// marks are cleared.
///
/// This implementation is not as efficient as it could be; we may walk
/// up the tree multiple times.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_init_marks")]
extern "C" fn xas_init_marks(xas: *const xa_state) {
let mut mark: u32 = 0;
loop {
if xa_track_free((unsafe { (*xas).xa }) as *const xarray) && mark == 0 {
xas_set_mark(xas, mark);
} else {
xas_clear_mark(xas, mark);
}
if mark == 2 {
break;
}
mark_inc!(mark);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_init_marks_249: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const xa_state)>, *mut ()>(Some(xas_init_marks)) };

fn node_get_marks(node: *mut xa_node, offset: u32) -> u32 {
let mut marks: u32 = 0;
let mut mark: u32 = 0;
loop {
if node_get_mark(node, offset, mark) {
marks |= (1i32 << mark) as u32;
}
if mark == 2 {
break;
}
mark_inc!(mark);
}
marks
}

fn node_mark_slots(node: *mut xa_node, sibs: u32, mark: u32) {
let mut i: i32 = 0;
if sibs == 0 {
node_mark_all(node, mark);
} else {
i = 0;
while (i as u64) < 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
node_set_mark(node, i as u32, mark);
i = (i as u32).wrapping_add(sibs.wrapping_add(1)) as i32;
}
}
}

fn node_set_marks(node: *mut xa_node, offset: u32, child: *mut xa_node, sibs: u32, marks: u32) {
let mut mark: u32 = 0;
loop {
if marks & ((1i32 << mark) as u32) != 0 {
node_set_mark(node, offset, mark);
if !child.is_null() {
node_mark_slots(child, sibs, mark);
}
}
if mark == 2 {
break;
}
mark_inc!(mark);
}
}

fn __xas_init_node_for_split(xas: *mut xa_state, node: *mut xa_node, entry: *mut ()) {
let mut i: u32 = 0;
let mut sibling: *mut () = core::ptr::null_mut::<()>();
let mask: u32 = (unsafe { (*xas).xa_sibs }) as u32;
if node.is_null() {
return;
}
unsafe {
(*node).array = (*xas).xa;
}
while (i as u64) < 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
if i & mask == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_250() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(i as isize), RCU_INITIALIZER!(entry));
if true {
break;
}
}
sibling = xa_mk_sibling(i);
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_251() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*node).slots.as_mut_ptr() as *mut *mut () }.wrapping_offset(i as isize), RCU_INITIALIZER!(sibling));
if true {
break;
}
}
}
i = i.wrapping_add(1);
}
}

// xas_split_alloc() - Allocate memory for splitting an entry.
// @xas: XArray operation state.
// @entry: New entry which will be stored in the array.
// @order: Current entry order.
// @gfp: Memory allocation flags.
//
// This function should be called before calling xas_split().
// If necessary, it will allocate new nodes (and fill them with @entry)
// to prepare for the upcoming split of an entry of @order size into
// entries of the order stored in the @xas.
//
// Context: May sleep if @gfp flags permit.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_split_alloc_255: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, *mut (), u32, u32)>, *mut ()>(Some(xas_split_alloc)) };

// xas_split() - Split a multi-index entry into smaller entries.
// @xas: XArray operation state.
// @entry: New entry to store in the array.
// @order: Current entry order.
//
// The size of the new entries is set in @xas.  The value in @entry is
// copied to all the replacement entries.
//
// Context: Any context.  The caller should hold the xa_lock.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_split_268: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, *mut (), u32)>, *mut ()>(Some(xas_split)) };

/// xas_try_split_min_order() - Minimal split order xas_try_split() can accept
/// @order: Current entry order.
///
/// xas_try_split() can split a multi-index entry to smaller than @order - 1 if
/// no new xa_node is needed. This function provides the minimal order
/// xas_try_split() supports.
///
/// Return: the minimal order xas_try_split() supports
///
/// Context: Any context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_try_split_min_order")]
extern "C" fn xas_try_split_min_order(order: u32) -> u32 {
if order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32) == 0 {
return if order == 0 { 0 } else { order.wrapping_sub(1) };
}
order.wrapping_sub(order % ((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_try_split_min_order_269: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> u32>, *mut ()>(Some(xas_try_split_min_order)) };

// xas_try_split() - Try to split a multi-index entry.
// @xas: XArray operation state.
// @entry: New entry to store in the array.
// @order: Current entry order.
//
// The size of the new entries is set in @xas.  The value in @entry is
// copied to all the replacement entries. If and only if one new xa_node is
// needed, the function will use GFP_NOWAIT to get one if xas->xa_alloc is
// NULL. If more new xa_node are needed, the function gives EINVAL error.
//
// NOTE: use xas_try_split_min_order() to get next split order instead of
// @order - 1 if you want to minmize xas_try_split() calls.
//
// Context: Any context.  The caller should hold the xa_lock.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_try_split_283: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, *mut (), u32)>, *mut ()>(Some(xas_try_split)) };

/// xas_pause() - Pause a walk to drop a lock.
/// @xas: XArray operation state.
///
/// Some users need to pause a walk and drop the lock they're holding in
/// order to yield to a higher priority thread or carry out an operation
/// on an entry.  Those users should call this function before they drop
/// the lock.  It resets the @xas to be suitable for the next iteration
/// of the loop after the user has reacquired the lock.  If most entries
/// found during a walk require you to call xas_pause(), the xa_for_each()
/// iterator may be more appropriate.
///
/// Note that xas_pause() only works for forward iteration.  If a user needs
/// to pause a reverse iteration, we will need a xas_pause_rev().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_pause")]
extern "C" fn xas_pause(xas: *mut xa_state) {
let node: *mut xa_node = unsafe { (*xas).xa_node };
if xas_invalid(xas as *const xa_state) {
return;
}
unsafe {
(*xas).xa_node = 3u64 as *mut xa_node;
}
if !node.is_null() {
let mut offset: u64 = (unsafe { (*xas).xa_offset }) as u64;
loop {
let __v2: u64 = offset.wrapping_add(1);
offset = __v2;
if __v2 >= 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
break;
}
if !xa_is_sibling(xa_entry((unsafe { (*xas).xa }) as *const xarray, node as *const xa_node, offset as u32) as *const ()) {
break;
}
}
let __v5: u64 = (unsafe { (*xas).xa_index }) & 18446744073709551615u64 << ((unsafe { (*node).shift }) as i32);
unsafe {
(*xas).xa_index = __v5;
}
let __v7: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v7.wrapping_add(offset.wrapping_sub(((*xas).xa_offset as i32) as u64) << ((*node).shift as i32));
}
if (unsafe { (*xas).xa_index }) == 0 {
unsafe {
(*xas).xa_node = 1u64 as *mut xa_node;
}
}
} else {
let __v10: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v10.wrapping_add(1);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_pause_284: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state)>, *mut ()>(Some(xas_pause)) };

/// __xas_prev() - Find the previous entry in the XArray.
/// @xas: XArray operation state.
///
/// Helper function for xas_prev() which handles all the complex cases
/// out of line.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xas_prev")]
extern "C" fn __xas_prev(xas: *mut xa_state) -> *mut () {
if !xas_frozen(unsafe { (*xas).xa_node }) {
let __v1: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v1.wrapping_sub(1);
}
}
if unsafe { (*xas).xa_node }.is_null() {
return set_bounds(xas);
}
if xas_not_node(unsafe { (*xas).xa_node }) {
return xas_load(xas);
}
if (((unsafe { (*xas).xa_offset }) as i32) as u32) != get_offset(unsafe { (*xas).xa_index }, unsafe { (*xas).xa_node }) {
unsafe {
(*xas).xa_offset = ((*xas).xa_offset as i32).wrapping_sub(1) as u8;
}
}
while ((unsafe { (*xas).xa_offset }) as i32) == 255 {
unsafe {
(*xas).xa_offset = ((*(*xas).xa_node).offset as i32).wrapping_sub(1) as u8;
}
unsafe {
(*xas).xa_node = xa_parent((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node);
}
if unsafe { (*xas).xa_node }.is_null() {
return set_bounds(xas);
}
}
loop {
let entry: *mut () = xa_entry((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, (unsafe { (*xas).xa_offset }) as u32);
if !xa_is_node(entry as *const ()) {
return entry;
}
unsafe {
(*xas).xa_node = xa_to_node(entry as *const ());
}
xas_set_offset(xas);
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xas_prev_285: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state) -> *mut ()>, *mut ()>(Some(__xas_prev)) };

/// __xas_next() - Find the next entry in the XArray.
/// @xas: XArray operation state.
///
/// Helper function for xas_next() which handles all the complex cases
/// out of line.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xas_next")]
extern "C" fn __xas_next(xas: *mut xa_state) -> *mut () {
if !xas_frozen(unsafe { (*xas).xa_node }) {
let __v1: u64 = unsafe { (*xas).xa_index };
unsafe {
(*xas).xa_index = __v1.wrapping_add(1);
}
}
if unsafe { (*xas).xa_node }.is_null() {
return set_bounds(xas);
}
if xas_not_node(unsafe { (*xas).xa_node }) {
return xas_load(xas);
}
if (((unsafe { (*xas).xa_offset }) as i32) as u32) != get_offset(unsafe { (*xas).xa_index }, unsafe { (*xas).xa_node }) {
unsafe {
(*xas).xa_offset = ((*xas).xa_offset as i32).wrapping_add(1) as u8;
}
}
while (((unsafe { (*xas).xa_offset }) as i32) as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
unsafe {
(*xas).xa_offset = ((*(*xas).xa_node).offset as i32).wrapping_add(1) as u8;
}
unsafe {
(*xas).xa_node = xa_parent((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node);
}
if unsafe { (*xas).xa_node }.is_null() {
return set_bounds(xas);
}
}
loop {
let entry: *mut () = xa_entry((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, (unsafe { (*xas).xa_offset }) as u32);
if !xa_is_node(entry as *const ()) {
return entry;
}
unsafe {
(*xas).xa_node = xa_to_node(entry as *const ());
}
xas_set_offset(xas);
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xas_next_286: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state) -> *mut ()>, *mut ()>(Some(__xas_next)) };

/// xas_find() - Find the next present entry in the XArray.
/// @xas: XArray operation state.
/// @max: Highest index to return.
///
/// If the @xas has not yet been walked to an entry, return the entry
/// which has an index >= xas.xa_index.  If it has been walked, the entry
/// currently being pointed at has been processed, and so we move to the
/// next entry.
///
/// If no entry is found and the array is smaller than @max, the iterator
/// is set to the smallest index not yet in the array.  This allows @xas
/// to be immediately passed to xas_store().
///
/// Return: The entry, if found, otherwise %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_find")]
extern "C" fn xas_find(xas: *mut xa_state, max: u64) -> *mut () {
let mut entry: *mut () = unsafe { core::mem::zeroed() };
if xas_error(xas as *const xa_state) != 0 || (unsafe { (*xas).xa_node }) == (1u64 as *mut xa_node) {
return core::ptr::null_mut::<()>();
}
if (unsafe { (*xas).xa_index }) > max {
return set_bounds(xas);
}
if unsafe { (*xas).xa_node }.is_null() {
unsafe {
(*xas).xa_index = 1;
}
return set_bounds(xas);
} else {
if (unsafe { (*xas).xa_node }) == (3u64 as *mut xa_node) {
entry = xas_load(xas);
let __v0: bool = !entry.is_null() || xas_not_node(unsafe { (*xas).xa_node });
if __v0 {
return entry;
}
} else {
if (unsafe { (*(*xas).xa_node).shift }) == 0 && (((unsafe { (*xas).xa_offset }) as i32) as u64) != (unsafe { (*xas).xa_index }) & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
unsafe {
(*xas).xa_offset = (unsafe { (*xas).xa_index }.wrapping_sub(1) & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)).wrapping_add(1) as u8;
}
}
}
}
xas_next_offset(xas);
while !unsafe { (*xas).xa_node }.is_null() && (unsafe { (*xas).xa_index }) <= max {
if unlikely!((((unsafe { (*xas).xa_offset }) as i32) as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) != 0 {
unsafe {
(*xas).xa_offset = ((*(*xas).xa_node).offset as i32).wrapping_add(1) as u8;
}
unsafe {
(*xas).xa_node = xa_parent((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node);
}
} else {
entry = xa_entry((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, (unsafe { (*xas).xa_offset }) as u32);
if xa_is_node(entry as *const ()) {
unsafe {
(*xas).xa_node = xa_to_node(entry as *const ());
}
unsafe {
(*xas).xa_offset = 0;
}
} else {
let __v2: bool = !entry.is_null() && !xa_is_sibling(entry as *const ());
if __v2 {
return entry;
}
xas_next_offset(xas);
}
}
}
if unsafe { (*xas).xa_node }.is_null() {
unsafe {
(*xas).xa_node = 1u64 as *mut xa_node;
}
}
core::ptr::null_mut::<()>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_find_287: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, u64) -> *mut ()>, *mut ()>(Some(xas_find)) };

/// xas_find_marked() - Find the next marked entry in the XArray.
/// @xas: XArray operation state.
/// @max: Highest index to return.
/// @mark: Mark number to search for.
///
/// If the @xas has not yet been walked to an entry, return the marked entry
/// which has an index >= xas.xa_index.  If it has been walked, the entry
/// currently being pointed at has been processed, and so we return the
/// first marked entry with an index > xas.xa_index.
///
/// If no marked entry is found and the array is smaller than @max, @xas is
/// set to the bounds state and xas->xa_index is set to the smallest index
/// not yet in the array.  This allows @xas to be immediately passed to
/// xas_store().
///
/// If no entry is found before @max is reached, @xas is set to the restart
/// state.
///
/// Return: The entry, if found, otherwise %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_find_marked")]
extern "C" fn xas_find_marked(xas: *mut xa_state, max: u64, mark: u32) -> *mut () {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut offset: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(offset) as *mut u32;
let mut advance: bool = unsafe { core::mem::zeroed() };
unsafe {
advance = true;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if xas_error(xas as *const xa_state) != 0 {
return core::ptr::null_mut::<()>();
} else {
'__join_0: {
if (*xas).xa_index <= max {
'__join_2: {
if unsafe { (*xas).xa_node }.is_null() {
(*xas).xa_index = 1;
} else {
if xas_top((*xas).xa_node) {
advance = false;
*__slate_slot_1 = xa_head((*xas).xa as *const xarray);
(*xas).xa_node = core::ptr::null_mut::<xa_node>();
if (*xas).xa_index > max_index(*__slate_slot_1) {
break '__join_2;
} else {
if !xa_is_node(*__slate_slot_1 as *const ()) {
if xa_marked((*xas).xa as *const xarray, mark) {
return *__slate_slot_1;
} else {
(*xas).xa_index = 1;
break '__join_2;
}
} else {
(*xas).xa_node = xa_to_node(*__slate_slot_1 as *const ());
(*xas).xa_offset = ((*xas).xa_index >> ((*(*xas).xa_node).shift as i32)) as u8;
}
}
}
loop {
if (*xas).xa_index <= max {
if unlikely!((((*xas).xa_offset as i32) as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) != 0 {
(*xas).xa_offset = ((*(*xas).xa_node).offset as i32).wrapping_add(1) as u8;
(*xas).xa_node = xa_parent((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node);
if unsafe { (*xas).xa_node }.is_null() {
break '__join_2;
} else {
advance = false;
}
} else {
if !advance {
*__slate_slot_1 = xa_entry((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node, (*xas).xa_offset as u32);
if xa_is_sibling(*__slate_slot_1 as *const ()) {
(*xas).xa_offset = xa_to_sibling(*__slate_slot_1 as *const ()) as u8;
xas_move_index(xas, (*xas).xa_offset as u64);
}
}
*__slate_slot_2 = xas_find_chunk(xas, advance, mark);
if *__slate_slot_2 > (((*xas).xa_offset as i32) as u32) {
advance = false;
xas_move_index(xas, *__slate_slot_2 as u64);
// Mind the wrap
if unsafe { (*xas).xa_index }.wrapping_sub(1) >= max {
break '__join_0;
} else {
(*xas).xa_offset = *__slate_slot_2 as u8;
if (*__slate_slot_2 as u64) == 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
continue;
}
}
}
*__slate_slot_1 = xa_entry((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node, (*xas).xa_offset as u32);
if (*__slate_slot_1).is_null() {
__v0 = !(xa_track_free((*xas).xa as *const xarray) && mark == 0);
} else {
__v0 = false;
}
if !__v0 {
if !xa_is_sibling(*__slate_slot_1 as *const ()) {
if !xa_is_node(*__slate_slot_1 as *const ()) {
break;
} else {
(*xas).xa_node = xa_to_node(*__slate_slot_1 as *const ());
xas_set_offset(xas);
}
}
}
}
} else {
break '__join_2;
}
}
return *__slate_slot_1;
}
}
if (*xas).xa_index <= max {
return set_bounds(xas);
}
}
}
(*xas).xa_node = 3u64 as *mut xa_node;
return core::ptr::null_mut::<()>();
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_find_marked_288: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state, u64, u32) -> *mut ()>, *mut ()>(Some(xas_find_marked)) };

/// xas_find_conflict() - Find the next present entry in a range.
/// @xas: XArray operation state.
///
/// The @xas describes both a range and a position within that range.
///
/// Context: Any context.  Expects xa_lock to be held.
/// Return: The next entry in the range covered by @xas or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_find_conflict")]
extern "C" fn xas_find_conflict(xas: *mut xa_state) -> *mut () {
let mut curr: *mut () = unsafe { core::mem::zeroed() };
if xas_error(xas as *const xa_state) != 0 {
return core::ptr::null_mut::<()>();
}
if unsafe { (*xas).xa_node }.is_null() {
return core::ptr::null_mut::<()>();
}
if xas_top(unsafe { (*xas).xa_node }) {
curr = xas_start(xas);
if curr.is_null() {
return core::ptr::null_mut::<()>();
}
while xa_is_node(curr as *const ()) {
let node: *mut xa_node = xa_to_node(curr as *const ());
curr = xas_descend(xas, node);
}
if !curr.is_null() {
return curr;
}
}
if ((unsafe { (*(*xas).xa_node).shift }) as i32) > ((unsafe { (*xas).xa_shift }) as i32) {
return core::ptr::null_mut::<()>();
}
'__slate_break_1: loop {
'__slate_continue_1: {
if ((unsafe { (*(*xas).xa_node).shift }) as i32) == ((unsafe { (*xas).xa_shift }) as i32) {
if ((unsafe { (*xas).xa_offset }) as i32) & ((unsafe { (*xas).xa_sibs }) as i32) == ((unsafe { (*xas).xa_sibs }) as i32) {
break '__slate_break_1;
}
} else {
if (((unsafe { (*xas).xa_offset }) as i32) as u64) == (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
unsafe {
(*xas).xa_offset = (*(*xas).xa_node).offset;
}
unsafe {
(*xas).xa_node = xa_parent_locked((*xas).xa as *const xarray, (*xas).xa_node as *const xa_node);
}
if unsafe { (*xas).xa_node }.is_null() {
break '__slate_break_1;
}
break '__slate_continue_1;
}
}
let __v4: u8 = ((unsafe { (*xas).xa_offset }) as i32).wrapping_add(1) as u8;
unsafe {
(*xas).xa_offset = __v4;
}
curr = xa_entry_locked((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, __v4 as u32);
if !xa_is_sibling(curr as *const ()) {
while xa_is_node(curr as *const ()) {
unsafe {
(*xas).xa_node = xa_to_node(curr as *const ());
}
unsafe {
(*xas).xa_offset = 0;
}
curr = xa_entry_locked((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, 0u32);
}
if !curr.is_null() {
return curr;
}
}
}
}
unsafe {
(*xas).xa_offset = ((*xas).xa_offset as i32).wrapping_sub((*xas).xa_sibs as i32) as u8;
}
core::ptr::null_mut::<()>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_find_conflict_289: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state) -> *mut ()>, *mut ()>(Some(xas_find_conflict)) };

/// xa_load() - Load an entry from an XArray.
/// @xa: XArray.
/// @index: index into array.
///
/// Context: Any context.  Takes and releases the RCU lock.
/// Return: The entry at @index in @xa.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_load")]
extern "C" fn xa_load(xa: *mut xarray, index: u64) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut entry: *mut () = unsafe { core::mem::zeroed() };
rcu_read_lock();
loop {
entry = xa_zero_to_null(xas_load(core::ptr::addr_of_mut!(xas)));
if !xas_retry(core::ptr::addr_of_mut!(xas), entry as *const ()) {
break;
}
}
rcu_read_unlock();
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_load_290: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64) -> *mut ()>, *mut ()>(Some(xa_load)) };

fn xas_result(xas: *mut xa_state, mut curr: *mut ()) -> *mut () {
if xas_error(xas as *const xa_state) != 0 {
curr = (unsafe { (*xas).xa_node }) as *mut ();
}
curr
}

/// __xa_erase() - Erase this entry from the XArray while locked.
/// @xa: XArray.
/// @index: Index into array.
///
/// After this function returns, loading from @index will return %NULL.
/// If the index is part of a multi-index entry, all indices will be erased
/// and none of the entries will be part of a multi-index entry.
///
/// Context: Any context.  Expects xa_lock to be held on entry.
/// Return: The entry which used to be at this index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_erase")]
extern "C" fn __xa_erase(xa: *mut xarray, index: u64) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
xas_result(core::ptr::addr_of_mut!(xas), xa_zero_to_null(xas_store(core::ptr::addr_of_mut!(xas), core::ptr::null_mut::<()>())))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_erase_291: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64) -> *mut ()>, *mut ()>(Some(__xa_erase)) };

/// xa_erase() - Erase this entry from the XArray.
/// @xa: XArray.
/// @index: Index of entry.
///
/// After this function returns, loading from @index will return %NULL.
/// If the index is part of a multi-index entry, all indices will be erased
/// and none of the entries will be part of a multi-index entry.
///
/// Context: Any context.  Takes and releases the xa_lock.
/// Return: The entry which used to be at this index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_erase")]
extern "C" fn xa_erase(xa: *mut xarray, index: u64) -> *mut () {
xa_lock!(xa);
let entry: *mut () = __xa_erase(xa, index);
xa_unlock!(xa);
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_erase_292: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64) -> *mut ()>, *mut ()>(Some(xa_erase)) };

/// __xa_store() - Store this entry in the XArray.
/// @xa: XArray.
/// @index: Index into array.
/// @entry: New entry.
/// @gfp: Memory allocation flags.
///
/// You must already be holding the xa_lock when calling this function.
/// It will drop the lock if needed to allocate memory, and then reacquire
/// it afterwards.
///
/// Context: Any context.  Expects xa_lock to be held on entry.  May
/// release and reacquire xa_lock if @gfp flags permit.
/// Return: The old entry at this index or xa_err() if an error happened.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_store")]
extern "C" fn __xa_store(xa: *mut xarray, index: u64, mut entry: *mut (), gfp: u32) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut curr: *mut () = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = !(!xa_is_advanced(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("293: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 293b - ., 3; .popsection\n# {0}", const 293 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___xa_store_1, sym __slate_asm_string___xa_store_2, const 1700 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("294: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 294b - ., 4; .popsection\n# {0}", const 294 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return XA_ERROR!(22i32.wrapping_neg()) as *mut ();
}
if xa_track_free(xa as *const xarray) && entry.is_null() {
entry = xa_mk_internal(257u64);
}
loop {
curr = xas_store(core::ptr::addr_of_mut!(xas), entry);
if xa_track_free(xa as *const xarray) {
xas_clear_mark(core::ptr::addr_of_mut!(xas) as *const xa_state, 0u32);
}
if !__xas_nomem(core::ptr::addr_of_mut!(xas), gfp) {
break;
}
}
xas_result(core::ptr::addr_of_mut!(xas), xa_zero_to_null(curr))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_store_295: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, *mut (), u32) -> *mut ()>, *mut ()>(Some(__xa_store)) };

/// xa_store() - Store this entry in the XArray.
/// @xa: XArray.
/// @index: Index into array.
/// @entry: New entry.
/// @gfp: Memory allocation flags.
///
/// After this function returns, loads from this index will return @entry.
/// Storing into an existing multi-index entry updates the entry of every index.
/// The marks associated with @index are unaffected unless @entry is %NULL.
///
/// Context: Any context.  Takes and releases the xa_lock.
/// May sleep if the @gfp flags permit.
/// Return: The old entry at this index on success, xa_err(-EINVAL) if @entry
/// cannot be stored in an XArray, or xa_err(-ENOMEM) if memory allocation
/// failed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_store")]
extern "C" fn xa_store(xa: *mut xarray, index: u64, entry: *mut (), gfp: u32) -> *mut () {
xa_lock!(xa);
let curr: *mut () = __xa_store(xa, index, entry, gfp);
xa_unlock!(xa);
curr
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_store_296: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, *mut (), u32) -> *mut ()>, *mut ()>(Some(xa_store)) };

/// __xa_cmpxchg() - Conditionally replace an entry in the XArray.
/// @xa: XArray.
/// @index: Index into array.
/// @old: Old value to test against.
/// @entry: New value to place in array.
/// @gfp: Memory allocation flags.
///
/// You must already be holding the xa_lock when calling this function.
/// It will drop the lock if needed to allocate memory, and then reacquire
/// it afterwards.
///
/// If the entry at @index is the same as @old, replace it with @entry.
/// If the return value is equal to @old, then the exchange was successful.
///
/// Context: Any context.  Expects xa_lock to be held on entry.  May
/// release and reacquire xa_lock if @gfp flags permit.
/// Return: The old value at this index or xa_err() if an error happened.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_cmpxchg")]
extern "C" fn __xa_cmpxchg(xa: *mut xarray, index: u64, old: *mut (), entry: *mut (), gfp: u32) -> *mut () {
xa_zero_to_null(__xa_cmpxchg_raw(xa, index, old, entry, gfp))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_cmpxchg_297: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, *mut (), *mut (), u32) -> *mut ()>, *mut ()>(Some(__xa_cmpxchg)) };

fn __xa_cmpxchg_raw(xa: *mut xarray, index: u64, old: *mut (), entry: *mut (), gfp: u32) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut curr: *mut () = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = !(!xa_is_advanced(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("298: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 298b - ., 3; .popsection\n# {0}", const 298 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___xa_cmpxchg_raw_1, sym __slate_asm_string___xa_cmpxchg_raw_2, const 1779 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("299: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 299b - ., 4; .popsection\n# {0}", const 299 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return XA_ERROR!(22i32.wrapping_neg()) as *mut ();
}
loop {
curr = xas_load(core::ptr::addr_of_mut!(xas));
if curr == old {
xas_store(core::ptr::addr_of_mut!(xas), entry);
if xa_track_free(xa as *const xarray) && !entry.is_null() && curr.is_null() {
xas_clear_mark(core::ptr::addr_of_mut!(xas) as *const xa_state, 0u32);
}
}
if !__xas_nomem(core::ptr::addr_of_mut!(xas), gfp) {
break;
}
}
xas_result(core::ptr::addr_of_mut!(xas), curr)
}

/// __xa_insert() - Store this entry in the XArray if no entry is present.
/// @xa: XArray.
/// @index: Index into array.
/// @entry: New entry.
/// @gfp: Memory allocation flags.
///
/// Inserting a NULL entry will store a reserved entry (like xa_reserve())
/// if no entry is present.  Inserting will fail if a reserved entry is
/// present, even though loading from this index will return NULL.
///
/// Context: Any context.  Expects xa_lock to be held on entry.  May
/// release and reacquire xa_lock if @gfp flags permit.
/// Return: 0 if the store succeeded.  -EBUSY if another entry was present.
/// -ENOMEM if memory could not be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_insert")]
extern "C" fn __xa_insert(xa: *mut xarray, index: u64, mut entry: *mut (), gfp: u32) -> i32 {
let mut errno: i32 = 0;
if entry.is_null() {
entry = xa_mk_internal(257u64);
}
let curr: *mut () = __xa_cmpxchg_raw(xa, index, core::ptr::null_mut::<()>(), entry, gfp);
errno = xa_err(curr);
if errno != 0 {
return errno;
}
if !curr.is_null() { 16i32.wrapping_neg() } else { 0 }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_insert_300: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, *mut (), u32) -> i32>, *mut ()>(Some(__xa_insert)) };

fn xas_set_range(xas: &mut xa_state, mut first: u64, last: u64) {
let mut shift: u32 = 0;
let mut sibs: u64 = last.wrapping_sub(first);
let mut offset: u32 = (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) as u32;
xas_set(xas, first);
while first & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) == 0 {
if sibs < (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
break;
}
if sibs == (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) && (offset as u64) < (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
break;
}
let __v1: u32 = shift;
shift = __v1.wrapping_add((if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }) as u32);
if (offset as u64) == (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
offset = (sibs & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) as u32;
}
let __v3: u64 = sibs;
sibs = __v3 >> if IS_ENABLED!() != 0 { 4i32 } else { 6i32 };
let __v5: u64 = first;
first = __v5 >> if IS_ENABLED!() != 0 { 4i32 } else { 6i32 };
}
offset = (first & (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1)) as u32;
if (offset as u64).wrapping_add(sibs) > (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1) {
sibs = (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 }).wrapping_sub(1).wrapping_sub(offset as u64);
}
if (first.wrapping_add(sibs).wrapping_add(1) << shift).wrapping_sub(1) > last {
sibs = sibs.wrapping_sub(1);
}
unsafe {
(*(xas as *mut xa_state)).xa_shift = shift as u8;
}
unsafe {
(*(xas as *mut xa_state)).xa_sibs = sibs as u8;
}
}

/// xa_store_range() - Store this entry at a range of indices in the XArray.
/// @xa: XArray.
/// @first: First index to affect.
/// @last: Last index to affect.
/// @entry: New entry.
/// @gfp: Memory allocation flags.
///
/// After this function returns, loads from any index between @first and @last,
/// inclusive will return @entry.
/// Storing into an existing multi-index entry updates the entry of every index.
/// The marks associated with @index are unaffected unless @entry is %NULL.
///
/// Context: Process context.  Takes and releases the xa_lock.  May sleep
/// if the @gfp flags permit.
/// Return: %NULL on success, xa_err(-EINVAL) if @entry cannot be stored in
/// an XArray, or xa_err(-ENOMEM) if memory allocation failed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_store_range")]
extern "C" fn xa_store_range(xa: *mut xarray, mut first: u64, last: u64, entry: *mut (), gfp: u32) -> *mut () {
let mut order: u32 = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut xas: core::mem::MaybeUninit<xa_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut xa_state = core::ptr::addr_of_mut!(xas) as *mut xa_state;
unsafe {
core::ptr::write(__slate_slot_7, {
let mut __t0: xa_state = core::mem::zeroed();
__t0.xa = xa;
__t0.xa_index = 0u64;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
});
__ret_warn_on = !(!xa_is_internal(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("301: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 301b - ., 3; .popsection\n# {0}", const 301 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xa_store_range_8, sym __slate_asm_string_xa_store_range_9, const 1880 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("302: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 302b - ., 4; .popsection\n# {0}", const 302 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v6: i64 = (__ret_warn_on != 0) as i64;
if __v6 != 0 {
return XA_ERROR!(22i32.wrapping_neg()) as *mut ();
} else {
if last < first {
return XA_ERROR!(22i32.wrapping_neg()) as *mut ();
} else {
loop {
'__join_1: {
xas_lock!(core::ptr::addr_of_mut!(*__slate_slot_7));
if !entry.is_null() {
order = 64;
if last.wrapping_add(1) != 0 {
order = __ffs!(last.wrapping_add(1)) as u32;
}
xas_set_order(&mut (*core::ptr::addr_of_mut!(*__slate_slot_7)), last, order);
xas_create(core::ptr::addr_of_mut!(*__slate_slot_7), true);
if xas_error(core::ptr::addr_of_mut!(*__slate_slot_7) as *const xa_state) != 0 {
break '__join_1;
}
}
loop {
xas_set_range(&mut (*core::ptr::addr_of_mut!(*__slate_slot_7)), first, last);
xas_store(core::ptr::addr_of_mut!(*__slate_slot_7), entry);
if xas_error(core::ptr::addr_of_mut!(*__slate_slot_7) as *const xa_state) != 0 {
break;
} else {
let __v2: u64 = xas_size(core::ptr::addr_of_mut!(*__slate_slot_7) as *const xa_state);
first = first.wrapping_add(__v2);
if first > last {
break;
}
}
}
}
xas_unlock!(core::ptr::addr_of_mut!(*__slate_slot_7));
if !xas_nomem(core::ptr::addr_of_mut!(*__slate_slot_7), gfp) {
break;
}
}
return xas_result(core::ptr::addr_of_mut!(*__slate_slot_7), core::ptr::null_mut::<()>());
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_store_range_303: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u64, *mut (), u32) -> *mut ()>, *mut ()>(Some(xa_store_range)) };

/// xas_get_order() - Get the order of an entry.
/// @xas: XArray operation state.
///
/// Called after xas_load, the xas should not be in an error state.
/// The xas should not be pointing to a sibling entry.
///
/// Return: A number between 0 and 63 indicating the order of the entry.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xas_get_order")]
extern "C" fn xas_get_order(xas: *mut xa_state) -> i32 {
let mut order: i32 = 0;
if unsafe { (*xas).xa_node }.is_null() {
return 0;
}
XA_NODE_BUG_ON!();
loop {
let slot: u32 = ((unsafe { (*xas).xa_offset }) as i32).wrapping_add(1i32 << order) as u32;
if (slot as u64) >= 1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } {
break;
}
if !xa_is_sibling(xa_entry((unsafe { (*xas).xa }) as *const xarray, (unsafe { (*xas).xa_node }) as *const xa_node, slot) as *const ()) {
break;
}
order = order.wrapping_add(1);
}
order = order.wrapping_add((unsafe { (*(*xas).xa_node).shift }) as i32);
order
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xas_get_order_304: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_state) -> i32>, *mut ()>(Some(xas_get_order)) };

/// xa_get_order() - Get the order of an entry.
/// @xa: XArray.
/// @index: Index of the entry.
///
/// Return: A number between 0 and 63 indicating the order of the entry.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_get_order")]
extern "C" fn xa_get_order(xa: *mut xarray, index: u64) -> i32 {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut order: i32 = 0;
rcu_read_lock();
let entry: *mut () = xas_load(core::ptr::addr_of_mut!(xas));
if !entry.is_null() {
order = xas_get_order(core::ptr::addr_of_mut!(xas));
}
rcu_read_unlock();
order
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_get_order_305: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64) -> i32>, *mut ()>(Some(xa_get_order)) };

/// __xa_alloc() - Find somewhere to store this entry in the XArray.
/// @xa: XArray.
/// @id: Pointer to ID.
/// @limit: Range for allocated ID.
/// @entry: New entry.
/// @gfp: Memory allocation flags.
///
/// Finds an empty entry in @xa between @limit.min and @limit.max,
/// stores the index into the @id pointer, then stores the entry at
/// that index.  A concurrent lookup will not see an uninitialised @id.
///
/// Must only be operated on an xarray initialized with flag XA_FLAGS_ALLOC set
/// in xa_init_flags().
///
/// Context: Any context.  Expects xa_lock to be held on entry.  May
/// release and reacquire xa_lock if @gfp flags permit.
/// Return: 0 on success, -ENOMEM if memory could not be allocated or
/// -EBUSY if there are no free entries in @limit.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_alloc")]
extern "C" fn __xa_alloc(xa: *mut xarray, id: *mut u32, mut entry: *mut (), limit: xa_limit, gfp: u32) -> i32 {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = 0u64;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!xa_is_advanced(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("306: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 306b - ., 3; .popsection\n# {0}", const 306 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___xa_alloc_1, sym __slate_asm_string___xa_alloc_2, const 1993 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("307: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 307b - ., 4; .popsection\n# {0}", const 307 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return 22i32.wrapping_neg();
}
let mut __v5: i64;
let __ret_warn_on: i32 = !(!(!xa_track_free(xa as *const xarray))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("308: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 308b - ., 3; .popsection\n# {0}", const 308 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___xa_alloc_6, sym __slate_asm_string___xa_alloc_7, const 1995 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("309: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 309b - ., 4; .popsection\n# {0}", const 309 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return 22i32.wrapping_neg();
}
if entry.is_null() {
entry = xa_mk_internal(257u64);
}
loop {
xas.xa_index = limit.min as u64;
xas_find_marked(core::ptr::addr_of_mut!(xas), limit.max as u64, 0u32);
if xas.xa_node == (3u64 as *mut xa_node) {
xas_set_err(core::ptr::addr_of_mut!(xas), 16i32.wrapping_neg() as i64);
} else {
unsafe {
*id = xas.xa_index as u32;
}
}
xas_store(core::ptr::addr_of_mut!(xas), entry);
xas_clear_mark(core::ptr::addr_of_mut!(xas) as *const xa_state, 0u32);
if !__xas_nomem(core::ptr::addr_of_mut!(xas), gfp) {
break;
}
}
xas_error(core::ptr::addr_of_mut!(xas) as *const xa_state)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_alloc_310: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, *mut u32, *mut (), xa_limit, u32) -> i32>, *mut ()>(Some(__xa_alloc)) };

// __xa_alloc_cyclic() - Find somewhere to store this entry in the XArray.
// @xa: XArray.
// @id: Pointer to ID.
// @entry: New entry.
// @limit: Range of allocated ID.
// @next: Pointer to next ID to allocate.
// @gfp: Memory allocation flags.
//
// Finds an empty entry in @xa between @limit.min and @limit.max,
// stores the index into the @id pointer, then stores the entry at
// that index.  A concurrent lookup will not see an uninitialised @id.
// The search for an empty entry will start at @next and will wrap
// around if necessary.
//
// Must only be operated on an xarray initialized with flag XA_FLAGS_ALLOC set
// in xa_init_flags().
//
// Context: Any context.  Expects xa_lock to be held on entry.  May
// release and reacquire xa_lock if @gfp flags permit.
// Return: 0 if the allocation succeeded without wrapping.  1 if the
// allocation succeeded after wrapping, -ENOMEM if memory could not be
// allocated or -EBUSY if there are no free entries in @limit.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_alloc_cyclic_314: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, *mut u32, *mut (), xa_limit, *mut u32, u32) -> i32>, *mut ()>(Some(__xa_alloc_cyclic)) };

/// __xa_set_mark() - Set this mark on this entry while locked.
/// @xa: XArray.
/// @index: Index of entry.
/// @mark: Mark number.
///
/// Attempting to set a mark on a %NULL entry does not succeed.
///
/// Context: Any context.  Expects xa_lock to be held on entry.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_set_mark")]
extern "C" fn __xa_set_mark(xa: *mut xarray, index: u64, mark: u32) {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let entry: *mut () = xas_load(core::ptr::addr_of_mut!(xas));
if !entry.is_null() {
xas_set_mark(core::ptr::addr_of_mut!(xas) as *const xa_state, mark);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_set_mark_315: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u32)>, *mut ()>(Some(__xa_set_mark)) };

/// __xa_clear_mark() - Clear this mark on this entry while locked.
/// @xa: XArray.
/// @index: Index of entry.
/// @mark: Mark number.
///
/// Context: Any context.  Expects xa_lock to be held on entry.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.__xa_clear_mark")]
extern "C" fn __xa_clear_mark(xa: *mut xarray, index: u64, mark: u32) {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let entry: *mut () = xas_load(core::ptr::addr_of_mut!(xas));
if !entry.is_null() {
xas_clear_mark(core::ptr::addr_of_mut!(xas) as *const xa_state, mark);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___xa_clear_mark_316: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u32)>, *mut ()>(Some(__xa_clear_mark)) };

/// xa_get_mark() - Inquire whether this mark is set on this entry.
/// @xa: XArray.
/// @index: Index of entry.
/// @mark: Mark number.
///
/// This function uses the RCU read lock, so the result may be out of date
/// by the time it returns.  If you need the result to be stable, use a lock.
///
/// Context: Any context.  Takes and releases the RCU lock.
/// Return: True if the entry at @index has this mark set, false if it doesn't.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_get_mark")]
extern "C" fn xa_get_mark(xa: *mut xarray, index: u64, mark: u32) -> bool {
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut xas: core::mem::MaybeUninit<xa_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut xa_state = core::ptr::addr_of_mut!(xas) as *mut xa_state;
unsafe {
core::ptr::write(__slate_slot_1, {
let mut __t0: xa_state = core::mem::zeroed();
__t0.xa = xa;
__t0.xa_index = index;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
});
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
*__slate_slot_0 = xas_start(core::ptr::addr_of_mut!(*__slate_slot_1));
while xas_get_mark(core::ptr::addr_of_mut!(*__slate_slot_1) as *const xa_state, mark) {
if !xa_is_node(*__slate_slot_0 as *const ()) {
rcu_read_unlock();
return true;
} else {
*__slate_slot_0 = xas_descend(core::ptr::addr_of_mut!(*__slate_slot_1), xa_to_node(*__slate_slot_0 as *const ()));
}
}
rcu_read_unlock();
return false;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_get_mark_317: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u32) -> bool>, *mut ()>(Some(xa_get_mark)) };

/// xa_set_mark() - Set this mark on this entry.
/// @xa: XArray.
/// @index: Index of entry.
/// @mark: Mark number.
///
/// Attempting to set a mark on a %NULL entry does not succeed.
///
/// Context: Process context.  Takes and releases the xa_lock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_set_mark")]
extern "C" fn xa_set_mark(xa: *mut xarray, index: u64, mark: u32) {
xa_lock!(xa);
__xa_set_mark(xa, index, mark);
xa_unlock!(xa);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_set_mark_318: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u32)>, *mut ()>(Some(xa_set_mark)) };

/// xa_clear_mark() - Clear this mark on this entry.
/// @xa: XArray.
/// @index: Index of entry.
/// @mark: Mark number.
///
/// Clearing a mark always succeeds.
///
/// Context: Process context.  Takes and releases the xa_lock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_clear_mark")]
extern "C" fn xa_clear_mark(xa: *mut xarray, index: u64, mark: u32) {
xa_lock!(xa);
__xa_clear_mark(xa, index, mark);
xa_unlock!(xa);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_clear_mark_319: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, u64, u32)>, *mut ()>(Some(xa_clear_mark)) };

/// xa_find() - Search the XArray for an entry.
/// @xa: XArray.
/// @indexp: Pointer to an index.
/// @max: Maximum index to search to.
/// @filter: Selection criterion.
///
/// Finds the entry in @xa which matches the @filter, and has the lowest
/// index that is at least @indexp and no more than @max.
/// If an entry is found, @indexp is updated to be the index of the entry.
/// This function is protected by the RCU read lock, so it may not find
/// entries which are being simultaneously added.  It will not return an
/// %XA_RETRY_ENTRY; if you need to see retry entries, use xas_find().
///
/// Context: Any context.  Takes and releases the RCU lock.
/// Return: The entry, if found, otherwise %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_find")]
extern "C" fn xa_find(xa: *mut xarray, indexp: *mut u64, max: u64, filter: u32) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = unsafe { *indexp };
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut entry: *mut () = unsafe { core::mem::zeroed() };
rcu_read_lock();
loop {
if filter < 3 {
entry = xas_find_marked(core::ptr::addr_of_mut!(xas), max, filter);
} else {
entry = xas_find(core::ptr::addr_of_mut!(xas), max);
}
if !xas_retry(core::ptr::addr_of_mut!(xas), entry as *const ()) {
break;
}
}
rcu_read_unlock();
if !entry.is_null() {
unsafe {
*indexp = xas.xa_index;
}
}
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_find_320: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, *mut u64, u64, u32) -> *mut ()>, *mut ()>(Some(xa_find)) };

fn xas_sibling(xas: &xa_state) -> bool {
let node: *mut xa_node = unsafe { (*(xas as *const xa_state)).xa_node };
let mut mask: u64 = 0;
if IS_ENABLED__2!() == 0 || node.is_null() {
return false;
}
mask = (1u64 << if IS_ENABLED!() != 0 { 4i32 } else { 6i32 } << ((unsafe { (*node).shift }) as i32)).wrapping_sub(1);
(unsafe { (*(xas as *const xa_state)).xa_index }) & mask > ((unsafe { (*(xas as *const xa_state)).xa_offset }) as u64) << ((unsafe { (*node).shift }) as i32)
}

/// xa_find_after() - Search the XArray for a present entry.
/// @xa: XArray.
/// @indexp: Pointer to an index.
/// @max: Maximum index to search to.
/// @filter: Selection criterion.
///
/// Finds the entry in @xa which matches the @filter and has the lowest
/// index that is above @indexp and no more than @max.
/// If an entry is found, @indexp is updated to be the index of the entry.
/// This function is protected by the RCU read lock, so it may miss entries
/// which are being simultaneously added.  It will not return an
/// %XA_RETRY_ENTRY; if you need to see retry entries, use xas_find().
///
/// Context: Any context.  Takes and releases the RCU lock.
/// Return: The pointer, if found, otherwise %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_find_after")]
extern "C" fn xa_find_after(xa: *mut xarray, indexp: *mut u64, max: u64, filter: u32) -> *mut () {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = unsafe { *indexp }.wrapping_add(1);
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
let mut entry: *mut () = unsafe { core::mem::zeroed() };
if xas.xa_index == 0 {
return core::ptr::null_mut::<()>();
}
rcu_read_lock();
loop {
if filter < 3 {
entry = xas_find_marked(core::ptr::addr_of_mut!(xas), max, filter);
} else {
entry = xas_find(core::ptr::addr_of_mut!(xas), max);
}
if xas_invalid(core::ptr::addr_of_mut!(xas) as *const xa_state) {
break;
}
if !xas_sibling(unsafe { &(*core::ptr::addr_of_mut!(xas)) }) {
if !xas_retry(core::ptr::addr_of_mut!(xas), entry as *const ()) {
break;
}
}
}
rcu_read_unlock();
if !entry.is_null() {
unsafe {
*indexp = xas.xa_index;
}
}
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_find_after_321: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, *mut u64, u64, u32) -> *mut ()>, *mut ()>(Some(xa_find_after)) };

fn xas_extract_present(xas: *mut xa_state, dst: *mut *mut (), max: u64, n: u32) -> u32 {
let mut entry: *mut () = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
rcu_read_lock();
entry = xas_find(xas, max);
while !entry.is_null() {
if !xas_retry(xas, entry as *const ()) {
let __v1: u32 = i;
i = __v1.wrapping_add(1);
unsafe {
*dst.wrapping_offset(__v1 as isize) = entry;
}
if i == n {
break;
}
}
entry = xas_next_entry(xas, max);
}
rcu_read_unlock();
i
}

fn xas_extract_marked(xas: *mut xa_state, dst: *mut *mut (), max: u64, n: u32, mark: u32) -> u32 {
let mut entry: *mut () = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
rcu_read_lock();
entry = xas_find_marked(xas, max, mark);
while !entry.is_null() {
if !xas_retry(xas, entry as *const ()) {
let __v1: u32 = i;
i = __v1.wrapping_add(1);
unsafe {
*dst.wrapping_offset(__v1 as isize) = entry;
}
if i == n {
break;
}
}
entry = xas_next_marked(xas, max, mark);
}
rcu_read_unlock();
i
}

/// xa_extract() - Copy selected entries from the XArray into a normal array.
/// @xa: The source XArray to copy from.
/// @dst: The buffer to copy entries into.
/// @start: The first index in the XArray eligible to be selected.
/// @max: The last index in the XArray eligible to be selected.
/// @n: The maximum number of entries to copy.
/// @filter: Selection criterion.
///
/// Copies up to @n entries that match @filter from the XArray.  The
/// copied entries will have indices between @start and @max, inclusive.
///
/// The @filter may be an XArray mark value, in which case entries which are
/// marked with that mark will be copied.  It may also be %XA_PRESENT, in
/// which case all entries which are not %NULL will be copied.
///
/// The entries returned may not represent a snapshot of the XArray at a
/// moment in time.  For example, if another thread stores to index 5, then
/// index 10, calling xa_extract() may return the old contents of index 5
/// and the new contents of index 10.  Indices not modified while this
/// function is running will not be skipped.
///
/// If you need stronger guarantees, holding the xa_lock across calls to this
/// function will prevent concurrent modification.
///
/// Context: Any context.  Takes and releases the RCU lock.
/// Return: The number of entries copied.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_extract")]
extern "C" fn xa_extract(xa: *mut xarray, dst: *mut *mut (), start: u64, max: u64, n: u32, filter: u32) -> u32 {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = xa;
__t0.xa_index = start;
__t0.xa_shift = 0u8;
__t0.xa_sibs = 0u8;
__t0.xa_offset = 0u8;
__t0.xa_pad = 0u8;
__t0.xa_node = 3u64 as *mut xa_node;
__t0.xa_alloc = core::ptr::null_mut::<xa_node>();
__t0.xa_update = None;
__t0.xa_lru = core::ptr::null_mut::<list_lru>();
__t0
};
if n == 0 {
return 0;
}
if filter < 3 {
return xas_extract_marked(core::ptr::addr_of_mut!(xas), dst, max, n, filter);
}
xas_extract_present(core::ptr::addr_of_mut!(xas), dst, max, n)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_extract_322: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray, *mut *mut (), u64, u64, u32, u32) -> u32>, *mut ()>(Some(xa_extract)) };

/// xa_delete_node() - Private interface for workingset code.
/// @node: Node to be removed from the tree.
/// @update: Function to call to update ancestor nodes.
///
/// Context: xa_lock must be held on entry and will not be released.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xarray.xa_delete_node")]
extern "C" fn xa_delete_node(node: *mut xa_node, update: Option<unsafe extern "C" fn(*mut xa_node)>) {
let mut xas: xa_state = {
let mut __t0: xa_state = unsafe { core::mem::zeroed() };
__t0.xa = unsafe { (*node).array };
__t0.xa_index = ((unsafe { (*node).offset }) as u64) << ((unsafe { (*node).shift }) as i32).wrapping_add(if IS_ENABLED!() != 0 { 4 } else { 6 });
__t0.xa_shift = ((unsafe { (*node).shift }) as i32).wrapping_add(if IS_ENABLED!() != 0 { 4 } else { 6 }) as u8;
__t0.xa_offset = unsafe { (*node).offset };
__t0.xa_node = xa_parent_locked((unsafe { (*node).array }) as *const xarray, node as *const xa_node);
__t0.xa_update = update;
__t0
};
xas_store(core::ptr::addr_of_mut!(xas), core::ptr::null_mut::<()>());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_delete_node_323: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xa_node, Option<unsafe extern "C" fn(*mut xa_node)>)>, *mut ()>(Some(xa_delete_node)) };

// For the benefit of the test suite
// xa_destroy() - Free all internal data structures.
// @xa: XArray.
//
// After calling this function, the XArray is empty and has freed all memory
// allocated for its internal data structures.  You are responsible for
// freeing the objects referenced by the XArray.
//
// Context: Any context.  Takes and releases the xa_lock, interrupt-safe.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xa_destroy_325: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xarray)>, *mut ()>(Some(xa_destroy)) };

static __slate_asm_string___xa_alloc_1: [u8; 1] = *b"\0";

static __slate_asm_string___xa_alloc_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";

static __slate_asm_string___xa_alloc_6: [u8; 1] = *b"\0";

static __slate_asm_string___xa_alloc_7: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";

static __slate_asm_string___xa_cmpxchg_raw_1: [u8; 1] = *b"\0";

static __slate_asm_string___xa_cmpxchg_raw_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";

static __slate_asm_string___xa_store_1: [u8; 1] = *b"\0";

static __slate_asm_string___xa_store_2: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_xa_store_range_8: [u8; 1] = *b"\0";

static __slate_asm_string_xa_store_range_9: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";

static __slate_asm_string_xas_split_alloc_15: [u8; 1] = *b"\0";

static __slate_asm_string_xas_split_alloc_16: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/xarray.c\0";
