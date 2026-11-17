
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
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! Generic associative array implementation.
//!
//! See Documentation/core-api/assoc_array.rst for information.
//!
//! Copyright (C) 2013 Red Hat, Inc. All Rights Reserved.
//! Written by David Howells (dhowells@redhat.com)
//!
//! #define DEBUG
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! The type of an index into the pagecache.
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Internal kernel timer frequency
//! Bitfields in the atomic value:
//!
//! When NR_CPUS < 16K
//!  0- 7: locked byte
//!  8-15: pending byte
//! 16-17: tail index
//! 18-31: tail cpu (+1)
//!
//! When NR_CPUS >= 16K
//!  0- 7: locked byte
//!     8: pending
//!  9-10: tail index
//! 11-31: tail cpu (+1)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! Defines x86 CPU feature bits
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! This counts to 15. Any more, it will return 16th argument.
//! Atomic operations are already serializing on x86
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! No such file or directory
//! Value too large for defined data type
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
//! @a is a power of 2 value
//! const_*() definitions provide good compile-time optimizations when
//! the passed arguments can be resolved at compile time.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! restart by calling sys_restart_syscall
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! If !LOCKDEP, the value is meaningless
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
//! get_task_state():
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
//! Helper macro to avoid gfp flags if they are the default one
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
//! Key data retrieved in chunks of this size
//! Number of slots per node
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
// __acquire() - function to acquire context lock exclusively
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x exclusively.
// __release() - function to release context lock exclusively
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x.
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
// __acquire_shared() - function to acquire context lock shared
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x with shared
// access.
// __release_shared() - function to release context lock shared
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x with shared
// access.
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
// &a[0] degrades to a pointer: a different type from an array
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
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// Non-serializing WRMSR
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "div0" AMD DIV0 speculation bug
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// Try again
// Out of memory
// Device or resource busy
// Invalid argument
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// Operation is not supported
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// abs - return absolute value of an argument
// @x: the value.
//
// If it is unsigned type, @x is converted to signed type first.
// char is treated as if it was signed (regardless of whether it really is)
// but the macro's return type is preserved as char.
//
// NOTE, for signed type if @x is the minimum, the returned result is undefined
// as there is not enough bits to represent it as a positive number.
//
// Return: an absolute value of @x.
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
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
// I am an IDLE thread
// Kernel thread cloned from userspace thread
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
// Dependency order vs. p above.
// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
// @v: The value to statically initialize with.
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE {
() => {
((core::mem::size_of::<[*mut assoc_array_ptr; 1]>() as u64) / (core::mem::size_of::<*mut assoc_array_ptr>() as u64)).wrapping_add(__must_be_array__18!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__5 {
() => {
((core::mem::size_of::<[*mut *mut assoc_array_ptr; 16]>() as u64) / (core::mem::size_of::<*mut *mut assoc_array_ptr>() as u64)).wrapping_add(__must_be_array__25!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__12 {
() => {
((core::mem::size_of::<[*mut assoc_array_ptr; 3]>() as u64) / (core::mem::size_of::<*mut assoc_array_ptr>() as u64)).wrapping_add(__must_be_array__13!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__13 {
() => {
((core::mem::size_of::<[__SlateRecord_assoc_array_priv_h_89; 2]>() as u64) / (core::mem::size_of::<__SlateRecord_assoc_array_priv_h_89>() as u64)).wrapping_add(__must_be_array__10!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__16 {
() => {
((core::mem::size_of::<[__SlateRecord_assoc_array_priv_h_93; 1]>() as u64) / (core::mem::size_of::<__SlateRecord_assoc_array_priv_h_93>() as u64)).wrapping_add(__must_be_array__5!() as u64)
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
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

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! UL {
() => {
_UL!()
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

macro_rules! __must_be_array__3 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_745_4>() as i32
};
}

macro_rules! __must_be_array__4 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_914_2>() as i32
};
}

macro_rules! __must_be_array__5 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1358>() as i32
};
}

macro_rules! __must_be_array__9 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_901_4>() as i32
};
}

macro_rules! __must_be_array__10 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1368>() as i32
};
}

macro_rules! __must_be_array__11 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_852_4>() as i32
};
}

macro_rules! __must_be_array__12 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1496_4>() as i32
};
}

macro_rules! __must_be_array__13 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1416>() as i32
};
}

macro_rules! __must_be_array__18 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1309>() as i32
};
}

macro_rules! __must_be_array__22 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1492_4>() as i32
};
}

macro_rules! __must_be_array__25 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_1363>() as i32
};
}

macro_rules! __must_be_array__26 {
() => {
core::mem::size_of::<__SlateRecord_assoc_array_c_866_2>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__17 {
() => {
((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__3 {
($y:expr) => {
$y.wrapping_sub(1)
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

/// check_add_overflow() - Calculate addition with overflow checking
/// @a: first addend
/// @b: second addend
/// @d: pointer to store sum
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted addition, regardless of whether
/// wrap-around occurred.
macro_rules! check_add_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_add(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_add(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__9!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__4!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__3 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__22!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__5 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__9!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__6 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__26!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__7 {
($count:expr) => {
size_mul(($count) as u64, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__12!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__8 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__11!() as u64))
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__10 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<u64>() as u64).wrapping_add(__must_be_array__3!() as u64))
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

macro_rules! kcsan_wmb {
() => {
while false {
}
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! offsetof__9 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, rcu) as u64
};
}

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down {
($x:expr, $y:expr) => {
($x) & !__round_mask__3!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__3!($y)).wrapping_add(1)
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size {
($count:expr) => {
size_add(24u64, flex_array_size__5!($count))
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__2 {
($count:expr) => {
size_add(24u64, flex_array_size__7!($count))
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__3 {
($count:expr) => {
size_add(24u64, flex_array_size__8!($count))
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__4 {
($count:expr) => {
size_add(24u64, flex_array_size__3!($count))
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__5 {
($count:expr) => {
size_add(24u64, flex_array_size__10!($count))
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t {
($count:expr) => {
struct_size__5!($count)
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__2 {
($count:expr) => {
struct_size__4!($count)
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__3 {
($count:expr) => {
struct_size!($count)
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__4 {
($count:expr) => {
struct_size__3!($count)
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__7 {
($x:expr) => {
!$x.is_null() as i64
};
}

unsafe extern "C" {
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __compiletime_assert_210();
    fn __compiletime_assert_211();
    fn __compiletime_assert_212();
    fn __compiletime_assert_213();
    fn __compiletime_assert_214();
    fn __compiletime_assert_215();
    fn __compiletime_assert_216();
    fn __compiletime_assert_217();
    fn __compiletime_assert_219();
    fn __compiletime_assert_220();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[inline(always)]
fn size_add(addend1: u64, addend2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_add_overflow!(addend1, addend2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

// __set_flex_counter() - Set the counter associated with the given flexible
//                        array member that has been annoated by __counted_by().
// @FAM: Instance of flexible array member within a given struct.
// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
//
// This is a no-op if no annotation exists. Count needs to be checked with
// overflows_flex_counter_type() before using this function.
#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_priv_h_89 {
    ptr: *mut *mut assoc_array_ptr,
    to: *mut assoc_array_ptr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_priv_h_93 {
    p: *mut u8,
    to: u8,
}


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
struct __SlateRecord_assoc_array_c_152 {
    /// Node in which leaf might be found
    node: *mut assoc_array_node,
    level: i32,
    slot: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_157 {
    shortcut: *mut assoc_array_shortcut,
    level: i32,
    sc_level: i32,
    sc_segments: u64,
    dissimilarity: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_745_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_852_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_866_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_901_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_914_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1309 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1358 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1363 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1368 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1416 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1492_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_assoc_array_c_1496_4 {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
#[inline(always)]
fn kmalloc_type(flags: u32, token: __SlateRecord_slab_h_532) -> u32 {
// The most common case is KMALLOC_NORMAL, so test for it
// with a single branch for all the relevant flags.
if likely!(flags & ((BIT!(4i32) as u32) | if IS_ENABLED__2!() != 0 { BIT!(0i32) as u32 } else { 0 } | if IS_ENABLED!() != 0 { BIT!(22i32) as u32 } else { 0 }) == 0) != 0 {
return 0;
}
// At least one of the flags has to be set. Their priorities in
// decreasing order are:
//  1) __GFP_DMA
//  2) __GFP_RECLAIMABLE
//  3) __GFP_ACCOUNT
if IS_ENABLED__2!() != 0 && flags & (BIT!(0i32) as u32) != 0 {
return 2;
}
if IS_ENABLED!() == 0 || flags & (BIT!(4i32) as u32) != 0 {
return 1;
} else {
return 0;
}
}

#[inline(always)]
fn __kmalloc_index(size: u64, size_is_constant: bool) -> u32 {
if size == 0 {
return 0;
}
if size <= 8u64 {
return 3;
}
if 8i32 <= 32 && size > 64 && size <= 96 {
return 1;
}
if 8i32 <= 64 && size > 128 && size <= 192 {
return 2;
}
if size <= 8 {
return 3;
}
if size <= 16 {
return 4;
}
if size <= 32 {
return 5;
}
if size <= 64 {
return 6;
}
if size <= 128 {
return 7;
}
if size <= 256 {
return 8;
}
if size <= 512 {
return 9;
}
if size <= 1024 {
return 10;
}
if size <= 2048u64 {
return 11;
}
if size <= 4096u64 {
return 12;
}
if size <= 8192u64 {
return 13;
}
if size <= 16384u64 {
return 14;
}
if size <= 32768u64 {
return 15;
}
if size <= 65536u64 {
return 16;
}
if size <= 131072u64 {
return 17;
}
if size <= 262144u64 {
return 18;
}
if size <= 524288u64 {
return 19;
}
if size <= 1048576u64 {
return 20;
}
if size <= 2097152u64 {
return 21;
}
if IS_ENABLED!() == 0 && size_is_constant {
loop {
if true {
unsafe { __compiletime_assert_208() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("209: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 209b - ., 3; .popsection\n# {0}", const 209 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___kmalloc_index_1, sym __slate_asm_string___kmalloc_index_2, const 834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
// Will never be reached. Needed because the compiler may complain
1i32.wrapping_neg() as u32
}

#[inline(always)]
fn _kmalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_noprof(size, flags) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, size) };
}
unsafe { __kmalloc_noprof(PASS_TOKEN_PARAMS!(size), flags) }
}

// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// __alloc_flex - Allocate an object that has a trailing flexible array
// @KMALLOC: kmalloc wrapper function to use for allocation.
// @GFP: GFP flags for the allocation.
// @TYPE: type of structure to allocate space for.
// @FAM: The name of the flexible array member of @TYPE structure.
// @COUNT: how many @FAM elements to allocate space for.
//
// Returns: Newly allocated pointer to @TYPE with @COUNT-many trailing
// @FAM elements, or NULL on failure or if @COUNT cannot be represented
// by the member of @TYPE that counts the @FAM elements (annotated via
// __counted_by()).
// kmalloc_flex - Allocate a single instance of the given flexible structure
// @VAR_OR_TYPE: Variable or type to allocate (with its flex array).
// @FAM: The name of the flexible array member of the structure.
// @COUNT: How many flexible array member elements are desired.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to @VAR_OR_TYPE on success, NULL on
// failure. If @FAM has been annotated with __counted_by(), the allocation
// will immediately fail if @COUNT is larger than what the type of the
// struct's counter variable can represent.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ops {
    get_key_chunk: Option<unsafe extern "C" fn(*const (), i32) -> u64>,
    get_object_key_chunk: Option<unsafe extern "C" fn(*const (), i32) -> u64>,
    compare_object: Option<unsafe extern "C" fn(*const (), *const ()) -> bool>,
    diff_objects: Option<unsafe extern "C" fn(*const (), *const ()) -> i32>,
    free_object: Option<unsafe extern "C" fn(*mut ())>,
}


#[unsafe(no_mangle)]
extern "C" fn assoc_array_iterate(array: *const assoc_array, iterator: Option<unsafe extern "C" fn(*const (), *mut ()) -> i32>, iterator_data: *mut ()) -> i32 {
let mut __v0: *mut assoc_array_ptr;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_215() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*array).root) as *const *mut assoc_array_ptr) };
let root: *mut assoc_array_ptr = __v0; // Address dependency.
if root.is_null() {
return 0;
}
assoc_array_subtree_iterate(root as *const assoc_array_ptr, core::ptr::null::<assoc_array_ptr>(), iterator, iterator_data)
}

#[unsafe(no_mangle)]
extern "C" fn assoc_array_find(array: *const assoc_array, ops: *const assoc_array_ops, index_key: *const ()) -> *mut () {
let mut result: assoc_array_walk_result = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(result) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
if (assoc_array_walk(array, ops, index_key, unsafe { &mut (*core::ptr::addr_of_mut!(result)) }) as u32) != 1 {
return core::ptr::null_mut::<()>();
}
let node: *const assoc_array_node = result.terminal_node.node as *const assoc_array_node;
// If the target key is available to us, it's has to be pointed to by
// the terminal node.
for slot in 0i32..16i32 {
let mut __v0: *mut assoc_array_ptr;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut assoc_array_ptr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_220() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(((*node).slots.as_ptr() as *const *mut assoc_array_ptr).wrapping_offset(slot as isize) as *const *mut assoc_array_ptr) };
let ptr: *const assoc_array_ptr = __v0 as *const assoc_array_ptr; // Address dependency.
let __v2: bool = !ptr.is_null() && assoc_array_ptr_is_leaf(ptr);
if __v2 {
// We need a barrier between the read of the pointer
// and dereferencing the pointer - but only if we are
// actually going to dereference it.
let leaf: *const () = assoc_array_ptr_to_leaf(ptr) as *const ();
if unsafe { unsafe { (*ops).compare_object }.unwrap()(leaf, index_key) } {
return leaf as *mut ();
}
}
}
core::ptr::null_mut::<()>()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_node {
    back_pointer: *mut assoc_array_ptr,
    parent_slot: u8,
    slots: [*mut assoc_array_ptr; 16],
    nr_leaves_on_branch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_shortcut {
    back_pointer: *mut assoc_array_ptr,
    parent_slot: i32,
    skip_to_level: i32,
    next_node: *mut assoc_array_ptr,
    index_key: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_edit {
    rcu: callback_head,
    array: *mut assoc_array,
    ops: *const assoc_array_ops,
    ops_for_excised_subtree: *const assoc_array_ops,
    leaf: *mut assoc_array_ptr,
    leaf_p: *mut *mut assoc_array_ptr,
    dead_leaf: *mut assoc_array_ptr,
    new_meta: [*mut assoc_array_ptr; 3],
    excised_meta: [*mut assoc_array_ptr; 1],
    excised_subtree: *mut assoc_array_ptr,
    set_backpointers: [*mut *mut assoc_array_ptr; 16],
    set_backpointers_to: *mut assoc_array_ptr,
    adjust_count_on: *mut assoc_array_node,
    adjust_count_by: i64,
    set: [__SlateRecord_assoc_array_priv_h_89; 2],
    set_parent_slot: [__SlateRecord_assoc_array_priv_h_93; 1],
    segment_cache: [u8; 17],
}


// Internal tree member pointers are marked in the bottom one or two bits to
// indicate what type they are so that we don't have to look behind every
// pointer to see what it points to.
//
// We provide functions to test type annotations and to create and translate
// the annotated pointers.
// Points to leaf (or nowhere)
// Points to node or shortcut
fn assoc_array_ptr_is_meta(x: *const assoc_array_ptr) -> bool {
(x as u64) & 1 != 0
}

fn assoc_array_ptr_is_leaf(x: *const assoc_array_ptr) -> bool {
!assoc_array_ptr_is_meta(x)
}

fn assoc_array_ptr_is_shortcut(x: *const assoc_array_ptr) -> bool {
(x as u64) & 2 != 0
}

fn assoc_array_ptr_is_node(x: *const assoc_array_ptr) -> bool {
!assoc_array_ptr_is_shortcut(x)
}

fn assoc_array_ptr_to_leaf(x: *const assoc_array_ptr) -> *mut () {
((x as u64) & !1) as *mut ()
}

fn __assoc_array_ptr_to_meta(x: *const assoc_array_ptr) -> u64 {
(x as u64) & !(2 | 1)
}

fn assoc_array_ptr_to_node(x: *const assoc_array_ptr) -> *mut assoc_array_node {
__assoc_array_ptr_to_meta(x) as *mut assoc_array_node
}

fn assoc_array_ptr_to_shortcut(x: *const assoc_array_ptr) -> *mut assoc_array_shortcut {
__assoc_array_ptr_to_meta(x) as *mut assoc_array_shortcut
}

fn __assoc_array_x_to_ptr(p: *const (), t: u64) -> *mut assoc_array_ptr {
((p as u64) | t) as *mut assoc_array_ptr
}

fn assoc_array_leaf_to_ptr(p: *const ()) -> *mut assoc_array_ptr {
__assoc_array_x_to_ptr(p, 0u64)
}

fn assoc_array_node_to_ptr(p: *const assoc_array_node) -> *mut assoc_array_ptr {
__assoc_array_x_to_ptr(p as *const (), 1u64)
}

fn assoc_array_shortcut_to_ptr(p: *const assoc_array_shortcut) -> *mut assoc_array_ptr {
__assoc_array_x_to_ptr(p as *const (), 3u64)
}

// Iterate over an associative array.  The caller must hold the RCU read lock
// or better.
fn assoc_array_subtree_iterate(root: *const assoc_array_ptr, stop: *const assoc_array_ptr, iterator: Option<unsafe extern "C" fn(*const (), *mut ()) -> i32>, iterator_data: *mut ()) -> i32 {
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(slot) as *mut i32;
let mut has_meta: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(has_meta) as *mut u64;
let mut parent: core::mem::MaybeUninit<*const assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *const assoc_array_ptr = core::ptr::addr_of_mut!(parent) as *mut *const assoc_array_ptr;
let mut ptr: core::mem::MaybeUninit<*const assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *const assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *const assoc_array_ptr;
let mut cursor: core::mem::MaybeUninit<*const assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *const assoc_array_ptr = core::ptr::addr_of_mut!(cursor) as *mut *const assoc_array_ptr;
let mut node: core::mem::MaybeUninit<*const assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *const assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *const assoc_array_node;
let mut shortcut: core::mem::MaybeUninit<*const assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *const assoc_array_shortcut = core::ptr::addr_of_mut!(shortcut) as *mut *const assoc_array_shortcut;
let mut __slate_state: usize = 31;
unsafe {
loop {
match __slate_state {
0 => {
// Ascend to next slot in parent node
*__slate_slot_19 = *__slate_slot_17;
let __v1: i32 = *__slate_slot_15;
*__slate_slot_15 = __v1.wrapping_add(1);
__slate_state = 14;
continue;
}
1 => {
return 0;
}
2 => {
let __v2: *mut assoc_array_ptr = core::ptr::read_volatile(core::ptr::addr_of!((*(*__slate_slot_21)).back_pointer) as *const *mut assoc_array_ptr);
*__slate_slot_17 = __v2 as *const assoc_array_ptr; // Address dependency.
*__slate_slot_15 = (*(*__slate_slot_21)).parent_slot;
if *__slate_slot_17 == stop {
__slate_state = 1;
continue;
} else {
__slate_state = 0;
continue;
}
}
3 => {
if false {
__slate_state = 3;
continue;
} else {
__slate_state = 2;
continue;
}
}
4 => {
*__slate_slot_21 = assoc_array_ptr_to_shortcut(*__slate_slot_17) as *const assoc_array_shortcut;
*__slate_slot_19 = *__slate_slot_17;
__slate_state = 3;
continue;
}
5 => {
if assoc_array_ptr_is_shortcut(*__slate_slot_17) {
__slate_state = 4;
continue;
} else {
__slate_state = 0;
continue;
}
}
6 => {
return 0;
}
7 => {
let __v3: *mut assoc_array_ptr = core::ptr::read_volatile(core::ptr::addr_of!((*(*__slate_slot_20)).back_pointer) as *const *mut assoc_array_ptr);
*__slate_slot_17 = __v3 as *const assoc_array_ptr; // Move up to the parent (may need to skip back over a shortcut)
// Address dependency.
*__slate_slot_15 = (*(*__slate_slot_20)).parent_slot as i32;
if *__slate_slot_17 == stop {
__slate_state = 6;
continue;
} else {
__slate_state = 5;
continue;
}
}
8 => {
if false {
__slate_state = 8;
continue;
} else {
__slate_state = 7;
continue;
}
}
9 => {
if *__slate_slot_15 < 16 {
__slate_state = 13;
continue;
} else {
__slate_state = 8;
continue;
}
}
10 => {
let __v5: i32 = *__slate_slot_15;
*__slate_slot_15 = __v5.wrapping_add(1);
__slate_state = 9;
continue;
}
11 => {
*__slate_slot_19 = *__slate_slot_18;
__slate_state = 30;
continue;
}
12 => {
let __v6: *mut assoc_array_ptr = core::ptr::read_volatile(((*(*__slate_slot_20)).slots.as_ptr() as *const *mut assoc_array_ptr).wrapping_offset(*__slate_slot_15 as isize) as *const *mut assoc_array_ptr);
*__slate_slot_18 = __v6 as *const assoc_array_ptr; // Address dependency.
if assoc_array_ptr_is_meta(*__slate_slot_18) {
__slate_state = 11;
continue;
} else {
__slate_state = 10;
continue;
}
}
13 => {
if false {
__slate_state = 13;
continue;
} else {
__slate_state = 12;
continue;
}
}
14 => {
*__slate_slot_20 = assoc_array_ptr_to_node(*__slate_slot_19) as *const assoc_array_node;
__slate_state = 9;
continue;
}
15 => {
*__slate_slot_15 = 0;
__slate_state = 14;
continue;
}
16 => {
// The second pass attends to all the metadata pointers.  If we follow
// one of these we may find that we don't come back here, but rather go
// back to a replacement node with the leaves in a different layout.
//
// We are guaranteed to make progress, however, as the slot number for
// a particular portion of the key space cannot change - and we
// continue at the back pointer + 1.
if *__slate_slot_16 & 1 == 0 {
__slate_state = 8;
continue;
} else {
__slate_state = 15;
continue;
}
}
17 => {
if *__slate_slot_15 < 16 {
__slate_state = 25;
continue;
} else {
__slate_state = 16;
continue;
}
}
18 => {
let __v8: i32 = *__slate_slot_15;
*__slate_slot_15 = __v8.wrapping_add(1);
__slate_state = 17;
continue;
}
19 => {
return *__slate_slot_14;
}
20 => {
// We need a barrier between the read of the pointer,
// which is supplied by the above READ_ONCE().
//
// Invoke the callback
*__slate_slot_14 = iterator.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_18) as *const (), iterator_data);
if *__slate_slot_14 != 0 {
__slate_state = 19;
continue;
} else {
__slate_state = 18;
continue;
}
}
21 => {
if __v9 {
__slate_state = 20;
continue;
} else {
__slate_state = 18;
continue;
}
}
22 => {
__v9 = assoc_array_ptr_is_leaf(*__slate_slot_18);
__slate_state = 21;
continue;
}
23 => {
__v9 = false;
__slate_state = 21;
continue;
}
24 => {
let __v12: *mut assoc_array_ptr = core::ptr::read_volatile(((*(*__slate_slot_20)).slots.as_ptr() as *const *mut assoc_array_ptr).wrapping_offset(*__slate_slot_15 as isize) as *const *mut assoc_array_ptr);
*__slate_slot_18 = __v12 as *const assoc_array_ptr; // Address dependency.
*__slate_slot_16 |= *__slate_slot_18 as u64;
if !(*__slate_slot_18).is_null() {
__slate_state = 22;
continue;
} else {
__slate_state = 23;
continue;
}
}
25 => {
if false {
__slate_state = 25;
continue;
} else {
__slate_state = 24;
continue;
}
}
26 => {
*__slate_slot_20 = assoc_array_ptr_to_node(*__slate_slot_19) as *const assoc_array_node;
*__slate_slot_15 = 0;
// We perform two passes of each node.
//
// The first pass does all the leaves in this node.  This means we
// don't miss any leaves if the node is split up by insertion whilst
// we're iterating over the branches rooted here (we may, however, see
// some leaves twice).
*__slate_slot_16 = 0;
__slate_state = 17;
continue;
}
27 => {
let __v13: *mut assoc_array_ptr = core::ptr::read_volatile(core::ptr::addr_of!((*(*__slate_slot_21)).next_node) as *const *mut assoc_array_ptr);
*__slate_slot_19 = __v13 as *const assoc_array_ptr; // Address dependency.
__slate_state = 26;
continue;
}
28 => {
if false {
__slate_state = 28;
continue;
} else {
__slate_state = 27;
continue;
}
}
29 => {
// Descend through a shortcut
*__slate_slot_21 = assoc_array_ptr_to_shortcut(*__slate_slot_19) as *const assoc_array_shortcut;
__slate_state = 28;
continue;
}
30 => {
if assoc_array_ptr_is_shortcut(*__slate_slot_19) {
__slate_state = 29;
continue;
} else {
__slate_state = 26;
continue;
}
}
31 => {
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_19 = root;
__slate_state = 30;
continue;
}
_ => {
core::hint::unreachable_unchecked();
}
}
}
}
unsafe { core::mem::zeroed() }
}

// assoc_array_iterate - Pass all objects in the array to a callback
// @array: The array to iterate over.
// @iterator: The callback function.
// @iterator_data: Private data for the callback function.
//
// Iterate over all the objects in an associative array.  Each one will be
// presented to the iterator function.
//
// If the array is being modified concurrently with the iteration then it is
// possible that some objects in the array will be passed to the iterator
// callback more than once - though every object should be passed at least
// once.  If this is undesirable then the caller must lock against modification
// for the duration of this function.
//
// The function will return 0 if no objects were in the array or else it will
// return the result of the last iterator function called.  Iteration stops
// immediately if any call to the iteration function results in a non-zero
// return.
//
// The caller should hold the RCU read lock or better if concurrent
// modification is possible.
#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_walk_result {
    terminal_node: __SlateRecord_assoc_array_c_152,
    wrong_shortcut: __SlateRecord_assoc_array_c_157,
}


// Navigate through the internal tree looking for the closest node to the key.
fn assoc_array_walk(array: *const assoc_array, ops: *const assoc_array_ops, index_key: *const (), result: &mut assoc_array_walk_result) -> u32 {
let mut slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(slot) as *mut i32;
let mut next_sc_level: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(next_sc_level) as *mut i32;
let mut sc_level: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(sc_level) as *mut i32;
let mut level: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(level) as *mut i32;
let mut segments: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u64 = core::ptr::addr_of_mut!(segments) as *mut u64;
let mut dissimilarity: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u64 = core::ptr::addr_of_mut!(dissimilarity) as *mut u64;
let mut sc_segments: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u64 = core::ptr::addr_of_mut!(sc_segments) as *mut u64;
let mut ptr: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *mut assoc_array_ptr;
let mut cursor: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(cursor) as *mut *mut assoc_array_ptr;
let mut node: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *mut assoc_array_node;
let mut shortcut: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(shortcut) as *mut *mut assoc_array_shortcut;
let mut __slate_state: usize = 29;
unsafe {
loop {
match __slate_state {
0 => {
*__slate_slot_18 = *__slate_slot_17;
__slate_state = 24;
continue;
}
1 => {
*__slate_slot_18 = *__slate_slot_17;
__slate_state = 23;
continue;
}
2 => {
// The shortcut matches the leaf's index to this point.
let __v0: *mut assoc_array_ptr = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_25)).next_node) as *const *mut assoc_array_ptr);
*__slate_slot_23 = __v0; // Address dependency.
if (*__slate_slot_18 ^ *__slate_slot_17) & -64i32 != 0 {
__slate_state = 0;
continue;
} else {
__slate_state = 1;
continue;
}
}
3 => {
if false {
__slate_state = 3;
continue;
} else {
__slate_state = 2;
continue;
}
}
4 => {
*__slate_slot_17 = *__slate_slot_16;
if *__slate_slot_17 < (*(*__slate_slot_25)).skip_to_level {
__slate_state = 11;
continue;
} else {
__slate_state = 3;
continue;
}
}
5 => {
// This shortcut points elsewhere
(*(result as *mut assoc_array_walk_result)).wrong_shortcut.shortcut = *__slate_slot_25;
(*(result as *mut assoc_array_walk_result)).wrong_shortcut.level = *__slate_slot_18;
(*(result as *mut assoc_array_walk_result)).wrong_shortcut.sc_level = *__slate_slot_17;
(*(result as *mut assoc_array_walk_result)).wrong_shortcut.sc_segments = *__slate_slot_21;
(*(result as *mut assoc_array_walk_result)).wrong_shortcut.dissimilarity = *__slate_slot_20;
return 2;
}
6 => {
if *__slate_slot_20 != 0 {
__slate_state = 5;
continue;
} else {
__slate_state = 4;
continue;
}
}
7 => {
// Trim segments that are beyond the shortcut
let shift: i32 = (*(*__slate_slot_25)).skip_to_level & 63i32;
let __v1: u64 = *__slate_slot_20 & !(18446744073709551615u64 << shift);
*__slate_slot_20 = __v1;
*__slate_slot_16 = (*(*__slate_slot_25)).skip_to_level;
__slate_state = 6;
continue;
}
8 => {
*__slate_slot_16 = (*__slate_slot_17).wrapping_add(64);
*__slate_slot_16 = round_down!(*__slate_slot_16, 64i32);
__slate_state = 6;
continue;
}
9 => {
*__slate_slot_21 = *(core::ptr::addr_of_mut!((*(*__slate_slot_25)).index_key) as *mut u64).wrapping_offset((*__slate_slot_17 >> ilog2!(64i32)) as isize);
*__slate_slot_20 = *__slate_slot_19 ^ *__slate_slot_21;
if (*(*__slate_slot_25)).skip_to_level < round_down!(*__slate_slot_17, 64i32).wrapping_add(64) {
__slate_state = 7;
continue;
} else {
__slate_state = 8;
continue;
}
}
10 => {
*__slate_slot_19 = unsafe { (*ops).get_key_chunk }.unwrap()(index_key, *__slate_slot_17);
__slate_state = 9;
continue;
}
11 => {
// Check the leaf against the shortcut's index key a word at a
// time, trimming the final word (the shortcut stores the index
// key completely from the root to the shortcut's target).
if *__slate_slot_17 & 63i32 == 0 {
__slate_state = 10;
continue;
} else {
__slate_state = 9;
continue;
}
}
12 => {
if false {
__slate_state = 15;
continue;
} else {
__slate_state = 11;
continue;
}
}
13 => {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_walk_26, sym __slate_asm_string_assoc_array_walk_27, const 245 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if false {
__slate_state = 13;
continue;
} else {
core::hint::unreachable_unchecked();
}
}
14 => {
core::arch::asm!("218: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 218b - ., 3; .popsection\n# {0}", const 218 as i32, options(nostack, att_syntax));
__slate_state = 13;
continue;
}
15 => {
if unlikely!(*__slate_slot_17 > (*(*__slate_slot_25)).skip_to_level) != 0 {
__slate_state = 14;
continue;
} else {
__slate_state = 12;
continue;
}
}
16 => {
*__slate_slot_25 = assoc_array_ptr_to_shortcut(*__slate_slot_23 as *const assoc_array_ptr);
*__slate_slot_17 = (*__slate_slot_18).wrapping_add(ilog2!(16i32));
__slate_state = 15;
continue;
}
17 => {
// There is a shortcut in the slot corresponding to the index key
// segment.  We follow the shortcut if its partial index key matches
// this leaf's.  Otherwise we need to split the shortcut.
*__slate_slot_23 = *__slate_slot_22;
__slate_state = 16;
continue;
}
18 => {
// There is a pointer to a node in the slot corresponding to
// this index key segment, so we need to follow it.
*__slate_slot_23 = *__slate_slot_22;
let __v6: i32 = *__slate_slot_18;
let __v5: i32 = __v6.wrapping_add(ilog2!(16i32));
*__slate_slot_18 = __v5;
if *__slate_slot_18 & 63i32 != 0 {
__slate_state = 23;
continue;
} else {
__slate_state = 24;
continue;
}
}
19 => {
if assoc_array_ptr_is_node(*__slate_slot_22 as *const assoc_array_ptr) {
__slate_state = 18;
continue;
} else {
__slate_state = 17;
continue;
}
}
20 => {
// The node doesn't have a node/shortcut pointer in the slot
// corresponding to the index key that we have to follow.
(*(result as *mut assoc_array_walk_result)).terminal_node.node = *__slate_slot_24;
(*(result as *mut assoc_array_walk_result)).terminal_node.level = *__slate_slot_18;
(*(result as *mut assoc_array_walk_result)).terminal_node.slot = *__slate_slot_15;
return 1;
}
21 => {
let __v9: *mut assoc_array_ptr = core::ptr::read_volatile(((*(*__slate_slot_24)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_15 as isize) as *const *mut assoc_array_ptr);
*__slate_slot_22 = __v9; // Address dependency.
if !assoc_array_ptr_is_meta(*__slate_slot_22 as *const assoc_array_ptr) {
__slate_state = 20;
continue;
} else {
__slate_state = 19;
continue;
}
}
22 => {
if false {
__slate_state = 22;
continue;
} else {
__slate_state = 21;
continue;
}
}
23 => {
*__slate_slot_24 = assoc_array_ptr_to_node(*__slate_slot_23 as *const assoc_array_ptr);
*__slate_slot_15 = (*__slate_slot_19 >> (*__slate_slot_18 & 63i32)) as i32;
*__slate_slot_15 &= 15i32;
__slate_state = 22;
continue;
}
24 => {
*__slate_slot_19 = unsafe { (*ops).get_key_chunk }.unwrap()(index_key, *__slate_slot_18);
if assoc_array_ptr_is_shortcut(*__slate_slot_23 as *const assoc_array_ptr) {
__slate_state = 16;
continue;
} else {
__slate_state = 23;
continue;
}
}
25 => {
*__slate_slot_18 = 0;
// Use segments from the key for the new leaf to navigate through the
// internal tree, skipping through nodes and shortcuts that are on
// route to the destination.  Eventually we'll come to a slot that is
// either empty or contains a leaf at which point we've found a node in
// which the leaf we're looking for might be found or into which it
// should be inserted.
__slate_state = 24;
continue;
}
26 => {
return 0;
}
27 => {
let __v13: *mut assoc_array_ptr = core::ptr::read_volatile(core::ptr::addr_of!((*array).root) as *const *mut assoc_array_ptr);
*__slate_slot_23 = __v13; // Address dependency.
if (*__slate_slot_23).is_null() {
__slate_state = 26;
continue;
} else {
__slate_state = 25;
continue;
}
}
28 => {
if false {
__slate_state = 28;
continue;
} else {
__slate_state = 27;
continue;
}
}
29 => {
let __slate_auto_bytes = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__slate_state = 28;
continue;
}
_ => {
core::hint::unreachable_unchecked();
}
}
}
}
unsafe { core::mem::zeroed() }
}

// assoc_array_find - Find an object by index key
// @array: The associative array to search.
// @ops: The operations to use.
// @index_key: The key to the object.
//
// Find an object in an associative array by walking through the internal tree
// to the node that should contain the object and then searching the leaves
// there.  NULL is returned if the requested object was not found in the array.
//
// The caller must hold the RCU read lock or better.
/// Destructively iterate over an associative array.  The caller must prevent
/// other simultaneous accesses.
fn assoc_array_destroy_subtree(root: *mut assoc_array_ptr, ops: *const assoc_array_ops) {
let mut ptr: *mut assoc_array_ptr = unsafe { core::mem::zeroed() };
let mut slot: i32 = unsafe { core::mem::zeroed() };
let mut parent: *mut assoc_array_ptr = unsafe { core::mem::zeroed() };
let mut cursor: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(cursor) as *mut *mut assoc_array_ptr;
let mut node: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *mut assoc_array_node;
let mut shortcut: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(shortcut) as *mut *mut assoc_array_shortcut;
unsafe {
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
parent = core::ptr::null_mut::<assoc_array_ptr>();
slot = 1i32.wrapping_neg();
*__slate_slot_16 = root;
if (*__slate_slot_16).is_null() {
return;
} else {
'__join_49: {
'__join_44: {
'__join_40: {
'__join_35: {
'__join_30: {
'__join_26: {
'__join_9: {
'__loop_52: loop {
if assoc_array_ptr_is_shortcut(*__slate_slot_16 as *const assoc_array_ptr) {
// Descend through a shortcut
loop {
if unlikely!(!assoc_array_ptr_is_shortcut(*__slate_slot_16 as *const assoc_array_ptr)) != 0 {
break '__join_49;
} else {
if true {
break;
}
}
}
*__slate_slot_18 = assoc_array_ptr_to_shortcut(*__slate_slot_16 as *const assoc_array_ptr);
loop {
if unlikely!((*(*__slate_slot_18)).back_pointer != parent) != 0 {
break '__join_44;
} else {
if true {
break;
}
}
}
loop {
if unlikely!(slot != 1i32.wrapping_neg() && (*(*__slate_slot_18)).parent_slot != slot) != 0 {
break '__join_40;
} else {
if true {
break;
}
}
}
parent = *__slate_slot_16;
*__slate_slot_16 = (*(*__slate_slot_18)).next_node;
slot = 1i32.wrapping_neg();
loop {
if unlikely!(!assoc_array_ptr_is_node(*__slate_slot_16 as *const assoc_array_ptr)) != 0 {
break '__join_35;
} else {
if true {
break;
}
}
}
}
*__slate_slot_17 = assoc_array_ptr_to_node(*__slate_slot_16 as *const assoc_array_ptr);
loop {
if unlikely!((*(*__slate_slot_17)).back_pointer != parent) != 0 {
break '__join_30;
} else {
if true {
break;
}
}
}
loop {
if unlikely!(slot != 1i32.wrapping_neg() && ((*(*__slate_slot_17)).parent_slot as i32) != slot) != 0 {
break '__join_26;
} else {
if true {
break;
}
}
}
slot = 0;
'__loop_22: loop {
while slot < 16 {
ptr = *((*(*__slate_slot_17)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(slot as isize);
if !ptr.is_null() {
if assoc_array_ptr_is_meta(ptr as *const assoc_array_ptr) {
break '__loop_22;
} else {
if !ops.is_null() {
unsafe { (*ops).free_object }.unwrap()(assoc_array_ptr_to_leaf(ptr as *const assoc_array_ptr));
}
}
}
slot = slot.wrapping_add(1);
}
parent = (*(*__slate_slot_17)).back_pointer;
slot = (*(*__slate_slot_17)).parent_slot as i32;
kfree(*__slate_slot_17 as *const ());
if parent.is_null() {
return;
} else {
// Done
// Move back up to the parent (may need to free a shortcut on
// the way up)
if assoc_array_ptr_is_shortcut(parent as *const assoc_array_ptr) {
*__slate_slot_18 = assoc_array_ptr_to_shortcut(parent as *const assoc_array_ptr);
loop {
if unlikely!((*(*__slate_slot_18)).next_node != *__slate_slot_16) != 0 {
break '__join_9;
} else {
if true {
break;
}
}
}
*__slate_slot_16 = parent;
parent = (*(*__slate_slot_18)).back_pointer;
slot = (*(*__slate_slot_18)).parent_slot;
kfree(*__slate_slot_18 as *const ());
if parent.is_null() {
return;
} else {
loop {
if unlikely!(!assoc_array_ptr_is_node(parent as *const assoc_array_ptr)) != 0 {
break '__loop_52;
} else {
if true {
break;
}
}
}
}
}
// Ascend to next slot in parent node
*__slate_slot_16 = parent;
*__slate_slot_17 = assoc_array_ptr_to_node(*__slate_slot_16 as *const assoc_array_ptr);
slot = slot.wrapping_add(1);
}
}
parent = *__slate_slot_16;
*__slate_slot_16 = ptr;
}
core::arch::asm!("228: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 228b - ., 3; .popsection\n# {0}", const 228 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_19, sym __slate_asm_string_assoc_array_destroy_subtree_20, const 418 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("227: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 227b - ., 3; .popsection\n# {0}", const 227 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_21, sym __slate_asm_string_assoc_array_destroy_subtree_22, const 409 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("226: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 226b - ., 3; .popsection\n# {0}", const 226 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_23, sym __slate_asm_string_assoc_array_destroy_subtree_24, const 377 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("225: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 225b - ., 3; .popsection\n# {0}", const 225 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_25, sym __slate_asm_string_assoc_array_destroy_subtree_26, const 376 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("224: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 224b - ., 3; .popsection\n# {0}", const 224 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_27, sym __slate_asm_string_assoc_array_destroy_subtree_28, const 371 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("223: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 223b - ., 3; .popsection\n# {0}", const 223 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_29, sym __slate_asm_string_assoc_array_destroy_subtree_30, const 367 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("222: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 222b - ., 3; .popsection\n# {0}", const 222 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_31, sym __slate_asm_string_assoc_array_destroy_subtree_32, const 366 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("221: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 221b - ., 3; .popsection\n# {0}", const 221 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_destroy_subtree_33, sym __slate_asm_string_assoc_array_destroy_subtree_34, const 364 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}

/// assoc_array_destroy - Destroy an associative array
/// @array: The array to destroy.
/// @ops: The operations to use.
///
/// Discard all metadata and free all objects in an associative array.  The
/// array will be empty and ready to use again upon completion.  This function
/// cannot fail.
///
/// The caller must prevent all other accesses whilst this takes place as no
/// attempt is made to adjust pointers gracefully to permit RCU readlock-holding
/// accesses to continue.  On the other hand, no memory allocation is required.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_destroy(array: *mut assoc_array, ops: *const assoc_array_ops) {
assoc_array_destroy_subtree(unsafe { (*array).root }, ops);
unsafe {
(*array).root = core::ptr::null_mut::<assoc_array_ptr>();
}
}

/// Handle insertion into an empty tree.
fn assoc_array_insert_in_empty_tree(edit: *mut assoc_array_edit) -> bool {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017-->%s()\n".as_ptr() as *const u8, c"assoc_array_insert_in_empty_tree".as_ptr() as *mut u8) };
}
__v0 = 0;
let mut __v1: *mut assoc_array_node;
let __obj_size: u64 = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut assoc_array_node;
let new_n0: *mut assoc_array_node = __v1;
if new_n0.is_null() {
return false;
}
unsafe {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
unsafe {
(*edit).leaf_p = ((*new_n0).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize);
}
unsafe {
(*edit).adjust_count_on = new_n0;
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = core::ptr::addr_of_mut!((*(*edit).array).root);
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017<--%s() = ok [no root]\n".as_ptr() as *const u8, c"assoc_array_insert_in_empty_tree".as_ptr() as *mut u8) };
}
__v7 = 0;
true
}

/// Handle insertion into a terminal node.
fn assoc_array_insert_into_terminal_node(edit: *mut assoc_array_edit, ops: *const assoc_array_ops, index_key: *const (), result: &assoc_array_walk_result) -> bool {
let mut __slate_compound_109: u64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_110: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_112: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_131: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_133: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_134: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_136: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_ptr: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut x: i32 = unsafe { core::mem::zeroed() };
let mut _old_1: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_68: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_1) as *mut *mut alloc_tag;
let mut _res_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_69: *mut *mut () = core::ptr::addr_of_mut!(_res_1) as *mut *mut ();
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_74: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_75: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size_2: u64 = unsafe { core::mem::zeroed() };
let mut __v86: bool = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_88: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_89: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut free_slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_90: *mut i32 = core::ptr::addr_of_mut!(free_slot) as *mut i32;
let mut next_slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_91: *mut i32 = core::ptr::addr_of_mut!(next_slot) as *mut i32;
let mut slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_92: *mut i32 = core::ptr::addr_of_mut!(slot) as *mut i32;
let mut diff: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_93: *mut i32 = core::ptr::addr_of_mut!(diff) as *mut i32;
let mut level: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_94: *mut i32 = core::ptr::addr_of_mut!(level) as *mut i32;
let mut have_meta: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_95: *mut bool = core::ptr::addr_of_mut!(have_meta) as *mut bool;
let mut keylen: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_96: *mut u64 = core::ptr::addr_of_mut!(keylen) as *mut u64;
let mut blank: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_97: *mut u64 = core::ptr::addr_of_mut!(blank) as *mut u64;
let mut base_seg: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_98: *mut u64 = core::ptr::addr_of_mut!(base_seg) as *mut u64;
let mut dissimilarity: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_99: *mut u64 = core::ptr::addr_of_mut!(dissimilarity) as *mut u64;
let mut ptr: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_100: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *mut assoc_array_ptr;
let mut side: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_101: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(side) as *mut *mut assoc_array_node;
let mut new_n1: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_102: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(new_n1) as *mut *mut assoc_array_node;
let mut new_n0: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_103: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(new_n0) as *mut *mut assoc_array_node;
let mut node: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_104: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *mut assoc_array_node;
let mut new_s0: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_105: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(new_s0) as *mut *mut assoc_array_shortcut;
let mut shortcut: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_106: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(shortcut) as *mut *mut assoc_array_shortcut;
unsafe {
let __slate_auto_bytes_4 = __slate_slot_106 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_105 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_104 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_103 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_102 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_101 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_100 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_99 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_12 = __slate_slot_98 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_13 = __slate_slot_97 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_14 = __slate_slot_96 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_15 = __slate_slot_95 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_16 = __slate_slot_94 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_17 = __slate_slot_93 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_18 = __slate_slot_92 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_18.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_19 = __slate_slot_91 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_19.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_20 = __slate_slot_90 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_20.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_21 = __slate_slot_89 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_21.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_22 = __slate_slot_88 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_22.add(0usize), 0u8, 4usize);
*__slate_slot_104 = (*(result as *const assoc_array_walk_result)).terminal_node.node;
*__slate_slot_94 = (*(result as *const assoc_array_walk_result)).terminal_node.level;
*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) = (*(result as *const assoc_array_walk_result)).terminal_node.slot as u8;
// We arrived at a node which doesn't have an onward node or shortcut
// pointer that we have to follow.  This means that (a) the leaf we
// want must go here (either by insertion or replacement) or (b) we
// need to split this node and insert in one of the fragments.
*__slate_slot_90 = 1i32.wrapping_neg();
// Firstly, we have to check the leaves in this node to see if there's
// a matching one we should replace in place.
*__slate_slot_89 = 0;
while *__slate_slot_89 < 16 {
*__slate_slot_100 = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
if (*__slate_slot_100).is_null() {
*__slate_slot_90 = *__slate_slot_89;
} else {
if assoc_array_ptr_is_leaf(*__slate_slot_100 as *const assoc_array_ptr) {
__v86 = unsafe { (*ops).compare_object }.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_100 as *const assoc_array_ptr) as *const (), index_key);
} else {
__v86 = false;
}
if __v86 {
(*edit).leaf_p = ((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
(*edit).dead_leaf = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
return true;
}
}
let __v83: i32 = *__slate_slot_89;
*__slate_slot_89 = __v83.wrapping_add(1);
}
// If there is a free slot in this node then we can just insert the
// leaf here.
if *__slate_slot_90 >= 0 {
(*edit).leaf_p = ((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_90 as isize);
(*edit).adjust_count_on = *__slate_slot_104;
return true;
} else {
__obj_size_2 = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_75 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_74 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_74 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_75 = _kzalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_134 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_134)
});
alloc_tag_restore!();
} else {
*__slate_slot_75 = _kzalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_136 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_136)
});
}
let __v77: *mut () = *__slate_slot_75;
// The node has no spare slots - so we're either going to have to split
// it or insert another node before it.
//
// Whatever, we're going to need at least two new nodes - so allocate
// those now.  We may also need a new shortcut, but we deal with that
// when we need it.
*__slate_slot_103 = __v77 as *mut assoc_array_node;
if (*__slate_slot_103).is_null() {
return false;
} else {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
__obj_size_1 = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let __slate_auto_bytes_2 = __slate_slot_69 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_68 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_68 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_69 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_131 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_131)
});
alloc_tag_restore!();
} else {
*__slate_slot_69 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_133 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_133)
});
}
*__slate_slot_102 = *__slate_slot_69 as *mut assoc_array_node;
if (*__slate_slot_102).is_null() {
return false;
} else {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(1 as isize) = assoc_array_node_to_ptr(*__slate_slot_102 as *const assoc_array_node);
// We need to find out how similar the leaves are.
*__slate_slot_95 = false;
*__slate_slot_89 = 0;
while *__slate_slot_89 < 16 {
*__slate_slot_100 = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
if assoc_array_ptr_is_meta(*__slate_slot_100 as *const assoc_array_ptr) {
*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) = 255;
*__slate_slot_95 = true;
} else {
*__slate_slot_98 = unsafe { (*ops).get_object_key_chunk }.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_100 as *const assoc_array_ptr) as *const (), *__slate_slot_94);
*__slate_slot_98 >>= *__slate_slot_94 & 63i32;
*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) = (*__slate_slot_98 & 15u64) as u8;
}
let __v64: i32 = *__slate_slot_89;
*__slate_slot_89 = __v64.wrapping_add(1);
}
'__join_102: {
if !(*__slate_slot_95) {
// The node contains only leaves
*__slate_slot_99 = 0;
*__slate_slot_98 = *((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as u64;
*__slate_slot_89 = 1;
while *__slate_slot_89 < 16 {
let __v61: u64 = *__slate_slot_99;
let __v60: u64 = __v61 | ((*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) as i32) as u64) ^ *__slate_slot_98;
*__slate_slot_99 = __v60;
let __v59: i32 = *__slate_slot_89;
*__slate_slot_89 = __v59.wrapping_add(1);
}
if *__slate_slot_99 & 15u64 == 0 {
// The old leaves all cluster in the same slot.  We will need
// to insert a shortcut if the new node wants to cluster with them.
if ((*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) as i32) as u64) ^ *__slate_slot_98 == 0 {
// All the leaves, new and old, want to cluster together in this node
// in the same slot, so we have to replace this node with a shortcut to
// skip over the identical parts of the key and then place a pair of
// nodes, one inside the other, at the end of the shortcut and
// distribute the keys between them.
//
// Firstly we need to work out where the leaves start diverging as a
// bit position into their keys so that we know how big the shortcut
// needs to be.
//
// We only need to make a single pass of N of the N+1 leaves because if
// any keys differ between themselves at bit X then at least one of
// them must also differ with the base key at bit X or before.
*__slate_slot_93 = 2147483647i32;
*__slate_slot_89 = 0;
'__join_34: {
while *__slate_slot_89 < 16 {
x = unsafe { (*ops).diff_objects }.unwrap()(assoc_array_ptr_to_leaf(*((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) as *const assoc_array_ptr) as *const (), index_key);
if x < *__slate_slot_93 {
loop {
if unlikely!(x < 0) != 0 {
break '__join_34;
} else {
if true {
break;
}
}
}
*__slate_slot_93 = x;
}
let __v27: i32 = *__slate_slot_89;
*__slate_slot_89 = __v27.wrapping_add(1);
}
'__join_27: {
loop {
if unlikely!(*__slate_slot_93 == 2147483647i32) != 0 {
break '__join_27;
} else {
if true {
break;
}
}
}
'__join_23: {
loop {
if unlikely!(*__slate_slot_93 < (*__slate_slot_94).wrapping_add(ilog2!(16i32))) != 0 {
break '__join_23;
} else {
if true {
break;
}
}
}
*__slate_slot_96 = round_up!(*__slate_slot_93, 64i32) as u64;
let __v25: u64 = *__slate_slot_96;
let __v24: u64 = __v25 >> ilog2!(64i32);
*__slate_slot_96 = __v24;
__count = *__slate_slot_96;
__obj_size = struct_size_t!(__count);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_16 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_17 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_110 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_110)
});
alloc_tag_restore!();
} else {
*__slate_slot_17 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_112 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_112)
});
}
__obj_ptr = *__slate_slot_17 as *mut assoc_array_shortcut;
if !__obj_ptr.is_null() {
let __v14: u64 = __count;
*core::ptr::addr_of_mut!(*{
__slate_compound_109 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_109)
}) = __v14;
}
*__slate_slot_105 = __obj_ptr;
if (*__slate_slot_105).is_null() {
return false;
} else {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(2 as isize) = assoc_array_shortcut_to_ptr(*__slate_slot_105 as *const assoc_array_shortcut);
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = assoc_array_shortcut_to_ptr(*__slate_slot_105 as *const assoc_array_shortcut);
(*(*__slate_slot_105)).back_pointer = (*(*__slate_slot_104)).back_pointer;
(*(*__slate_slot_105)).parent_slot = (*(*__slate_slot_104)).parent_slot as i32;
(*(*__slate_slot_105)).next_node = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
(*(*__slate_slot_103)).back_pointer = assoc_array_shortcut_to_ptr(*__slate_slot_105 as *const assoc_array_shortcut);
(*(*__slate_slot_103)).parent_slot = 0;
(*(*__slate_slot_102)).back_pointer = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
(*(*__slate_slot_102)).parent_slot = 1i32.wrapping_neg() as u8;
// Need to calculate this
let __v13: i32 = *__slate_slot_93 & !ilog2!(16i32).wrapping_sub(1);
*__slate_slot_94 = __v13;
(*(*__slate_slot_105)).skip_to_level = __v13;
'__join_11: {
loop {
if unlikely!(*__slate_slot_94 <= 0) != 0 {
break '__join_11;
} else {
if true {
break;
}
}
}
*__slate_slot_89 = 0;
while (*__slate_slot_89 as u64) < *__slate_slot_96 {
*(core::ptr::addr_of_mut!((*(*__slate_slot_105)).index_key) as *mut u64).wrapping_offset(*__slate_slot_89 as isize) = unsafe { (*ops).get_key_chunk }.unwrap()(index_key, (*__slate_slot_89).wrapping_mul(64));
let __v11: i32 = *__slate_slot_89;
*__slate_slot_89 = __v11.wrapping_add(1);
}
if *__slate_slot_94 & 63i32 != 0 {
*__slate_slot_97 = 18446744073709551615u64 << (*__slate_slot_94 & 63i32);
let __v8: *mut u64 = (core::ptr::addr_of_mut!((*(*__slate_slot_105)).index_key) as *mut u64).wrapping_offset((*__slate_slot_96).wrapping_sub(1) as isize);
let __v6: u64 = *__v8 & !(*__slate_slot_97);
*__v8 = __v6;
}
// This now reduces to a node splitting exercise for which we'll need
// to regenerate the disparity table.
*__slate_slot_89 = 0;
while *__slate_slot_89 < 16 {
*__slate_slot_100 = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
*__slate_slot_98 = unsafe { (*ops).get_object_key_chunk }.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_100 as *const assoc_array_ptr) as *const (), *__slate_slot_94);
*__slate_slot_98 >>= *__slate_slot_94 & 63i32;
*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) = (*__slate_slot_98 & 15u64) as u8;
let __v3: i32 = *__slate_slot_89;
*__slate_slot_89 = __v3.wrapping_add(1);
}
*__slate_slot_98 = unsafe { (*ops).get_key_chunk }.unwrap()(index_key, *__slate_slot_94);
*__slate_slot_98 >>= *__slate_slot_94 & 63i32;
*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) = (*__slate_slot_98 & 15u64) as u8;
break '__join_102;
}
core::arch::asm!("238: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 238b - ., 3; .popsection\n# {0}", const 238 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_107, sym __slate_asm_string_assoc_array_insert_into_terminal_node_108, const 761 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
core::arch::asm!("237: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 237b - ., 3; .popsection\n# {0}", const 237 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_113, sym __slate_asm_string_assoc_array_insert_into_terminal_node_114, const 740 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("236: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 236b - ., 3; .popsection\n# {0}", const 236 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_115, sym __slate_asm_string_assoc_array_insert_into_terminal_node_116, const 739 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("235: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 235b - ., 3; .popsection\n# {0}", const 235 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_117, sym __slate_asm_string_assoc_array_insert_into_terminal_node_118, const 735 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
// Otherwise all the old leaves cluster in the same slot, but
// the new leaf wants to go into a different slot - so we
// create a new node (n0) to hold the new leaf and a pointer to
// a new node (n1) holding all the old leaves.
//
// This can be done by falling through to the node splitting
// path.
}
}
// We need to split the current node.  The node must contain anything
// from a single leaf (in the one leaf case, this leaf will cluster
// with the new leaf) and the rest meta-pointers, to all leaves, some
// of which may cluster.
//
// It won't contain the case in which all the current leaves plus the
// new leaves want to cluster in the same slot.
//
// We need to expel at least two leaves out of a set consisting of the
// leaves in the node and the new leaf.  The current meta pointers can
// just be copied as they shouldn't cluster with any of the leaves.
//
// We need a new node (n0) to replace the current one and a new node to
// take the expelled nodes (n1).
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
(*(*__slate_slot_103)).back_pointer = (*(*__slate_slot_104)).back_pointer;
(*(*__slate_slot_103)).parent_slot = (*(*__slate_slot_104)).parent_slot;
(*(*__slate_slot_102)).back_pointer = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
(*(*__slate_slot_102)).parent_slot = 1i32.wrapping_neg() as u8;
// Need to calculate this
}
(*(*__slate_slot_103)).nr_leaves_on_branch = (*(*__slate_slot_104)).nr_leaves_on_branch;
(*(*__slate_slot_102)).nr_leaves_on_branch = 0;
// Begin by finding two matching leaves.  There have to be at least two
// that match - even if there are meta pointers - because any leaf that
// would match a slot with a meta pointer in it must be somewhere
// behind that meta pointer and cannot be here.  Further, given N
// remaining leaf slots, we now have N+1 leaves to go in them.
*__slate_slot_89 = 0;
'__loop_95: while *__slate_slot_89 < 16 {
*__slate_slot_92 = *((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) as i32;
if *__slate_slot_92 != 255 {
*__slate_slot_88 = (*__slate_slot_89).wrapping_add(1);
while *__slate_slot_88 < 17i32 {
if (*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_88 as isize) as i32) == *__slate_slot_92 {
break '__loop_95;
} else {
let __v53: i32 = *__slate_slot_88;
*__slate_slot_88 = __v53.wrapping_add(1);
}
}
}
let __v51: i32 = *__slate_slot_89;
*__slate_slot_89 = __v51.wrapping_add(1);
}
'__join_92: {
loop {
if unlikely!(*__slate_slot_89 >= 16) != 0 {
break '__join_92;
} else {
if true {
break;
}
}
}
'__join_88: {
loop {
if unlikely!(*__slate_slot_88 >= 17i32) != 0 {
break '__join_88;
} else {
if true {
break;
}
}
}
'__join_84: {
loop {
if unlikely!(*__slate_slot_92 >= 16) != 0 {
break '__join_84;
} else {
if true {
break;
}
}
}
(*(*__slate_slot_102)).parent_slot = *__slate_slot_92 as u8;
// Metadata pointers cannot change slot
*__slate_slot_89 = 0;
while *__slate_slot_89 < 16 {
if assoc_array_ptr_is_meta(*((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) as *const assoc_array_ptr) {
*((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
} else {
*((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) = core::ptr::null_mut::<assoc_array_ptr>();
}
let __v48: i32 = *__slate_slot_89;
*__slate_slot_89 = __v48.wrapping_add(1);
}
'__join_74: {
loop {
if unlikely!(!unsafe { *((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_92 as isize) }.is_null()) != 0 {
break '__join_74;
} else {
if true {
break;
}
}
}
*((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_92 as isize) = assoc_array_node_to_ptr(*__slate_slot_102 as *const assoc_array_node);
// Filter the leaf pointers between the new nodes
*__slate_slot_90 = 1i32.wrapping_neg();
*__slate_slot_91 = 0;
*__slate_slot_89 = 0;
while *__slate_slot_89 < 16 {
if !assoc_array_ptr_is_meta(*((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) as *const assoc_array_ptr) {
if (*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) as i32) == *__slate_slot_92 {
let __v44: i32 = *__slate_slot_91;
*__slate_slot_91 = __v44.wrapping_add(1);
*((*(*__slate_slot_102)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(__v44 as isize) = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
let __v42: *mut assoc_array_node = *__slate_slot_102;
let __v41: u64 = (*__v42).nr_leaves_on_branch;
(*__v42).nr_leaves_on_branch = __v41.wrapping_add(1);
} else {
loop {
let __v46: i32 = *__slate_slot_90;
*__slate_slot_90 = __v46.wrapping_add(1);
if unsafe { *((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_90 as isize) }.is_null() {
break;
}
}
*((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_90 as isize) = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
}
}
let __v39: i32 = *__slate_slot_89;
*__slate_slot_89 = __v39.wrapping_add(1);
}
if (*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) as i32) != *__slate_slot_92 {
loop {
let __v34: i32 = *__slate_slot_90;
*__slate_slot_90 = __v34.wrapping_add(1);
if unsafe { *((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_90 as isize) }.is_null() {
break;
}
}
(*edit).leaf_p = ((*(*__slate_slot_103)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_90 as isize);
(*edit).adjust_count_on = *__slate_slot_103;
} else {
let __v36: i32 = *__slate_slot_91;
*__slate_slot_91 = __v36.wrapping_add(1);
(*edit).leaf_p = ((*(*__slate_slot_102)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(__v36 as isize);
(*edit).adjust_count_on = *__slate_slot_102;
}
'__join_58: {
loop {
if unlikely!(*__slate_slot_91 <= 1) != 0 {
break '__join_58;
} else {
if true {
break;
}
}
}
(*edit).set_backpointers_to = assoc_array_node_to_ptr(*__slate_slot_103 as *const assoc_array_node);
*__slate_slot_89 = 0;
'__join_51: {
while *__slate_slot_89 < 16 {
if (*((*edit).segment_cache.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_89 as isize) as i32) == 255 {
*__slate_slot_100 = *((*(*__slate_slot_104)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize);
loop {
if unlikely!(assoc_array_ptr_is_leaf(*__slate_slot_100 as *const assoc_array_ptr)) != 0 {
break '__join_51;
} else {
if true {
break;
}
}
}
if assoc_array_ptr_is_node(*__slate_slot_100 as *const assoc_array_ptr) {
*__slate_slot_101 = assoc_array_ptr_to_node(*__slate_slot_100 as *const assoc_array_ptr);
*((*edit).set_backpointers.as_mut_ptr() as *mut *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) = core::ptr::addr_of_mut!((*(*__slate_slot_101)).back_pointer);
} else {
*__slate_slot_106 = assoc_array_ptr_to_shortcut(*__slate_slot_100 as *const assoc_array_ptr);
*((*edit).set_backpointers.as_mut_ptr() as *mut *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_89 as isize) = core::ptr::addr_of_mut!((*(*__slate_slot_106)).back_pointer);
}
}
let __v32: i32 = *__slate_slot_89;
*__slate_slot_89 = __v32.wrapping_add(1);
}
*__slate_slot_100 = (*(*__slate_slot_104)).back_pointer;
if (*__slate_slot_100).is_null() {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = core::ptr::addr_of_mut!((*(*edit).array).root);
} else {
if assoc_array_ptr_is_node(*__slate_slot_100 as *const assoc_array_ptr) {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = ((*assoc_array_ptr_to_node(*__slate_slot_100 as *const assoc_array_ptr)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(((*(*__slate_slot_104)).parent_slot as i32) as isize);
} else {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = core::ptr::addr_of_mut!((*assoc_array_ptr_to_shortcut(*__slate_slot_100 as *const assoc_array_ptr)).next_node);
}
}
*((*edit).excised_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_node_to_ptr(*__slate_slot_104 as *const assoc_array_node);
return true;
}
core::arch::asm!("234: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 234b - ., 3; .popsection\n# {0}", const 234 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_119, sym __slate_asm_string_assoc_array_insert_into_terminal_node_120, const 692 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("233: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 233b - ., 3; .popsection\n# {0}", const 233 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_121, sym __slate_asm_string_assoc_array_insert_into_terminal_node_122, const 686 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("232: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 232b - ., 3; .popsection\n# {0}", const 232 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_123, sym __slate_asm_string_assoc_array_insert_into_terminal_node_124, const 653 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("231: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 231b - ., 3; .popsection\n# {0}", const 231 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_125, sym __slate_asm_string_assoc_array_insert_into_terminal_node_126, const 643 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("230: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 230b - ., 3; .popsection\n# {0}", const 230 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_127, sym __slate_asm_string_assoc_array_insert_into_terminal_node_128, const 642 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("229: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 229b - ., 3; .popsection\n# {0}", const 229 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_into_terminal_node_129, sym __slate_asm_string_assoc_array_insert_into_terminal_node_130, const 641 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
}
unsafe { core::mem::zeroed() }
}

// Handle insertion into the middle of a shortcut.
fn assoc_array_insert_mid_shortcut(edit: *mut assoc_array_edit, ops: *const assoc_array_ops, result: &assoc_array_walk_result) -> bool {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: u64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_44: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_46: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_49: u64 = unsafe { core::mem::zeroed() };
let mut shortcut: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut new_s0: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut new_s1: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut side: *mut assoc_array_node = unsafe { core::mem::zeroed() };
let mut sc_segments: u64 = 0;
let mut dissimilarity: u64 = 0;
let mut blank: u64 = 0;
let mut keylen: u64 = 0;
let mut level: i32 = 0;
let mut sc_level: i32 = 0;
let mut diff: i32 = 0;
let mut sc_slot: i32 = 0;
shortcut = unsafe { (*(result as *const assoc_array_walk_result)).wrong_shortcut.shortcut };
level = unsafe { (*(result as *const assoc_array_walk_result)).wrong_shortcut.level };
sc_level = unsafe { (*(result as *const assoc_array_walk_result)).wrong_shortcut.sc_level };
sc_segments = unsafe { (*(result as *const assoc_array_walk_result)).wrong_shortcut.sc_segments };
dissimilarity = unsafe { (*(result as *const assoc_array_walk_result)).wrong_shortcut.dissimilarity };
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017-->%s(ix=%d dis=%lx scix=%d)\n".as_ptr() as *const u8, c"assoc_array_insert_mid_shortcut".as_ptr() as *mut u8, level, dissimilarity, sc_level) };
}
__v0 = 0;
// We need to split a shortcut and insert a node between the two
// pieces.  Zero-length pieces will be dispensed with entirely.
//
// First of all, we need to find out in which level the first
// difference was.
diff = __ffs!(dissimilarity) as i32;
let __v1: i32 = diff;
diff = __v1 & !ilog2!(16i32).wrapping_sub(1);
let __v4: i32 = diff.wrapping_add(sc_level & -64i32);
diff = __v4;
let mut __v5: i32;
if false {
unsafe { _printk(c"\x017diff=%d\n".as_ptr() as *const u8, diff) };
}
__v5 = 0;
if unsafe { (*shortcut).back_pointer }.is_null() {
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = core::ptr::addr_of_mut!((*(*edit).array).root);
}
} else {
if assoc_array_ptr_is_node((unsafe { (*shortcut).back_pointer }) as *const assoc_array_ptr) {
let node: *mut assoc_array_node = assoc_array_ptr_to_node((unsafe { (*shortcut).back_pointer }) as *const assoc_array_ptr);
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = ((*node).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset((*shortcut).parent_slot as isize);
}
} else {
loop {
unsafe {
core::arch::asm!("239: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 239b - ., 3; .popsection\n# {0}", const 239 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_mid_shortcut_6, sym __slate_asm_string_assoc_array_insert_mid_shortcut_7, const 830 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
}
unsafe {
*((*edit).excised_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_shortcut_to_ptr(shortcut as *const assoc_array_shortcut);
}
// Create a new node now since we're going to need it anyway
let mut __v10: *mut assoc_array_node;
{
let __obj_size: u64 = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let mut __v11: *mut ();
let mut __v12: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
__v12 = _res;
__v11 = __v12;
__v10 = __v11 as *mut assoc_array_node;
}
let new_n0: *mut assoc_array_node = __v10;
if new_n0.is_null() {
return false;
}
unsafe {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
unsafe {
(*edit).adjust_count_on = new_n0;
}
// Insert a new shortcut before the new node if this segment isn't of
// zero length - otherwise we just connect the new node directly to the
// parent.
let __v16: i32 = level;
level = __v16.wrapping_add(ilog2!(16i32));
if diff > level {
let mut __v18: i32;
if false {
unsafe { _printk(c"\x017pre-shortcut %d...%d\n".as_ptr() as *const u8, level, diff) };
}
__v18 = 0;
keylen = round_up!(diff, 64i32) as u64;
let __v19: u64 = keylen;
keylen = __v19 >> ilog2!(64i32);
let mut __v21: *mut assoc_array_shortcut;
let __count: u64 = keylen;
let __obj_size: u64 = struct_size_t__4!(__count);
let mut __v22: *mut ();
let mut __v23: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_24 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_24)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
} });
}
__v23 = _res;
__v22 = __v23;
let __obj_ptr: *mut assoc_array_shortcut = __v22 as *mut assoc_array_shortcut;
if !__obj_ptr.is_null() {
let mut __v27: u64;
let __v28: u64 = __count;
unsafe {
*core::ptr::addr_of_mut!(*{
__slate_compound_29 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_29)
}) = __v28;
}
__v27 = __v28;
}
__v21 = __obj_ptr;
new_s0 = __v21;
if new_s0.is_null() {
return false;
}
unsafe {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(1 as isize) = assoc_array_shortcut_to_ptr(new_s0 as *const assoc_array_shortcut);
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = assoc_array_shortcut_to_ptr(new_s0 as *const assoc_array_shortcut);
}
unsafe {
(*new_s0).back_pointer = (*shortcut).back_pointer;
}
unsafe {
(*new_s0).parent_slot = (*shortcut).parent_slot;
}
unsafe {
(*new_s0).next_node = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
unsafe {
(*new_s0).skip_to_level = diff;
}
unsafe {
(*new_n0).back_pointer = assoc_array_shortcut_to_ptr(new_s0 as *const assoc_array_shortcut);
}
unsafe {
(*new_n0).parent_slot = 0;
}
unsafe { memcpy((core::ptr::addr_of_mut!((*new_s0).index_key) as *mut u64) as *mut (), (core::ptr::addr_of_mut!((*shortcut).index_key) as *mut u64) as *const (), flex_array_size__6!(keylen)) };
blank = 18446744073709551615u64 << (diff & 63i32);
let mut __v30: i32;
if false {
unsafe { _printk(c"\x017blank off [%zu] %d: %lx\n".as_ptr() as *const u8, keylen.wrapping_sub(1), diff, blank) };
}
__v30 = 0;
let __v31: *mut u64 = unsafe { core::ptr::addr_of_mut!((*new_s0).index_key) as *mut u64 }.wrapping_offset(keylen.wrapping_sub(1) as isize);
let __v33: u64 = (unsafe { *__v31 }) & !blank;
unsafe {
*__v31 = __v33;
}
} else {
let mut __v34: i32;
if false {
unsafe { _printk(c"\x017no pre-shortcut\n".as_ptr() as *const u8) };
}
__v34 = 0;
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
unsafe {
(*new_n0).back_pointer = (*shortcut).back_pointer;
}
unsafe {
(*new_n0).parent_slot = (*shortcut).parent_slot as u8;
}
}
side = assoc_array_ptr_to_node((unsafe { (*shortcut).next_node }) as *const assoc_array_ptr);
unsafe {
(*new_n0).nr_leaves_on_branch = (*side).nr_leaves_on_branch;
}
// We need to know which slot in the new node is going to take a
// metadata pointer.
sc_slot = (sc_segments >> (diff & 63i32)) as i32;
sc_slot &= 15i32;
let mut __v37: i32;
if false {
unsafe { _printk(c"\x017new slot %lx >> %d -> %d\n".as_ptr() as *const u8, sc_segments, diff & 63i32, sc_slot) };
}
__v37 = 0;
// Determine whether we need to follow the new node with a replacement
// for the current shortcut.  We could in theory reuse the current
// shortcut if its parent slot number doesn't change - but that's a
// 1-in-16 chance so not worth expending the code upon.
level = diff.wrapping_add(ilog2!(16i32));
if level < unsafe { (*shortcut).skip_to_level } {
let mut __v38: i32;
if false {
unsafe { _printk(c"\x017post-shortcut %d...%d\n".as_ptr() as *const u8, level, (*shortcut).skip_to_level) };
}
__v38 = 0;
keylen = round_up!(unsafe { (*shortcut).skip_to_level }, 64i32) as u64;
let __v39: u64 = keylen;
keylen = __v39 >> ilog2!(64i32);
let mut __v41: *mut assoc_array_shortcut;
let __count: u64 = keylen;
let __obj_size: u64 = struct_size_t__3!(__count);
let mut __v42: *mut ();
let mut __v43: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_44 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_44)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_46 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_46)
} });
}
__v43 = _res;
__v42 = __v43;
let __obj_ptr: *mut assoc_array_shortcut = __v42 as *mut assoc_array_shortcut;
if !__obj_ptr.is_null() {
let mut __v47: u64;
let __v48: u64 = __count;
unsafe {
*core::ptr::addr_of_mut!(*{
__slate_compound_49 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_49)
}) = __v48;
}
__v47 = __v48;
}
__v41 = __obj_ptr;
new_s1 = __v41;
if new_s1.is_null() {
return false;
}
unsafe {
*((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(2 as isize) = assoc_array_shortcut_to_ptr(new_s1 as *const assoc_array_shortcut);
}
unsafe {
(*new_s1).back_pointer = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
unsafe {
(*new_s1).parent_slot = sc_slot;
}
unsafe {
(*new_s1).next_node = (*shortcut).next_node;
}
unsafe {
(*new_s1).skip_to_level = (*shortcut).skip_to_level;
}
unsafe {
*((*new_n0).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(sc_slot as isize) = assoc_array_shortcut_to_ptr(new_s1 as *const assoc_array_shortcut);
}
unsafe { memcpy((core::ptr::addr_of_mut!((*new_s1).index_key) as *mut u64) as *mut (), (core::ptr::addr_of_mut!((*shortcut).index_key) as *mut u64) as *const (), flex_array_size!(keylen)) };
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*side).back_pointer);
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).to = assoc_array_shortcut_to_ptr(new_s1 as *const assoc_array_shortcut);
}
} else {
let mut __v50: i32;
if false {
unsafe { _printk(c"\x017no post-shortcut\n".as_ptr() as *const u8) };
}
__v50 = 0;
// We don't have to replace the pointed-to node as long as we
// use memory barriers to make sure the parent slot number is
// changed before the back pointer (the parent slot number is
// irrelevant to the old parent shortcut).
unsafe {
*((*new_n0).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(sc_slot as isize) = (*shortcut).next_node;
}
unsafe {
(*((*edit).set_parent_slot.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_93).wrapping_offset(0 as isize)).p = core::ptr::addr_of_mut!((*side).parent_slot);
}
unsafe {
(*((*edit).set_parent_slot.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_93).wrapping_offset(0 as isize)).to = sc_slot as u8;
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*side).back_pointer);
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).to = assoc_array_node_to_ptr(new_n0 as *const assoc_array_node);
}
}
// Install the new leaf in a spare slot in the new node.
if sc_slot == 0 {
unsafe {
(*edit).leaf_p = ((*new_n0).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(1 as isize);
}
} else {
unsafe {
(*edit).leaf_p = ((*new_n0).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize);
}
}
let mut __v51: i32;
if false {
unsafe { _printk(c"\x017<--%s() = ok [split shortcut]\n".as_ptr() as *const u8, c"assoc_array_insert_mid_shortcut".as_ptr() as *mut u8) };
}
__v51 = 0;
true
}

/// assoc_array_insert - Script insertion of an object into an associative array
/// @array: The array to insert into.
/// @ops: The operations to use.
/// @index_key: The key to insert at.
/// @object: The object to insert.
///
/// Precalculate and preallocate a script for the insertion or replacement of an
/// object in an associative array.  This results in an edit script that can
/// either be applied or cancelled.
///
/// The function returns a pointer to an edit script or -ENOMEM.
///
/// The caller should lock against other modifications and must continue to hold
/// the lock until assoc_array_apply_edit() has been called.
///
/// Accesses to the tree may take place concurrently with this function,
/// provided they hold the RCU read lock.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_insert(array: *mut assoc_array, ops: *const assoc_array_ops, index_key: *const (), object: *mut ()) -> *mut assoc_array_edit {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut edit: core::mem::MaybeUninit<*mut assoc_array_edit> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut assoc_array_edit = core::ptr::addr_of_mut!(edit) as *mut *mut assoc_array_edit;
let mut result: core::mem::MaybeUninit<assoc_array_walk_result> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut assoc_array_walk_result = core::ptr::addr_of_mut!(result) as *mut assoc_array_walk_result;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 48usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
// The leaf pointer we're given must not have the bottom bit set as we
// use those for type-marking the pointer.  NULL pointers are also not
// allowed as they indicate an empty slot but we have to allow them
// here as they can be updated later.
'__join_15: {
loop {
if unlikely!(assoc_array_ptr_is_meta(object as *const assoc_array_ptr)) != 0 {
break '__join_15;
} else {
if true {
break;
}
}
}
__obj_size = size_mul(core::mem::size_of::<assoc_array_edit>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
alloc_tag_restore!();
} else {
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
}
*__slate_slot_8 = *__slate_slot_2 as *mut assoc_array_edit;
if (*__slate_slot_8).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut assoc_array_edit;
} else {
(*(*__slate_slot_8)).array = array;
(*(*__slate_slot_8)).ops = ops;
(*(*__slate_slot_8)).leaf = assoc_array_leaf_to_ptr(object as *const ());
(*(*__slate_slot_8)).adjust_count_by = 1;
let __t0: u32 = assoc_array_walk(array as *const assoc_array, ops, index_key, &mut (*core::ptr::addr_of_mut!(*__slate_slot_9))) as u32;
if __t0 == 0 {
if assoc_array_insert_in_empty_tree(*__slate_slot_8) {
// Allocate a root node if there isn't one yet
return *__slate_slot_8;
}
} else {
if __t0 == 1 {
if assoc_array_insert_into_terminal_node(*__slate_slot_8, ops, index_key, &(*core::ptr::addr_of_mut!(*__slate_slot_9))) {
// We found a node that doesn't have a node/shortcut pointer in
// the slot corresponding to the index key that we have to
// follow.
return *__slate_slot_8;
}
} else {
if __t0 == 2 {
if assoc_array_insert_mid_shortcut(*__slate_slot_8, ops, &(*core::ptr::addr_of_mut!(*__slate_slot_9))) {
// We found a shortcut that didn't match our key in a slot we
// needed to follow.
return *__slate_slot_8;
}
}
}
}
// Clean up after an out of memory error
assoc_array_cancel_edit(*__slate_slot_8);
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut assoc_array_edit;
}
}
core::arch::asm!("240: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 240b - ., 3; .popsection\n# {0}", const 240 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_13, sym __slate_asm_string_assoc_array_insert_14, const 977 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

/// assoc_array_insert_set_object - Set the new object pointer in an edit script
/// @edit: The edit script to modify.
/// @object: The object pointer to set.
///
/// Change the object to be inserted in an edit script.  The object pointed to
/// by the old object is not freed.  This must be done prior to applying the
/// script.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_insert_set_object(edit: *mut assoc_array_edit, object: *mut ()) {
loop {
if unlikely!(object.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("241: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 241b - ., 3; .popsection\n# {0}", const 241 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_insert_set_object_0, sym __slate_asm_string_assoc_array_insert_set_object_1, const 1031 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
unsafe {
(*edit).leaf = assoc_array_leaf_to_ptr(object as *const ());
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_delete_collapse_context {
    node: *mut assoc_array_node,
    skip_leaf: *const (),
    slot: i32,
}


/// Subtree collapse to node iterator.
#[unsafe(link_section = ".text.slate_distinct.assoc_array.assoc_array_delete_collapse_iterator")]
extern "C" fn assoc_array_delete_collapse_iterator(leaf: *const (), iterator_data: *mut ()) -> i32 {
let collapse: *mut assoc_array_delete_collapse_context = iterator_data as *mut assoc_array_delete_collapse_context;
if leaf == unsafe { (*collapse).skip_leaf } {
return 0;
}
loop {
if unlikely!((unsafe { (*collapse).slot }) >= 16) != 0 {
loop {
unsafe {
core::arch::asm!("242: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 242b - ., 3; .popsection\n# {0}", const 242 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_delete_collapse_iterator_0, sym __slate_asm_string_assoc_array_delete_collapse_iterator_1, const 1052 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
let __v6: i32 = unsafe { (*collapse).slot };
unsafe {
(*collapse).slot = __v6.wrapping_add(1);
}
unsafe {
*((*(*collapse).node).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(__v6 as isize) = assoc_array_leaf_to_ptr(leaf);
}
0
}

/// assoc_array_delete - Script deletion of an object from an associative array
/// @array: The array to search.
/// @ops: The operations to use.
/// @index_key: The key to the object.
///
/// Precalculate and preallocate a script for the deletion of an object from an
/// associative array.  This results in an edit script that can either be
/// applied or cancelled.
///
/// The function returns a pointer to an edit script if the object was found,
/// NULL if the object was not found or -ENOMEM.
///
/// The caller should lock against other modifications and must continue to hold
/// the lock until assoc_array_apply_edit() has been called.
///
/// Accesses to the tree may take place concurrently with this function,
/// provided they hold the RCU read lock.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_delete(array: *mut assoc_array, ops: *const assoc_array_ops, index_key: *const ()) -> *mut assoc_array_edit {
let mut __slate_compound_45: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_47: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_50: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_52: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut s: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ptr: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *mut assoc_array_ptr;
let mut grandparent: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(grandparent) as *mut *mut assoc_array_node;
let mut parent: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(parent) as *mut *mut assoc_array_node;
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut __v23: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut i32 = core::ptr::addr_of_mut!(slot) as *mut i32;
let mut has_meta: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut bool = core::ptr::addr_of_mut!(has_meta) as *mut bool;
let mut ptr: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *mut assoc_array_ptr;
let mut edit: core::mem::MaybeUninit<*mut assoc_array_edit> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut assoc_array_edit = core::ptr::addr_of_mut!(edit) as *mut *mut assoc_array_edit;
let mut new_n0: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(new_n0) as *mut *mut assoc_array_node;
let mut node: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *mut assoc_array_node;
let mut result: core::mem::MaybeUninit<assoc_array_walk_result> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut assoc_array_walk_result = core::ptr::addr_of_mut!(result) as *mut assoc_array_walk_result;
let mut collapse: core::mem::MaybeUninit<assoc_array_delete_collapse_context> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut assoc_array_delete_collapse_context = core::ptr::addr_of_mut!(collapse) as *mut assoc_array_delete_collapse_context;
unsafe {
let __slate_auto_bytes_5 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 24usize);
let __slate_auto_bytes_6 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 48usize);
let __slate_auto_bytes_7 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_12 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
__obj_size_1 = size_mul(core::mem::size_of::<assoc_array_edit>() as u64, 1u64);
let __slate_auto_bytes_14 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_25 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_26 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_50 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_50)
});
alloc_tag_restore!();
} else {
*__slate_slot_26 = _kzalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_52 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_52)
});
}
*__slate_slot_36 = *__slate_slot_26 as *mut assoc_array_edit;
if (*__slate_slot_36).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut assoc_array_edit;
} else {
'__join_41: {
(*(*__slate_slot_36)).array = array;
(*(*__slate_slot_36)).ops = ops;
(*(*__slate_slot_36)).adjust_count_by = 1i32.wrapping_neg() as i64;
let __t0: u32 = assoc_array_walk(array as *const assoc_array, ops, index_key, &mut (*core::ptr::addr_of_mut!(*__slate_slot_39))) as u32;
if __t0 == 1 {
// We found a node that should contain the leaf we've been
// asked to remove - *if* it's in the tree.
*__slate_slot_38 = (*__slate_slot_39).terminal_node.node;
*__slate_slot_33 = 0;
'__join_40: {
while *__slate_slot_33 < 16 {
*__slate_slot_35 = *((*(*__slate_slot_38)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_33 as isize);
if !(*__slate_slot_35).is_null() {
__v23 = assoc_array_ptr_is_leaf(*__slate_slot_35 as *const assoc_array_ptr);
} else {
__v23 = false;
}
if __v23 {
__v22 = unsafe { (*ops).compare_object }.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_35 as *const assoc_array_ptr) as *const (), index_key);
} else {
__v22 = false;
}
if __v22 {
break '__join_40;
} else {
let __v21: i32 = *__slate_slot_33;
*__slate_slot_33 = __v21.wrapping_add(1);
}
}
break '__join_41;
}
'__join_39: {
loop {
if unlikely!((*array).nr_leaves_on_tree <= 0) != 0 {
break '__join_39;
} else {
if true {
break;
}
}
}
// In the simplest form of deletion we just clear the slot and release
// the leaf after a suitable interval.
(*(*__slate_slot_36)).dead_leaf = *((*(*__slate_slot_38)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_33 as isize);
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = ((*(*__slate_slot_38)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_33 as isize);
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = core::ptr::null_mut::<assoc_array_ptr>();
(*(*__slate_slot_36)).adjust_count_on = *__slate_slot_38;
// If that concludes erasure of the last leaf, then delete the entire
// internal array.
if (*array).nr_leaves_on_tree == 1 {
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*array).root);
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).to = core::ptr::null_mut::<assoc_array_ptr>();
(*(*__slate_slot_36)).adjust_count_on = core::ptr::null_mut::<assoc_array_node>();
(*(*__slate_slot_36)).excised_subtree = (*array).root;
return *__slate_slot_36;
} else {
// However, we'd also like to clear up some metadata blocks if we
// possibly can.
//
// We go for a simple algorithm of: if this node has FAN_OUT or fewer
// leaves in it, then attempt to collapse it - and attempt to
// recursively collapse up the tree.
//
// We could also try and collapse in partially filled subtrees to take
// up space in this node.
if (*(*__slate_slot_38)).nr_leaves_on_branch <= 17u64 {
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
// First of all, we need to know if this node has metadata so
// that we don't try collapsing if all the leaves are already
// here.
*__slate_slot_34 = false;
*__slate_slot_32 = 0;
'__join_28: {
loop {
if *__slate_slot_32 < 16 {
*__slate_slot_15 = *((*(*__slate_slot_38)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_32 as isize);
if assoc_array_ptr_is_meta(*__slate_slot_15 as *const assoc_array_ptr) {
break;
} else {
let __v14: i32 = *__slate_slot_32;
*__slate_slot_32 = __v14.wrapping_add(1);
}
} else {
break '__join_28;
}
}
*__slate_slot_34 = true;
}
// Look further up the tree to see if we can collapse this node
// into a more proximal node too.
*__slate_slot_17 = *__slate_slot_38;
loop {
*__slate_slot_15 = (*(*__slate_slot_17)).back_pointer;
if (*__slate_slot_15).is_null() {
break;
} else {
if assoc_array_ptr_is_shortcut(*__slate_slot_15 as *const assoc_array_ptr) {
let s: *mut assoc_array_shortcut = assoc_array_ptr_to_shortcut(*__slate_slot_15 as *const assoc_array_ptr);
*__slate_slot_15 = (*s).back_pointer;
if (*__slate_slot_15).is_null() {
break;
}
}
*__slate_slot_16 = assoc_array_ptr_to_node(*__slate_slot_15 as *const assoc_array_ptr);
if (*(*__slate_slot_16)).nr_leaves_on_branch <= 17u64 {
*__slate_slot_17 = *__slate_slot_16;
} else {
break;
}
}
}
'__join_2: {
if *__slate_slot_34 || *__slate_slot_17 != *__slate_slot_38 {
*__slate_slot_38 = *__slate_slot_17;
__obj_size = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_45 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_45)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_47 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_47)
});
}
let __v7: *mut () = *__slate_slot_5;
// Create a new node to collapse into
*__slate_slot_37 = __v7 as *mut assoc_array_node;
if (*__slate_slot_37).is_null() {
// Clean up after an out of memory error
assoc_array_cancel_edit(*__slate_slot_36);
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut assoc_array_edit;
} else {
*((*(*__slate_slot_36)).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(0 as isize) = assoc_array_node_to_ptr(*__slate_slot_37 as *const assoc_array_node);
(*(*__slate_slot_37)).back_pointer = (*(*__slate_slot_38)).back_pointer;
(*(*__slate_slot_37)).parent_slot = (*(*__slate_slot_38)).parent_slot;
(*(*__slate_slot_37)).nr_leaves_on_branch = (*(*__slate_slot_38)).nr_leaves_on_branch;
(*(*__slate_slot_36)).adjust_count_on = *__slate_slot_37;
(*__slate_slot_40).node = *__slate_slot_37;
(*__slate_slot_40).skip_leaf = assoc_array_ptr_to_leaf((*(*__slate_slot_36)).dead_leaf as *const assoc_array_ptr) as *const ();
(*__slate_slot_40).slot = 0;
assoc_array_subtree_iterate(assoc_array_node_to_ptr(*__slate_slot_38 as *const assoc_array_node) as *const assoc_array_ptr, (*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr, Some(assoc_array_delete_collapse_iterator), core::ptr::addr_of_mut!(*__slate_slot_40) as *mut ());
'__join_15: {
loop {
if unlikely!(((*__slate_slot_40).slot as u64) != unsafe { (*(*__slate_slot_37)).nr_leaves_on_branch }.wrapping_sub(1)) != 0 {
break '__join_15;
} else {
if true {
break;
}
}
}
if unsafe { (*(*__slate_slot_38)).back_pointer }.is_null() {
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*array).root);
} else {
if assoc_array_ptr_is_leaf((*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr) {
core::arch::asm!("245: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 245b - ., 3; .popsection\n# {0}", const 245 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_delete_41, sym __slate_asm_string_assoc_array_delete_42, const 1231 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
} else {
if assoc_array_ptr_is_node((*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr) {
let p: *mut assoc_array_node = assoc_array_ptr_to_node((*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr);
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = ((*p).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(((*(*__slate_slot_38)).parent_slot as i32) as isize);
} else {
if assoc_array_ptr_is_shortcut((*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr) {
s = assoc_array_ptr_to_shortcut((*(*__slate_slot_38)).back_pointer as *const assoc_array_ptr);
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*s).next_node);
}
}
}
}
(*((*(*__slate_slot_36)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).to = assoc_array_node_to_ptr(*__slate_slot_37 as *const assoc_array_node);
(*(*__slate_slot_36)).excised_subtree = assoc_array_node_to_ptr(*__slate_slot_38 as *const assoc_array_node);
break '__join_2;
}
core::arch::asm!("244: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 244b - ., 3; .popsection\n# {0}", const 244 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_delete_43, sym __slate_asm_string_assoc_array_delete_44, const 1226 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
// There's no point collapsing if the original node has no meta
// pointers to discard and if we didn't merge into one of that
// node's ancestry.
}
return *__slate_slot_36;
}
}
core::arch::asm!("243: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 243b - ., 3; .popsection\n# {0}", const 243 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_delete_48, sym __slate_asm_string_assoc_array_delete_49, const 1124 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
assoc_array_cancel_edit(*__slate_slot_36);
return core::ptr::null_mut::<assoc_array_edit>();
}
}
}

/// assoc_array_clear - Script deletion of all objects from an associative array
/// @array: The array to clear.
/// @ops: The operations to use.
///
/// Precalculate and preallocate a script for the deletion of all the objects
/// from an associative array.  This results in an edit script that can either
/// be applied or cancelled.
///
/// The function returns a pointer to an edit script if there are objects to be
/// deleted, NULL if there are no objects in the array or -ENOMEM.
///
/// The caller should lock against other modifications and must continue to hold
/// the lock until assoc_array_apply_edit() has been called.
///
/// Accesses to the tree may take place concurrently with this function,
/// provided they hold the RCU read lock.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_clear(array: *mut assoc_array, ops: *const assoc_array_ops) -> *mut assoc_array_edit {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017-->%s()\n".as_ptr() as *const u8, c"assoc_array_clear".as_ptr() as *mut u8) };
}
__v0 = 0;
if unsafe { (*array).root }.is_null() {
return core::ptr::null_mut::<assoc_array_edit>();
}
let mut __v1: *mut assoc_array_edit;
let __obj_size: u64 = size_mul(core::mem::size_of::<assoc_array_edit>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut assoc_array_edit;
let edit: *mut assoc_array_edit = __v1;
if edit.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut assoc_array_edit;
}
unsafe {
(*edit).array = array;
}
unsafe {
(*edit).ops = ops;
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).ptr = core::ptr::addr_of_mut!((*array).root);
}
unsafe {
(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(1 as isize)).to = core::ptr::null_mut::<assoc_array_ptr>();
}
unsafe {
(*edit).excised_subtree = (*array).root;
}
unsafe {
(*edit).ops_for_excised_subtree = ops;
}
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017all gone\n".as_ptr() as *const u8) };
}
__v7 = 0;
edit
}

/// Handle the deferred destruction after an applied edit.
#[unsafe(link_section = ".text.slate_distinct.assoc_array.assoc_array_rcu_cleanup")]
extern "C" fn assoc_array_rcu_cleanup(head: *mut callback_head) {
let edit: *mut assoc_array_edit = container_of__2!(head, assoc_array_edit);
let mut i: i32 = 0;
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017-->%s()\n".as_ptr() as *const u8, c"assoc_array_rcu_cleanup".as_ptr() as *mut u8) };
}
__v1 = 0;
if !unsafe { (*edit).dead_leaf }.is_null() {
unsafe { unsafe { (*(*edit).ops).free_object }.unwrap()(assoc_array_ptr_to_leaf((*edit).dead_leaf as *const assoc_array_ptr)) };
}
while (i as u64) < ARRAY_SIZE!() {
if !unsafe { *((*edit).excised_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(i as isize) }.is_null() {
unsafe { kfree(assoc_array_ptr_to_node(*((*edit).excised_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(i as isize) as *const assoc_array_ptr) as *const ()) };
}
i = i.wrapping_add(1);
}
if !unsafe { (*edit).excised_subtree }.is_null() {
loop {
if unlikely!(assoc_array_ptr_is_leaf((unsafe { (*edit).excised_subtree }) as *const assoc_array_ptr)) != 0 {
loop {
unsafe {
core::arch::asm!("246: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 246b - ., 3; .popsection\n# {0}", const 246 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_rcu_cleanup_5, sym __slate_asm_string_assoc_array_rcu_cleanup_6, const 1314 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
if assoc_array_ptr_is_node((unsafe { (*edit).excised_subtree }) as *const assoc_array_ptr) {
let n: *mut assoc_array_node = assoc_array_ptr_to_node((unsafe { (*edit).excised_subtree }) as *const assoc_array_ptr);
unsafe {
(*n).back_pointer = core::ptr::null_mut::<assoc_array_ptr>();
}
} else {
let s: *mut assoc_array_shortcut = assoc_array_ptr_to_shortcut((unsafe { (*edit).excised_subtree }) as *const assoc_array_ptr);
unsafe {
(*s).back_pointer = core::ptr::null_mut::<assoc_array_ptr>();
}
}
assoc_array_destroy_subtree(unsafe { (*edit).excised_subtree }, unsafe { (*edit).ops_for_excised_subtree });
}
unsafe { kfree(edit as *const ()) };
}

/// assoc_array_apply_edit - Apply an edit script to an associative array
/// @edit: The script to apply.
///
/// Apply an edit script to an associative array to effect an insertion,
/// deletion or clearance.  As the edit script includes preallocated memory,
/// this is guaranteed not to fail.
///
/// The edit script, dead objects and dead metadata will be scheduled for
/// destruction after an RCU grace period to permit those doing read-only
/// accesses on the array to continue to do so under the RCU read lock whilst
/// the edit is taking place.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_apply_edit(edit: *mut assoc_array_edit) {
let mut node: *mut assoc_array_node = unsafe { core::mem::zeroed() };
let mut ptr: *mut assoc_array_ptr = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017-->%s()\n".as_ptr() as *const u8, c"assoc_array_apply_edit".as_ptr() as *mut u8) };
}
__v0 = 0;
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if !unsafe { (*edit).leaf_p }.is_null() {
unsafe {
*(*edit).leaf_p = (*edit).leaf;
}
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
while (i as u64) < ARRAY_SIZE__16!() {
if !unsafe { (*((*edit).set_parent_slot.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_93).wrapping_offset(i as isize)).p }.is_null() {
unsafe {
*(*((*edit).set_parent_slot.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_93).wrapping_offset(i as isize)).p = (*((*edit).set_parent_slot.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_93).wrapping_offset(i as isize)).to;
}
}
i = i.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
i = 0;
while (i as u64) < ARRAY_SIZE__5!() {
if !unsafe { *((*edit).set_backpointers.as_mut_ptr() as *mut *mut *mut assoc_array_ptr).wrapping_offset(i as isize) }.is_null() {
unsafe {
*(*((*edit).set_backpointers.as_mut_ptr() as *mut *mut *mut assoc_array_ptr).wrapping_offset(i as isize)) = (*edit).set_backpointers_to;
}
}
i = i.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
i = 0;
while (i as u64) < ARRAY_SIZE__13!() {
if !unsafe { (*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(i as isize)).ptr }.is_null() {
unsafe {
*(*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(i as isize)).ptr = (*((*edit).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(i as isize)).to;
}
}
i = i.wrapping_add(1);
}
if unsafe { (*(*edit).array).root }.is_null() {
unsafe {
(*(*edit).array).nr_leaves_on_tree = 0;
}
} else {
if !unsafe { (*edit).adjust_count_on }.is_null() {
node = unsafe { (*edit).adjust_count_on };
loop {
let __v19: u64 = unsafe { (*node).nr_leaves_on_branch };
unsafe {
(*node).nr_leaves_on_branch = __v19.wrapping_add((*edit).adjust_count_by as u64);
}
ptr = unsafe { (*node).back_pointer };
if ptr.is_null() {
break;
}
if assoc_array_ptr_is_shortcut(ptr as *const assoc_array_ptr) {
let shortcut: *mut assoc_array_shortcut = assoc_array_ptr_to_shortcut(ptr as *const assoc_array_ptr);
ptr = unsafe { (*shortcut).back_pointer };
if ptr.is_null() {
break;
}
}
loop {
if unlikely!(!assoc_array_ptr_is_node(ptr as *const assoc_array_ptr)) != 0 {
loop {
unsafe {
core::arch::asm!("247: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 247b - ., 3; .popsection\n# {0}", const 247 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_apply_edit_22, sym __slate_asm_string_assoc_array_apply_edit_23, const 1388 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
node = assoc_array_ptr_to_node(ptr as *const assoc_array_ptr);
}
let __v27: *mut assoc_array = unsafe { (*edit).array };
let __v28: u64 = unsafe { (*__v27).nr_leaves_on_tree };
unsafe {
(*__v27).nr_leaves_on_tree = __v28.wrapping_add((*edit).adjust_count_by as u64);
}
}
}
unsafe { call_rcu(core::ptr::addr_of_mut!((*edit).rcu), Some(assoc_array_rcu_cleanup)) };
}

/// assoc_array_cancel_edit - Discard an edit script.
/// @edit: The script to discard.
///
/// Free an edit script and all the preallocated data it holds without making
/// any changes to the associative array it was intended for.
///
/// NOTE!  In the case of an insertion script, this does _not_ release the leaf
/// that was to be inserted.  That is left to the caller.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_cancel_edit(edit: *mut assoc_array_edit) {
// Clean up after an out of memory error
let mut i: i32 = 0;
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017-->%s()\n".as_ptr() as *const u8, c"assoc_array_cancel_edit".as_ptr() as *mut u8) };
}
__v0 = 0;
while (i as u64) < ARRAY_SIZE__12!() {
let ptr: *mut assoc_array_ptr = unsafe { *((*edit).new_meta.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(i as isize) };
if !ptr.is_null() {
if assoc_array_ptr_is_node(ptr as *const assoc_array_ptr) {
unsafe { kfree(assoc_array_ptr_to_node(ptr as *const assoc_array_ptr) as *const ()) };
} else {
unsafe { kfree(assoc_array_ptr_to_shortcut(ptr as *const assoc_array_ptr) as *const ()) };
}
}
i = i.wrapping_add(1);
}
unsafe { kfree(edit as *const ()) };
}

/// assoc_array_gc - Garbage collect an associative array.
/// @array: The array to clean.
/// @ops: The operations to use.
/// @iterator: A callback function to pass judgement on each object.
/// @iterator_data: Private data for the callback function.
///
/// Collect garbage from an associative array and pack down the internal tree to
/// save memory.
///
/// The iterator function is asked to pass judgement upon each object in the
/// array.  If it returns false, the object is discard and if it returns true,
/// the object is kept.  If it returns true, it must increment the object's
/// usage count (or whatever it needs to do to retain it) before returning.
///
/// This function returns 0 if successful or -ENOMEM if out of memory.  In the
/// latter case, the array is not changed.
///
/// The caller should lock against other modifications and must continue to hold
/// the lock until assoc_array_apply_edit() has been called.
///
/// Accesses to the tree may take place concurrently with this function,
/// provided they hold the RCU read lock.
#[unsafe(no_mangle)]
extern "C" fn assoc_array_gc(array: *mut assoc_array, ops: *const assoc_array_ops, iterator: Option<unsafe extern "C" fn(*mut (), *mut ()) -> bool>, iterator_data: *mut ()) -> i32 {
let mut __slate_compound_102: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_104: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_105: u64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_106: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_108: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_109: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_111: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut n: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(n) as *mut *mut assoc_array_node;
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut p: *mut assoc_array_ptr = unsafe { core::mem::zeroed() };
let mut __v34: bool = unsafe { core::mem::zeroed() };
let mut child: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(child) as *mut *mut assoc_array_node;
let mut s_1: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(s_1) as *mut *mut assoc_array_shortcut;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_51: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut _old_1: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_1) as *mut *mut alloc_tag;
let mut _res_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_61: *mut *mut () = core::ptr::addr_of_mut!(_res_1) as *mut *mut ();
let mut __obj_ptr: *mut assoc_array_shortcut = unsafe { core::mem::zeroed() };
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_70: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_71: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size_2: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_77: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut next_slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_78: *mut i32 = core::ptr::addr_of_mut!(next_slot) as *mut i32;
let mut nr_free: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_79: *mut i32 = core::ptr::addr_of_mut!(nr_free) as *mut i32;
let mut slot: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_80: *mut i32 = core::ptr::addr_of_mut!(slot) as *mut i32;
let mut keylen: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_81: *mut i32 = core::ptr::addr_of_mut!(keylen) as *mut i32;
let mut retained: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_82: *mut bool = core::ptr::addr_of_mut!(retained) as *mut bool;
let mut nr_leaves_on_tree: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_83: *mut u64 = core::ptr::addr_of_mut!(nr_leaves_on_tree) as *mut u64;
let mut new_ptr_pp: core::mem::MaybeUninit<*mut *mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_84: *mut *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(new_ptr_pp) as *mut *mut *mut assoc_array_ptr;
let mut new_parent: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_85: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(new_parent) as *mut *mut assoc_array_ptr;
let mut new_root: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_86: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(new_root) as *mut *mut assoc_array_ptr;
let mut ptr: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_87: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(ptr) as *mut *mut assoc_array_ptr;
let mut cursor: core::mem::MaybeUninit<*mut assoc_array_ptr> = core::mem::MaybeUninit::uninit();
let __slate_slot_88: *mut *mut assoc_array_ptr = core::ptr::addr_of_mut!(cursor) as *mut *mut assoc_array_ptr;
let mut edit: core::mem::MaybeUninit<*mut assoc_array_edit> = core::mem::MaybeUninit::uninit();
let __slate_slot_89: *mut *mut assoc_array_edit = core::ptr::addr_of_mut!(edit) as *mut *mut assoc_array_edit;
let mut new_n: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_90: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(new_n) as *mut *mut assoc_array_node;
let mut node: core::mem::MaybeUninit<*mut assoc_array_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_91: *mut *mut assoc_array_node = core::ptr::addr_of_mut!(node) as *mut *mut assoc_array_node;
let mut new_s: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_92: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(new_s) as *mut *mut assoc_array_shortcut;
let mut shortcut: core::mem::MaybeUninit<*mut assoc_array_shortcut> = core::mem::MaybeUninit::uninit();
let __slate_slot_93: *mut *mut assoc_array_shortcut = core::ptr::addr_of_mut!(shortcut) as *mut *mut assoc_array_shortcut;
unsafe {
let __slate_auto_bytes_5 = __slate_slot_93 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_92 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_91 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_90 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_89 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_88 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_87 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_12 = __slate_slot_86 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_13 = __slate_slot_85 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_14 = __slate_slot_84 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_15 = __slate_slot_83 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_16 = __slate_slot_82 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_17 = __slate_slot_81 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_18 = __slate_slot_80 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_18.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_19 = __slate_slot_79 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_19.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_20 = __slate_slot_78 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_20.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_21 = __slate_slot_77 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_21.add(0usize), 0u8, 4usize);
if unsafe { (*array).root }.is_null() {
return 0;
} else {
__obj_size_2 = size_mul(core::mem::size_of::<assoc_array_edit>() as u64, 1u64);
let __slate_auto_bytes_4 = __slate_slot_71 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_70 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_70 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_71 = _kzalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_109 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_109)
});
alloc_tag_restore!();
} else {
*__slate_slot_71 = _kzalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_111 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_111)
});
}
*__slate_slot_89 = *__slate_slot_71 as *mut assoc_array_edit;
if (*__slate_slot_89).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_89)).array = array;
(*(*__slate_slot_89)).ops = ops;
(*(*__slate_slot_89)).ops_for_excised_subtree = ops;
(*((*(*__slate_slot_89)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).ptr = core::ptr::addr_of_mut!((*array).root);
(*(*__slate_slot_89)).excised_subtree = (*array).root;
*__slate_slot_85 = core::ptr::null_mut::<assoc_array_ptr>();
*__slate_slot_86 = core::ptr::null_mut::<assoc_array_ptr>();
*__slate_slot_84 = core::ptr::addr_of_mut!(*__slate_slot_86);
*__slate_slot_88 = (*array).root;
'__join_0: {
'__join_57: {
'__join_49: {
'__join_43: {
'__join_1: {
'__join_22: {
'__join_19: {
'__join_5: {
'__loop_97: loop {
if assoc_array_ptr_is_shortcut(*__slate_slot_88 as *const assoc_array_ptr) {
*__slate_slot_93 = assoc_array_ptr_to_shortcut(*__slate_slot_88 as *const assoc_array_ptr);
*__slate_slot_81 = round_up!(unsafe { (*(*__slate_slot_93)).skip_to_level }, 64i32);
let __v69: i32 = *__slate_slot_81;
let __v68: i32 = __v69 >> ilog2!(64i32);
*__slate_slot_81 = __v68;
__count = *__slate_slot_81 as u64;
__obj_size_1 = struct_size_t__2!(__count);
let __slate_auto_bytes_3 = __slate_slot_61 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_60 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_60 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_61 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_106 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_106)
});
alloc_tag_restore!();
} else {
*__slate_slot_61 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_108 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_108)
});
}
__obj_ptr = *__slate_slot_61 as *mut assoc_array_shortcut;
if !__obj_ptr.is_null() {
let __v58: u64 = __count;
*core::ptr::addr_of_mut!(*{
__slate_compound_105 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_105)
}) = __v58;
}
*__slate_slot_92 = __obj_ptr;
if (*__slate_slot_92).is_null() {
break '__join_0;
} else {
memcpy(*__slate_slot_92 as *mut (), *__slate_slot_93 as *const (), struct_size__2!(*__slate_slot_81));
(*(*__slate_slot_92)).back_pointer = *__slate_slot_85;
(*(*__slate_slot_92)).parent_slot = (*(*__slate_slot_93)).parent_slot;
let __v56: *mut assoc_array_ptr = assoc_array_shortcut_to_ptr(*__slate_slot_92 as *const assoc_array_shortcut);
*__slate_slot_85 = __v56;
*(*__slate_slot_84) = __v56;
*__slate_slot_84 = core::ptr::addr_of_mut!((*(*__slate_slot_92)).next_node);
*__slate_slot_88 = (*(*__slate_slot_93)).next_node;
}
}
// If this point is a shortcut, then we need to duplicate it and
// advance the target cursor.
// Duplicate the node at this position
*__slate_slot_91 = assoc_array_ptr_to_node(*__slate_slot_88 as *const assoc_array_ptr);
__obj_size = size_mul(core::mem::size_of::<assoc_array_node>() as u64, 1u64);
let __slate_auto_bytes_2 = __slate_slot_51 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_50 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_51 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_102 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_102)
});
alloc_tag_restore!();
} else {
*__slate_slot_51 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_104 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_104)
});
}
*__slate_slot_90 = *__slate_slot_51 as *mut assoc_array_node;
if (*__slate_slot_90).is_null() {
break '__join_0;
} else {
(*(*__slate_slot_90)).back_pointer = *__slate_slot_85;
(*(*__slate_slot_90)).parent_slot = (*(*__slate_slot_91)).parent_slot;
let __v48: *mut assoc_array_ptr = assoc_array_node_to_ptr(*__slate_slot_90 as *const assoc_array_node);
*__slate_slot_85 = __v48;
*(*__slate_slot_84) = __v48;
*__slate_slot_84 = core::ptr::null_mut::<*mut assoc_array_ptr>();
*__slate_slot_80 = 0;
loop {
if *__slate_slot_80 < 16 {
*__slate_slot_87 = *((*(*__slate_slot_91)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize);
if !(*__slate_slot_87).is_null() {
if assoc_array_ptr_is_leaf(*__slate_slot_87 as *const assoc_array_ptr) {
if iterator.unwrap()(assoc_array_ptr_to_leaf(*__slate_slot_87 as *const assoc_array_ptr), iterator_data) {
*((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize) = *__slate_slot_87;
}
// The iterator will have done any reference
// counting on the object for us.
} else {
break;
}
}
let __v47: i32 = *__slate_slot_80;
*__slate_slot_80 = __v47.wrapping_add(1);
} else {
// Filter across any leaves and gc any subtrees
loop {
// Count up the number of empty slots in this node and work out the
// subtree leaf count.
(*(*__slate_slot_90)).nr_leaves_on_branch = 0;
*__slate_slot_79 = 0;
*__slate_slot_80 = 0;
while *__slate_slot_80 < 16 {
*__slate_slot_87 = *((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize);
if (*__slate_slot_87).is_null() {
let __v41: i32 = *__slate_slot_79;
*__slate_slot_79 = __v41.wrapping_add(1);
} else {
if assoc_array_ptr_is_leaf(*__slate_slot_87 as *const assoc_array_ptr) {
let __v44: *mut assoc_array_node = *__slate_slot_90;
let __v43: u64 = (*__v44).nr_leaves_on_branch;
(*__v44).nr_leaves_on_branch = __v43.wrapping_add(1);
}
}
let __v39: i32 = *__slate_slot_80;
*__slate_slot_80 = __v39.wrapping_add(1);
}
// See what we can fold in
*__slate_slot_82 = false;
*__slate_slot_78 = 0;
*__slate_slot_80 = 0;
while *__slate_slot_80 < 16 {
let __slate_auto_bytes_1 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_87 = *((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize);
if (*__slate_slot_87).is_null() {
__v34 = true;
} else {
__v34 = assoc_array_ptr_is_leaf(*__slate_slot_87 as *const assoc_array_ptr);
}
if !__v34 {
*__slate_slot_36 = core::ptr::null_mut::<assoc_array_shortcut>();
if assoc_array_ptr_is_shortcut(*__slate_slot_87 as *const assoc_array_ptr) {
*__slate_slot_36 = assoc_array_ptr_to_shortcut(*__slate_slot_87 as *const assoc_array_ptr);
*__slate_slot_87 = (*(*__slate_slot_36)).next_node;
}
*__slate_slot_35 = assoc_array_ptr_to_node(*__slate_slot_87 as *const assoc_array_ptr);
let __v33: *mut assoc_array_node = *__slate_slot_90;
let __v32: u64 = (*__v33).nr_leaves_on_branch;
(*__v33).nr_leaves_on_branch = __v32.wrapping_add((*(*__slate_slot_35)).nr_leaves_on_branch);
if (*(*__slate_slot_35)).nr_leaves_on_branch <= ((*__slate_slot_79).wrapping_add(1) as u64) {
// Fold the child node into this one
// We would already have reaped an intervening shortcut
// on the way back up the tree.
loop {
if unlikely__7!(*__slate_slot_36) != 0 {
break '__join_57;
} else {
if true {
break;
}
}
}
*((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize) = core::ptr::null_mut::<assoc_array_ptr>();
let __v28: i32 = *__slate_slot_79;
*__slate_slot_79 = __v28.wrapping_add(1);
if *__slate_slot_80 < *__slate_slot_78 {
*__slate_slot_78 = *__slate_slot_80;
}
*__slate_slot_77 = 0;
while *__slate_slot_77 < 16 {
p = *((*(*__slate_slot_35)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_77 as isize);
if !p.is_null() {
loop {
if unlikely!(assoc_array_ptr_is_meta(p as *const assoc_array_ptr)) != 0 {
break '__join_49;
} else {
if true {
break;
}
}
}
while !unsafe { *((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_78 as isize) }.is_null() {
let __v25: i32 = *__slate_slot_78;
*__slate_slot_78 = __v25.wrapping_add(1);
}
loop {
if unlikely!(*__slate_slot_78 >= 16) != 0 {
break '__join_43;
} else {
if true {
break;
}
}
}
let __v23: i32 = *__slate_slot_78;
*__slate_slot_78 = __v23.wrapping_add(1);
*((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(__v23 as isize) = p;
let __v21: i32 = *__slate_slot_79;
*__slate_slot_79 = __v21.wrapping_sub(1);
}
let __v19: i32 = *__slate_slot_77;
*__slate_slot_77 = __v19.wrapping_add(1);
}
kfree(*__slate_slot_35 as *const ());
} else {
*__slate_slot_82 = true;
}
}
let __v17: i32 = *__slate_slot_80;
*__slate_slot_80 = __v17.wrapping_add(1);
}
if !(*__slate_slot_82 && (*(*__slate_slot_90)).nr_leaves_on_branch <= 16) {
break;
}
}
'__join_9: {
*__slate_slot_83 = (*(*__slate_slot_90)).nr_leaves_on_branch;
// Excise this node if it is singly occupied by a shortcut
if *__slate_slot_79 == 15i32 {
*__slate_slot_80 = 0;
while *__slate_slot_80 < 16 {
let __v13: *mut assoc_array_ptr = *((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize);
*__slate_slot_87 = __v13;
if !__v13.is_null() {
break;
} else {
let __v12: i32 = *__slate_slot_80;
*__slate_slot_80 = __v12.wrapping_add(1);
}
}
if assoc_array_ptr_is_meta(*__slate_slot_87 as *const assoc_array_ptr) {
__v10 = assoc_array_ptr_is_shortcut(*__slate_slot_87 as *const assoc_array_ptr);
} else {
__v10 = false;
}
if __v10 {
*__slate_slot_92 = assoc_array_ptr_to_shortcut(*__slate_slot_87 as *const assoc_array_ptr);
*__slate_slot_85 = (*(*__slate_slot_90)).back_pointer;
*__slate_slot_80 = (*(*__slate_slot_90)).parent_slot as i32;
kfree(*__slate_slot_90 as *const ());
if (*__slate_slot_85).is_null() {
break '__join_22;
} else {
if assoc_array_ptr_is_shortcut(*__slate_slot_85 as *const assoc_array_ptr) {
// We can discard any preceding shortcut also
let s: *mut assoc_array_shortcut = assoc_array_ptr_to_shortcut(*__slate_slot_85 as *const assoc_array_ptr);
let __v6: *mut assoc_array_ptr = (*s).back_pointer;
(*(*__slate_slot_92)).back_pointer = __v6;
*__slate_slot_85 = __v6;
let __v5: i32 = (*s).parent_slot;
(*(*__slate_slot_92)).parent_slot = __v5;
*__slate_slot_80 = __v5;
kfree(s as *const ());
if (*__slate_slot_85).is_null() {
break '__join_19;
}
}
(*(*__slate_slot_92)).back_pointer = *__slate_slot_85;
(*(*__slate_slot_92)).parent_slot = *__slate_slot_80;
*__slate_slot_90 = assoc_array_ptr_to_node(*__slate_slot_85 as *const assoc_array_ptr);
*((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize) = *__slate_slot_87;
break '__join_9;
}
}
}
// Excise any shortcuts we might encounter that point to nodes that
// only contain leaves.
*__slate_slot_87 = (*(*__slate_slot_90)).back_pointer;
if (*__slate_slot_87).is_null() {
break '__join_1;
} else {
if assoc_array_ptr_is_shortcut(*__slate_slot_87 as *const assoc_array_ptr) {
*__slate_slot_92 = assoc_array_ptr_to_shortcut(*__slate_slot_87 as *const assoc_array_ptr);
*__slate_slot_85 = (*(*__slate_slot_92)).back_pointer;
*__slate_slot_80 = (*(*__slate_slot_92)).parent_slot;
if (*(*__slate_slot_90)).nr_leaves_on_branch <= 16 {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
(*(*__slate_slot_90)).back_pointer = *__slate_slot_85;
(*(*__slate_slot_90)).parent_slot = *__slate_slot_80 as u8;
kfree(*__slate_slot_92 as *const ());
if (*__slate_slot_85).is_null() {
break '__loop_97;
} else {
*__slate_slot_4 = assoc_array_ptr_to_node(*__slate_slot_85 as *const assoc_array_ptr);
*((*(*__slate_slot_4)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize) = assoc_array_node_to_ptr(*__slate_slot_90 as *const assoc_array_node);
}
}
} else {
*__slate_slot_85 = *__slate_slot_87;
}
*__slate_slot_90 = assoc_array_ptr_to_node(*__slate_slot_85 as *const assoc_array_ptr);
}
}
*__slate_slot_87 = (*(*__slate_slot_91)).back_pointer;
if assoc_array_ptr_is_shortcut(*__slate_slot_87 as *const assoc_array_ptr) {
*__slate_slot_93 = assoc_array_ptr_to_shortcut(*__slate_slot_87 as *const assoc_array_ptr);
*__slate_slot_80 = (*(*__slate_slot_93)).parent_slot;
*__slate_slot_88 = (*(*__slate_slot_93)).back_pointer;
if (*__slate_slot_88).is_null() {
break '__join_1;
}
} else {
*__slate_slot_80 = (*(*__slate_slot_91)).parent_slot as i32;
*__slate_slot_88 = *__slate_slot_87;
}
loop {
if unlikely!((*__slate_slot_88).is_null()) != 0 {
break '__join_5;
} else {
if true {
break;
}
}
}
*__slate_slot_91 = assoc_array_ptr_to_node(*__slate_slot_88 as *const assoc_array_ptr);
let __v2: i32 = *__slate_slot_80;
*__slate_slot_80 = __v2.wrapping_add(1);
}
}
*__slate_slot_84 = ((*(*__slate_slot_90)).slots.as_mut_ptr() as *mut *mut assoc_array_ptr).wrapping_offset(*__slate_slot_80 as isize);
*__slate_slot_88 = *__slate_slot_87;
}
}
*__slate_slot_86 = assoc_array_node_to_ptr(*__slate_slot_90 as *const assoc_array_node);
break '__join_1;
}
core::arch::asm!("251: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 251b - ., 3; .popsection\n# {0}", const 251 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_gc_94, sym __slate_asm_string_assoc_array_gc_95, const 1708 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
(*(*__slate_slot_92)).back_pointer = core::ptr::null_mut::<assoc_array_ptr>();
(*(*__slate_slot_92)).parent_slot = 0;
*__slate_slot_86 = *__slate_slot_87;
break '__join_1;
}
(*(*__slate_slot_92)).back_pointer = core::ptr::null_mut::<assoc_array_ptr>();
(*(*__slate_slot_92)).parent_slot = 0;
*__slate_slot_86 = *__slate_slot_87;
}
(*((*(*__slate_slot_89)).set.as_mut_ptr() as *mut __SlateRecord_assoc_array_priv_h_89).wrapping_offset(0 as isize)).to = *__slate_slot_86;
assoc_array_apply_edit(*__slate_slot_89);
(*array).nr_leaves_on_tree = *__slate_slot_83;
return 0;
}
core::arch::asm!("250: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 250b - ., 3; .popsection\n# {0}", const 250 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_gc_96, sym __slate_asm_string_assoc_array_gc_97, const 1597 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("249: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 249b - ., 3; .popsection\n# {0}", const 249 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_gc_98, sym __slate_asm_string_assoc_array_gc_99, const 1594 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("248: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 248b - ., 3; .popsection\n# {0}", const 248 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_assoc_array_gc_100, sym __slate_asm_string_assoc_array_gc_101, const 1584 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
assoc_array_destroy_subtree(*__slate_slot_86, (*(*__slate_slot_89)).ops);
kfree(*__slate_slot_89 as *const ());
return 12i32.wrapping_neg();
}
}
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_assoc_array_apply_edit_22: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_apply_edit_23: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_delete_41: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_delete_42: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_delete_43: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_delete_44: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_delete_48: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_delete_49: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_delete_collapse_iterator_0: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_delete_collapse_iterator_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_19: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_20: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_21: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_22: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_23: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_24: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_25: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_26: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_27: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_28: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_29: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_30: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_31: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_32: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_destroy_subtree_33: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_destroy_subtree_34: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_gc_100: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_gc_101: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_gc_94: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_gc_95: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_gc_96: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_gc_97: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_gc_98: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_gc_99: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_13: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_14: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_107: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_108: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_113: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_114: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_115: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_116: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_117: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_118: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_119: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_120: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_121: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_122: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_123: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_124: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_125: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_126: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_127: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_128: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_129: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_into_terminal_node_130: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_mid_shortcut_6: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_mid_shortcut_7: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_insert_set_object_0: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_insert_set_object_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_rcu_cleanup_5: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_rcu_cleanup_6: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";

static __slate_asm_string_assoc_array_walk_26: [u8; 1] = *b"\0";

static __slate_asm_string_assoc_array_walk_27: [u8; 47] = *b"/home/takashi/c-corpus/linux/lib/assoc_array.c\0";
