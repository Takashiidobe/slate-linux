
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! Various workarounds for chipset bugs.
//! This code runs very early and can't use the regular PCI subsystem
//! The entries are keyed to PCI bridges which usually identify chipsets
//! uniquely.
//! This is only for whole classes of chipsets with specific problems which
//! need early invasive action (e.g. before the timers are initialized).
//! Most PCI device specific workarounds can be done later and should be
//! in standard PCI quirks
//! Mainboard specific bugs should be handled by DMI entries.
//! CPU specific bugs in setup.c
//! This counts to 15. Any more, it will return 16th argument.
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
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
//! Defines x86 CPU feature bits
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! @a is a power of 2 value
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! strscpy - Copy a C-string into a sized buffer
//! @dst: Where to copy the string to
//! @src: Where to copy the string from
//! @...: Size of destination buffer (optional)
//!
//! Copy the source string @src, or as much of it as fits, into the
//! destination @dst buffer. The behavior is undefined if the string
//! buffers overlap. The destination @dst buffer is always NUL terminated,
//! unless it's zero-sized.
//!
//! The size argument @... is only required when @dst is not an array, or
//! when the copy needs to be smaller than sizeof(@dst).
//!
//! Preferred to strncpy() since it always returns a valid string, and
//! doesn't unnecessarily force the tail of the destination buffer to be
//! zero padded. If padding is desired please use strscpy_pad().
//!
//! Returns the number of characters copied in @dst (not including the
//! trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
//! truncated.
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! for_each_if - helper for handling conditionals in various for_each macros
//! @condition: The condition to check
//!
//! Typical use::
//!
//! 	#define for_each_foo_bar(x, y) \'
//! 		list_for_each_entry(x, y->list, head) \'
//! 			for_each_if(x->something == SOMETHING)
//!
//! The for_each_if() macro makes the use of for_each_foo_bar() less error
//! prone.
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Sign Flag
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! writeable
//! 64bit FS base
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
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! If !LOCKDEP, the value is meaningless
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! Convenience macros for the sake of set_current_state:
//! wait_queue_entry::flags
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
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
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
//! Docs say 0 for future compat.
//! Each PCI slot may be a combo card with its own bus.  4 IRQ pins per slot.
//! Helper macro to avoid gfp flags if they are the default one
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! VMA locks do not behave like most ordinary locks found in the kernel, so we
//! cannot quite have full lockdep tracking in the way we would ideally prefer.
//!
//! Read locks act as shared locks which exclude an exclusive lock being
//! taken. We therefore mark these accordingly on read lock acquire/release.
//!
//! Write locks are acquired exclusively per-VMA, but released in a shared
//! fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
//! that write lock is released.
//!
//! We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
//! is the fact that, of course, we do lockdep-track the mmap lock rwsem which
//! must be held when taking a VMA write lock.
//!
//! We do, however, want to indicate that during either acquisition of a VMA
//! write lock or detachment of a VMA that we require the lock held be exclusive,
//! so we utilise lockdep to do so.
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
//! This is the smallest unsigned integer type that can fit a value
//! (up to 'long long')
//! CAP_BPF allows the following BPF operations:
//! - Creating all types of BPF maps
//! - Advanced verifier features
//!   - Indirect variable access
//!   - Bounded loops
//!   - BPF to BPF function calls
//!   - Scalar precision tracking
//!   - Larger complexity limits
//!   - Dead code elimination
//!   - And potentially other features
//! - Loading BPF Type Format (BTF) data
//! - Retrieve xlated and JITed code of BPF programs
//! - Use bpf_spin_lock() helper
//!
//! CAP_PERFMON relaxes the verifier checks further:
//! - BPF progs can use of pointer-to-integer conversions
//! - speculation attack hardening measures are bypassed
//! - bpf_probe_read to read arbitrary kernel memory is allowed
//! - bpf_trace_printk to print kernel memory is allowed
//!
//! CAP_SYS_ADMIN is required to use bpf_probe_write_user.
//!
//! CAP_SYS_ADMIN is required to iterate system wide loaded
//! programs, maps, links, BTFs and convert their IDs to file descriptors.
//!
//! CAP_PERFMON and CAP_BPF are required to load tracing programs.
//! CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
//! return the first ptr or next element in an RCU protected list
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
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! no-op
//! per-IO O_SYNC
//! File supports atomic writes
//! The structure layout of user_xstateregs, used for exporting the
//! extended register state through ptrace and core-dump (NT_X86_XSTATE note)
//! interfaces will be same as the memory layout of xsave used by the processor
//! (except for the bytes 464..511, which can be used by the software) and hence
//! the size of this structure varies depending on the features supported by the
//! processor and OS. The size of the structure that users need to use can be
//! obtained by doing:
//!     cpuid_count(0xd, 0, &eax, &ptrace_xstateregs_struct_size, &ecx, &edx);
//! i.e., cpuid.(eax=0xd,ecx=0).ebx will be the size that user (debuggers, etc.)
//! need to use.
//!
//! For now, only the first 8 bytes of the software usable bytes[464..471] will
//! be used and will be set to OS enabled xstate mask (which is same as the
//! 64bit mask returned by the xgetbv's xCR0).  Users (analyzing core dump
//! remotely, etc.) can use this mask as well as the mask saved in the
//! xstate_hdr bytes and interpret what states the processor/OS supports
//! and what states are in modified/initialized conditions for the
//! particular process/thread.
//!
//! Also when the user modifies certain state FP/SSE/etc through the
//! ptrace interface, they must ensure that the header.xfeatures
//! bytes[512..519] of the memory layout are updated correspondingly.
//! i.e., for example when FP state is modified to a non-init state,
//! header.xfeatures's bit 0 must be set to '1', when SSE is modified to
//! non-init state, header.xfeatures's bit 1 must to be set to '1', etc.
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! High Precision Event Timer table
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! Helpful configuration macro to be used in tables etc.
//! device 2 has a read-only mirror
//! I855_GM
//! 8237 DMA controllers
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Aperture control register bits.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
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
// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
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
// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
//
// Assert that there are no concurrent writes to @var; other readers are
// allowed. This assertion can be used to specify properties of concurrent code,
// where violation cannot be detected as a normal data race.
//
// For example, if we only have a single writer, but multiple concurrent
// readers, to avoid data races, all these accesses must be marked; even
// concurrent marked writes racing with the single writer are bugs.
// Unfortunately, due to being marked, they are no longer data races. For cases
// like these, we can use the macro as follows:
//
// .. code-block:: c
//
// 	void writer(void) {
// 		spin_lock(&update_foo_lock);
// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
// 		WRITE_ONCE(shared_foo, ...);
// 		spin_unlock(&update_foo_lock);
// 	}
// 	void reader(void) {
// 		// update_foo_lock does not need to be held!
// 		... = READ_ONCE(shared_foo);
// 	}
//
// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
// checking if a clear scope where no concurrent writes are expected exists.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_ACCESS - assert no concurrent accesses to @var
//
// Assert that there are no concurrent accesses to @var (no readers nor
// writers). This assertion can be used to specify properties of concurrent
// code, where violation cannot be detected as a normal data race.
//
// For example, where exclusive access is expected after determining no other
// users of an object are left, but the object is not actually freed. We can
// check that this property actually holds as follows:
//
// .. code-block:: c
//
// 	if (refcount_dec_and_test(&obj->refcnt)) {
// 		ASSERT_EXCLUSIVE_ACCESS(*obj);
// 		do_some_cleanup(obj);
// 		release_for_reuse(obj);
// 	}
//
// Note:
//
// 1. ASSERT_EXCLUSIVE_ACCESS_SCOPED(), if applicable, performs more thorough
//    checking if a clear scope where no concurrent accesses are expected exists.
//
// 2. For cases where the object is freed, `KASAN <kasan.html>`_ is a better
//    fit to detect use-after-free bugs.
//
// @var: variable to assert on
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
// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
// Combine multiple asm inline constraint args into a single arg for passing to
// another macro.
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
// informational
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// FW_BUG
// Add this to a message where you are sure the firmware is buggy or behaves
// really stupid or out of spec. Be aware that the responsible BIOS developer
// should be able to fix this issue or at least get a concrete idea of the
// problem by reading your message without the need of looking at the kernel
// code.
//
// Use it for definite and high priority BIOS bugs.
//
// FW_WARN
// Use it for not that clear (e.g. could the kernel messed up things already?)
// and medium priority BIOS bugs.
//
// FW_INFO
// Use this one if you want to tell the user or vendor about something
// suspicious, but generally harmless related to the firmware.
//
// Use it for information or very low priority BIOS bugs.
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
// pr_err - Print an error-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ERR loglevel. It uses pr_fmt() to
// generate the format string.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
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
// Atomic operations are already serializing on x86
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Operations for contexts that are safe from preemption/interrupts.  These
// operations verify that preemption is disabled.
// Operations with implied preemption/interrupt protection.  These
// operations can be used without worrying about preemption or interrupt.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "x2apic" X2APIC
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Xen paravirtual guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
// "11ap" Bad local APIC aka 11AP
// "l1tf" CPU is affected by L1 Terminal Fault
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
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
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// Overflow Flag
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// disable sas during sighandling
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// raw_read_seqcount() - read the raw seqcount_t counter value
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// raw_read_seqcount opens a read critical section of the given
// seqcount_t, without any lockdep checking, and without checking or
// masking the sequence counter LSB. Calling code is responsible for
// handling that.
//
// Return: count to be passed to read_seqcount_retry()
// raw_seqcount_try_begin() - begin a seqcount_t read critical section
//                            w/o lockdep and w/o counter stabilization
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count to be passed to read_seqcount_retry()
//
// Similar to raw_seqcount_begin(), except it enables eliding the critical
// section entirely if odd, instead of doing the speculation knowing it will
// fail.
//
// Useful when counter stabilization is more or less equivalent to taking
// the lock and there is a slowpath that does that.
//
// If true, start will be set to the (even) sequence count read.
//
// Return: true when a read critical section is started.
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
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
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
// The bottom two bits of the slot determine how the remaining bits in the
// slot are interpreted:
//
// 00 - data pointer
// 10 - internal entry
// x1 - value entry
//
// The internal entry may be a pointer to the next level in the tree, a
// sibling entry, or an indicator that the entry in this slot has been moved
// to another location in the tree and the lookup should be restarted.  While
// NULL fits the 'data pointer' pattern, it means that there is no entry in
// the tree for this index (no matter what level of the tree it is found at).
// This means that storing a NULL entry in the tree is the same as deleting
// the entry from the tree.
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Implementation specific max size
// list_entry_rcu - get the struct for this entry
// @ptr:        the &struct list_head pointer.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_next_or_null_rcu - get the next element from a list
// @head:	the head for the list.
// @ptr:        the list head to take the next element from.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// Note that if the ptr is at the end of the list, NULL is returned.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Please ignore the name of this function.  It should be called
// switch_to_kernel_thread().
//
// enter_lazy_tlb() is a hint from the scheduler that we are entering a
// kernel thread or other context without an mm.  Acceptable implementations
// include doing nothing whatsoever, switching to init_mm, or various clever
// lazy tricks to try to minimize TLB flushes.
//
// The scheduler reserves the right to call enter_lazy_tlb() several times
// in a row.  It will notify us that we're going back to a real mm by
// calling switch_mm_irqs_off().
// Keep stack-allocated flush_tlb_info cacheline aligned, but cap the
// alignment to avoid excessive stack usage on large-cacheline systems.
// This is used for both get_user() and __get_user() to expand to
// the proper special function call that has odd calling conventions
// due to returning both a value and an error, and that depends on
// the size of the pointer passed in.
//
// Careful: we have to cast the result to the type of the pointer
// for sign reasons.
//
// The use of _ASM_DX as the register specifier is a bit of a
// simplification, as gcc only cares about it as the starting point
// and not size: for a 64-bit value it will use %ecx:%edx on 32 bits
// (%ecx being the next register in gcc's x86 register sequence), and
// %rdx on 64 bits.
//
// Clang/LLVM cares about the size of the register, but still wants
// the base register for something that ends up being a pair.
// get_user - Get a simple variable from user space.
// @x:   Variable to store result.
// @ptr: Source address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple variable from user space to kernel
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and the result of
// dereferencing @ptr must be assignable to @x without a cast.
//
// Return: zero on success, or -EFAULT on error.
// On error, the variable @x is set to zero.
// direct disk access hint
// Reserved kernel ranges [-100], [-10000, -40000].
// Special value for dirfd used to
// indicate openat should use the
// current working directory.
// Reserved for per-syscall flags	0xff.
// Do not follow symbolic
// links.
// Allow empty relative
// pathname to operate on dirfd
// directly.
// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// See the comment in asm-generic/io.h about ioremap_np().
// 16 bits
// 16 bits
// Revision ID
// Device class
// 8 bits
// Multi-Function Device (possible)
// Base addresses specify locations in memory or I/O space.
// Decoded size can be determined by writing a value of
// 0xffffffff to the register, and reading it back.  Only
// 1 bits are decoded.
// 32 bits
// 32 bits [htype 0,1 only]
// Secondary bus number
// PM control and status register
// Current power state (D0 to D3)
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
// Error values that may be returned by PCI functions
// PCI_DEVICE - macro used to describe a specific PCI device
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device.  The subvendor and subdevice fields will be set to
// PCI_ANY_ID.
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 3
// DWORD 5
// Graphics Mode Select
// device 2 has a read-only mirror from i85x/i865 onwards
// I865_G
// I915_G
// E7221_G
// I915_GM
// I945_G
// I945_GM
// I945_GME
// I946_GZ
// G35_G
// I965_Q
// I965_G
// Q35_G
// G33_G
// Q33_G
// I965_GM
// I965_GME
// GM45_G
// IGD_E_G
// Q45_G
// G45_G
// G41_G
// B43_G
// B43_G.1
// GT1 mobile
// GT2 mobile
// GT1 desktop
// GT1 server
// GT2 desktop
// GT2 server
// ULT GT1 desktop
// ULT GT1 mobile
// ULT GT1 server
// ULT GT1 reserved
// ULX GT1 mobile
// GT1 desktop
// GT1 mobile
// GT1 server
// GT1 reserved
// GT1 reserved
// SDV GT1 desktop
// SDV GT1 mobile
// SDV GT1 server
// SDV GT1 reserved
// SDV GT1 reserved
// CRW GT1 desktop
// CRW GT1 mobile
// CRW GT1 server
// CRW GT1 reserved
// CRW GT1 reserved
// ULT GT2 desktop
// ULT GT2 mobile
// ULT GT2 server
// ULT GT2 reserved
// ULX GT2 mobile
// GT2 desktop
// GT2 mobile
// GT2 server
// GT2 reserved
// GT2 reserved
// SDV GT2 desktop
// SDV GT2 mobile
// SDV GT2 server
// SDV GT2 reserved
// SDV GT2 reserved
// CRW GT2 desktop
// CRW GT2 mobile
// CRW GT2 server
// CRW GT2 reserved
// CRW GT2 reserved
// ULT GT3 desktop
// ULT GT3 mobile
// ULT GT3 server
// ULT GT3 reserved
// ULT GT3 reserved
// GT3 desktop
// GT3 mobile
// GT3 server
// GT3 reserved
// GT3 reserved
// SDV GT3 desktop
// SDV GT3 mobile
// SDV GT3 server
// SDV GT3 reserved
// SDV GT3 reserved
// CRW GT3 desktop
// CRW GT3 mobile
// CRW GT3 server
// CRW GT3 reserved
// CRW GT3 reserved
// GT1 ULT
// GT1 Iris
// GT1 ULX
// GT1 ULT
// GT1 Server
// GT1 Workstation
// GT2 ULT
// GT2 ULT
// GT2 ULX
// GT2 Halo
// GT2 Server
// GT2 Workstation
// ULT
// Iris
// ULX
// ULT
// Server
// Workstation
// ULT
// Iris
// ULX
// ULT
// Server
// Workstation
// ULT GT1
// ULT GT1.5
// ULX GT1
// ULX GT1.5
// DT  GT1
// SRV GT1
// Halo GT1
// DT  GT1.5
// ULT GT2
// ULT GT2F
// ULX GT2
// DT  GT2
// SRV GT2
// Halo GT2
// WKS GT2
// ULT GT3
// ULT GT3e
// ULT GT3e
// SRV GT3
// Halo GT3e
// SRV GT3e
// DT GT4
// SRV GT4e
// Halo GT4e
// WKS GT4e
// APL HD Graphics 505
// APL HD Graphics 500
// ULT GT1
// ULT GT1.5
// ULX GT1
// ULX GT1.5
// DT  GT1
// Halo GT1
// SRV GT1
// Halo GT1
// ULT GT2
// ULT GT2F
// ULX GT2
// DT  GT2
// Mobile GT2
// SRV GT2
// Halo GT2
// WKS GT2
// ULT GT3
// ULT GT3
// ULT GT3
// Halo GT4
// AML/KBL Y GT2
// ULX GT2
// ULX GT2
// AML/CFL Y GT2
// CML GT1
// CML GT2
// CFL S
// SRV GT1
// SRV GT1
// SRV GT1
// SRV GT2
// SRV GT2
// SRV GT2
// SRV GT2
// SRV GT2
// CFL H
// Halo GT2
// Halo GT2
// CFL U GT2
// CFL U GT3
// ULT GT3
// ULT GT3
// ULT GT3
// ULT GT3
// WHL/CFL U GT1
// WHL/CFL U GT2
// WHL/CFL U GT3
// CNL
// ICL
// EHL
// JSL
// TGL
// RKL
// ADL-S
// ADL-P
// ADL-N
// RPL-S
// RPL-U
// RPL-P
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
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
// max number of signatures to store
// K8 On-cpu GART registers
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
macro_rules! ARRAY_SIZE__25 {
() => {
((core::mem::size_of::<[pci_device_id; 335]>() as u64) / (core::mem::size_of::<pci_device_id>() as u64)).wrapping_add(__must_be_array__10!() as u64)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
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

macro_rules! KB {
($x:expr) => {
(($x) as u64).wrapping_mul(1024)
};
}

macro_rules! KB__2 {
($x:expr) => {
$x.wrapping_mul(1024)
};
}

macro_rules! MB {
($x:expr) => {
KB__2!(KB!($x))
};
}

macro_rules! ____is_defined {
() => {
0i32
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

macro_rules! __must_be_array__10 {
() => {
core::mem::size_of::<__SlateRecord_early_quirks_c_607>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! bcma_aread32 {
($reg:expr, $mmio:expr) => {
unsafe { ioread32(((((($mmio) as *const u8).wrapping_offset(4096i32 as isize) as *mut ()) as *const u8).wrapping_offset(($reg) as isize) as *mut ()) as *const ()) };
};
}

macro_rules! bcma_aread32__2 {
($reg:expr, $mmio:expr) => {
unsafe { ioread32(((((($mmio) as *const u8).wrapping_offset(4096i32 as isize) as *mut ()) as *const u8).wrapping_offset(($reg) as isize) as *mut ()) as *const ()) }
};
}

macro_rules! bcma_awrite32 {
($reg:expr, $val:expr, $mmio:expr) => {
unsafe { iowrite32(($val) as u32, (((($mmio) as *const u8).wrapping_offset(4096i32 as isize) as *mut ()) as *const u8).wrapping_offset(($reg) as isize) as *mut ()) };
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay {
($n:expr) => {
if true && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
}
}
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut acpi_skip_timer_override: i32;
    static mut acpi_use_timer_override: i32;
    static mut acpi_fix_pin2_polarity: i32;
    static mut x86_apple_machine: bool;
    static mut boot_hpet_disable: bool;
    static mut e820_table: *mut e820_table;
    fn iomem_get_mapping() -> *mut address_space;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_41();
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn early_ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn early_iounmap(arg0: *mut (), arg1: u64);
    fn ioread32(arg0: *const ()) -> u32;
    fn iowrite32(arg0: u32, arg1: *mut ());
    fn acpi_table_parse(arg0: *mut u8, arg1: Option<unsafe extern "C" fn(*mut acpi_table_header) -> i32>) -> i32;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn read_pci_config(arg0: u8, arg1: u8, arg2: u8, arg3: u8) -> u32;
    fn read_pci_config_byte(arg0: u8, arg1: u8, arg2: u8, arg3: u8) -> u8;
    fn read_pci_config_16(arg0: u8, arg1: u8, arg2: u8, arg3: u8) -> u16;
    fn write_pci_config(arg0: u8, arg1: u8, arg2: u8, arg3: u8, arg4: u32);
    fn write_pci_config_byte(arg0: u8, arg1: u8, arg2: u8, arg3: u8, arg4: u8);
    fn write_pci_config_16(arg0: u8, arg1: u8, arg2: u8, arg3: u8, arg4: u16);
    fn early_pci_allowed() -> i32;
    fn e820__range_add(arg0: u64, arg1: u64, arg2: u32);
    fn e820__update_table(arg0: *mut e820_table) -> i32;
    fn set_irq_remapping_broken();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_202 {
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_head {
    first: *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_node {
    next: *mut hlist_node,
    pprev: *mut *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcuwait {
    task: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// helpers to define resources
fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ratelimit_state {
    lock: raw_spinlock,
    interval: i32,
    burst: i32,
    rs_n_left: __SlateRecord_types_h_190,
    missed: __SlateRecord_types_h_190,
    flags: u32,
    begin: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qrwlock_types_h_27 {
    cnts: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qrwlock_types_h_27_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qrwlock_types_h_27_2 {
    wlocked: u8,
    __lstate: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_41() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = core::ptr::null_mut::<task_struct>();
return __v0;
}
let mut __v2: *mut task_struct;
let mut pscr_ret__: *mut task_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<*mut task_struct>() }) as u64 {
1 => {
let mut __v4: *mut task_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v4 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v4;
}
2 => {
let mut __v6: *mut task_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v6 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v6;
}
4 => {
let mut __v7: *mut task_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v7 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v7;
}
8 => {
let mut __v8: *mut task_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v8 = pfo_val__ as *mut task_struct;
pscr_ret__ = __v8;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v2 = pscr_ret__;
__v2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
}


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


// was written to (raised by CPU)
// on 4KB pages
// On 2MB or 1GB pages
// Protection Keys, bit 1/4
// Saved Dirty bit (leaf)
// The hardware requires shadow stack to be Write=0,Dirty=1. However,
// there are valid cases where the kernel might create read-only PTEs that
// are dirty (e.g., fork(), mprotect(), userfaultfd, soft-dirty tracking). In
// this case, the _PAGE_SAVED_DIRTY bit is used instead of the HW-dirty bit,
// to avoid creating a wrong "shadow stack" PTEs. Such PTEs have
// (Write=0,SavedDirty=1,Dirty=0) set.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct pgprot {
    pgprot: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits153U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_29 {
    futex: __SlateRecord_restart_block_h_31,
    nanosleep: __SlateRecord_restart_block_h_40,
    poll: __SlateRecord_restart_block_h_50,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_31 {
    uaddr: *mut u32,
    val: u32,
    flags: u32,
    bitset: u32,
    time: i64,
    uaddr2: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_40 {
    clockid: i32,
    r#type: u32,
    __slate_anon_2: __SlateRecord_restart_block_h_43,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_43 {
    rmtp: *mut __kernel_timespec,
    compat_rmtp: *mut old_timespec32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_50 {
    ufds: *mut pollfd,
    nfds: i32,
    has_timeout: i32,
    end_time: timespec64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits154U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_struct {
    tls_array: [desc_struct; 3],
    sp: u64,
    es: u16,
    ds: u16,
    fsindex: u16,
    gsindex: u16,
    fsbase: u64,
    gsbase: u64,
    ptrace_bps: [*mut perf_event; 4],
    virtual_dr6: u64,
    ptrace_dr7: u64,
    cr2: u64,
    trap_nr: u64,
    error_code: u64,
    io_bitmap: *mut io_bitmap,
    iopl_emul: u64,
    __slate_bits_0: __slate_bits::__SlateBits155U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_sched_data {
    robust_list: *mut robust_list_head,
    compat_robust_list: *mut compat_robust_list_head,
    pi_state_list: list_head,
    pi_state_cache: *mut futex_pi_state,
    exit_mutex: mutex,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_phash {
    lock: mutex,
    hash: *mut futex_private_hash,
    hash_new: *mut futex_private_hash,
    batches: u64,
    rcu: callback_head,
    atomic: __SlateRecord_types_h_197,
    r#ref: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer {
    node: timerqueue_linked_node,
    base: *mut hrtimer_clock_base,
    is_queued: bool,
    is_rel: bool,
    is_soft: bool,
    is_hard: bool,
    is_lazy: bool,
    _softexpires: i64,
    function: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_io_accounting {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_39 {
    _pid: i32,
    _uid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_45 {
    _tid: i32,
    _overrun: i32,
    _sigval: sigval,
    _sys_private: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_53 {
    _pid: i32,
    _uid: u32,
    _sigval: sigval,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_60 {
    _pid: i32,
    _uid: u32,
    _status: i32,
    _utime: i64,
    _stime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_69 {
    _addr: *mut (),
    __slate_anon_1: __SlateRecord_siginfo_h_74,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_siginfo_h_74 {
    _trapno: i32,
    _addr_lsb: i16,
    _addr_bnd: __SlateRecord_siginfo_h_83,
    _addr_pkey: __SlateRecord_siginfo_h_89,
    _perf: __SlateRecord_siginfo_h_94,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_83 {
    _dummy_bnd: [u8; 8],
    _lower: *mut (),
    _upper: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_89 {
    _dummy_pkey: [u8; 8],
    _pkey: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_94 {
    _data: u64,
    _type: u32,
    _flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_103 {
    _band: i64,
    _fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_109 {
    _call_addr: *mut (),
    _syscall: i32,
    _arch: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_mm_cid {
    pcpu: *mut mm_cid_pcpu,
    mode: u32,
    max_cids: u32,
    irq_work: irq_work,
    work: work_struct,
    lock: raw_spinlock,
    mutex: mutex,
    user_list: hlist_head,
    nr_cpus_allowed: u32,
    users: u32,
    pcpu_thrs: u32,
    update_deferred: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_38 {
    all: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_40,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_40 {
    __slate_anon_0: __SlateRecord_rseq_types_h_41,
    has_rseq: u8,
    __pad: u8,
    __slate_anon_3: __SlateRecord_rseq_types_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_41 {
    events: u32,
    __slate_anon_1: __SlateRecord_rseq_types_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_43 {
    sched_switch: u8,
    ids_changed: u8,
    user_irq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_52 {
    error: u16,
    __slate_anon_1: __SlateRecord_rseq_types_h_54,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_54 {
    fatal: u8,
    slowpath: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_74 {
    cpu_cid: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_76,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_76 {
    cpu_id: u32,
    mm_cid: u32,
}


// get_task_state():
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
// set_special_state() should be used for those states when the blocking task
// can not use the regular condition based wait-loop. In that case we must
// serialize against wakeups such that any possible in-flight TASK_RUNNING
// stores will not collide with our state change.
// may shadow
#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_info {
    pcount: u64,
    run_delay: u64,
    max_run_delay: u64,
    min_run_delay: u64,
    last_arrival: u64,
    last_queued: u64,
    max_run_delay_ts: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_avg {
    last_update_time: u64,
    load_sum: u64,
    runnable_sum: u64,
    util_sum: u32,
    period_contrib: u32,
    load_avg: u64,
    runnable_avg: u64,
    util_avg: u64,
    util_est: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_statistics {
    wait_start: u64,
    wait_max: u64,
    wait_count: u64,
    wait_sum: u64,
    iowait_count: u64,
    iowait_sum: u64,
    sleep_start: u64,
    sleep_max: u64,
    sum_sleep_runtime: i64,
    block_start: u64,
    block_max: u64,
    sum_block_runtime: i64,
    exec_max: i64,
    slice_max: u64,
    nr_failed_migrations_affine: u64,
    nr_failed_migrations_running: u64,
    nr_failed_migrations_hot: u64,
    nr_forced_migrations: u64,
    nr_wakeups: u64,
    nr_wakeups_sync: u64,
    nr_wakeups_migrate: u64,
    nr_wakeups_local: u64,
    nr_wakeups_remote: u64,
    nr_wakeups_affine: u64,
    nr_wakeups_affine_attempts: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_entity {
    load: load_weight,
    h_load: load_weight,
    run_node: rb_node,
    deadline: u64,
    min_vruntime: u64,
    min_slice: u64,
    max_slice: u64,
    group_node: list_head,
    on_rq: u8,
    sched_delayed: u8,
    rel_deadline: u8,
    custom_slice: u8,
    exec_start: u64,
    sum_exec_runtime: u64,
    prev_sum_exec_runtime: u64,
    vruntime: u64,
    vlag: i64,
    vprot: u64,
    slice: u64,
    nr_migrations: u64,
    depth: i32,
    parent: *mut sched_entity,
    cfs_rq: *mut cfs_rq,
    my_q: *mut cfs_rq,
    runnable_weight: u64,
    avg: sched_avg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_rt_entity {
    run_list: list_head,
    timeout: u64,
    watchdog_stamp: u64,
    time_slice: u32,
    on_rq: u16,
    on_list: u16,
    back: *mut sched_rt_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_dl_entity {
    rb_node: rb_node,
    dl_runtime: u64,
    dl_deadline: u64,
    dl_period: u64,
    dl_bw: u64,
    dl_density: u64,
    runtime: i64,
    deadline: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits279U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_struct {
    thread_info: thread_info,
    __state: u32,
    saved_state: u32,
    stack: *mut (),
    usage: refcount_struct,
    flags: u32,
    ptrace: u32,
    on_cpu: u8,
    on_rq: u8,
    is_blocked: u8,
    __pad: u8,
    wake_entry: __call_single_node,
    wakee_flips: u32,
    wakee_flip_decay_ts: u64,
    last_wakee: *mut task_struct,
    recent_used_cpu: i32,
    wake_cpu: i32,
    prio: i32,
    static_prio: i32,
    normal_prio: i32,
    rt_priority: u32,
    se: sched_entity,
    rt: sched_rt_entity,
    dl: sched_dl_entity,
    dl_server: *mut sched_dl_entity,
    sched_class: *const sched_class,
    sched_task_group: *mut task_group,
    stats: sched_statistics,
    btrace_seq: u32,
    policy: u32,
    max_allowed_capacity: u64,
    nr_cpus_allowed: i32,
    cpus_ptr: *const cpumask,
    user_cpus_ptr: *mut cpumask,
    cpus_mask: cpumask,
    migration_pending: *mut (),
    migration_disabled: u16,
    migration_flags: u16,
    rcu_read_lock_nesting: i32,
    rcu_read_unlock_special: rcu_special,
    rcu_node_entry: list_head,
    rcu_blocked_node: *mut rcu_node,
    rcu_tasks_nvcsw: u64,
    rcu_tasks_holdout: u8,
    rcu_tasks_idx: u8,
    rcu_tasks_idle_cpu: i32,
    rcu_tasks_holdout_list: list_head,
    rcu_tasks_exit_cpu: i32,
    rcu_tasks_exit_list: list_head,
    trc_reader_nesting: i32,
    trc_reader_scp: *mut srcu_ctr,
    sched_info: sched_info,
    tasks: list_head,
    pushable_tasks: plist_node,
    pushable_dl_tasks: rb_node,
    mm: *mut mm_struct,
    active_mm: *mut mm_struct,
    exec_state: *mut task_exec_state,
    exit_state: i32,
    exit_code: i32,
    exit_signal: i32,
    pdeath_signal: i32,
    jobctl: u64,
    personality: u32,
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
    atomic_flags: u64,
    restart_block: restart_block,
    pid: i32,
    tgid: i32,
    stack_canary: u64,
    real_parent: *mut task_struct,
    parent: *mut task_struct,
    children: list_head,
    sibling: list_head,
    group_leader: *mut task_struct,
    ptraced: list_head,
    ptrace_entry: list_head,
    thread_pid: *mut pid,
    pid_links: [hlist_node; 4],
    thread_node: list_head,
    vfork_done: *mut completion,
    set_child_tid: *mut i32,
    clear_child_tid: *mut i32,
    worker_private: *mut (),
    utime: u64,
    stime: u64,
    gtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    start_time: u64,
    start_boottime: u64,
    min_flt: u64,
    maj_flt: u64,
    posix_cputimers: posix_cputimers,
    posix_cputimers_work: posix_cputimers_work,
    ptracer_cred: *const cred,
    real_cred: *const cred,
    cred: *const cred,
    cached_requested_key: *mut key,
    comm: [u8; 16],
    nameidata: *mut nameidata,
    sysvsem: sysv_sem,
    sysvshm: sysv_shm,
    real_fs: *mut fs_struct,
    fs: *mut fs_struct,
    files: *mut files_struct,
    io_uring: *mut io_uring_task,
    io_uring_restrict: *mut io_restriction,
    nsproxy: *mut nsproxy,
    signal: *mut signal_struct,
    sighand: *mut sighand_struct,
    blocked: __SlateRecord_signal_h_25,
    real_blocked: __SlateRecord_signal_h_25,
    saved_sigmask: __SlateRecord_signal_h_25,
    pending: sigpending,
    sas_ss_sp: u64,
    sas_ss_size: u64,
    sas_ss_flags: u32,
    task_works: *mut callback_head,
    audit_context: *mut audit_context,
    loginuid: __SlateRecord_uidgid_types_h_9,
    sessionid: u32,
    seccomp: seccomp,
    syscall_dispatch: syscall_user_dispatch,
    parent_exec_id: u64,
    self_exec_id: u64,
    alloc_lock: spinlock,
    pi_lock: raw_spinlock,
    wake_q: wake_q_node,
    pi_waiters: rb_root_cached,
    pi_top_task: *mut task_struct,
    pi_blocked_on: *mut rt_mutex_waiter,
    blocked_on: *mut mutex,
    blocked_lock: raw_spinlock,
    blocked_donor: *mut task_struct,
    journal_info: *mut (),
    bio_list: *mut bio_list,
    plug: *mut blk_plug,
    reclaim_state: *mut reclaim_state,
    io_context: *mut io_context,
    capture_control: *mut capture_control,
    ptrace_message: u64,
    last_siginfo: *mut kernel_siginfo,
    ioac: task_io_accounting,
    acct_rss_mem1: u64,
    acct_vm_mem1: u64,
    acct_timexpd: u64,
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    mems_allowed_seq: seqcount_spinlock,
    cpuset_mem_spread_rotor: i32,
    cgroups: *mut css_set,
    cg_list: list_head,
    futex: futex_sched_data,
    perf_recursion: [u8; 4],
    perf_event_ctxp: *mut perf_event_context,
    perf_event_mutex: mutex,
    perf_event_list: list_head,
    perf_ctx_data: *mut perf_ctx_data,
    ipi_mask: task_ipi_mask,
    mempolicy: *mut mempolicy,
    il_prev: i16,
    il_weight: u8,
    pref_node_fork: i16,
    cache_work: callback_head,
    sched_cache_grp: *mut sched_cache_group,
    preferred_llc: i32,
    pref_llc_queued: i32,
    rseq: rseq_data,
    mm_cid: sched_mm_cid,
    tlb_ubc: tlbflush_unmap_batch,
    splice_pipe: *mut pipe_inode_info,
    task_frag: page_frag,
    delays: *mut task_delay_info,
    nr_dirtied: i32,
    nr_dirtied_pause: i32,
    dirty_paused_when: u64,
    timer_slack_ns: u64,
    default_timer_slack_ns: u64,
    trace_recursion: u64,
    throttle_disk: *mut gendisk,
    utask: *mut uprobe_task,
    kmap_ctrl: kmap_ctrl,
    rcu: callback_head,
    rcu_users: refcount_struct,
    pagefault_disabled: i32,
    oom_reaper_list: *mut task_struct,
    oom_reaper_timer: timer_list,
    stack_vm_area: *mut vm_struct,
    stack_refcount: refcount_struct,
    security: *mut (),
    bpf_net_context: *mut bpf_net_context,
    mce_vaddr: *mut (),
    mce_kflags: u64,
    mce_addr: u64,
    __slate_bits_2: __slate_bits::__SlateBits69U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


// Per process flags
// I'm a virtual CPU
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
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_cache_group {
    pcpu_sched: *mut sched_cache_time,
    lock: raw_spinlock,
    epoch: u64,
    nr_running_avg: u64,
    next_scan: u64,
    footprint: u64,
    cpu: i32,
    refcnt: refcount_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Dependency order vs. p above.
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
// rcu_replace_pointer() - replace an RCU pointer, returning its old value
// @rcu_ptr: RCU pointer, whose old value is returned
// @ptr: regular pointer
// @c: the lockdep conditions under which the dereference will take place
//
// Perform a replacement, where @rcu_ptr is an RCU-annotated
// pointer and @c is the lockdep argument that is passed to the
// rcu_dereference_protected() call used to read that pointer.  The old
// value of @rcu_ptr is returned, and @rcu_ptr is set to @ptr.
// rcu_access_pointer() - fetch RCU pointer with no dereferencing
// @p: The pointer to read
//
// Return the value of the specified RCU-protected pointer, but omit the
// lockdep checks for being in an RCU read-side critical section.  This is
// useful when the value of this pointer is accessed, but the pointer is
// not dereferenced, for example, when testing an RCU-protected pointer
// against NULL.  Within an RCU read-side critical section, there is little
// reason to use rcu_access_pointer().  Although rcu_access_pointer() may
// also be used in cases where update-side locks prevent the value of the
// pointer from changing, you should instead use rcu_dereference_protected()
// for this use case.  It is also permissible to use rcu_access_pointer()
// within lockless updaters to obtain the old value for an atomic operation,
// for example, for cmpxchg().
//
// It is usually best to test the rcu_access_pointer() return value
// directly in order to avoid accidental dereferences being introduced
// by later inattentive changes.  In other words, assigning the
// rcu_access_pointer() return value to a local variable results in an
// accident waiting to happen.
//
// It is also permissible to use rcu_access_pointer() when read-side
// access to the pointer was removed at least one grace period ago, as is
// the case in the context of the RCU callback that is freeing up the data,
// or after a synchronize_rcu() returns.  This can be useful when tearing
// down multi-linked structures after a grace period has elapsed.  However,
// rcu_dereference_protected() is normally preferred for this use case.
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
// rcu_dereference_protected() - fetch RCU pointer when updates prevented
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Return the value of the specified RCU-protected pointer, but omit
// the READ_ONCE().  This is useful in cases where update-side locks
// prevent the value of the pointer from changing.  Please note that this
// primitive does *not* prevent the compiler from repeating this reference
// or combining it with other references, so it should not be used without
// protection of appropriate locks.
//
// This function is only for update-side use.  Using this function
// when protected only by rcu_read_lock() will result in infrequent
// but very ugly failures.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct srcu_data {
    srcu_ctrs: [srcu_ctr; 2],
    srcu_reader_flavor: i32,
    __slate_pad_2: [u8; 28],
    lock: raw_spinlock,
    srcu_cblist: rcu_segcblist,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_cblist_invoking: bool,
    delay_work: timer_list,
    work: work_struct,
    srcu_barrier_head: callback_head,
    srcu_ec_head: callback_head,
    srcu_ec_state: i32,
    mynode: *mut srcu_node,
    grpmask: u64,
    cpu: i32,
    ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_node {
    lock: raw_spinlock,
    srcu_have_cbs: [u64; 4],
    srcu_data_have_cbs: [u64; 4],
    srcu_gp_seq_needed_exp: u64,
    srcu_parent: *mut srcu_node,
    grplo: i32,
    grphi: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_usage {
    node: *mut srcu_node,
    level: [*mut srcu_node; 3],
    srcu_size_state: i32,
    srcu_cb_mutex: mutex,
    lock: raw_spinlock,
    srcu_gp_mutex: mutex,
    srcu_gp_seq: u64,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_gp_start: u64,
    srcu_last_gp_end: u64,
    srcu_size_jiffies: u64,
    srcu_n_lock_retries: u64,
    srcu_n_exp_nodelay: u64,
    sda_is_static: bool,
    srcu_barrier_seq: u64,
    srcu_barrier_mutex: mutex,
    srcu_barrier_completion: completion,
    srcu_barrier_cpu_cnt: __SlateRecord_types_h_190,
    reschedule_jiffies: u64,
    reschedule_count: u64,
    work: delayed_work,
    irq_work: irq_work,
    srcu_ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct files_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_restriction {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_task {
    state: u32,
    depth: u32,
    return_instances: *mut return_instance,
    ri_pool: *mut return_instance,
    ri_timer: timer_list,
    ri_seqcount: seqcount,
    __slate_anon_6: __SlateRecord_uprobes_h_136,
    active_uprobe: *mut uprobe,
    xol_vaddr: u64,
    signal_denied: bool,
    auprobe: *mut arch_uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct return_instance {
    hprobe: hprobe,
    func: u64,
    stack: u64,
    orig_ret_vaddr: u64,
    chained: bool,
    cons_cnt: i32,
    next: *mut return_instance,
    rcu: callback_head,
    consumer: return_consumer,
    extra_consumers: *mut return_consumer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mmu_h_84 {
    ctx_id: u64,
    tlb_gen: __SlateRecord_types_h_197,
    next_trim_cpumask: u64,
    ldt_usr_sem: rw_semaphore,
    ldt: *mut ldt_struct,
    flags: u64,
    lock: mutex,
    vdso: *mut (),
    vdso_image: *const vdso_image,
    perf_rdpmc_allowed: __SlateRecord_types_h_190,
    pkey_allocation_map: u16,
    execute_only_pkey: i16,
    global_asid: u16,
    asid_transition: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct page {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_89,
    __slate_anon_2: __SlateRecord_mm_types_h_155,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits396U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_31 {
    insn: [u8; 16],
    ixol: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_38 {
    branch: __SlateRecord_uprobes_h_39,
    defparam: __SlateRecord_uprobes_h_44,
    push: __SlateRecord_uprobes_h_48,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_39 {
    offs: i32,
    ilen: u8,
    opc1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_44 {
    fixups: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_48 {
    reg_offset: u8,
    ilen: u8,
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_area_desc {
    mm: *mut mm_struct,
    file: *mut file,
    start: u64,
    end: u64,
    pgoff: u64,
    vm_file: *mut file,
    vma_flags: __SlateRecord_mm_types_h_871,
    page_prot: pgprot,
    vm_ops: *const vm_operations_struct,
    private_data: *mut (),
    action: mmap_action,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct vm_area_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_926,
    vm_mm: *mut mm_struct,
    vm_page_prot: pgprot,
    __slate_anon_3: __SlateRecord_mm_types_h_947,
    vm_lock_seq: u32,
    __vm_anon_pgoff_lo: u32,
    anon_vma_chain: list_head,
    anon_vma: *mut anon_vma,
    vm_ops: *const vm_operations_struct,
    vm_pgoff: u64,
    vm_file: *mut file,
    vm_private_data: *mut (),
    swap_readahead_info: __SlateRecord_types_h_197,
    vm_policy: *mut mempolicy,
    __slate_pad_14: [u8; 8],
    vm_refcnt: refcount_struct,
    __vm_anon_pgoff_hi: u32,
    shared: __SlateRecord_mm_types_h_1062,
    vm_userfaultfd_ctx: vm_userfaultfd_ctx,
    pfnmap_track_ctx: *mut pfnmap_track_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_136 {
    __slate_anon_0: __SlateRecord_uprobes_h_137,
    __slate_anon_1: __SlateRecord_uprobes_h_142,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_137 {
    autask: arch_uprobe_task,
    vaddr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_142 {
    dup_xol_work: callback_head,
    dup_xol_addr: u64,
}


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_89 {
    __slate_anon_0: __SlateRecord_mm_types_h_90,
    __slate_anon_1: __SlateRecord_mm_types_h_118,
    __slate_anon_2: __SlateRecord_mm_types_h_129,
    __slate_anon_3: __SlateRecord_mm_types_h_132,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_90 {
    __slate_anon_0: __SlateRecord_mm_types_h_96,
    mapping: *mut address_space,
    __slate_anon_2: __SlateRecord_mm_types_h_105,
    private: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_96 {
    lru: list_head,
    buddy_list: list_head,
    pcp_list: list_head,
    pcp_llist: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_105 {
    __folio_index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_118 {
    pp_magic: u64,
    pp: *mut page_pool,
    _pp_mapping_pad: u64,
    dma_addr: u64,
    pp_ref_count: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_129 {
    compound_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_132 {
    _unused_pgmap_compound_info: *mut (),
    zone_device_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_155 {
    page_type: u32,
    _mapcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_node {
    count: __SlateRecord_types_h_190,
    active: __SlateRecord_types_h_190,
    __parent: *mut kernfs_node,
    name: *const u8,
    rb: rb_node,
    ns: *const ns_common,
    hash: u32,
    flags: u16,
    mode: u16,
    __slate_anon_9: __SlateRecord_kernfs_h_221,
    id: u64,
    r#priv: *mut (),
    iattr: *mut kernfs_iattrs,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_file {
    kn: *mut kernfs_node,
    file: *mut file,
    seq_file: *mut seq_file,
    r#priv: *mut (),
    mutex: mutex,
    prealloc_mutex: mutex,
    event: i32,
    list: list_head,
    prealloc_buf: *mut u8,
    atomic_write_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits443U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_406 {
    __slate_anon_0: __SlateRecord_mm_types_h_407,
    page: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_407 {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_410,
    mapping: *mut address_space,
    __slate_anon_3: __SlateRecord_mm_types_h_425,
    __slate_anon_4: __SlateRecord_mm_types_h_429,
    _mapcount: __SlateRecord_types_h_190,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_410 {
    lru: list_head,
    __slate_anon_1: __SlateRecord_mm_types_h_414,
    pgmap: *mut dev_pagemap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_414 {
    __filler: *mut (),
    mlock_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_ops {
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    read: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    atomic_write_len: u64,
    prealloc: bool,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    mmap: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut vm_area_struct) -> i32>,
    llseek: Option<unsafe extern "C" fn(*mut kernfs_open_file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_425 {
    index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_429 {
    private: *mut (),
    swap: __SlateRecord_mm_types_h_288,
    migrate_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_451 {
    __slate_anon_0: __SlateRecord_mm_types_h_452,
    __page_1: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_452 {
    _flags_1: u64,
    _head_1: u64,
    __slate_anon_2: __SlateRecord_mm_types_h_455,
    _mapcount_1: __SlateRecord_types_h_190,
    _refcount_1: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_455 {
    __slate_anon_0: __SlateRecord_mm_types_h_456,
    _usable_1: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_456 {
    _large_mapcount: __SlateRecord_types_h_190,
    _nr_pages_mapped: __SlateRecord_types_h_190,
    _entire_mapcount: __SlateRecord_types_h_190,
    _pincount: __SlateRecord_types_h_190,
    _mm_id_mapcount: [i32; 2],
    __slate_anon_5: __SlateRecord_mm_types_h_465,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_465 {
    _mm_id: [u32; 2],
    _mm_ids: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_483 {
    __slate_anon_0: __SlateRecord_mm_types_h_484,
    __page_2: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_484 {
    _flags_2: u64,
    _head_2: u64,
    _deferred_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_497 {
    __slate_anon_0: __SlateRecord_mm_types_h_498,
    __page_3: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_498 {
    _flags_3: u64,
    _head_3: u64,
    _hugetlb_subpool: *mut (),
    _hugetlb_cgroup: *mut (),
    _hugetlb_cgroup_rsvd: *mut (),
    _hugetlb_hwpoison: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_ns_type_operations {
    r#type: u32,
    current_may_mount: Option<unsafe extern "C" fn() -> bool>,
    grab_current_ns: Option<unsafe extern "C" fn() -> *mut ns_common>,
    netlink_ns: Option<unsafe extern "C" fn(*mut sock) -> *const ns_common>,
    initial_ns: Option<unsafe extern "C" fn() -> *const ns_common>,
    drop_ns: Option<unsafe extern "C" fn(*mut ns_common)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstat {
    result_mask: u32,
    mode: u16,
    nlink: u32,
    blksize: u32,
    attributes: u64,
    attributes_mask: u64,
    ino: u64,
    dev: u32,
    rdev: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    size: i64,
    atime: timespec64,
    mtime: timespec64,
    ctime: timespec64,
    btime: timespec64,
    blocks: u64,
    mnt_id: u64,
    change_cookie: u64,
    subvol: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    dio_read_offset_align: u32,
    atomic_write_unit_min: u32,
    atomic_write_unit_max: u32,
    atomic_write_unit_max_opt: u32,
    atomic_write_segments_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_group {
    name: *const u8,
    __slate_anon_1: __SlateRecord_sysfs_h_107,
    is_bin_visible: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u16>,
    bin_size: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u64>,
    __slate_anon_4: __SlateRecord_sysfs_h_118,
    bin_attrs: *const *const bin_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_830 {
    remap: __SlateRecord_mm_types_h_831,
    simple_ioremap: __SlateRecord_mm_types_h_837,
    map_kernel: __SlateRecord_mm_types_h_841,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_831 {
    start: u64,
    start_pfn: u64,
    size: u64,
    pgprot: pgprot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_837 {
    start_phys_addr: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_841 {
    start: u64,
    pages: *mut *mut page,
    nr_pages: u64,
    pgoff: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bin_attribute {
    attr: attribute,
    size: u64,
    private: *mut (),
    f_mapping: Option<unsafe extern "C" fn() -> *mut address_space>,
    read: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, i64, i32) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


// buffer for the variables
#[repr(C)]
#[derive(Clone, Copy)]
struct kobject {
    name: *const u8,
    entry: list_head,
    parent: *mut kobject,
    kset: *mut kset,
    ktype: *const kobj_type,
    sd: *mut kernfs_node,
    kref: kref,
    __slate_bits_0: __slate_bits::__SlateBits445U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_926 {
    __slate_anon_0: __SlateRecord_mm_types_h_927,
    vm_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_927 {
    vm_start: u64,
    vm_end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_947 {
    vm_flags: u64,
    flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_type {
    release: Option<unsafe extern "C" fn(*mut kobject)>,
    sysfs_ops: *const sysfs_ops,
    default_groups: *mut *const attribute_group,
    child_ns_type: Option<unsafe extern "C" fn(*const kobject) -> *const kobj_ns_type_operations>,
    namespace: Option<unsafe extern "C" fn(*const kobject) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const kobject, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1176 {
    __slate_anon_0: __SlateRecord_mm_types_h_1181,
    mm_mt: maple_tree,
    mmap_base: u64,
    mmap_legacy_base: u64,
    mmap_compat_base: u64,
    mmap_compat_legacy_base: u64,
    task_size: u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    membarrier_state: __SlateRecord_types_h_190,
    mm_users: __SlateRecord_types_h_190,
    mm_cid: mm_mm_cid,
    sched_cache_grp: *mut sched_cache_group,
    pgtables_bytes: __SlateRecord_types_h_197,
    map_count: i32,
    page_table_lock: spinlock,
    mmap_lock: rw_semaphore,
    mmlist: list_head,
    vma_writer_wait: rcuwait,
    mm_lock_seq: seqcount,
    futex: futex_mm_data,
    hiwater_rss: u64,
    hiwater_vm: u64,
    total_vm: u64,
    locked_vm: u64,
    pinned_vm: __SlateRecord_types_h_197,
    data_vm: u64,
    exec_vm: u64,
    stack_vm: u64,
    __slate_anon_28: __SlateRecord_mm_types_h_1289,
    write_protect_seq: seqcount,
    arg_lock: spinlock,
    start_code: u64,
    end_code: u64,
    start_data: u64,
    end_data: u64,
    start_brk: u64,
    brk: u64,
    start_stack: u64,
    arg_start: u64,
    arg_end: u64,
    env_start: u64,
    env_end: u64,
    saved_auxv: [u64; 56],
    rss_stat: [percpu_counter; 4],
    binfmt: *mut linux_binfmt,
    context: __SlateRecord_mmu_h_84,
    flags: __SlateRecord_mm_types_h_1171,
    ioctx_lock: spinlock,
    ioctx_table: *mut kioctx_table,
    exe_file: *mut file,
    notifier_subscriptions: *mut mmu_notifier_subscriptions,
    tlb_flush_pending: __SlateRecord_types_h_190,
    tlb_flush_batched: __SlateRecord_types_h_190,
    uprobes_state: uprobes_state,
    hugetlb_usage: __SlateRecord_types_h_197,
    async_put_work: work_struct,
    iommu_mm: *mut iommu_mm_data,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1181 {
    mm_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_1289 {
    def_flags: u64,
    def_vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct linux_binfmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_subscriptions {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits470U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits470U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits470U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits470U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iov_iter {
    iter_type: u8,
    nofault: bool,
    data_source: bool,
    iov_offset: u64,
    __slate_anon_4: __SlateRecord_uio_h_58,
    __slate_anon_5: __SlateRecord_uio_h_78,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrinker {
    count_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    scan_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    batch: i64,
    seeks: i32,
    flags: u32,
    refcount: refcount_struct,
    done: completion,
    rcu: callback_head,
    private_data: *mut (),
    list: list_head,
    nr_deferred: *mut __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_107 {
    is_visible: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, i32) -> u16>,
    is_visible_const: Option<unsafe extern "C" fn(*mut kobject, *const attribute, i32) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_118 {
    attrs: *mut *mut attribute,
    attrs_const: *const *const attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
    usage: refcount_struct,
    serial: i32,
    __slate_anon_2: __SlateRecord_key_h_198,
    sem: rw_semaphore,
    user: *mut key_user,
    security: *mut (),
    __slate_anon_6: __SlateRecord_key_h_208,
    last_used_at: i64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    perm: u32,
    quotalen: u16,
    datalen: u16,
    state: i16,
    flags: u64,
    __slate_anon_15: __SlateRecord_key_h_246,
    __slate_anon_16: __SlateRecord_key_h_261,
    restrict_link: *mut key_restriction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_struct {
    __count: refcount_struct,
    epoll_watches: percpu_counter,
    unix_inflight: u64,
    pipe_bufs: __SlateRecord_types_h_197,
    uidhash_node: hlist_node,
    uid: __SlateRecord_uidgid_types_h_9,
    locked_vm: __SlateRecord_types_h_197,
    ratelimit: ratelimit_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
    usage: __SlateRecord_types_h_197,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    suid: __SlateRecord_uidgid_types_h_9,
    sgid: __SlateRecord_uidgid_types_h_13,
    euid: __SlateRecord_uidgid_types_h_9,
    egid: __SlateRecord_uidgid_types_h_13,
    fsuid: __SlateRecord_uidgid_types_h_9,
    fsgid: __SlateRecord_uidgid_types_h_13,
    securebits: u32,
    cap_inheritable: __SlateRecord_capability_h_24,
    cap_permitted: __SlateRecord_capability_h_24,
    cap_effective: __SlateRecord_capability_h_24,
    cap_bset: __SlateRecord_capability_h_24,
    cap_ambient: __SlateRecord_capability_h_24,
    jit_keyring: u8,
    session_keyring: *mut key,
    process_keyring: *mut key,
    thread_keyring: *mut key,
    request_key_auth: *mut key,
    security: *mut (),
    user: *mut user_struct,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    group_info: *mut group_info,
    __slate_anon_25: __SlateRecord_cred_h_147,
}


// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
    count: refcount_struct,
    level: u32,
    lock: spinlock,
    __slate_anon_3: __SlateRecord_pid_h_62,
    tasks: [hlist_head; 4],
    inodes: hlist_head,
    wait_pidfd: wait_queue_head,
    rcu: callback_head,
    numbers: [upid; 0],
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pacct_struct {
    ac_flag: i32,
    ac_exitcode: i64,
    ac_mem: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_58 {
    __ubuf_iovec: iovec,
    __slate_anon_1: __SlateRecord_uio_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uio_h_65 {
    __slate_anon_0: __SlateRecord_uio_h_66,
    count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_66 {
    __iov: *const iovec,
    kvec: *const kvec,
    bvec: *const bio_vec,
    folioq: *const folio_queue,
    xarray: *mut xarray,
    ubuf: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
    sigcnt: refcount_struct,
    live: __SlateRecord_types_h_190,
    nr_threads: i32,
    quick_threads: i32,
    thread_head: list_head,
    wait_chldexit: wait_queue_head,
    curr_target: *mut task_struct,
    shared_pending: sigpending,
    multiprocess: hlist_head,
    group_exit_code: i32,
    notify_count: i32,
    group_exec_task: *mut task_struct,
    group_stop_count: i32,
    flags: u32,
    core_state: *mut core_state,
    __slate_bits_0: __slate_bits::__SlateBits282U0,
    next_posix_timer_id: __SlateRecord_types_h_190,
    posix_timers: hlist_head,
    ignored_posix_timers: hlist_head,
    real_timer: hrtimer,
    it_real_incr: i64,
    it: [cpu_itimer; 2],
    cputimer: thread_group_cputimer,
    posix_cputimers: posix_cputimers,
    pids: [*mut pid; 4],
    tty_old_pgrp: *mut pid,
    leader: i32,
    tty: *mut tty_struct,
    stats_lock: seqlock,
    utime: u64,
    stime: u64,
    cutime: u64,
    cstime: u64,
    gtime: u64,
    cgtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    cnvcsw: u64,
    cnivcsw: u64,
    min_flt: u64,
    maj_flt: u64,
    cmin_flt: u64,
    cmaj_flt: u64,
    inblock: u64,
    oublock: u64,
    cinblock: u64,
    coublock: u64,
    maxrss: u64,
    cmaxrss: u64,
    ioac: task_io_accounting,
    sum_sched_runtime: u64,
    rlim: [rlimit; 16],
    pacct: pacct_struct,
    stats: *mut taskstats,
    audit_tty: u32,
    tty_audit_buf: *mut tty_audit_buf,
    cgroup_threadgroup_rwsem: rw_semaphore,
    oom_flag_origin: bool,
    oom_score_adj: i16,
    oom_score_adj_min: i16,
    oom_mm: *mut mm_struct,
    cred_guard_mutex: mutex,
    exec_update_lock: rw_semaphore,
}


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_388() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqblk {
    dqb_bhardlimit: i64,
    dqb_bsoftlimit: i64,
    dqb_curspace: i64,
    dqb_rsvspace: i64,
    dqb_ihardlimit: i64,
    dqb_isoftlimit: i64,
    dqb_curinodes: i64,
    dqb_btime: i64,
    dqb_itime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_117 {
    __slate_anon_0: __SlateRecord_key_h_118,
    x: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_118 {
    desc_len: u16,
    desc: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqinfo {
    dqi_format: *mut quota_format_type,
    dqi_fmt_id: i32,
    dqi_dirty_list: list_head,
    dqi_flags: u64,
    dqi_bgrace: u32,
    dqi_igrace: u32,
    dqi_max_spc_limit: i64,
    dqi_max_ino_limit: i64,
    dqi_priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot {
    dq_hash: hlist_node,
    dq_inuse: list_head,
    dq_free: list_head,
    dq_dirty: list_head,
    dq_lock: mutex,
    dq_dqb_lock: spinlock,
    dq_count: __SlateRecord_types_h_190,
    dq_sb: *mut super_block,
    dq_id: kqid,
    dq_off: i64,
    dq_flags: u64,
    dq_dqb: mem_dqblk,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_198 {
    graveyard_link: list_head,
    serial_node: rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_user {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_208 {
    expiry: i64,
    revoked_at: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_246 {
    index_key: keyring_index_key,
    __slate_anon_1: __SlateRecord_key_h_248,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_248 {
    hash: u64,
    len_desc: u64,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_261 {
    payload: key_payload,
    __slate_anon_1: __SlateRecord_key_h_263,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_263 {
    name_link: list_head,
    keys: assoc_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_ops {
    check_quota_file: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    write_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    free_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    commit_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot_operations {
    write_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    alloc_dquot: Option<unsafe extern "C" fn(*mut super_block, i32) -> *mut dquot>,
    destroy_dquot: Option<unsafe extern "C" fn(*mut dquot)>,
    acquire_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    mark_dirty: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    get_reserved_space: Option<unsafe extern "C" fn(*mut inode) -> *mut i64>,
    get_projid: Option<unsafe extern "C" fn(*mut inode, *mut __SlateRecord_projid_h_24) -> i32>,
    get_inode_usage: Option<unsafe extern "C" fn(*mut inode, *mut i64) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_dqblk {
    d_fieldmask: i32,
    d_spc_hardlimit: u64,
    d_spc_softlimit: u64,
    d_ino_hardlimit: u64,
    d_ino_softlimit: u64,
    d_space: u64,
    d_ino_count: u64,
    d_ino_timer: i64,
    d_spc_timer: i64,
    d_ino_warns: i32,
    d_spc_warns: i32,
    d_rt_spc_hardlimit: u64,
    d_rt_spc_softlimit: u64,
    d_rt_space: u64,
    d_rt_spc_timer: i64,
    d_rt_spc_warns: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_type_state {
    flags: u32,
    spc_timelimit: u32,
    ino_timelimit: u32,
    rt_spc_timelimit: u32,
    spc_warnlimit: u32,
    ino_warnlimit: u32,
    rt_spc_warnlimit: u32,
    ino: u64,
    blocks: u64,
    nextents: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_info {
    i_fieldmask: i32,
    i_flags: u32,
    i_spc_timelimit: u32,
    i_ino_timelimit: u32,
    i_rt_spc_timelimit: u32,
    i_spc_warnlimit: u32,
    i_ino_warnlimit: u32,
    i_rt_spc_warnlimit: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quotactl_ops {
    quota_on: Option<unsafe extern "C" fn(*mut super_block, i32, i32, *const path) -> i32>,
    quota_off: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    quota_enable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_disable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_sync: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    set_info: Option<unsafe extern "C" fn(*mut super_block, i32, *mut qc_info) -> i32>,
    get_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_nextdqblk: Option<unsafe extern "C" fn(*mut super_block, *mut kqid, *mut qc_dqblk) -> i32>,
    set_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut super_block, *mut qc_state) -> i32>,
    rm_xquota: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_operations {
    alloc_inode: Option<unsafe extern "C" fn(*mut super_block) -> *mut inode>,
    destroy_inode: Option<unsafe extern "C" fn(*mut inode)>,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
    dirty_inode: Option<unsafe extern "C" fn(*mut inode, i32)>,
    write_inode: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    sync_inode_metadata: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    drop_inode: Option<unsafe extern "C" fn(*mut inode) -> i32>,
    evict_inode: Option<unsafe extern "C" fn(*mut inode)>,
    put_super: Option<unsafe extern "C" fn(*mut super_block)>,
    sync_fs: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    freeze_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    freeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    thaw_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    unfreeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    statfs: Option<unsafe extern "C" fn(*mut dentry, *mut kstatfs) -> i32>,
    umount_begin: Option<unsafe extern "C" fn(*mut super_block)>,
    show_options: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_devname: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_path: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_stats: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    quota_read: Option<unsafe extern "C" fn(*mut super_block, i32, *mut u8, u64, i64) -> i64>,
    quota_write: Option<unsafe extern "C" fn(*mut super_block, i32, *const u8, u64, i64) -> i64>,
    get_dquots: Option<unsafe extern "C" fn(*mut inode) -> *mut *mut dquot>,
    nr_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    free_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    remove_bdev: Option<unsafe extern "C" fn(*mut super_block, *mut block_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut super_block)>,
    report_error: Option<unsafe extern "C" fn(*const fserror_event)>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct super_block {
    s_list: list_head,
    s_dev: u32,
    s_super_dev: *mut super_dev,
    s_blocksize_bits: u8,
    s_blocksize: u64,
    s_maxbytes: i64,
    s_type: *mut file_system_type,
    s_op: *const super_operations,
    dq_op: *const dquot_operations,
    s_qcop: *const quotactl_ops,
    s_export_op: *const export_operations,
    s_flags: u64,
    s_iflags: u64,
    s_magic: u64,
    s_root: *mut dentry,
    s_umount: rw_semaphore,
    s_passive: refcount_struct,
    s_active: __SlateRecord_types_h_190,
    s_security: *mut (),
    s_xattr: *const *const xattr_handler,
    s_roots: hlist_head,
    s_roots_lock: spinlock,
    s_mounts: *mut mount,
    s_bdev: *mut block_device,
    s_bdev_file: *mut file,
    s_bdi: *mut backing_dev_info,
    s_mtd: *mut mtd_info,
    s_instances: hlist_node,
    s_quota_types: u32,
    s_dquot: quota_info,
    s_writers: sb_writers,
    s_fs_info: *mut (),
    s_time_gran: u32,
    s_time_min: i64,
    s_time_max: i64,
    s_fsnotify_mask: u32,
    s_fsnotify_info: *mut fsnotify_sb_info,
    s_id: [u8; 32],
    s_uuid: __SlateRecord_uuid_h_21,
    s_uuid_len: u8,
    s_sysfs_name: [u8; 37],
    s_max_links: u32,
    s_d_flags: u32,
    s_vfs_rename_mutex: mutex,
    s_subtype: *const u8,
    __s_d_op: *const dentry_operations,
    s_shrink: *mut shrinker,
    s_remove_count: __SlateRecord_types_h_197,
    s_readonly_remount: i32,
    s_wb_err: u32,
    s_dio_done_wq: *mut workqueue_struct,
    s_pins: hlist_head,
    s_user_ns: *mut user_namespace,
    s_dentry_lru: list_lru,
    s_inode_lru: list_lru,
    rcu: callback_head,
    destroy_work: work_struct,
    s_sync_lock: mutex,
    s_stack_depth: i32,
    __slate_pad_59: [u8; 12],
    s_inode_list_lock: spinlock,
    s_inodes: list_head,
    s_inode_wblist_lock: spinlock,
    s_inodes_wb: list_head,
    s_min_writeback_pages: i64,
    s_pending_errors: refcount_struct,
}


// sb->s_flags.  Note that these mirror the equivalent MS_* flags where
// represented in both.
// Mount read-only
// Update the on-disk [acm]times lazily
#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
    d_flags: u32,
    d_seq: seqcount_spinlock,
    d_hash: hlist_bl_node,
    d_parent: *mut dentry,
    __slate_anon_4: __SlateRecord_dcache_h_99,
    d_inode: *mut inode,
    d_shortname: shortname_store,
    d_op: *const dentry_operations,
    d_sb: *mut super_block,
    d_time: u64,
    d_fsdata: *mut (),
    d_lockref: lockref,
    d_lru: list_head,
    d_sib: hlist_node,
    d_children: hlist_head,
    __slate_anon_15: __SlateRecord_dcache_h_126,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dentry_operations {
    d_revalidate: Option<unsafe extern "C" fn(*mut inode, *const qstr, *mut dentry, u32) -> i32>,
    d_weak_revalidate: Option<unsafe extern "C" fn(*mut dentry, u32) -> i32>,
    d_hash: Option<unsafe extern "C" fn(*const dentry, *mut qstr) -> i32>,
    d_compare: Option<unsafe extern "C" fn(*const dentry, u32, *const u8, *const qstr) -> i32>,
    d_delete: Option<unsafe extern "C" fn(*const dentry) -> i32>,
    d_init: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
    d_release: Option<unsafe extern "C" fn(*mut dentry)>,
    d_prune: Option<unsafe extern "C" fn(*mut dentry)>,
    d_iput: Option<unsafe extern "C" fn(*mut dentry, *mut inode)>,
    d_dname: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> *mut u8>,
    d_automount: Option<unsafe extern "C" fn(*mut path) -> *mut vfsmount>,
    d_manage: Option<unsafe extern "C" fn(*const path, bool) -> i32>,
    d_real: Option<unsafe extern "C" fn(*mut dentry, u32) -> *mut dentry>,
    d_unalias_trylock: Option<unsafe extern "C" fn(*const dentry) -> bool>,
    d_unalias_unlock: Option<unsafe extern "C" fn(*const dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_409() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(mnt as *const vfsmount)).mnt_idmap) as *const *mut mnt_idmap) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits596U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct export_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_sb_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstatfs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mount {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mtd_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


// File is embedded in backing_file object
// File is capable of returning -EAGAIN if I/O will block
// The two FMODE_NONOTIFY* define which fsnotify events should not be generated
// for an open file. These are the possible values of
// (f->f_mode & FMODE_FSNOTIFY_MASK) and their meaning:
//
// FMODE_NONOTIFY - suppress all (incl. non-permission) events.
// FMODE_NONOTIFY_PERM - suppress permission (incl. pre-content) events.
// FMODE_NONOTIFY | FMODE_NONOTIFY_PERM - suppress only FAN_ACCESS_PERM.
// Whiteout is represented by a char device.  The following constants define the
// mode and device number to use.
#[repr(C)]
#[derive(Clone, Copy)]
struct iattr {
    ia_valid: u32,
    ia_mode: u16,
    __slate_anon_2: __SlateRecord_fs_h_272,
    __slate_anon_3: __SlateRecord_fs_h_276,
    ia_size: i64,
    ia_atime: timespec64,
    ia_mtime: timespec64,
    ia_ctime: timespec64,
    ia_file: *mut file,
}


// non-RWF related bits - start at 16
#[repr(C)]
#[derive(Clone, Copy)]
struct kiocb {
    ki_filp: *mut file,
    ki_pos: i64,
    ki_complete: Option<unsafe extern "C" fn(*mut kiocb, i64)>,
    private: *mut (),
    ki_flags: i32,
    ki_ioprio: u16,
    ki_write_stream: u8,
    ki_waitq: *mut wait_page_queue,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_lockref_h_26 {
    lock_count: u64,
    __slate_anon_1: __SlateRecord_lockref_h_30,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_lockref_h_30 {
    lock: spinlock,
    count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space_operations {
    read_folio: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>,
    writepages: Option<unsafe extern "C" fn(*mut address_space, *mut writeback_control) -> i32>,
    dirty_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> bool>,
    readahead: Option<unsafe extern "C" fn(*mut readahead_control)>,
    write_begin: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, *mut *mut folio, *mut *mut ()) -> i32>,
    write_end: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, u32, *mut folio, *mut ()) -> i32>,
    bmap: Option<unsafe extern "C" fn(*mut address_space, u64) -> u64>,
    invalidate_folio: Option<unsafe extern "C" fn(*mut folio, u64, u64)>,
    release_folio: Option<unsafe extern "C" fn(*mut folio, u32) -> bool>,
    free_folio: Option<unsafe extern "C" fn(*mut folio)>,
    direct_IO: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    migrate_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio, *mut folio, u32) -> i32>,
    launder_folio: Option<unsafe extern "C" fn(*mut folio) -> i32>,
    is_partially_uptodate: Option<unsafe extern "C" fn(*mut folio, u64, u64) -> bool>,
    is_dirty_writeback: Option<unsafe extern "C" fn(*mut folio, *mut bool, *mut bool)>,
    error_remove_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> i32>,
    swap_activate: Option<unsafe extern "C" fn(*mut swap_info_struct, *mut file, *mut u64) -> i32>,
    swap_deactivate: Option<unsafe extern "C" fn(*mut file)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
    host: *mut inode,
    i_pages: xarray,
    invalidate_lock: rw_semaphore,
    gfp_mask: u32,
    i_mmap_writable: __SlateRecord_types_h_190,
    i_mmap: rb_root_cached,
    nrpages: u64,
    writeback_index: u64,
    a_ops: *const address_space_operations,
    flags: u64,
    wb_err: u32,
    i_private_lock: spinlock,
    i_mmap_rwsem: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_50 {
    __slate_anon_0: __SlateRecord_dcache_h_51,
    hash_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dcache_h_51 {
    hash: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_99 {
    __d_name: qstr,
    d_name: qstr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_126 {
    d_alias: hlist_node,
    d_in_lookup_hash: hlist_bl_node,
    d_rcu: callback_head,
    waiters: *mut completion_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
    i_mode: u16,
    i_opflags: u16,
    i_flags: u32,
    i_acl: *mut posix_acl,
    i_default_acl: *mut posix_acl,
    i_uid: __SlateRecord_uidgid_types_h_9,
    i_gid: __SlateRecord_uidgid_types_h_13,
    i_op: *const inode_operations,
    i_sb: *mut super_block,
    i_mapping: *mut address_space,
    i_security: *mut (),
    i_ino: u64,
    __slate_anon_12: __SlateRecord_fs_h_790,
    i_rdev: u32,
    i_size: i64,
    i_atime_sec: i64,
    i_mtime_sec: i64,
    i_ctime_sec: i64,
    i_atime_nsec: u32,
    i_mtime_nsec: u32,
    i_ctime_nsec: u32,
    i_generation: u32,
    i_lock: spinlock,
    i_bytes: u16,
    i_blkbits: u8,
    i_write_hint: u8,
    i_blocks: u64,
    i_state: inode_state_flags,
    i_rwsem: rw_semaphore,
    dirtied_when: u64,
    dirtied_time_when: u64,
    i_hash: hlist_node,
    i_io_list: list_head,
    i_lru: list_head,
    i_sb_list: list_head,
    i_wb_list: list_head,
    __slate_anon_36: __SlateRecord_fs_h_834,
    i_version: __SlateRecord_types_h_197,
    i_sequence: __SlateRecord_types_h_197,
    i_count: __SlateRecord_types_h_190,
    i_dio_count: __SlateRecord_types_h_190,
    i_writecount: __SlateRecord_types_h_190,
    i_readcount: __SlateRecord_types_h_190,
    __slate_anon_43: __SlateRecord_fs_h_846,
    i_flctx: *mut file_lock_context,
    i_data: address_space,
    __slate_anon_46: __SlateRecord_fs_h_852,
    __slate_anon_47: __SlateRecord_fs_h_856,
    i_fsnotify_mask: u32,
    i_fsnotify_marks: *mut fsnotify_mark_connector,
    i_private: *mut (),
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
#[repr(C)]
#[derive(Clone, Copy)]
struct fown_struct {
    file: *mut file,
    lock: rwlock,
    pid: *mut pid,
    pid_type: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    euid: __SlateRecord_uidgid_types_h_9,
    signum: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_ra_state {
    start: u64,
    size: u32,
    async_size: u32,
    ra_pages: u32,
    order: u16,
    mmap_miss: u16,
    prev_pos: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
    f_lock: spinlock,
    f_mode: u32,
    f_op: *const file_operations,
    f_mapping: *mut address_space,
    private_data: *mut (),
    f_inode: *mut inode,
    f_flags: u32,
    f_iocb_flags: u32,
    f_cred: *const cred,
    f_owner: *mut fown_struct,
    __slate_anon_10: __SlateRecord_fs_h_1267,
    __slate_anon_11: __SlateRecord_fs_h_1271,
    f_pos: i64,
    f_security: *mut (),
    f_wb_err: u32,
    f_sb_err: u32,
    f_ep: *mut hlist_head,
    __slate_anon_17: __SlateRecord_fs_h_1287,
    f_ref: __SlateRecord_file_ref_h_52,
}


// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
#[repr(C)]
#[derive(Clone, Copy)]
struct dir_context {
    actor: Option<unsafe extern "C" fn(*mut dir_context, *const u8, i32, i64, u64, u32) -> bool>,
    pos: i64,
    count: i32,
    dt_flags_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_operations {
    owner: *mut module,
    fop_flags: u32,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    write_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    iopoll: Option<unsafe extern "C" fn(*mut kiocb, *mut io_comp_batch, u32) -> i32>,
    iterate_shared: Option<unsafe extern "C" fn(*mut file, *mut dir_context) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    flush: Option<unsafe extern "C" fn(*mut file, *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    fsync: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    fasync: Option<unsafe extern "C" fn(i32, *mut file, i32) -> i32>,
    lock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    check_flags: Option<unsafe extern "C" fn(i32) -> i32>,
    flock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    splice_write: Option<unsafe extern "C" fn(*mut pipe_inode_info, *mut file, *mut i64, u64, u32) -> i64>,
    splice_read: Option<unsafe extern "C" fn(*mut file, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut file)>,
    setlease: Option<unsafe extern "C" fn(*mut file, i32, *mut *mut file_lease, *mut *mut ()) -> i32>,
    fallocate: Option<unsafe extern "C" fn(*mut file, i32, i64, i64) -> i64>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut file)>,
    copy_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, u64, u32) -> i64>,
    remap_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, i64, u32) -> i64>,
    fadvise: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    uring_cmd: Option<unsafe extern "C" fn(*mut io_uring_cmd, u32) -> i32>,
    uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut io_uring_cmd, *mut io_comp_batch, u32) -> i32>,
    mmap_prepare: Option<unsafe extern "C" fn(*mut vm_area_desc) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


// File system supports uncached read/write buffered IO
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inode_operations {
    lookup: Option<unsafe extern "C" fn(*mut inode, *mut dentry, u32) -> *mut dentry>,
    get_link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut delayed_call) -> *const u8>,
    permission: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, i32) -> i32>,
    get_inode_acl: Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>,
    readlink: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> i32>,
    create: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> i32>,
    link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut dentry) -> i32>,
    unlink: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    symlink: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *const u8) -> i32>,
    mkdir: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> *mut dentry>,
    rmdir: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    mknod: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16, u32) -> i32>,
    rename: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *mut inode, *mut dentry, u32) -> i32>,
    setattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>,
    getattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>,
    listxattr: Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>,
    fiemap: Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>,
    update_time: Option<unsafe extern "C" fn(*mut inode, u32, u32) -> i32>,
    sync_lazytime: Option<unsafe extern "C" fn(*mut inode)>,
    atomic_open: Option<unsafe extern "C" fn(*mut inode, *mut dentry, *mut file, u32, u16) -> i32>,
    tmpfile: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut file, u16) -> i32>,
    get_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, i32) -> *mut posix_acl>,
    set_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>,
    fileattr_set: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>,
    fileattr_get: Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>,
    get_offset_ctx: Option<unsafe extern "C" fn(*mut inode) -> *mut offset_ctx>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_272 {
    ia_uid: __SlateRecord_uidgid_types_h_9,
    ia_vfsuid: __SlateRecord_mnt_idmapping_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_276 {
    ia_gid: __SlateRecord_uidgid_types_h_13,
    ia_vfsgid: __SlateRecord_mnt_idmapping_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
}


// no suid or xattr security attributes
// Note that nosuid etc flags are inode-specific: setting some file-system
// flags just means all the inodes inherit those flags by default. It might be
// possible to override it selectively if you really wanted to with some
// ioctl() that is not currently implemented.
//
// Exception: SB_RDONLY is always applied to the entire file system.
//
// Unfortunately, it is possible to change a filesystems flags with it mounted
// with files in use.  This means that all of the inodes will not have their
// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
#[repr(C)]
#[derive(Clone, Copy)]
struct file_system_type {
    name: *const u8,
    fs_flags: i32,
    init_fs_context: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    parameters: *const fs_parameter_spec,
    kill_sb: Option<unsafe extern "C" fn(*mut super_block)>,
    owner: *mut module,
    list: hlist_node,
    fs_supers: hlist_head,
    s_lock_key: lock_class_key,
    s_umount_key: lock_class_key,
    s_vfs_rename_key: lock_class_key,
    s_writers_key: [lock_class_key; 3],
    i_lock_key: lock_class_key,
    i_mutex_key: lock_class_key,
    invalidate_lock_key: lock_class_key,
    i_mutex_dir_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_790 {
    i_nlink: u32,
    __i_nlink: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_834 {
    i_dentry: hlist_head,
    i_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_846 {
    i_fop: *const file_operations,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_852 {
    i_devices: list_head,
    i_linklen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_856 {
    i_pipe: *mut pipe_inode_info,
    i_cdev: *mut cdev,
    i_link: *mut u8,
    i_dir_seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
    data: *mut (),
    size: u64,
    alt: u64,
    alt_len: u64,
    extable_base: u64,
    extable_len: u64,
    extable: *const (),
    sym___kernel_sigreturn: i64,
    sym___kernel_rt_sigreturn: i64,
    sym___kernel_vsyscall: i64,
    sym_int80_landing_pad: i64,
    sym_vdso32_sigreturn_landing_pad: i64,
    sym_vdso32_rt_sigreturn_landing_pad: i64,
    sym___futex_list64_try_unlock_cs_start: i64,
    sym___futex_list64_try_unlock_cs_end: i64,
    sym___futex_list32_try_unlock_cs_start: i64,
    sym___futex_list32_try_unlock_cs_end: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1267 {
    f_path: path,
    __f_path: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1271 {
    f_pos_lock: mutex,
    f_pipe: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1287 {
    f_task_work: callback_head,
    f_llist: llist_node,
    f_ra: file_ra_state,
    f_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lease {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param {
    name: *const u8,
    r#mod: *mut module,
    ops: *const kernel_param_ops,
    perm: u16,
    level: i8,
    flags: u8,
    __slate_anon_6: __SlateRecord_moduleparam_h_89,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_array {
    max: u32,
    elemsize: u32,
    num: *mut u32,
    ops: *const kernel_param_ops,
    elem: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_arch_specific {
    num_orcs: u32,
    orc_unwind_ip: *mut i32,
    orc_unwind: *mut orc_entry,
    its_pages: its_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_kobject {
    kobj: kobject,
    r#mod: *mut module,
    drivers_dir: *mut kobject,
    mp: *mut module_param_attrs,
    kobj_completion: *mut completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct module {
    state: u32,
    list: list_head,
    name: [u8; 56],
    mkobj: module_kobject,
    modinfo_attrs: *mut module_attribute,
    version: *const u8,
    srcversion: *const u8,
    imported_namespaces: *const u8,
    holders_dir: *mut kobject,
    syms: *const kernel_symbol,
    crcs: *const u32,
    flagstab: *const u8,
    num_syms: u32,
    param_lock: mutex,
    kp: *mut kernel_param,
    num_kp: u32,
    using_gplonly_symbols: bool,
    async_probe_requested: bool,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    init: Option<unsafe extern "C" fn() -> i32>,
    __slate_pad_21: [u8; 8],
    mem: [module_memory; 7],
    arch: mod_arch_specific,
    taints: u64,
    num_bugs: u32,
    bug_list: list_head,
    bug_table: *mut bug_entry,
    kallsyms: *mut mod_kallsyms,
    core_kallsyms: mod_kallsyms,
    sect_attrs: *mut module_sect_attrs,
    notes_attrs: *mut module_notes_attrs,
    percpu: *mut (),
    percpu_size: u32,
    noinstr_text_start: *mut (),
    noinstr_text_size: u32,
    num_tracepoints: u32,
    tracepoints_ptrs: *const i32,
    num_srcu_structs: u32,
    srcu_struct_ptrs: *mut *mut srcu_struct,
    jump_entries: *mut jump_entry,
    num_jump_entries: u32,
    num_trace_bprintk_fmt: u32,
    trace_bprintk_fmt_start: *mut *const u8,
    trace_events: *mut *mut trace_event_call,
    num_trace_events: u32,
    trace_evals: *mut *mut trace_eval_map,
    num_trace_evals: u32,
    kprobes_text_start: *mut (),
    kprobes_text_size: u32,
    kprobe_blacklist: *mut u64,
    num_kprobe_blacklist: u32,
    num_static_call_sites: i32,
    static_call_sites: *mut static_call_site,
    source_list: list_head,
    target_list: list_head,
    exit: Option<unsafe extern "C" fn()>,
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_symbol {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_sect_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_notes_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits467U0,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_desc {
    mask: cpumask,
    __slate_bits_0: __slate_bits::__SlateBits701U0,
}


#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inb(mut port: u16) -> u8 {
let mut value: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inb %dx, %al", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
value
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_instance_cookie {
    value: u64,
    ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_table_header {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    asl_compiler_id: [u8; 4],
    asl_compiler_revision: u32,
}


// The microseconds/nanosecond delay multiplicators are used to convert a
// constant microseconds/nanoseconds value to a value which can be used by the
// architectures specific implementation to transform it into loops.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
#[inline(always)]
fn udelay(usec: u64) {
if unsafe { __slate_intrinsic_f217c4720f04f6db(usec) } {
if usec >= 20000 {
unsafe { __bad_udelay() };
} else {
unsafe { __const_udelay(usec.wrapping_mul(4295u64)) };
}
} else {
unsafe { __udelay(usec) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_alloc_info {
    pin: i32,
    node: i32,
    __slate_bits_0: __slate_bits::__SlateBits729U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uv_alloc_info {
    limit: i32,
    blade: i32,
    offset: u64,
    name: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_alloc_info {
    r#type: u32,
    flags: u32,
    devid: u32,
    hwirq: u64,
    mask: *const cpumask,
    desc: *mut msi_desc,
    data: *mut (),
    __slate_anon_7: __SlateRecord_hw_irq_h_82,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits747U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_49 {
    __slate_anon_0: __SlateRecord_msi_h_49_2,
    __slate_anon_1: __SlateRecord_msi_h_49_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_2 {
    __slate_bits_0: __slate_bits::__SlateBits751U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits752U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_62 {
    address_lo: u32,
    arch_addr_lo: x86_msi_addr_lo,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_66 {
    address_hi: u32,
    arch_addr_hi: x86_msi_addr_hi,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_70 {
    data: u32,
    arch_data: x86_msi_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_109 {
    msi_mask: u32,
    msix_ctrl: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits761U0,
    default_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_122 {
    mask_pos: u8,
    mask_base: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_data {
    __slate_anon_0: core::mem::MaybeUninit<[u8; 4]>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_early_quirks_c_607 {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_addr_lo {
    __slate_anon_0: __SlateRecord_msi_h_49,
}


type __SlateImage_intel_early_ids = __SlateAlign16<[pci_device_id; 335]>;

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits279U0 {
        #[bits(1)]
        pub dl_throttled: u32,
        #[bits(1)]
        pub dl_yielded: u32,
        #[bits(1)]
        pub dl_non_contending: u32,
        #[bits(1)]
        pub dl_overrun: u32,
        #[bits(1)]
        pub dl_server: u32,
        #[bits(1)]
        pub dl_server_active: u32,
        #[bits(1)]
        pub dl_defer: u32,
        #[bits(1)]
        pub dl_defer_armed: u32,
        #[bits(1)]
        pub dl_defer_running: u32,
        #[bits(1)]
        pub dl_defer_idle: u32,
        #[bits(1)]
        pub dl_bw_attached: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U0 {
        #[bits(1)]
        pub sched_reset_on_fork: u32,
        #[bits(1)]
        pub sched_contributes_to_load: u32,
        #[bits(1)]
        pub sched_migrated: u32,
        #[bits(1)]
        pub sched_task_hot: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U1 {
        #[bits(1)]
        pub sched_remote_wakeup: u32,
        #[bits(1)]
        pub sched_rt_mutex: u32,
        #[bits(1)]
        pub in_execve: u32,
        #[bits(1)]
        pub in_iowait: u32,
        #[bits(1)]
        pub restore_sigmask: u32,
        #[bits(1)]
        pub no_cgroup_migration: u32,
        #[bits(1)]
        pub frozen: u32,
        #[bits(1)]
        pub use_memdelay: u32,
        #[bits(1)]
        pub in_eventfd: u32,
        #[bits(1)]
        pub pasid_activated: u32,
        #[bits(1)]
        pub reported_split_lock: u32,
        #[bits(1)]
        pub in_thrashing: u32,
        #[bits(1)]
        pub in_nf_duplicate: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits282U0 {
        #[bits(1)]
        pub is_child_subreaper: u32,
        #[bits(1)]
        pub has_child_subreaper: u32,
        #[bits(1)]
        pub autoreap: u32,
        #[bits(1)]
        pub timer_create_restore_ids: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits153U0 {
        #[bits(8)]
        pub base1: u16,
        #[bits(4)]
        pub r#type: u16,
        #[bits(1)]
        pub s: u16,
        #[bits(2)]
        pub dpl: u16,
        #[bits(1)]
        pub p: u16,
        #[bits(4)]
        pub limit1: u16,
        #[bits(1)]
        pub avl: u16,
        #[bits(1)]
        pub l: u16,
        #[bits(1)]
        pub d: u16,
        #[bits(1)]
        pub g: u16,
        #[bits(8)]
        pub base2: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits155U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits396U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits596U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits443U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits445U0 {
        #[bits(1)]
        pub state_initialized: u32,
        #[bits(1)]
        pub state_in_sysfs: u32,
        #[bits(1)]
        pub state_add_uevent_sent: u32,
        #[bits(1)]
        pub state_remove_uevent_sent: u32,
        #[bits(1)]
        pub uevent_suppress: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits154U0 {
        #[bits(4)]
        pub sp_reg: u32,
        #[bits(4)]
        pub bp_reg: u32,
        #[bits(3)]
        pub r#type: u32,
        #[bits(1)]
        pub signal: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits470U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits467U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits470U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits470U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits470U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits751U0 {
        #[bits(2)]
        pub reserved_0: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub redirect_hint: u32,
        #[bits(1)]
        pub reserved_1: u32,
        #[bits(7)]
        pub virt_destid_8_14: u32,
        #[bits(8)]
        pub destid_0_7: u32,
        #[bits(12)]
        pub base_address: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits752U0 {
        #[bits(2)]
        pub dmar_reserved_0: u32,
        #[bits(1)]
        pub dmar_index_15: u32,
        #[bits(1)]
        pub dmar_subhandle_valid: u32,
        #[bits(1)]
        pub dmar_format: u32,
        #[bits(15)]
        pub dmar_index_0_14: u32,
        #[bits(12)]
        pub dmar_base_address: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits754U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits747U0 {
        #[bits(8)]
        pub vector: u32,
        #[bits(3)]
        pub delivery_mode: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(2)]
        pub reserved: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub is_level: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits729U0 {
        #[bits(1)]
        pub is_level: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub valid: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits701U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits761U0 {
        #[bits(1)]
        pub is_msix: u8,
        #[bits(3)]
        pub multiple: u8,
        #[bits(3)]
        pub multi_cap: u8,
        #[bits(1)]
        pub can_mask: u8,
        #[bits(1)]
        pub is_64: u8,
        #[bits(1)]
        pub is_virtual: u8,
        #[bits(6, access = na)]
        pub __slate_pad_6: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct x86_msi_addr_hi {
    __slate_bits_0: __slate_bits::__SlateBits754U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
    __slate_anon_0: __SlateRecord_msi_h_62,
    __slate_anon_1: __SlateRecord_msi_h_66,
    __slate_anon_2: __SlateRecord_msi_h_70,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_msi_desc {
    __slate_anon_0: __SlateRecord_msi_h_109,
    msi_attrib: __SlateRecord_msi_h_113,
    __slate_anon_2: __SlateRecord_msi_h_122,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_domain_cookie {
    value: u64,
    ptr: *mut (),
    iobase: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc_data {
    dcookie: msi_domain_cookie,
    icookie: msi_instance_cookie,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
    irq: u32,
    nvec_used: u32,
    dev: *mut device,
    msg: msi_msg,
    affinity: *mut irq_affinity_desc,
    sysfs_attrs: *mut device_attribute,
    write_msi_msg: Option<unsafe extern "C" fn(*mut msi_desc, *mut ())>,
    write_msi_msg_data: *mut (),
    msi_index: u16,
    __slate_anon_9: __SlateRecord_msi_h_202,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_ops {
    get_hwirq: Option<unsafe extern "C" fn(*mut msi_domain_info, *mut irq_alloc_info) -> u64>,
    msi_init: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32, u64, *mut irq_alloc_info) -> i32>,
    msi_free: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32)>,
    msi_prepare: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32, *mut irq_alloc_info) -> i32>,
    msi_teardown: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info)>,
    prepare_desc: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info, *mut msi_desc)>,
    set_desc: Option<unsafe extern "C" fn(*mut irq_alloc_info, *mut msi_desc)>,
    domain_alloc_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32) -> i32>,
    domain_free_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device)>,
    msi_translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_info {
    flags: u32,
    bus_token: u32,
    hwsize: u32,
    ops: *mut msi_domain_ops,
    dev: *mut device,
    chip: *mut irq_chip,
    chip_data: *mut (),
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    handler_data: *mut (),
    handler_name: *const u8,
    alloc_data: *mut irq_alloc_info,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
    supported_flags: u32,
    required_flags: u32,
    chip_flags: u32,
    bus_select_token: u32,
    bus_select_mask: u32,
    prefix: *const u8,
    init_dev_msi_info: Option<unsafe extern "C" fn(*mut device, *mut irq_domain, *mut irq_domain, *mut msi_domain_info) -> bool>,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct e820_entry {
    addr: u64,
    size: u64,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct e820_table {
    nr_entries: u32,
    entries: [e820_entry; 320],
}


// Various well-known legacy memory ranges in physical memory:
#[unsafe(link_section = ".init.text")]
extern "C" fn fix_hypertransport_config(num: i32, slot: i32, func: i32) {
let mut htcfg: u32 = 0;
// we found a hypertransport bus
// make sure that we are broadcasting
// interrupts to all cpus on the ht bus
// if we're using extended apic ids
htcfg = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 104u8) };
if htcfg & 262144u32 != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Detected use of extended apic ids on hypertransport bus\n".as_ptr() as *const u8) };
if htcfg & 131072u32 == 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016Enabling hypertransport extended apic interrupt broadcast\n".as_ptr() as *const u8) };
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016Note this is a bios bug, please contact your hw vendor\n".as_ptr() as *const u8) };
htcfg |= 131072u32;
unsafe { write_pci_config(num as u8, slot as u8, func as u8, 104u8, htcfg) };
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn via_bugs(num: i32, slot: i32, func: i32) {
}

#[unsafe(link_section = ".init.text")]
extern "C" fn nvidia_hpet_check(header: *mut acpi_table_header) -> i32 {
0
}

#[unsafe(link_section = ".init.text")]
extern "C" fn nvidia_bugs(num: i32, slot: i32, func: i32) {
// Only applies to Nvidia root ports (bus 0) and not to
// Nvidia graphics cards with PCI ports on secondary buses.
if num != 0 {
return;
}
// All timer overrides on Nvidia are
// wrong unless HPET is enabled.
// Unfortunately that's not true on many Asus boards.
// We don't know yet how to detect this automatically, but
// at least allow a command line override.
if (unsafe { acpi_use_timer_override }) != 0 {
return;
}
if (unsafe { acpi_table_parse(c"HPET".as_ptr() as *mut u8, Some(nvidia_hpet_check)) }) != 0 {
unsafe {
acpi_skip_timer_override = 1;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Nvidia board detected. Ignoring ACPI timer override.\n".as_ptr() as *const u8) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016If you got timer trouble try acpi_use_timer_override\n".as_ptr() as *const u8) };
}
// RED-PEN skip them on mptables too?
}

#[unsafe(link_section = ".init.text")]
fn ati_ixp4x0_rev(num: i32, slot: i32, func: i32) -> u32 {
let mut d: u32 = 0;
let mut b: u8 = unsafe { read_pci_config_byte(num as u8, slot as u8, func as u8, 172u8) };
let __v1: u8 = ((b as i32) & -33i32) as u8;
b = __v1;
unsafe { write_pci_config_byte(num as u8, slot as u8, func as u8, 172u8, b) };
d = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 112u8) };
d |= 256u32;
unsafe { write_pci_config(num as u8, slot as u8, func as u8, 112u8, d) };
d = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 8u8) };
d &= 255;
d
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_intel_graphics_stolen_res: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym intel_graphics_stolen_res, options(att_syntax));

#[unsafe(link_section = ".init.text")]
extern "C" fn ati_bugs(num: i32, slot: i32, func: i32) {
let mut d: u32 = 0;
let mut b: u8 = 0;
if (unsafe { acpi_use_timer_override }) != 0 {
return;
}
d = ati_ixp4x0_rev(num, slot, func);
if d < 130 {
unsafe {
acpi_skip_timer_override = 1;
}
} else {
// check for IRQ0 interrupt swap
__outb(114u8, 3286u16);
b = __inb(3287u16);
if (b as i32) & 2 == 0 {
unsafe {
acpi_skip_timer_override = 1;
}
}
}
if (unsafe { acpi_skip_timer_override }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016SB4X0 revision 0x%x\n".as_ptr() as *const u8, d) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016Ignoring ACPI timer override.\n".as_ptr() as *const u8) };
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016If you got timer trouble try acpi_use_timer_override\n".as_ptr() as *const u8) };
}
}

#[unsafe(link_section = ".init.text")]
fn ati_sbx00_rev(num: i32, slot: i32, func: i32) -> u32 {
let mut d: u32 = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 8u8) };
d &= 255;
d
}

#[unsafe(link_section = ".init.text")]
extern "C" fn ati_bugs_contd(num: i32, slot: i32, func: i32) {
let mut d: u32 = 0;
let rev: u32 = ati_sbx00_rev(num, slot, func);
if rev >= 64 {
unsafe {
acpi_fix_pin2_polarity = 1;
}
}
// SB600: revisions 0x11, 0x12, 0x13, 0x14, ...
// SB700: revisions 0x39, 0x3a, ...
// SB800: revisions 0x40, 0x41, ...
if rev >= 57 {
return;
}
if (unsafe { acpi_use_timer_override }) != 0 {
return;
}
// check for IRQ0 interrupt swap
d = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 100u8) };
if d & 16384u32 == 0 {
unsafe {
acpi_skip_timer_override = 1;
}
}
if (unsafe { acpi_skip_timer_override }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016SB600 revision 0x%x\n".as_ptr() as *const u8, rev) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016Ignoring ACPI timer override.\n".as_ptr() as *const u8) };
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016If you got timer trouble try acpi_use_timer_override\n".as_ptr() as *const u8) };
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn intel_remapping_check(num: i32, slot: i32, func: i32) {
let mut revision: u8 = 0;
let device: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 2u8) };
revision = unsafe { read_pci_config_byte(num as u8, slot as u8, func as u8, 8u8) };
// Revision <= 13 of all triggering devices id in this quirk
// have a problem draining interrupts when irq remapping is
// enabled, and should be flagged as broken. Additionally
// revision 0x22 of device id 0x3405 has this problem.
if (revision as i32) <= 19 {
unsafe { set_irq_remapping_broken() };
} else {
if (device as i32) == 13317 && (revision as i32) == 34 {
unsafe { set_irq_remapping_broken() };
}
}
}

// Systems with Intel graphics controllers set aside memory exclusively
// for gfx driver use.  This memory is not marked in the E820 as reserved
// or as RAM, and so is subject to overlap from E820 manipulation later
// in the boot process.  On some systems, MMIO space is allocated on top,
// despite the efforts of the "RAM buffer" approach, which simply rounds
// memory boundaries up to 64M to try to catch space that may decode
// as RAM and so is not suitable for MMIO.
#[unsafe(link_section = ".init.text")]
fn i830_tseg_size() -> u64 {
let esmramc: u8 = unsafe { read_pci_config_byte(0u8, 0u8, 0u8, 145u8) };
if (esmramc as i32) & 1i32 == 0 {
return 0;
}
if (esmramc as i32) & 2i32 != 0 {
return MB!(1i32);
} else {
return KB!(512i32);
}
}

#[unsafe(link_section = ".init.text")]
fn i845_tseg_size() -> u64 {
let esmramc: u8 = unsafe { read_pci_config_byte(0u8, 0u8, 0u8, 158u8) };
let tseg_size: u8 = ((esmramc as i32) & 6i32) as u8;
if (esmramc as i32) & 1i32 == 0 {
return 0;
}
match tseg_size as i32 {
4 => {
return KB!(512i32);
}
6 => {
return MB!(1i32);
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Unknown ESMRAMC value: %x!\n".as_ptr() as *const u8, esmramc as i32) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_i845_tseg_size_3, sym __slate_asm_string_i845_tseg_size_4, const 270 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, esmramc as i32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_552: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".init.text")]
fn i85x_tseg_size() -> u64 {
let esmramc: u8 = unsafe { read_pci_config_byte(0u8, 0u8, 0u8, 97u8) };
if (esmramc as i32) & 1i32 == 0 {
return 0;
}
MB!(1i32)
}

#[unsafe(link_section = ".init.text")]
fn i830_mem_size() -> u64 {
(((unsafe { read_pci_config_byte(0u8, 0u8, 0u8, 99u8) }) as i32) as u64).wrapping_mul(MB!(32i32))
}

#[unsafe(link_section = ".init.text")]
fn i85x_mem_size() -> u64 {
(((unsafe { read_pci_config_byte(0u8, 0u8, 1u8, 67u8) }) as i32) as u64).wrapping_mul(MB!(32i32))
}

/// On 830/845/85x the stolen memory base isn't available in any
/// register. We need to calculate it as TOM-TSEG_SIZE-stolen_size.
#[unsafe(link_section = ".init.text")]
extern "C" fn i830_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
i830_mem_size().wrapping_sub(i830_tseg_size()).wrapping_sub(stolen_size)
}

#[unsafe(link_section = ".init.text")]
extern "C" fn i845_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
i830_mem_size().wrapping_sub(i845_tseg_size()).wrapping_sub(stolen_size)
}

#[unsafe(link_section = ".init.text")]
extern "C" fn i85x_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
i85x_mem_size().wrapping_sub(i85x_tseg_size()).wrapping_sub(stolen_size)
}

#[unsafe(link_section = ".init.text")]
extern "C" fn i865_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
let toud: u16 = unsafe { read_pci_config_16(0u8, 0u8, 0u8, 196u8) };
((toud as i32) as u64).wrapping_mul(KB!(64i32)).wrapping_add(i845_tseg_size())
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gen3_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
let mut bsm: u32 = 0;
// Almost universally we can find the Graphics Base of Stolen Memory
// at register BSM (0x5c) in the igfx configuration space. On a few
// (desktop) machines this is also mirrored in the bridge device at
// different locations, or in the MCHBAR.
bsm = unsafe { read_pci_config(num as u8, slot as u8, func as u8, 92u8) };
(bsm & 1048576u32.wrapping_neg()) as u64
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gen11_stolen_base(num: i32, slot: i32, func: i32, stolen_size: u64) -> u64 {
let mut bsm: u64 = (unsafe { read_pci_config(num as u8, slot as u8, func as u8, 192u8) }) as u64;
bsm &= 1048576u32.wrapping_neg() as u64;
let __v2: u64 = ((unsafe { read_pci_config(num as u8, slot as u8, func as u8, 196u8) }) as u64) << 32i32;
bsm |= __v2;
bsm
}

#[unsafe(link_section = ".init.text")]
extern "C" fn i830_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(0u8, 0u8, 0u8, 82u8) };
let gms: u16 = ((gmch_ctrl as i32) & 112i32) as u16;
match gms as i32 {
32 => {
return KB!(512i32);
}
48 => {
return MB!(1i32);
}
64 => {
return MB!(8i32);
// local memory isn't part of the normal address space
}
16 => {
return 0;
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Unknown GMCH_CTRL value: %x!\n".as_ptr() as *const u8, gmch_ctrl as i32) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_i830_stolen_size_3, sym __slate_asm_string_i830_stolen_size_4, const 369 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, gmch_ctrl as i32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_553: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".init.text")]
extern "C" fn gen3_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(0u8, 0u8, 0u8, 82u8) };
let gms: u16 = ((gmch_ctrl as i32) & 240i32) as u16;
match gms as i32 {
16 => {
return MB!(1i32);
}
32 => {
return MB!(4i32);
}
48 => {
return MB!(8i32);
}
64 => {
return MB!(16i32);
}
80 => {
return MB!(32i32);
}
96 => {
return MB!(48i32);
}
112 => {
return MB!(64i32);
}
128 => {
return MB!(128i32);
}
144 => {
return MB!(256i32);
}
160 => {
return MB!(96i32);
}
176 => {
return MB!(160i32);
}
192 => {
return MB!(224i32);
}
208 => {
return MB!(352i32);
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Unknown GMCH_CTRL value: %x!\n".as_ptr() as *const u8, gmch_ctrl as i32) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_gen3_stolen_size_3, sym __slate_asm_string_gen3_stolen_size_4, const 398 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, gmch_ctrl as i32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_554: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".init.text")]
extern "C" fn gen6_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 80u8) };
let gms: u16 = ((gmch_ctrl as i32) >> 3i32 & 31) as u16;
((gms as i32) as u64).wrapping_mul(MB!(32i32))
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gen8_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 80u8) };
let gms: u16 = ((gmch_ctrl as i32) >> 8i32 & 255) as u16;
((gms as i32) as u64).wrapping_mul(MB!(32i32))
}

#[unsafe(link_section = ".init.text")]
extern "C" fn chv_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 80u8) };
let gms: u16 = ((gmch_ctrl as i32) >> 3i32 & 31) as u16;
// 0x0  to 0x10: 32MB increments starting at 0MB
// 0x11 to 0x16: 4MB increments starting at 8MB
// 0x17 to 0x1d: 4MB increments start at 36MB
if (gms as i32) < 17 {
return ((gms as i32) as u64).wrapping_mul(MB!(32i32));
} else {
if (gms as i32) < 23 {
return ((gms as i32).wrapping_sub(17) as u64).wrapping_mul(MB!(4i32)).wrapping_add(MB!(8i32));
} else {
return ((gms as i32).wrapping_sub(23) as u64).wrapping_mul(MB!(4i32)).wrapping_add(MB!(36i32));
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gen9_stolen_size(num: i32, slot: i32, func: i32) -> u64 {
let gmch_ctrl: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 80u8) };
let gms: u16 = ((gmch_ctrl as i32) >> 8i32 & 255) as u16;
// 0x0  to 0xef: 32MB increments starting at 0MB
//
// 0xf0 to 0xfe: 4MB increments starting at 4MB
if (gms as i32) < 240 {
return ((gms as i32) as u64).wrapping_mul(MB!(32i32));
} else {
return ((gms as i32).wrapping_sub(240) as u64).wrapping_mul(MB!(4i32)).wrapping_add(MB!(4i32));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_early_ops {
    stolen_size: Option<unsafe extern "C" fn(i32, i32, i32) -> u64>,
    stolen_base: Option<unsafe extern "C" fn(i32, i32, i32, u64) -> u64>,
}


#[unsafe(link_section = ".init.rodata")]
static i830_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(i830_stolen_size), stolen_base: Some(i830_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static i845_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(i830_stolen_size), stolen_base: Some(i845_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static i85x_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen3_stolen_size), stolen_base: Some(i85x_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static i865_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen3_stolen_size), stolen_base: Some(i865_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static gen3_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen3_stolen_size), stolen_base: Some(gen3_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static gen6_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen6_stolen_size), stolen_base: Some(gen3_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static gen8_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen8_stolen_size), stolen_base: Some(gen3_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static gen9_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen9_stolen_size), stolen_base: Some(gen3_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static chv_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(chv_stolen_size), stolen_base: Some(gen3_stolen_base) });

#[unsafe(link_section = ".init.rodata")]
static gen11_early_ops: __SlateConst<intel_early_ops> = __SlateConst(intel_early_ops { stolen_size: Some(gen9_stolen_size), stolen_base: Some(gen11_stolen_base) });

/// Intel integrated GPUs for which we need to reserve "stolen memory"
#[unsafe(link_section = ".init.rodata")]
static mut intel_early_ids: core::mem::MaybeUninit<__SlateImage_intel_early_ids> = unsafe {
let mut __t0: core::mem::MaybeUninit<__SlateImage_intel_early_ids> = core::mem::MaybeUninit::zeroed();
let __t1 = __t0.as_mut_ptr();
(*__t1).0[0].vendor = 32902u32;
(*__t1).0[0].device = 13687u32;
(*__t1).0[0].subvendor = 4294967295u32;
(*__t1).0[0].subdevice = 4294967295u32;
(*__t1).0[0].class = 196608u32;
(*__t1).0[0].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[0].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(i830_early_ops.0) }) });
(*__t1).0[1].vendor = 32902u32;
(*__t1).0[1].device = 9570u32;
(*__t1).0[1].subvendor = 4294967295u32;
(*__t1).0[1].subdevice = 4294967295u32;
(*__t1).0[1].class = 196608u32;
(*__t1).0[1].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[1].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(i845_early_ops.0) }) });
(*__t1).0[2].vendor = 32902u32;
(*__t1).0[2].device = 13698u32;
(*__t1).0[2].subvendor = 4294967295u32;
(*__t1).0[2].subdevice = 4294967295u32;
(*__t1).0[2].class = 196608u32;
(*__t1).0[2].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[2].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(i85x_early_ops.0) }) });
(*__t1).0[3].vendor = 32902u32;
(*__t1).0[3].device = 13710u32;
(*__t1).0[3].subvendor = 4294967295u32;
(*__t1).0[3].subdevice = 4294967295u32;
(*__t1).0[3].class = 196608u32;
(*__t1).0[3].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[3].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(i85x_early_ops.0) }) });
(*__t1).0[4].vendor = 32902u32;
(*__t1).0[4].device = 9586u32;
(*__t1).0[4].subvendor = 4294967295u32;
(*__t1).0[4].subdevice = 4294967295u32;
(*__t1).0[4].class = 196608u32;
(*__t1).0[4].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[4].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(i865_early_ops.0) }) });
(*__t1).0[5].vendor = 32902u32;
(*__t1).0[5].device = 9602u32;
(*__t1).0[5].subvendor = 4294967295u32;
(*__t1).0[5].subdevice = 4294967295u32;
(*__t1).0[5].class = 196608u32;
(*__t1).0[5].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[5].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[6].vendor = 32902u32;
(*__t1).0[6].device = 9610u32;
(*__t1).0[6].subvendor = 4294967295u32;
(*__t1).0[6].subdevice = 4294967295u32;
(*__t1).0[6].class = 196608u32;
(*__t1).0[6].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[6].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[7].vendor = 32902u32;
(*__t1).0[7].device = 9618u32;
(*__t1).0[7].subvendor = 4294967295u32;
(*__t1).0[7].subdevice = 4294967295u32;
(*__t1).0[7].class = 196608u32;
(*__t1).0[7].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[7].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[8].vendor = 32902u32;
(*__t1).0[8].device = 10098u32;
(*__t1).0[8].subvendor = 4294967295u32;
(*__t1).0[8].subdevice = 4294967295u32;
(*__t1).0[8].class = 196608u32;
(*__t1).0[8].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[8].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[9].vendor = 32902u32;
(*__t1).0[9].device = 10146u32;
(*__t1).0[9].subvendor = 4294967295u32;
(*__t1).0[9].subdevice = 4294967295u32;
(*__t1).0[9].class = 196608u32;
(*__t1).0[9].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[9].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[10].vendor = 32902u32;
(*__t1).0[10].device = 10158u32;
(*__t1).0[10].subvendor = 4294967295u32;
(*__t1).0[10].subdevice = 4294967295u32;
(*__t1).0[10].class = 196608u32;
(*__t1).0[10].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[10].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[11].vendor = 32902u32;
(*__t1).0[11].device = 3888u32;
(*__t1).0[11].subvendor = 4294967295u32;
(*__t1).0[11].subdevice = 4294967295u32;
(*__t1).0[11].class = 196608u32;
(*__t1).0[11].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[11].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[12].vendor = 32902u32;
(*__t1).0[12].device = 3889u32;
(*__t1).0[12].subvendor = 4294967295u32;
(*__t1).0[12].subdevice = 4294967295u32;
(*__t1).0[12].class = 196608u32;
(*__t1).0[12].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[12].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[13].vendor = 32902u32;
(*__t1).0[13].device = 3890u32;
(*__t1).0[13].subvendor = 4294967295u32;
(*__t1).0[13].subdevice = 4294967295u32;
(*__t1).0[13].class = 196608u32;
(*__t1).0[13].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[13].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[14].vendor = 32902u32;
(*__t1).0[14].device = 3891u32;
(*__t1).0[14].subvendor = 4294967295u32;
(*__t1).0[14].subdevice = 4294967295u32;
(*__t1).0[14].class = 196608u32;
(*__t1).0[14].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[14].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[15].vendor = 32902u32;
(*__t1).0[15].device = 40961u32;
(*__t1).0[15].subvendor = 4294967295u32;
(*__t1).0[15].subdevice = 4294967295u32;
(*__t1).0[15].class = 196608u32;
(*__t1).0[15].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[15].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[16].vendor = 32902u32;
(*__t1).0[16].device = 40977u32;
(*__t1).0[16].subvendor = 4294967295u32;
(*__t1).0[16].subdevice = 4294967295u32;
(*__t1).0[16].class = 196608u32;
(*__t1).0[16].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[16].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[17].vendor = 32902u32;
(*__t1).0[17].device = 10610u32;
(*__t1).0[17].subvendor = 4294967295u32;
(*__t1).0[17].subdevice = 4294967295u32;
(*__t1).0[17].class = 196608u32;
(*__t1).0[17].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[17].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[18].vendor = 32902u32;
(*__t1).0[18].device = 10626u32;
(*__t1).0[18].subvendor = 4294967295u32;
(*__t1).0[18].subdevice = 4294967295u32;
(*__t1).0[18].class = 196608u32;
(*__t1).0[18].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[18].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[19].vendor = 32902u32;
(*__t1).0[19].device = 10642u32;
(*__t1).0[19].subvendor = 4294967295u32;
(*__t1).0[19].subdevice = 4294967295u32;
(*__t1).0[19].class = 196608u32;
(*__t1).0[19].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[19].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[20].vendor = 32902u32;
(*__t1).0[20].device = 10658u32;
(*__t1).0[20].subvendor = 4294967295u32;
(*__t1).0[20].subdevice = 4294967295u32;
(*__t1).0[20].class = 196608u32;
(*__t1).0[20].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[20].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[21].vendor = 32902u32;
(*__t1).0[21].device = 10674u32;
(*__t1).0[21].subvendor = 4294967295u32;
(*__t1).0[21].subdevice = 4294967295u32;
(*__t1).0[21].class = 196608u32;
(*__t1).0[21].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[21].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[22].vendor = 32902u32;
(*__t1).0[22].device = 10690u32;
(*__t1).0[22].subvendor = 4294967295u32;
(*__t1).0[22].subdevice = 4294967295u32;
(*__t1).0[22].class = 196608u32;
(*__t1).0[22].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[22].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[23].vendor = 32902u32;
(*__t1).0[23].device = 10706u32;
(*__t1).0[23].subvendor = 4294967295u32;
(*__t1).0[23].subdevice = 4294967295u32;
(*__t1).0[23].class = 196608u32;
(*__t1).0[23].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[23].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[24].vendor = 32902u32;
(*__t1).0[24].device = 10754u32;
(*__t1).0[24].subvendor = 4294967295u32;
(*__t1).0[24].subdevice = 4294967295u32;
(*__t1).0[24].class = 196608u32;
(*__t1).0[24].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[24].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[25].vendor = 32902u32;
(*__t1).0[25].device = 10770u32;
(*__t1).0[25].subvendor = 4294967295u32;
(*__t1).0[25].subdevice = 4294967295u32;
(*__t1).0[25].class = 196608u32;
(*__t1).0[25].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[25].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[26].vendor = 32902u32;
(*__t1).0[26].device = 10818u32;
(*__t1).0[26].subvendor = 4294967295u32;
(*__t1).0[26].subdevice = 4294967295u32;
(*__t1).0[26].class = 196608u32;
(*__t1).0[26].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[26].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[27].vendor = 32902u32;
(*__t1).0[27].device = 11778u32;
(*__t1).0[27].subvendor = 4294967295u32;
(*__t1).0[27].subdevice = 4294967295u32;
(*__t1).0[27].class = 196608u32;
(*__t1).0[27].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[27].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[28].vendor = 32902u32;
(*__t1).0[28].device = 11794u32;
(*__t1).0[28].subvendor = 4294967295u32;
(*__t1).0[28].subdevice = 4294967295u32;
(*__t1).0[28].class = 196608u32;
(*__t1).0[28].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[28].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[29].vendor = 32902u32;
(*__t1).0[29].device = 11810u32;
(*__t1).0[29].subvendor = 4294967295u32;
(*__t1).0[29].subdevice = 4294967295u32;
(*__t1).0[29].class = 196608u32;
(*__t1).0[29].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[29].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[30].vendor = 32902u32;
(*__t1).0[30].device = 11826u32;
(*__t1).0[30].subvendor = 4294967295u32;
(*__t1).0[30].subdevice = 4294967295u32;
(*__t1).0[30].class = 196608u32;
(*__t1).0[30].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[30].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[31].vendor = 32902u32;
(*__t1).0[31].device = 11842u32;
(*__t1).0[31].subvendor = 4294967295u32;
(*__t1).0[31].subdevice = 4294967295u32;
(*__t1).0[31].class = 196608u32;
(*__t1).0[31].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[31].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[32].vendor = 32902u32;
(*__t1).0[32].device = 11922u32;
(*__t1).0[32].subvendor = 4294967295u32;
(*__t1).0[32].subdevice = 4294967295u32;
(*__t1).0[32].class = 196608u32;
(*__t1).0[32].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[32].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[33].vendor = 32902u32;
(*__t1).0[33].device = 66u32;
(*__t1).0[33].subvendor = 4294967295u32;
(*__t1).0[33].subdevice = 4294967295u32;
(*__t1).0[33].class = 196608u32;
(*__t1).0[33].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[33].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[34].vendor = 32902u32;
(*__t1).0[34].device = 70u32;
(*__t1).0[34].subvendor = 4294967295u32;
(*__t1).0[34].subdevice = 4294967295u32;
(*__t1).0[34].class = 196608u32;
(*__t1).0[34].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[34].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen3_early_ops.0) }) });
(*__t1).0[35].vendor = 32902u32;
(*__t1).0[35].device = 258u32;
(*__t1).0[35].subvendor = 4294967295u32;
(*__t1).0[35].subdevice = 4294967295u32;
(*__t1).0[35].class = 196608u32;
(*__t1).0[35].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[35].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[36].vendor = 32902u32;
(*__t1).0[36].device = 266u32;
(*__t1).0[36].subvendor = 4294967295u32;
(*__t1).0[36].subdevice = 4294967295u32;
(*__t1).0[36].class = 196608u32;
(*__t1).0[36].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[36].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[37].vendor = 32902u32;
(*__t1).0[37].device = 274u32;
(*__t1).0[37].subvendor = 4294967295u32;
(*__t1).0[37].subdevice = 4294967295u32;
(*__t1).0[37].class = 196608u32;
(*__t1).0[37].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[37].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[38].vendor = 32902u32;
(*__t1).0[38].device = 290u32;
(*__t1).0[38].subvendor = 4294967295u32;
(*__t1).0[38].subdevice = 4294967295u32;
(*__t1).0[38].class = 196608u32;
(*__t1).0[38].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[38].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[39].vendor = 32902u32;
(*__t1).0[39].device = 262u32;
(*__t1).0[39].subvendor = 4294967295u32;
(*__t1).0[39].subdevice = 4294967295u32;
(*__t1).0[39].class = 196608u32;
(*__t1).0[39].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[39].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[40].vendor = 32902u32;
(*__t1).0[40].device = 278u32;
(*__t1).0[40].subvendor = 4294967295u32;
(*__t1).0[40].subdevice = 4294967295u32;
(*__t1).0[40].class = 196608u32;
(*__t1).0[40].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[40].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[41].vendor = 32902u32;
(*__t1).0[41].device = 294u32;
(*__t1).0[41].subvendor = 4294967295u32;
(*__t1).0[41].subdevice = 4294967295u32;
(*__t1).0[41].class = 196608u32;
(*__t1).0[41].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[41].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[42].vendor = 32902u32;
(*__t1).0[42].device = 342u32;
(*__t1).0[42].subvendor = 4294967295u32;
(*__t1).0[42].subdevice = 4294967295u32;
(*__t1).0[42].class = 196608u32;
(*__t1).0[42].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[42].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[43].vendor = 32902u32;
(*__t1).0[43].device = 358u32;
(*__t1).0[43].subvendor = 4294967295u32;
(*__t1).0[43].subdevice = 4294967295u32;
(*__t1).0[43].class = 196608u32;
(*__t1).0[43].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[43].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[44].vendor = 32902u32;
(*__t1).0[44].device = 338u32;
(*__t1).0[44].subvendor = 4294967295u32;
(*__t1).0[44].subdevice = 4294967295u32;
(*__t1).0[44].class = 196608u32;
(*__t1).0[44].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[44].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[45].vendor = 32902u32;
(*__t1).0[45].device = 346u32;
(*__t1).0[45].subvendor = 4294967295u32;
(*__t1).0[45].subdevice = 4294967295u32;
(*__t1).0[45].class = 196608u32;
(*__t1).0[45].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[45].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[46].vendor = 32902u32;
(*__t1).0[46].device = 354u32;
(*__t1).0[46].subvendor = 4294967295u32;
(*__t1).0[46].subdevice = 4294967295u32;
(*__t1).0[46].class = 196608u32;
(*__t1).0[46].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[46].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[47].vendor = 32902u32;
(*__t1).0[47].device = 362u32;
(*__t1).0[47].subvendor = 4294967295u32;
(*__t1).0[47].subdevice = 4294967295u32;
(*__t1).0[47].class = 196608u32;
(*__t1).0[47].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[47].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[48].vendor = 32902u32;
(*__t1).0[48].device = 2562u32;
(*__t1).0[48].subvendor = 4294967295u32;
(*__t1).0[48].subdevice = 4294967295u32;
(*__t1).0[48].class = 196608u32;
(*__t1).0[48].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[48].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[49].vendor = 32902u32;
(*__t1).0[49].device = 2566u32;
(*__t1).0[49].subvendor = 4294967295u32;
(*__t1).0[49].subdevice = 4294967295u32;
(*__t1).0[49].class = 196608u32;
(*__t1).0[49].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[49].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[50].vendor = 32902u32;
(*__t1).0[50].device = 2570u32;
(*__t1).0[50].subvendor = 4294967295u32;
(*__t1).0[50].subdevice = 4294967295u32;
(*__t1).0[50].class = 196608u32;
(*__t1).0[50].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[50].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[51].vendor = 32902u32;
(*__t1).0[51].device = 2571u32;
(*__t1).0[51].subvendor = 4294967295u32;
(*__t1).0[51].subdevice = 4294967295u32;
(*__t1).0[51].class = 196608u32;
(*__t1).0[51].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[51].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[52].vendor = 32902u32;
(*__t1).0[52].device = 2574u32;
(*__t1).0[52].subvendor = 4294967295u32;
(*__t1).0[52].subdevice = 4294967295u32;
(*__t1).0[52].class = 196608u32;
(*__t1).0[52].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[52].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[53].vendor = 32902u32;
(*__t1).0[53].device = 1026u32;
(*__t1).0[53].subvendor = 4294967295u32;
(*__t1).0[53].subdevice = 4294967295u32;
(*__t1).0[53].class = 196608u32;
(*__t1).0[53].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[53].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[54].vendor = 32902u32;
(*__t1).0[54].device = 1030u32;
(*__t1).0[54].subvendor = 4294967295u32;
(*__t1).0[54].subdevice = 4294967295u32;
(*__t1).0[54].class = 196608u32;
(*__t1).0[54].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[54].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[55].vendor = 32902u32;
(*__t1).0[55].device = 1034u32;
(*__t1).0[55].subvendor = 4294967295u32;
(*__t1).0[55].subdevice = 4294967295u32;
(*__t1).0[55].class = 196608u32;
(*__t1).0[55].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[55].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[56].vendor = 32902u32;
(*__t1).0[56].device = 1035u32;
(*__t1).0[56].subvendor = 4294967295u32;
(*__t1).0[56].subdevice = 4294967295u32;
(*__t1).0[56].class = 196608u32;
(*__t1).0[56].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[56].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[57].vendor = 32902u32;
(*__t1).0[57].device = 1038u32;
(*__t1).0[57].subvendor = 4294967295u32;
(*__t1).0[57].subdevice = 4294967295u32;
(*__t1).0[57].class = 196608u32;
(*__t1).0[57].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[57].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[58].vendor = 32902u32;
(*__t1).0[58].device = 3074u32;
(*__t1).0[58].subvendor = 4294967295u32;
(*__t1).0[58].subdevice = 4294967295u32;
(*__t1).0[58].class = 196608u32;
(*__t1).0[58].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[58].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[59].vendor = 32902u32;
(*__t1).0[59].device = 3078u32;
(*__t1).0[59].subvendor = 4294967295u32;
(*__t1).0[59].subdevice = 4294967295u32;
(*__t1).0[59].class = 196608u32;
(*__t1).0[59].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[59].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[60].vendor = 32902u32;
(*__t1).0[60].device = 3082u32;
(*__t1).0[60].subvendor = 4294967295u32;
(*__t1).0[60].subdevice = 4294967295u32;
(*__t1).0[60].class = 196608u32;
(*__t1).0[60].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[60].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[61].vendor = 32902u32;
(*__t1).0[61].device = 3083u32;
(*__t1).0[61].subvendor = 4294967295u32;
(*__t1).0[61].subdevice = 4294967295u32;
(*__t1).0[61].class = 196608u32;
(*__t1).0[61].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[61].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[62].vendor = 32902u32;
(*__t1).0[62].device = 3086u32;
(*__t1).0[62].subvendor = 4294967295u32;
(*__t1).0[62].subdevice = 4294967295u32;
(*__t1).0[62].class = 196608u32;
(*__t1).0[62].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[62].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[63].vendor = 32902u32;
(*__t1).0[63].device = 3330u32;
(*__t1).0[63].subvendor = 4294967295u32;
(*__t1).0[63].subdevice = 4294967295u32;
(*__t1).0[63].class = 196608u32;
(*__t1).0[63].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[63].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[64].vendor = 32902u32;
(*__t1).0[64].device = 3334u32;
(*__t1).0[64].subvendor = 4294967295u32;
(*__t1).0[64].subdevice = 4294967295u32;
(*__t1).0[64].class = 196608u32;
(*__t1).0[64].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[64].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[65].vendor = 32902u32;
(*__t1).0[65].device = 3338u32;
(*__t1).0[65].subvendor = 4294967295u32;
(*__t1).0[65].subdevice = 4294967295u32;
(*__t1).0[65].class = 196608u32;
(*__t1).0[65].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[65].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[66].vendor = 32902u32;
(*__t1).0[66].device = 3339u32;
(*__t1).0[66].subvendor = 4294967295u32;
(*__t1).0[66].subdevice = 4294967295u32;
(*__t1).0[66].class = 196608u32;
(*__t1).0[66].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[66].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[67].vendor = 32902u32;
(*__t1).0[67].device = 3342u32;
(*__t1).0[67].subvendor = 4294967295u32;
(*__t1).0[67].subdevice = 4294967295u32;
(*__t1).0[67].class = 196608u32;
(*__t1).0[67].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[67].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[68].vendor = 32902u32;
(*__t1).0[68].device = 2578u32;
(*__t1).0[68].subvendor = 4294967295u32;
(*__t1).0[68].subdevice = 4294967295u32;
(*__t1).0[68].class = 196608u32;
(*__t1).0[68].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[68].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[69].vendor = 32902u32;
(*__t1).0[69].device = 2582u32;
(*__t1).0[69].subvendor = 4294967295u32;
(*__t1).0[69].subdevice = 4294967295u32;
(*__t1).0[69].class = 196608u32;
(*__t1).0[69].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[69].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[70].vendor = 32902u32;
(*__t1).0[70].device = 2586u32;
(*__t1).0[70].subvendor = 4294967295u32;
(*__t1).0[70].subdevice = 4294967295u32;
(*__t1).0[70].class = 196608u32;
(*__t1).0[70].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[70].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[71].vendor = 32902u32;
(*__t1).0[71].device = 2587u32;
(*__t1).0[71].subvendor = 4294967295u32;
(*__t1).0[71].subdevice = 4294967295u32;
(*__t1).0[71].class = 196608u32;
(*__t1).0[71].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[71].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[72].vendor = 32902u32;
(*__t1).0[72].device = 2590u32;
(*__t1).0[72].subvendor = 4294967295u32;
(*__t1).0[72].subdevice = 4294967295u32;
(*__t1).0[72].class = 196608u32;
(*__t1).0[72].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[72].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[73].vendor = 32902u32;
(*__t1).0[73].device = 1042u32;
(*__t1).0[73].subvendor = 4294967295u32;
(*__t1).0[73].subdevice = 4294967295u32;
(*__t1).0[73].class = 196608u32;
(*__t1).0[73].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[73].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[74].vendor = 32902u32;
(*__t1).0[74].device = 1046u32;
(*__t1).0[74].subvendor = 4294967295u32;
(*__t1).0[74].subdevice = 4294967295u32;
(*__t1).0[74].class = 196608u32;
(*__t1).0[74].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[74].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[75].vendor = 32902u32;
(*__t1).0[75].device = 1050u32;
(*__t1).0[75].subvendor = 4294967295u32;
(*__t1).0[75].subdevice = 4294967295u32;
(*__t1).0[75].class = 196608u32;
(*__t1).0[75].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[75].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[76].vendor = 32902u32;
(*__t1).0[76].device = 1051u32;
(*__t1).0[76].subvendor = 4294967295u32;
(*__t1).0[76].subdevice = 4294967295u32;
(*__t1).0[76].class = 196608u32;
(*__t1).0[76].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[76].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[77].vendor = 32902u32;
(*__t1).0[77].device = 1054u32;
(*__t1).0[77].subvendor = 4294967295u32;
(*__t1).0[77].subdevice = 4294967295u32;
(*__t1).0[77].class = 196608u32;
(*__t1).0[77].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[77].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[78].vendor = 32902u32;
(*__t1).0[78].device = 3090u32;
(*__t1).0[78].subvendor = 4294967295u32;
(*__t1).0[78].subdevice = 4294967295u32;
(*__t1).0[78].class = 196608u32;
(*__t1).0[78].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[78].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[79].vendor = 32902u32;
(*__t1).0[79].device = 3094u32;
(*__t1).0[79].subvendor = 4294967295u32;
(*__t1).0[79].subdevice = 4294967295u32;
(*__t1).0[79].class = 196608u32;
(*__t1).0[79].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[79].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[80].vendor = 32902u32;
(*__t1).0[80].device = 3098u32;
(*__t1).0[80].subvendor = 4294967295u32;
(*__t1).0[80].subdevice = 4294967295u32;
(*__t1).0[80].class = 196608u32;
(*__t1).0[80].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[80].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[81].vendor = 32902u32;
(*__t1).0[81].device = 3099u32;
(*__t1).0[81].subvendor = 4294967295u32;
(*__t1).0[81].subdevice = 4294967295u32;
(*__t1).0[81].class = 196608u32;
(*__t1).0[81].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[81].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[82].vendor = 32902u32;
(*__t1).0[82].device = 3102u32;
(*__t1).0[82].subvendor = 4294967295u32;
(*__t1).0[82].subdevice = 4294967295u32;
(*__t1).0[82].class = 196608u32;
(*__t1).0[82].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[82].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[83].vendor = 32902u32;
(*__t1).0[83].device = 3346u32;
(*__t1).0[83].subvendor = 4294967295u32;
(*__t1).0[83].subdevice = 4294967295u32;
(*__t1).0[83].class = 196608u32;
(*__t1).0[83].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[83].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[84].vendor = 32902u32;
(*__t1).0[84].device = 3350u32;
(*__t1).0[84].subvendor = 4294967295u32;
(*__t1).0[84].subdevice = 4294967295u32;
(*__t1).0[84].class = 196608u32;
(*__t1).0[84].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[84].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[85].vendor = 32902u32;
(*__t1).0[85].device = 3354u32;
(*__t1).0[85].subvendor = 4294967295u32;
(*__t1).0[85].subdevice = 4294967295u32;
(*__t1).0[85].class = 196608u32;
(*__t1).0[85].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[85].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[86].vendor = 32902u32;
(*__t1).0[86].device = 3355u32;
(*__t1).0[86].subvendor = 4294967295u32;
(*__t1).0[86].subdevice = 4294967295u32;
(*__t1).0[86].class = 196608u32;
(*__t1).0[86].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[86].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[87].vendor = 32902u32;
(*__t1).0[87].device = 3358u32;
(*__t1).0[87].subvendor = 4294967295u32;
(*__t1).0[87].subdevice = 4294967295u32;
(*__t1).0[87].class = 196608u32;
(*__t1).0[87].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[87].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[88].vendor = 32902u32;
(*__t1).0[88].device = 2594u32;
(*__t1).0[88].subvendor = 4294967295u32;
(*__t1).0[88].subdevice = 4294967295u32;
(*__t1).0[88].class = 196608u32;
(*__t1).0[88].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[88].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[89].vendor = 32902u32;
(*__t1).0[89].device = 2598u32;
(*__t1).0[89].subvendor = 4294967295u32;
(*__t1).0[89].subdevice = 4294967295u32;
(*__t1).0[89].class = 196608u32;
(*__t1).0[89].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[89].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[90].vendor = 32902u32;
(*__t1).0[90].device = 2602u32;
(*__t1).0[90].subvendor = 4294967295u32;
(*__t1).0[90].subdevice = 4294967295u32;
(*__t1).0[90].class = 196608u32;
(*__t1).0[90].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[90].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[91].vendor = 32902u32;
(*__t1).0[91].device = 2603u32;
(*__t1).0[91].subvendor = 4294967295u32;
(*__t1).0[91].subdevice = 4294967295u32;
(*__t1).0[91].class = 196608u32;
(*__t1).0[91].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[91].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[92].vendor = 32902u32;
(*__t1).0[92].device = 2606u32;
(*__t1).0[92].subvendor = 4294967295u32;
(*__t1).0[92].subdevice = 4294967295u32;
(*__t1).0[92].class = 196608u32;
(*__t1).0[92].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[92].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[93].vendor = 32902u32;
(*__t1).0[93].device = 1058u32;
(*__t1).0[93].subvendor = 4294967295u32;
(*__t1).0[93].subdevice = 4294967295u32;
(*__t1).0[93].class = 196608u32;
(*__t1).0[93].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[93].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[94].vendor = 32902u32;
(*__t1).0[94].device = 1062u32;
(*__t1).0[94].subvendor = 4294967295u32;
(*__t1).0[94].subdevice = 4294967295u32;
(*__t1).0[94].class = 196608u32;
(*__t1).0[94].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[94].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[95].vendor = 32902u32;
(*__t1).0[95].device = 1066u32;
(*__t1).0[95].subvendor = 4294967295u32;
(*__t1).0[95].subdevice = 4294967295u32;
(*__t1).0[95].class = 196608u32;
(*__t1).0[95].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[95].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[96].vendor = 32902u32;
(*__t1).0[96].device = 1067u32;
(*__t1).0[96].subvendor = 4294967295u32;
(*__t1).0[96].subdevice = 4294967295u32;
(*__t1).0[96].class = 196608u32;
(*__t1).0[96].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[96].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[97].vendor = 32902u32;
(*__t1).0[97].device = 1070u32;
(*__t1).0[97].subvendor = 4294967295u32;
(*__t1).0[97].subdevice = 4294967295u32;
(*__t1).0[97].class = 196608u32;
(*__t1).0[97].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[97].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[98].vendor = 32902u32;
(*__t1).0[98].device = 3106u32;
(*__t1).0[98].subvendor = 4294967295u32;
(*__t1).0[98].subdevice = 4294967295u32;
(*__t1).0[98].class = 196608u32;
(*__t1).0[98].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[98].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[99].vendor = 32902u32;
(*__t1).0[99].device = 3110u32;
(*__t1).0[99].subvendor = 4294967295u32;
(*__t1).0[99].subdevice = 4294967295u32;
(*__t1).0[99].class = 196608u32;
(*__t1).0[99].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[99].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[100].vendor = 32902u32;
(*__t1).0[100].device = 3114u32;
(*__t1).0[100].subvendor = 4294967295u32;
(*__t1).0[100].subdevice = 4294967295u32;
(*__t1).0[100].class = 196608u32;
(*__t1).0[100].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[100].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[101].vendor = 32902u32;
(*__t1).0[101].device = 3115u32;
(*__t1).0[101].subvendor = 4294967295u32;
(*__t1).0[101].subdevice = 4294967295u32;
(*__t1).0[101].class = 196608u32;
(*__t1).0[101].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[101].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[102].vendor = 32902u32;
(*__t1).0[102].device = 3118u32;
(*__t1).0[102].subvendor = 4294967295u32;
(*__t1).0[102].subdevice = 4294967295u32;
(*__t1).0[102].class = 196608u32;
(*__t1).0[102].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[102].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[103].vendor = 32902u32;
(*__t1).0[103].device = 3362u32;
(*__t1).0[103].subvendor = 4294967295u32;
(*__t1).0[103].subdevice = 4294967295u32;
(*__t1).0[103].class = 196608u32;
(*__t1).0[103].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[103].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[104].vendor = 32902u32;
(*__t1).0[104].device = 3366u32;
(*__t1).0[104].subvendor = 4294967295u32;
(*__t1).0[104].subdevice = 4294967295u32;
(*__t1).0[104].class = 196608u32;
(*__t1).0[104].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[104].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[105].vendor = 32902u32;
(*__t1).0[105].device = 3370u32;
(*__t1).0[105].subvendor = 4294967295u32;
(*__t1).0[105].subdevice = 4294967295u32;
(*__t1).0[105].class = 196608u32;
(*__t1).0[105].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[105].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[106].vendor = 32902u32;
(*__t1).0[106].device = 3371u32;
(*__t1).0[106].subvendor = 4294967295u32;
(*__t1).0[106].subdevice = 4294967295u32;
(*__t1).0[106].class = 196608u32;
(*__t1).0[106].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[106].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[107].vendor = 32902u32;
(*__t1).0[107].device = 3374u32;
(*__t1).0[107].subvendor = 4294967295u32;
(*__t1).0[107].subdevice = 4294967295u32;
(*__t1).0[107].class = 196608u32;
(*__t1).0[107].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[107].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen6_early_ops.0) }) });
(*__t1).0[108].vendor = 32902u32;
(*__t1).0[108].device = 5638u32;
(*__t1).0[108].subvendor = 4294967295u32;
(*__t1).0[108].subdevice = 4294967295u32;
(*__t1).0[108].class = 196608u32;
(*__t1).0[108].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[108].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[109].vendor = 32902u32;
(*__t1).0[109].device = 5643u32;
(*__t1).0[109].subvendor = 4294967295u32;
(*__t1).0[109].subdevice = 4294967295u32;
(*__t1).0[109].class = 196608u32;
(*__t1).0[109].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[109].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[110].vendor = 32902u32;
(*__t1).0[110].device = 5646u32;
(*__t1).0[110].subvendor = 4294967295u32;
(*__t1).0[110].subdevice = 4294967295u32;
(*__t1).0[110].class = 196608u32;
(*__t1).0[110].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[110].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[111].vendor = 32902u32;
(*__t1).0[111].device = 5634u32;
(*__t1).0[111].subvendor = 4294967295u32;
(*__t1).0[111].subdevice = 4294967295u32;
(*__t1).0[111].class = 196608u32;
(*__t1).0[111].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[111].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[112].vendor = 32902u32;
(*__t1).0[112].device = 5642u32;
(*__t1).0[112].subvendor = 4294967295u32;
(*__t1).0[112].subdevice = 4294967295u32;
(*__t1).0[112].class = 196608u32;
(*__t1).0[112].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[112].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[113].vendor = 32902u32;
(*__t1).0[113].device = 5645u32;
(*__t1).0[113].subvendor = 4294967295u32;
(*__t1).0[113].subdevice = 4294967295u32;
(*__t1).0[113].class = 196608u32;
(*__t1).0[113].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[113].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[114].vendor = 32902u32;
(*__t1).0[114].device = 5654u32;
(*__t1).0[114].subvendor = 4294967295u32;
(*__t1).0[114].subdevice = 4294967295u32;
(*__t1).0[114].class = 196608u32;
(*__t1).0[114].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[114].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[115].vendor = 32902u32;
(*__t1).0[115].device = 5659u32;
(*__t1).0[115].subvendor = 4294967295u32;
(*__t1).0[115].subdevice = 4294967295u32;
(*__t1).0[115].class = 196608u32;
(*__t1).0[115].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[115].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[116].vendor = 32902u32;
(*__t1).0[116].device = 5662u32;
(*__t1).0[116].subvendor = 4294967295u32;
(*__t1).0[116].subdevice = 4294967295u32;
(*__t1).0[116].class = 196608u32;
(*__t1).0[116].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[116].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[117].vendor = 32902u32;
(*__t1).0[117].device = 5650u32;
(*__t1).0[117].subvendor = 4294967295u32;
(*__t1).0[117].subdevice = 4294967295u32;
(*__t1).0[117].class = 196608u32;
(*__t1).0[117].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[117].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[118].vendor = 32902u32;
(*__t1).0[118].device = 5658u32;
(*__t1).0[118].subvendor = 4294967295u32;
(*__t1).0[118].subdevice = 4294967295u32;
(*__t1).0[118].class = 196608u32;
(*__t1).0[118].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[118].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[119].vendor = 32902u32;
(*__t1).0[119].device = 5661u32;
(*__t1).0[119].subvendor = 4294967295u32;
(*__t1).0[119].subdevice = 4294967295u32;
(*__t1).0[119].class = 196608u32;
(*__t1).0[119].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[119].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[120].vendor = 32902u32;
(*__t1).0[120].device = 5670u32;
(*__t1).0[120].subvendor = 4294967295u32;
(*__t1).0[120].subdevice = 4294967295u32;
(*__t1).0[120].class = 196608u32;
(*__t1).0[120].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[120].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[121].vendor = 32902u32;
(*__t1).0[121].device = 5675u32;
(*__t1).0[121].subvendor = 4294967295u32;
(*__t1).0[121].subdevice = 4294967295u32;
(*__t1).0[121].class = 196608u32;
(*__t1).0[121].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[121].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[122].vendor = 32902u32;
(*__t1).0[122].device = 5678u32;
(*__t1).0[122].subvendor = 4294967295u32;
(*__t1).0[122].subdevice = 4294967295u32;
(*__t1).0[122].class = 196608u32;
(*__t1).0[122].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[122].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[123].vendor = 32902u32;
(*__t1).0[123].device = 5666u32;
(*__t1).0[123].subvendor = 4294967295u32;
(*__t1).0[123].subdevice = 4294967295u32;
(*__t1).0[123].class = 196608u32;
(*__t1).0[123].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[123].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[124].vendor = 32902u32;
(*__t1).0[124].device = 5674u32;
(*__t1).0[124].subvendor = 4294967295u32;
(*__t1).0[124].subdevice = 4294967295u32;
(*__t1).0[124].class = 196608u32;
(*__t1).0[124].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[124].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[125].vendor = 32902u32;
(*__t1).0[125].device = 5677u32;
(*__t1).0[125].subvendor = 4294967295u32;
(*__t1).0[125].subdevice = 4294967295u32;
(*__t1).0[125].class = 196608u32;
(*__t1).0[125].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[125].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[126].vendor = 32902u32;
(*__t1).0[126].device = 5686u32;
(*__t1).0[126].subvendor = 4294967295u32;
(*__t1).0[126].subdevice = 4294967295u32;
(*__t1).0[126].class = 196608u32;
(*__t1).0[126].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[126].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[127].vendor = 32902u32;
(*__t1).0[127].device = 5691u32;
(*__t1).0[127].subvendor = 4294967295u32;
(*__t1).0[127].subdevice = 4294967295u32;
(*__t1).0[127].class = 196608u32;
(*__t1).0[127].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[127].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[128].vendor = 32902u32;
(*__t1).0[128].device = 5694u32;
(*__t1).0[128].subvendor = 4294967295u32;
(*__t1).0[128].subdevice = 4294967295u32;
(*__t1).0[128].class = 196608u32;
(*__t1).0[128].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[128].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[129].vendor = 32902u32;
(*__t1).0[129].device = 5682u32;
(*__t1).0[129].subvendor = 4294967295u32;
(*__t1).0[129].subdevice = 4294967295u32;
(*__t1).0[129].class = 196608u32;
(*__t1).0[129].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[129].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[130].vendor = 32902u32;
(*__t1).0[130].device = 5690u32;
(*__t1).0[130].subvendor = 4294967295u32;
(*__t1).0[130].subdevice = 4294967295u32;
(*__t1).0[130].class = 196608u32;
(*__t1).0[130].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[130].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[131].vendor = 32902u32;
(*__t1).0[131].device = 5693u32;
(*__t1).0[131].subvendor = 4294967295u32;
(*__t1).0[131].subdevice = 4294967295u32;
(*__t1).0[131].class = 196608u32;
(*__t1).0[131].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[131].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen8_early_ops.0) }) });
(*__t1).0[132].vendor = 32902u32;
(*__t1).0[132].device = 8880u32;
(*__t1).0[132].subvendor = 4294967295u32;
(*__t1).0[132].subdevice = 4294967295u32;
(*__t1).0[132].class = 196608u32;
(*__t1).0[132].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[132].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(chv_early_ops.0) }) });
(*__t1).0[133].vendor = 32902u32;
(*__t1).0[133].device = 8881u32;
(*__t1).0[133].subvendor = 4294967295u32;
(*__t1).0[133].subdevice = 4294967295u32;
(*__t1).0[133].class = 196608u32;
(*__t1).0[133].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[133].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(chv_early_ops.0) }) });
(*__t1).0[134].vendor = 32902u32;
(*__t1).0[134].device = 8882u32;
(*__t1).0[134].subvendor = 4294967295u32;
(*__t1).0[134].subdevice = 4294967295u32;
(*__t1).0[134].class = 196608u32;
(*__t1).0[134].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[134].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(chv_early_ops.0) }) });
(*__t1).0[135].vendor = 32902u32;
(*__t1).0[135].device = 8883u32;
(*__t1).0[135].subvendor = 4294967295u32;
(*__t1).0[135].subdevice = 4294967295u32;
(*__t1).0[135].class = 196608u32;
(*__t1).0[135].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[135].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(chv_early_ops.0) }) });
(*__t1).0[136].vendor = 32902u32;
(*__t1).0[136].device = 6406u32;
(*__t1).0[136].subvendor = 4294967295u32;
(*__t1).0[136].subdevice = 4294967295u32;
(*__t1).0[136].class = 196608u32;
(*__t1).0[136].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[136].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[137].vendor = 32902u32;
(*__t1).0[137].device = 6419u32;
(*__t1).0[137].subvendor = 4294967295u32;
(*__t1).0[137].subdevice = 4294967295u32;
(*__t1).0[137].class = 196608u32;
(*__t1).0[137].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[137].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[138].vendor = 32902u32;
(*__t1).0[138].device = 6414u32;
(*__t1).0[138].subvendor = 4294967295u32;
(*__t1).0[138].subdevice = 4294967295u32;
(*__t1).0[138].class = 196608u32;
(*__t1).0[138].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[138].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[139].vendor = 32902u32;
(*__t1).0[139].device = 6421u32;
(*__t1).0[139].subvendor = 4294967295u32;
(*__t1).0[139].subdevice = 4294967295u32;
(*__t1).0[139].class = 196608u32;
(*__t1).0[139].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[139].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[140].vendor = 32902u32;
(*__t1).0[140].device = 6402u32;
(*__t1).0[140].subvendor = 4294967295u32;
(*__t1).0[140].subdevice = 4294967295u32;
(*__t1).0[140].class = 196608u32;
(*__t1).0[140].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[140].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[141].vendor = 32902u32;
(*__t1).0[141].device = 6410u32;
(*__t1).0[141].subvendor = 4294967295u32;
(*__t1).0[141].subdevice = 4294967295u32;
(*__t1).0[141].class = 196608u32;
(*__t1).0[141].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[141].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[142].vendor = 32902u32;
(*__t1).0[142].device = 6411u32;
(*__t1).0[142].subvendor = 4294967295u32;
(*__t1).0[142].subdevice = 4294967295u32;
(*__t1).0[142].class = 196608u32;
(*__t1).0[142].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[142].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[143].vendor = 32902u32;
(*__t1).0[143].device = 6423u32;
(*__t1).0[143].subvendor = 4294967295u32;
(*__t1).0[143].subdevice = 4294967295u32;
(*__t1).0[143].class = 196608u32;
(*__t1).0[143].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[143].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[144].vendor = 32902u32;
(*__t1).0[144].device = 6422u32;
(*__t1).0[144].subvendor = 4294967295u32;
(*__t1).0[144].subdevice = 4294967295u32;
(*__t1).0[144].class = 196608u32;
(*__t1).0[144].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[144].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[145].vendor = 32902u32;
(*__t1).0[145].device = 6433u32;
(*__t1).0[145].subvendor = 4294967295u32;
(*__t1).0[145].subdevice = 4294967295u32;
(*__t1).0[145].class = 196608u32;
(*__t1).0[145].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[145].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[146].vendor = 32902u32;
(*__t1).0[146].device = 6430u32;
(*__t1).0[146].subvendor = 4294967295u32;
(*__t1).0[146].subdevice = 4294967295u32;
(*__t1).0[146].class = 196608u32;
(*__t1).0[146].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[146].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[147].vendor = 32902u32;
(*__t1).0[147].device = 6418u32;
(*__t1).0[147].subvendor = 4294967295u32;
(*__t1).0[147].subdevice = 4294967295u32;
(*__t1).0[147].class = 196608u32;
(*__t1).0[147].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[147].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[148].vendor = 32902u32;
(*__t1).0[148].device = 6426u32;
(*__t1).0[148].subvendor = 4294967295u32;
(*__t1).0[148].subdevice = 4294967295u32;
(*__t1).0[148].class = 196608u32;
(*__t1).0[148].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[148].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[149].vendor = 32902u32;
(*__t1).0[149].device = 6427u32;
(*__t1).0[149].subvendor = 4294967295u32;
(*__t1).0[149].subdevice = 4294967295u32;
(*__t1).0[149].class = 196608u32;
(*__t1).0[149].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[149].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[150].vendor = 32902u32;
(*__t1).0[150].device = 6429u32;
(*__t1).0[150].subvendor = 4294967295u32;
(*__t1).0[150].subdevice = 4294967295u32;
(*__t1).0[150].class = 196608u32;
(*__t1).0[150].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[150].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[151].vendor = 32902u32;
(*__t1).0[151].device = 6435u32;
(*__t1).0[151].subvendor = 4294967295u32;
(*__t1).0[151].subdevice = 4294967295u32;
(*__t1).0[151].class = 196608u32;
(*__t1).0[151].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[151].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[152].vendor = 32902u32;
(*__t1).0[152].device = 6438u32;
(*__t1).0[152].subvendor = 4294967295u32;
(*__t1).0[152].subdevice = 4294967295u32;
(*__t1).0[152].class = 196608u32;
(*__t1).0[152].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[152].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[153].vendor = 32902u32;
(*__t1).0[153].device = 6439u32;
(*__t1).0[153].subvendor = 4294967295u32;
(*__t1).0[153].subdevice = 4294967295u32;
(*__t1).0[153].class = 196608u32;
(*__t1).0[153].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[153].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[154].vendor = 32902u32;
(*__t1).0[154].device = 6442u32;
(*__t1).0[154].subvendor = 4294967295u32;
(*__t1).0[154].subdevice = 4294967295u32;
(*__t1).0[154].class = 196608u32;
(*__t1).0[154].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[154].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[155].vendor = 32902u32;
(*__t1).0[155].device = 6443u32;
(*__t1).0[155].subvendor = 4294967295u32;
(*__t1).0[155].subdevice = 4294967295u32;
(*__t1).0[155].class = 196608u32;
(*__t1).0[155].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[155].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[156].vendor = 32902u32;
(*__t1).0[156].device = 6445u32;
(*__t1).0[156].subvendor = 4294967295u32;
(*__t1).0[156].subdevice = 4294967295u32;
(*__t1).0[156].class = 196608u32;
(*__t1).0[156].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[156].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[157].vendor = 32902u32;
(*__t1).0[157].device = 6450u32;
(*__t1).0[157].subvendor = 4294967295u32;
(*__t1).0[157].subdevice = 4294967295u32;
(*__t1).0[157].class = 196608u32;
(*__t1).0[157].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[157].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[158].vendor = 32902u32;
(*__t1).0[158].device = 6458u32;
(*__t1).0[158].subvendor = 4294967295u32;
(*__t1).0[158].subdevice = 4294967295u32;
(*__t1).0[158].class = 196608u32;
(*__t1).0[158].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[158].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[159].vendor = 32902u32;
(*__t1).0[159].device = 6459u32;
(*__t1).0[159].subvendor = 4294967295u32;
(*__t1).0[159].subdevice = 4294967295u32;
(*__t1).0[159].class = 196608u32;
(*__t1).0[159].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[159].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[160].vendor = 32902u32;
(*__t1).0[160].device = 6461u32;
(*__t1).0[160].subvendor = 4294967295u32;
(*__t1).0[160].subdevice = 4294967295u32;
(*__t1).0[160].class = 196608u32;
(*__t1).0[160].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[160].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[161].vendor = 32902u32;
(*__t1).0[161].device = 2692u32;
(*__t1).0[161].subvendor = 4294967295u32;
(*__t1).0[161].subdevice = 4294967295u32;
(*__t1).0[161].class = 196608u32;
(*__t1).0[161].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[161].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[162].vendor = 32902u32;
(*__t1).0[162].device = 6788u32;
(*__t1).0[162].subvendor = 4294967295u32;
(*__t1).0[162].subdevice = 4294967295u32;
(*__t1).0[162].class = 196608u32;
(*__t1).0[162].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[162].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[163].vendor = 32902u32;
(*__t1).0[163].device = 6789u32;
(*__t1).0[163].subvendor = 4294967295u32;
(*__t1).0[163].subdevice = 4294967295u32;
(*__t1).0[163].class = 196608u32;
(*__t1).0[163].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[163].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[164].vendor = 32902u32;
(*__t1).0[164].device = 23172u32;
(*__t1).0[164].subvendor = 4294967295u32;
(*__t1).0[164].subdevice = 4294967295u32;
(*__t1).0[164].class = 196608u32;
(*__t1).0[164].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[164].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[165].vendor = 32902u32;
(*__t1).0[165].device = 23173u32;
(*__t1).0[165].subvendor = 4294967295u32;
(*__t1).0[165].subdevice = 4294967295u32;
(*__t1).0[165].class = 196608u32;
(*__t1).0[165].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[165].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[166].vendor = 32902u32;
(*__t1).0[166].device = 22790u32;
(*__t1).0[166].subvendor = 4294967295u32;
(*__t1).0[166].subdevice = 4294967295u32;
(*__t1).0[166].class = 196608u32;
(*__t1).0[166].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[166].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[167].vendor = 32902u32;
(*__t1).0[167].device = 22803u32;
(*__t1).0[167].subvendor = 4294967295u32;
(*__t1).0[167].subdevice = 4294967295u32;
(*__t1).0[167].class = 196608u32;
(*__t1).0[167].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[167].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[168].vendor = 32902u32;
(*__t1).0[168].device = 22798u32;
(*__t1).0[168].subvendor = 4294967295u32;
(*__t1).0[168].subdevice = 4294967295u32;
(*__t1).0[168].class = 196608u32;
(*__t1).0[168].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[168].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[169].vendor = 32902u32;
(*__t1).0[169].device = 22805u32;
(*__t1).0[169].subvendor = 4294967295u32;
(*__t1).0[169].subdevice = 4294967295u32;
(*__t1).0[169].class = 196608u32;
(*__t1).0[169].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[169].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[170].vendor = 32902u32;
(*__t1).0[170].device = 22786u32;
(*__t1).0[170].subvendor = 4294967295u32;
(*__t1).0[170].subdevice = 4294967295u32;
(*__t1).0[170].class = 196608u32;
(*__t1).0[170].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[170].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[171].vendor = 32902u32;
(*__t1).0[171].device = 22792u32;
(*__t1).0[171].subvendor = 4294967295u32;
(*__t1).0[171].subdevice = 4294967295u32;
(*__t1).0[171].class = 196608u32;
(*__t1).0[171].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[171].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[172].vendor = 32902u32;
(*__t1).0[172].device = 22794u32;
(*__t1).0[172].subvendor = 4294967295u32;
(*__t1).0[172].subdevice = 4294967295u32;
(*__t1).0[172].class = 196608u32;
(*__t1).0[172].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[172].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[173].vendor = 32902u32;
(*__t1).0[173].device = 22795u32;
(*__t1).0[173].subvendor = 4294967295u32;
(*__t1).0[173].subdevice = 4294967295u32;
(*__t1).0[173].class = 196608u32;
(*__t1).0[173].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[173].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[174].vendor = 32902u32;
(*__t1).0[174].device = 22806u32;
(*__t1).0[174].subvendor = 4294967295u32;
(*__t1).0[174].subdevice = 4294967295u32;
(*__t1).0[174].class = 196608u32;
(*__t1).0[174].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[174].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[175].vendor = 32902u32;
(*__t1).0[175].device = 22817u32;
(*__t1).0[175].subvendor = 4294967295u32;
(*__t1).0[175].subdevice = 4294967295u32;
(*__t1).0[175].class = 196608u32;
(*__t1).0[175].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[175].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[176].vendor = 32902u32;
(*__t1).0[176].device = 22814u32;
(*__t1).0[176].subvendor = 4294967295u32;
(*__t1).0[176].subdevice = 4294967295u32;
(*__t1).0[176].class = 196608u32;
(*__t1).0[176].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[176].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[177].vendor = 32902u32;
(*__t1).0[177].device = 22802u32;
(*__t1).0[177].subvendor = 4294967295u32;
(*__t1).0[177].subdevice = 4294967295u32;
(*__t1).0[177].class = 196608u32;
(*__t1).0[177].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[177].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[178].vendor = 32902u32;
(*__t1).0[178].device = 22807u32;
(*__t1).0[178].subvendor = 4294967295u32;
(*__t1).0[178].subdevice = 4294967295u32;
(*__t1).0[178].class = 196608u32;
(*__t1).0[178].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[178].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[179].vendor = 32902u32;
(*__t1).0[179].device = 22810u32;
(*__t1).0[179].subvendor = 4294967295u32;
(*__t1).0[179].subdevice = 4294967295u32;
(*__t1).0[179].class = 196608u32;
(*__t1).0[179].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[179].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[180].vendor = 32902u32;
(*__t1).0[180].device = 22811u32;
(*__t1).0[180].subvendor = 4294967295u32;
(*__t1).0[180].subdevice = 4294967295u32;
(*__t1).0[180].class = 196608u32;
(*__t1).0[180].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[180].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[181].vendor = 32902u32;
(*__t1).0[181].device = 22813u32;
(*__t1).0[181].subvendor = 4294967295u32;
(*__t1).0[181].subdevice = 4294967295u32;
(*__t1).0[181].class = 196608u32;
(*__t1).0[181].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[181].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[182].vendor = 32902u32;
(*__t1).0[182].device = 22822u32;
(*__t1).0[182].subvendor = 4294967295u32;
(*__t1).0[182].subdevice = 4294967295u32;
(*__t1).0[182].class = 196608u32;
(*__t1).0[182].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[182].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[183].vendor = 32902u32;
(*__t1).0[183].device = 22819u32;
(*__t1).0[183].subvendor = 4294967295u32;
(*__t1).0[183].subdevice = 4294967295u32;
(*__t1).0[183].class = 196608u32;
(*__t1).0[183].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[183].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[184].vendor = 32902u32;
(*__t1).0[184].device = 22823u32;
(*__t1).0[184].subvendor = 4294967295u32;
(*__t1).0[184].subdevice = 4294967295u32;
(*__t1).0[184].class = 196608u32;
(*__t1).0[184].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[184].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[185].vendor = 32902u32;
(*__t1).0[185].device = 22843u32;
(*__t1).0[185].subvendor = 4294967295u32;
(*__t1).0[185].subdevice = 4294967295u32;
(*__t1).0[185].class = 196608u32;
(*__t1).0[185].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[185].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[186].vendor = 32902u32;
(*__t1).0[186].device = 22812u32;
(*__t1).0[186].subvendor = 4294967295u32;
(*__t1).0[186].subdevice = 4294967295u32;
(*__t1).0[186].class = 196608u32;
(*__t1).0[186].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[186].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[187].vendor = 32902u32;
(*__t1).0[187].device = 34752u32;
(*__t1).0[187].subvendor = 4294967295u32;
(*__t1).0[187].subdevice = 4294967295u32;
(*__t1).0[187].class = 196608u32;
(*__t1).0[187].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[187].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[188].vendor = 32902u32;
(*__t1).0[188].device = 16016u32;
(*__t1).0[188].subvendor = 4294967295u32;
(*__t1).0[188].subdevice = 4294967295u32;
(*__t1).0[188].class = 196608u32;
(*__t1).0[188].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[188].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[189].vendor = 32902u32;
(*__t1).0[189].device = 16019u32;
(*__t1).0[189].subvendor = 4294967295u32;
(*__t1).0[189].subdevice = 4294967295u32;
(*__t1).0[189].class = 196608u32;
(*__t1).0[189].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[189].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[190].vendor = 32902u32;
(*__t1).0[190].device = 16025u32;
(*__t1).0[190].subvendor = 4294967295u32;
(*__t1).0[190].subdevice = 4294967295u32;
(*__t1).0[190].class = 196608u32;
(*__t1).0[190].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[190].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[191].vendor = 32902u32;
(*__t1).0[191].device = 16017u32;
(*__t1).0[191].subvendor = 4294967295u32;
(*__t1).0[191].subdevice = 4294967295u32;
(*__t1).0[191].class = 196608u32;
(*__t1).0[191].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[191].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[192].vendor = 32902u32;
(*__t1).0[192].device = 16018u32;
(*__t1).0[192].subvendor = 4294967295u32;
(*__t1).0[192].subdevice = 4294967295u32;
(*__t1).0[192].class = 196608u32;
(*__t1).0[192].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[192].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[193].vendor = 32902u32;
(*__t1).0[193].device = 16022u32;
(*__t1).0[193].subvendor = 4294967295u32;
(*__t1).0[193].subdevice = 4294967295u32;
(*__t1).0[193].class = 196608u32;
(*__t1).0[193].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[193].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[194].vendor = 32902u32;
(*__t1).0[194].device = 16024u32;
(*__t1).0[194].subvendor = 4294967295u32;
(*__t1).0[194].subdevice = 4294967295u32;
(*__t1).0[194].class = 196608u32;
(*__t1).0[194].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[194].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[195].vendor = 32902u32;
(*__t1).0[195].device = 16026u32;
(*__t1).0[195].subvendor = 4294967295u32;
(*__t1).0[195].subdevice = 4294967295u32;
(*__t1).0[195].class = 196608u32;
(*__t1).0[195].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[195].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[196].vendor = 32902u32;
(*__t1).0[196].device = 16028u32;
(*__t1).0[196].subvendor = 4294967295u32;
(*__t1).0[196].subdevice = 4294967295u32;
(*__t1).0[196].class = 196608u32;
(*__t1).0[196].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[196].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[197].vendor = 32902u32;
(*__t1).0[197].device = 16020u32;
(*__t1).0[197].subvendor = 4294967295u32;
(*__t1).0[197].subdevice = 4294967295u32;
(*__t1).0[197].class = 196608u32;
(*__t1).0[197].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[197].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[198].vendor = 32902u32;
(*__t1).0[198].device = 16027u32;
(*__t1).0[198].subvendor = 4294967295u32;
(*__t1).0[198].subdevice = 4294967295u32;
(*__t1).0[198].class = 196608u32;
(*__t1).0[198].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[198].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[199].vendor = 32902u32;
(*__t1).0[199].device = 16041u32;
(*__t1).0[199].subvendor = 4294967295u32;
(*__t1).0[199].subdevice = 4294967295u32;
(*__t1).0[199].class = 196608u32;
(*__t1).0[199].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[199].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[200].vendor = 32902u32;
(*__t1).0[200].device = 16037u32;
(*__t1).0[200].subvendor = 4294967295u32;
(*__t1).0[200].subdevice = 4294967295u32;
(*__t1).0[200].class = 196608u32;
(*__t1).0[200].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[200].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[201].vendor = 32902u32;
(*__t1).0[201].device = 16038u32;
(*__t1).0[201].subvendor = 4294967295u32;
(*__t1).0[201].subdevice = 4294967295u32;
(*__t1).0[201].class = 196608u32;
(*__t1).0[201].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[201].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[202].vendor = 32902u32;
(*__t1).0[202].device = 16039u32;
(*__t1).0[202].subvendor = 4294967295u32;
(*__t1).0[202].subdevice = 4294967295u32;
(*__t1).0[202].class = 196608u32;
(*__t1).0[202].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[202].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[203].vendor = 32902u32;
(*__t1).0[203].device = 16040u32;
(*__t1).0[203].subvendor = 4294967295u32;
(*__t1).0[203].subdevice = 4294967295u32;
(*__t1).0[203].class = 196608u32;
(*__t1).0[203].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[203].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[204].vendor = 32902u32;
(*__t1).0[204].device = 34762u32;
(*__t1).0[204].subvendor = 4294967295u32;
(*__t1).0[204].subdevice = 4294967295u32;
(*__t1).0[204].class = 196608u32;
(*__t1).0[204].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[204].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[205].vendor = 32902u32;
(*__t1).0[205].device = 16033u32;
(*__t1).0[205].subvendor = 4294967295u32;
(*__t1).0[205].subdevice = 4294967295u32;
(*__t1).0[205].class = 196608u32;
(*__t1).0[205].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[205].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[206].vendor = 32902u32;
(*__t1).0[206].device = 16036u32;
(*__t1).0[206].subvendor = 4294967295u32;
(*__t1).0[206].subdevice = 4294967295u32;
(*__t1).0[206].class = 196608u32;
(*__t1).0[206].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[206].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[207].vendor = 32902u32;
(*__t1).0[207].device = 16032u32;
(*__t1).0[207].subvendor = 4294967295u32;
(*__t1).0[207].subdevice = 4294967295u32;
(*__t1).0[207].class = 196608u32;
(*__t1).0[207].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[207].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[208].vendor = 32902u32;
(*__t1).0[208].device = 16035u32;
(*__t1).0[208].subvendor = 4294967295u32;
(*__t1).0[208].subdevice = 4294967295u32;
(*__t1).0[208].class = 196608u32;
(*__t1).0[208].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[208].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[209].vendor = 32902u32;
(*__t1).0[209].device = 16034u32;
(*__t1).0[209].subvendor = 4294967295u32;
(*__t1).0[209].subdevice = 4294967295u32;
(*__t1).0[209].class = 196608u32;
(*__t1).0[209].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[209].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[210].vendor = 32902u32;
(*__t1).0[210].device = 39842u32;
(*__t1).0[210].subvendor = 4294967295u32;
(*__t1).0[210].subdevice = 4294967295u32;
(*__t1).0[210].class = 196608u32;
(*__t1).0[210].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[210].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[211].vendor = 32902u32;
(*__t1).0[211].device = 39844u32;
(*__t1).0[211].subvendor = 4294967295u32;
(*__t1).0[211].subdevice = 4294967295u32;
(*__t1).0[211].class = 196608u32;
(*__t1).0[211].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[211].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[212].vendor = 32902u32;
(*__t1).0[212].device = 39845u32;
(*__t1).0[212].subvendor = 4294967295u32;
(*__t1).0[212].subdevice = 4294967295u32;
(*__t1).0[212].class = 196608u32;
(*__t1).0[212].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[212].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[213].vendor = 32902u32;
(*__t1).0[213].device = 39848u32;
(*__t1).0[213].subvendor = 4294967295u32;
(*__t1).0[213].subdevice = 4294967295u32;
(*__t1).0[213].class = 196608u32;
(*__t1).0[213].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[213].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[214].vendor = 32902u32;
(*__t1).0[214].device = 39874u32;
(*__t1).0[214].subvendor = 4294967295u32;
(*__t1).0[214].subdevice = 4294967295u32;
(*__t1).0[214].class = 196608u32;
(*__t1).0[214].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[214].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[215].vendor = 32902u32;
(*__t1).0[215].device = 39876u32;
(*__t1).0[215].subvendor = 4294967295u32;
(*__t1).0[215].subdevice = 4294967295u32;
(*__t1).0[215].class = 196608u32;
(*__t1).0[215].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[215].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[216].vendor = 32902u32;
(*__t1).0[216].device = 39877u32;
(*__t1).0[216].subvendor = 4294967295u32;
(*__t1).0[216].subdevice = 4294967295u32;
(*__t1).0[216].class = 196608u32;
(*__t1).0[216].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[216].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[217].vendor = 32902u32;
(*__t1).0[217].device = 39878u32;
(*__t1).0[217].subvendor = 4294967295u32;
(*__t1).0[217].subdevice = 4294967295u32;
(*__t1).0[217].class = 196608u32;
(*__t1).0[217].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[217].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[218].vendor = 32902u32;
(*__t1).0[218].device = 39880u32;
(*__t1).0[218].subvendor = 4294967295u32;
(*__t1).0[218].subdevice = 4294967295u32;
(*__t1).0[218].class = 196608u32;
(*__t1).0[218].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[218].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[219].vendor = 32902u32;
(*__t1).0[219].device = 39910u32;
(*__t1).0[219].subvendor = 4294967295u32;
(*__t1).0[219].subdevice = 4294967295u32;
(*__t1).0[219].class = 196608u32;
(*__t1).0[219].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[219].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[220].vendor = 32902u32;
(*__t1).0[220].device = 39926u32;
(*__t1).0[220].subvendor = 4294967295u32;
(*__t1).0[220].subdevice = 4294967295u32;
(*__t1).0[220].class = 196608u32;
(*__t1).0[220].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[220].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[221].vendor = 32902u32;
(*__t1).0[221].device = 39713u32;
(*__t1).0[221].subvendor = 4294967295u32;
(*__t1).0[221].subdevice = 4294967295u32;
(*__t1).0[221].class = 196608u32;
(*__t1).0[221].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[221].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[222].vendor = 32902u32;
(*__t1).0[222].device = 39850u32;
(*__t1).0[222].subvendor = 4294967295u32;
(*__t1).0[222].subdevice = 4294967295u32;
(*__t1).0[222].class = 196608u32;
(*__t1).0[222].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[222].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[223].vendor = 32902u32;
(*__t1).0[223].device = 39852u32;
(*__t1).0[223].subvendor = 4294967295u32;
(*__t1).0[223].subdevice = 4294967295u32;
(*__t1).0[223].class = 196608u32;
(*__t1).0[223].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[223].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[224].vendor = 32902u32;
(*__t1).0[224].device = 39745u32;
(*__t1).0[224].subvendor = 4294967295u32;
(*__t1).0[224].subdevice = 4294967295u32;
(*__t1).0[224].class = 196608u32;
(*__t1).0[224].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[224].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[225].vendor = 32902u32;
(*__t1).0[225].device = 39882u32;
(*__t1).0[225].subvendor = 4294967295u32;
(*__t1).0[225].subdevice = 4294967295u32;
(*__t1).0[225].class = 196608u32;
(*__t1).0[225].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[225].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[226].vendor = 32902u32;
(*__t1).0[226].device = 39884u32;
(*__t1).0[226].subvendor = 4294967295u32;
(*__t1).0[226].subdevice = 4294967295u32;
(*__t1).0[226].class = 196608u32;
(*__t1).0[226].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[226].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[227].vendor = 32902u32;
(*__t1).0[227].device = 12676u32;
(*__t1).0[227].subvendor = 4294967295u32;
(*__t1).0[227].subdevice = 4294967295u32;
(*__t1).0[227].class = 196608u32;
(*__t1).0[227].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[227].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[228].vendor = 32902u32;
(*__t1).0[228].device = 12677u32;
(*__t1).0[228].subvendor = 4294967295u32;
(*__t1).0[228].subdevice = 4294967295u32;
(*__t1).0[228].class = 196608u32;
(*__t1).0[228].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[228].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[229].vendor = 32902u32;
(*__t1).0[229].device = 23108u32;
(*__t1).0[229].subvendor = 4294967295u32;
(*__t1).0[229].subdevice = 4294967295u32;
(*__t1).0[229].class = 196608u32;
(*__t1).0[229].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[229].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[230].vendor = 32902u32;
(*__t1).0[230].device = 23116u32;
(*__t1).0[230].subvendor = 4294967295u32;
(*__t1).0[230].subdevice = 4294967295u32;
(*__t1).0[230].class = 196608u32;
(*__t1).0[230].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[230].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[231].vendor = 32902u32;
(*__t1).0[231].device = 23124u32;
(*__t1).0[231].subvendor = 4294967295u32;
(*__t1).0[231].subdevice = 4294967295u32;
(*__t1).0[231].class = 196608u32;
(*__t1).0[231].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[231].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[232].vendor = 32902u32;
(*__t1).0[232].device = 23132u32;
(*__t1).0[232].subvendor = 4294967295u32;
(*__t1).0[232].subdevice = 4294967295u32;
(*__t1).0[232].class = 196608u32;
(*__t1).0[232].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[232].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[233].vendor = 32902u32;
(*__t1).0[233].device = 23104u32;
(*__t1).0[233].subvendor = 4294967295u32;
(*__t1).0[233].subdevice = 4294967295u32;
(*__t1).0[233].class = 196608u32;
(*__t1).0[233].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[233].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[234].vendor = 32902u32;
(*__t1).0[234].device = 23105u32;
(*__t1).0[234].subvendor = 4294967295u32;
(*__t1).0[234].subdevice = 4294967295u32;
(*__t1).0[234].class = 196608u32;
(*__t1).0[234].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[234].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[235].vendor = 32902u32;
(*__t1).0[235].device = 23106u32;
(*__t1).0[235].subvendor = 4294967295u32;
(*__t1).0[235].subdevice = 4294967295u32;
(*__t1).0[235].class = 196608u32;
(*__t1).0[235].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[235].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[236].vendor = 32902u32;
(*__t1).0[236].device = 23113u32;
(*__t1).0[236].subvendor = 4294967295u32;
(*__t1).0[236].subdevice = 4294967295u32;
(*__t1).0[236].class = 196608u32;
(*__t1).0[236].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[236].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[237].vendor = 32902u32;
(*__t1).0[237].device = 23114u32;
(*__t1).0[237].subvendor = 4294967295u32;
(*__t1).0[237].subdevice = 4294967295u32;
(*__t1).0[237].class = 196608u32;
(*__t1).0[237].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[237].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[238].vendor = 32902u32;
(*__t1).0[238].device = 23120u32;
(*__t1).0[238].subvendor = 4294967295u32;
(*__t1).0[238].subdevice = 4294967295u32;
(*__t1).0[238].class = 196608u32;
(*__t1).0[238].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[238].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[239].vendor = 32902u32;
(*__t1).0[239].device = 23121u32;
(*__t1).0[239].subvendor = 4294967295u32;
(*__t1).0[239].subdevice = 4294967295u32;
(*__t1).0[239].class = 196608u32;
(*__t1).0[239].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[239].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[240].vendor = 32902u32;
(*__t1).0[240].device = 23122u32;
(*__t1).0[240].subvendor = 4294967295u32;
(*__t1).0[240].subdevice = 4294967295u32;
(*__t1).0[240].class = 196608u32;
(*__t1).0[240].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[240].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[241].vendor = 32902u32;
(*__t1).0[241].device = 23129u32;
(*__t1).0[241].subvendor = 4294967295u32;
(*__t1).0[241].subdevice = 4294967295u32;
(*__t1).0[241].class = 196608u32;
(*__t1).0[241].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[241].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[242].vendor = 32902u32;
(*__t1).0[242].device = 23130u32;
(*__t1).0[242].subvendor = 4294967295u32;
(*__t1).0[242].subdevice = 4294967295u32;
(*__t1).0[242].class = 196608u32;
(*__t1).0[242].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[242].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen9_early_ops.0) }) });
(*__t1).0[243].vendor = 32902u32;
(*__t1).0[243].device = 35408u32;
(*__t1).0[243].subvendor = 4294967295u32;
(*__t1).0[243].subdevice = 4294967295u32;
(*__t1).0[243].class = 196608u32;
(*__t1).0[243].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[243].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[244].vendor = 32902u32;
(*__t1).0[244].device = 35410u32;
(*__t1).0[244].subvendor = 4294967295u32;
(*__t1).0[244].subdevice = 4294967295u32;
(*__t1).0[244].class = 196608u32;
(*__t1).0[244].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[244].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[245].vendor = 32902u32;
(*__t1).0[245].device = 35411u32;
(*__t1).0[245].subvendor = 4294967295u32;
(*__t1).0[245].subdevice = 4294967295u32;
(*__t1).0[245].class = 196608u32;
(*__t1).0[245].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[245].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[246].vendor = 32902u32;
(*__t1).0[246].device = 35412u32;
(*__t1).0[246].subvendor = 4294967295u32;
(*__t1).0[246].subdevice = 4294967295u32;
(*__t1).0[246].class = 196608u32;
(*__t1).0[246].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[246].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[247].vendor = 32902u32;
(*__t1).0[247].device = 35414u32;
(*__t1).0[247].subvendor = 4294967295u32;
(*__t1).0[247].subdevice = 4294967295u32;
(*__t1).0[247].class = 196608u32;
(*__t1).0[247].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[247].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[248].vendor = 32902u32;
(*__t1).0[248].device = 35415u32;
(*__t1).0[248].subvendor = 4294967295u32;
(*__t1).0[248].subdevice = 4294967295u32;
(*__t1).0[248].class = 196608u32;
(*__t1).0[248].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[248].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[249].vendor = 32902u32;
(*__t1).0[249].device = 35416u32;
(*__t1).0[249].subvendor = 4294967295u32;
(*__t1).0[249].subdevice = 4294967295u32;
(*__t1).0[249].class = 196608u32;
(*__t1).0[249].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[249].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[250].vendor = 32902u32;
(*__t1).0[250].device = 35417u32;
(*__t1).0[250].subvendor = 4294967295u32;
(*__t1).0[250].subdevice = 4294967295u32;
(*__t1).0[250].class = 196608u32;
(*__t1).0[250].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[250].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[251].vendor = 32902u32;
(*__t1).0[251].device = 35418u32;
(*__t1).0[251].subvendor = 4294967295u32;
(*__t1).0[251].subdevice = 4294967295u32;
(*__t1).0[251].class = 196608u32;
(*__t1).0[251].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[251].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[252].vendor = 32902u32;
(*__t1).0[252].device = 35419u32;
(*__t1).0[252].subvendor = 4294967295u32;
(*__t1).0[252].subdevice = 4294967295u32;
(*__t1).0[252].class = 196608u32;
(*__t1).0[252].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[252].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[253].vendor = 32902u32;
(*__t1).0[253].device = 35420u32;
(*__t1).0[253].subvendor = 4294967295u32;
(*__t1).0[253].subdevice = 4294967295u32;
(*__t1).0[253].class = 196608u32;
(*__t1).0[253].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[253].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[254].vendor = 32902u32;
(*__t1).0[254].device = 35440u32;
(*__t1).0[254].subvendor = 4294967295u32;
(*__t1).0[254].subdevice = 4294967295u32;
(*__t1).0[254].class = 196608u32;
(*__t1).0[254].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[254].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[255].vendor = 32902u32;
(*__t1).0[255].device = 35441u32;
(*__t1).0[255].subvendor = 4294967295u32;
(*__t1).0[255].subdevice = 4294967295u32;
(*__t1).0[255].class = 196608u32;
(*__t1).0[255].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[255].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[256].vendor = 32902u32;
(*__t1).0[256].device = 35409u32;
(*__t1).0[256].subvendor = 4294967295u32;
(*__t1).0[256].subdevice = 4294967295u32;
(*__t1).0[256].class = 196608u32;
(*__t1).0[256].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[256].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[257].vendor = 32902u32;
(*__t1).0[257].device = 35421u32;
(*__t1).0[257].subvendor = 4294967295u32;
(*__t1).0[257].subdevice = 4294967295u32;
(*__t1).0[257].class = 196608u32;
(*__t1).0[257].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[257].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[258].vendor = 32902u32;
(*__t1).0[258].device = 17729u32;
(*__t1).0[258].subvendor = 4294967295u32;
(*__t1).0[258].subdevice = 4294967295u32;
(*__t1).0[258].class = 196608u32;
(*__t1).0[258].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[258].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[259].vendor = 32902u32;
(*__t1).0[259].device = 17745u32;
(*__t1).0[259].subvendor = 4294967295u32;
(*__t1).0[259].subdevice = 4294967295u32;
(*__t1).0[259].class = 196608u32;
(*__t1).0[259].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[259].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[260].vendor = 32902u32;
(*__t1).0[260].device = 17749u32;
(*__t1).0[260].subvendor = 4294967295u32;
(*__t1).0[260].subdevice = 4294967295u32;
(*__t1).0[260].class = 196608u32;
(*__t1).0[260].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[260].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[261].vendor = 32902u32;
(*__t1).0[261].device = 17751u32;
(*__t1).0[261].subvendor = 4294967295u32;
(*__t1).0[261].subdevice = 4294967295u32;
(*__t1).0[261].class = 196608u32;
(*__t1).0[261].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[261].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[262].vendor = 32902u32;
(*__t1).0[262].device = 17776u32;
(*__t1).0[262].subvendor = 4294967295u32;
(*__t1).0[262].subdevice = 4294967295u32;
(*__t1).0[262].class = 196608u32;
(*__t1).0[262].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[262].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[263].vendor = 32902u32;
(*__t1).0[263].device = 17777u32;
(*__t1).0[263].subvendor = 4294967295u32;
(*__t1).0[263].subdevice = 4294967295u32;
(*__t1).0[263].class = 196608u32;
(*__t1).0[263].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[263].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[264].vendor = 32902u32;
(*__t1).0[264].device = 20049u32;
(*__t1).0[264].subvendor = 4294967295u32;
(*__t1).0[264].subdevice = 4294967295u32;
(*__t1).0[264].class = 196608u32;
(*__t1).0[264].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[264].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[265].vendor = 32902u32;
(*__t1).0[265].device = 20053u32;
(*__t1).0[265].subvendor = 4294967295u32;
(*__t1).0[265].subdevice = 4294967295u32;
(*__t1).0[265].class = 196608u32;
(*__t1).0[265].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[265].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[266].vendor = 32902u32;
(*__t1).0[266].device = 20055u32;
(*__t1).0[266].subvendor = 4294967295u32;
(*__t1).0[266].subdevice = 4294967295u32;
(*__t1).0[266].class = 196608u32;
(*__t1).0[266].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[266].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[267].vendor = 32902u32;
(*__t1).0[267].device = 20065u32;
(*__t1).0[267].subvendor = 4294967295u32;
(*__t1).0[267].subdevice = 4294967295u32;
(*__t1).0[267].class = 196608u32;
(*__t1).0[267].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[267].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[268].vendor = 32902u32;
(*__t1).0[268].device = 20081u32;
(*__t1).0[268].subvendor = 4294967295u32;
(*__t1).0[268].subdevice = 4294967295u32;
(*__t1).0[268].class = 196608u32;
(*__t1).0[268].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[268].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[269].vendor = 32902u32;
(*__t1).0[269].device = 39520u32;
(*__t1).0[269].subvendor = 4294967295u32;
(*__t1).0[269].subdevice = 4294967295u32;
(*__t1).0[269].class = 196608u32;
(*__t1).0[269].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[269].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[270].vendor = 32902u32;
(*__t1).0[270].device = 39528u32;
(*__t1).0[270].subvendor = 4294967295u32;
(*__t1).0[270].subdevice = 4294967295u32;
(*__t1).0[270].class = 196608u32;
(*__t1).0[270].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[270].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[271].vendor = 32902u32;
(*__t1).0[271].device = 39536u32;
(*__t1).0[271].subvendor = 4294967295u32;
(*__t1).0[271].subdevice = 4294967295u32;
(*__t1).0[271].class = 196608u32;
(*__t1).0[271].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[271].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[272].vendor = 32902u32;
(*__t1).0[272].device = 39488u32;
(*__t1).0[272].subvendor = 4294967295u32;
(*__t1).0[272].subdevice = 4294967295u32;
(*__t1).0[272].class = 196608u32;
(*__t1).0[272].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[272].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[273].vendor = 32902u32;
(*__t1).0[273].device = 39497u32;
(*__t1).0[273].subvendor = 4294967295u32;
(*__t1).0[273].subdevice = 4294967295u32;
(*__t1).0[273].class = 196608u32;
(*__t1).0[273].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[273].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[274].vendor = 32902u32;
(*__t1).0[274].device = 39513u32;
(*__t1).0[274].subvendor = 4294967295u32;
(*__t1).0[274].subdevice = 4294967295u32;
(*__t1).0[274].class = 196608u32;
(*__t1).0[274].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[274].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[275].vendor = 32902u32;
(*__t1).0[275].device = 39544u32;
(*__t1).0[275].subvendor = 4294967295u32;
(*__t1).0[275].subdevice = 4294967295u32;
(*__t1).0[275].class = 196608u32;
(*__t1).0[275].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[275].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[276].vendor = 32902u32;
(*__t1).0[276].device = 39616u32;
(*__t1).0[276].subvendor = 4294967295u32;
(*__t1).0[276].subdevice = 4294967295u32;
(*__t1).0[276].class = 196608u32;
(*__t1).0[276].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[276].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[277].vendor = 32902u32;
(*__t1).0[277].device = 39625u32;
(*__t1).0[277].subvendor = 4294967295u32;
(*__t1).0[277].subdevice = 4294967295u32;
(*__t1).0[277].class = 196608u32;
(*__t1).0[277].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[277].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[278].vendor = 32902u32;
(*__t1).0[278].device = 39641u32;
(*__t1).0[278].subvendor = 4294967295u32;
(*__t1).0[278].subdevice = 4294967295u32;
(*__t1).0[278].class = 196608u32;
(*__t1).0[278].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[278].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[279].vendor = 32902u32;
(*__t1).0[279].device = 39672u32;
(*__t1).0[279].subvendor = 4294967295u32;
(*__t1).0[279].subdevice = 4294967295u32;
(*__t1).0[279].class = 196608u32;
(*__t1).0[279].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[279].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[280].vendor = 32902u32;
(*__t1).0[280].device = 19584u32;
(*__t1).0[280].subvendor = 4294967295u32;
(*__t1).0[280].subdevice = 4294967295u32;
(*__t1).0[280].class = 196608u32;
(*__t1).0[280].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[280].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[281].vendor = 32902u32;
(*__t1).0[281].device = 19594u32;
(*__t1).0[281].subvendor = 4294967295u32;
(*__t1).0[281].subdevice = 4294967295u32;
(*__t1).0[281].class = 196608u32;
(*__t1).0[281].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[281].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[282].vendor = 32902u32;
(*__t1).0[282].device = 19595u32;
(*__t1).0[282].subvendor = 4294967295u32;
(*__t1).0[282].subdevice = 4294967295u32;
(*__t1).0[282].class = 196608u32;
(*__t1).0[282].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[282].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[283].vendor = 32902u32;
(*__t1).0[283].device = 19596u32;
(*__t1).0[283].subvendor = 4294967295u32;
(*__t1).0[283].subdevice = 4294967295u32;
(*__t1).0[283].class = 196608u32;
(*__t1).0[283].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[283].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[284].vendor = 32902u32;
(*__t1).0[284].device = 19600u32;
(*__t1).0[284].subvendor = 4294967295u32;
(*__t1).0[284].subdevice = 4294967295u32;
(*__t1).0[284].class = 196608u32;
(*__t1).0[284].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[284].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[285].vendor = 32902u32;
(*__t1).0[285].device = 19610u32;
(*__t1).0[285].subvendor = 4294967295u32;
(*__t1).0[285].subdevice = 4294967295u32;
(*__t1).0[285].class = 196608u32;
(*__t1).0[285].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[285].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[286].vendor = 32902u32;
(*__t1).0[286].device = 18048u32;
(*__t1).0[286].subvendor = 4294967295u32;
(*__t1).0[286].subdevice = 4294967295u32;
(*__t1).0[286].class = 196608u32;
(*__t1).0[286].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[286].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[287].vendor = 32902u32;
(*__t1).0[287].device = 18050u32;
(*__t1).0[287].subvendor = 4294967295u32;
(*__t1).0[287].subdevice = 4294967295u32;
(*__t1).0[287].class = 196608u32;
(*__t1).0[287].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[287].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[288].vendor = 32902u32;
(*__t1).0[288].device = 18056u32;
(*__t1).0[288].subvendor = 4294967295u32;
(*__t1).0[288].subdevice = 4294967295u32;
(*__t1).0[288].class = 196608u32;
(*__t1).0[288].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[288].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[289].vendor = 32902u32;
(*__t1).0[289].device = 18058u32;
(*__t1).0[289].subvendor = 4294967295u32;
(*__t1).0[289].subdevice = 4294967295u32;
(*__t1).0[289].class = 196608u32;
(*__t1).0[289].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[289].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[290].vendor = 32902u32;
(*__t1).0[290].device = 18059u32;
(*__t1).0[290].subvendor = 4294967295u32;
(*__t1).0[290].subdevice = 4294967295u32;
(*__t1).0[290].class = 196608u32;
(*__t1).0[290].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[290].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[291].vendor = 32902u32;
(*__t1).0[291].device = 18064u32;
(*__t1).0[291].subvendor = 4294967295u32;
(*__t1).0[291].subdevice = 4294967295u32;
(*__t1).0[291].class = 196608u32;
(*__t1).0[291].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[291].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[292].vendor = 32902u32;
(*__t1).0[292].device = 18066u32;
(*__t1).0[292].subvendor = 4294967295u32;
(*__t1).0[292].subdevice = 4294967295u32;
(*__t1).0[292].class = 196608u32;
(*__t1).0[292].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[292].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[293].vendor = 32902u32;
(*__t1).0[293].device = 18067u32;
(*__t1).0[293].subvendor = 4294967295u32;
(*__t1).0[293].subdevice = 4294967295u32;
(*__t1).0[293].class = 196608u32;
(*__t1).0[293].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[293].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[294].vendor = 32902u32;
(*__t1).0[294].device = 18080u32;
(*__t1).0[294].subvendor = 4294967295u32;
(*__t1).0[294].subdevice = 4294967295u32;
(*__t1).0[294].class = 196608u32;
(*__t1).0[294].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[294].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[295].vendor = 32902u32;
(*__t1).0[295].device = 18081u32;
(*__t1).0[295].subvendor = 4294967295u32;
(*__t1).0[295].subdevice = 4294967295u32;
(*__t1).0[295].class = 196608u32;
(*__t1).0[295].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[295].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[296].vendor = 32902u32;
(*__t1).0[296].device = 18082u32;
(*__t1).0[296].subvendor = 4294967295u32;
(*__t1).0[296].subdevice = 4294967295u32;
(*__t1).0[296].class = 196608u32;
(*__t1).0[296].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[296].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[297].vendor = 32902u32;
(*__t1).0[297].device = 18083u32;
(*__t1).0[297].subvendor = 4294967295u32;
(*__t1).0[297].subdevice = 4294967295u32;
(*__t1).0[297].class = 196608u32;
(*__t1).0[297].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[297].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[298].vendor = 32902u32;
(*__t1).0[298].device = 18086u32;
(*__t1).0[298].subvendor = 4294967295u32;
(*__t1).0[298].subdevice = 4294967295u32;
(*__t1).0[298].class = 196608u32;
(*__t1).0[298].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[298].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[299].vendor = 32902u32;
(*__t1).0[299].device = 18088u32;
(*__t1).0[299].subvendor = 4294967295u32;
(*__t1).0[299].subdevice = 4294967295u32;
(*__t1).0[299].class = 196608u32;
(*__t1).0[299].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[299].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[300].vendor = 32902u32;
(*__t1).0[300].device = 18090u32;
(*__t1).0[300].subvendor = 4294967295u32;
(*__t1).0[300].subdevice = 4294967295u32;
(*__t1).0[300].class = 196608u32;
(*__t1).0[300].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[300].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[301].vendor = 32902u32;
(*__t1).0[301].device = 17962u32;
(*__t1).0[301].subvendor = 4294967295u32;
(*__t1).0[301].subdevice = 4294967295u32;
(*__t1).0[301].class = 196608u32;
(*__t1).0[301].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[301].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[302].vendor = 32902u32;
(*__t1).0[302].device = 17958u32;
(*__t1).0[302].subvendor = 4294967295u32;
(*__t1).0[302].subdevice = 4294967295u32;
(*__t1).0[302].class = 196608u32;
(*__t1).0[302].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[302].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[303].vendor = 32902u32;
(*__t1).0[303].device = 17960u32;
(*__t1).0[303].subvendor = 4294967295u32;
(*__t1).0[303].subdevice = 4294967295u32;
(*__t1).0[303].class = 196608u32;
(*__t1).0[303].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[303].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[304].vendor = 32902u32;
(*__t1).0[304].device = 18096u32;
(*__t1).0[304].subvendor = 4294967295u32;
(*__t1).0[304].subdevice = 4294967295u32;
(*__t1).0[304].class = 196608u32;
(*__t1).0[304].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[304].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[305].vendor = 32902u32;
(*__t1).0[305].device = 18097u32;
(*__t1).0[305].subvendor = 4294967295u32;
(*__t1).0[305].subdevice = 4294967295u32;
(*__t1).0[305].class = 196608u32;
(*__t1).0[305].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[305].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[306].vendor = 32902u32;
(*__t1).0[306].device = 18098u32;
(*__t1).0[306].subvendor = 4294967295u32;
(*__t1).0[306].subdevice = 4294967295u32;
(*__t1).0[306].class = 196608u32;
(*__t1).0[306].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[306].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[307].vendor = 32902u32;
(*__t1).0[307].device = 18099u32;
(*__t1).0[307].subvendor = 4294967295u32;
(*__t1).0[307].subdevice = 4294967295u32;
(*__t1).0[307].class = 196608u32;
(*__t1).0[307].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[307].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[308].vendor = 32902u32;
(*__t1).0[308].device = 18112u32;
(*__t1).0[308].subvendor = 4294967295u32;
(*__t1).0[308].subdevice = 4294967295u32;
(*__t1).0[308].class = 196608u32;
(*__t1).0[308].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[308].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[309].vendor = 32902u32;
(*__t1).0[309].device = 18113u32;
(*__t1).0[309].subvendor = 4294967295u32;
(*__t1).0[309].subdevice = 4294967295u32;
(*__t1).0[309].class = 196608u32;
(*__t1).0[309].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[309].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[310].vendor = 32902u32;
(*__t1).0[310].device = 18114u32;
(*__t1).0[310].subvendor = 4294967295u32;
(*__t1).0[310].subdevice = 4294967295u32;
(*__t1).0[310].class = 196608u32;
(*__t1).0[310].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[310].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[311].vendor = 32902u32;
(*__t1).0[311].device = 18115u32;
(*__t1).0[311].subvendor = 4294967295u32;
(*__t1).0[311].subdevice = 4294967295u32;
(*__t1).0[311].class = 196608u32;
(*__t1).0[311].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[311].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[312].vendor = 32902u32;
(*__t1).0[312].device = 18128u32;
(*__t1).0[312].subvendor = 4294967295u32;
(*__t1).0[312].subdevice = 4294967295u32;
(*__t1).0[312].class = 196608u32;
(*__t1).0[312].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[312].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[313].vendor = 32902u32;
(*__t1).0[313].device = 18129u32;
(*__t1).0[313].subvendor = 4294967295u32;
(*__t1).0[313].subdevice = 4294967295u32;
(*__t1).0[313].class = 196608u32;
(*__t1).0[313].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[313].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[314].vendor = 32902u32;
(*__t1).0[314].device = 18130u32;
(*__t1).0[314].subvendor = 4294967295u32;
(*__t1).0[314].subdevice = 4294967295u32;
(*__t1).0[314].class = 196608u32;
(*__t1).0[314].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[314].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[315].vendor = 32902u32;
(*__t1).0[315].device = 18131u32;
(*__t1).0[315].subvendor = 4294967295u32;
(*__t1).0[315].subdevice = 4294967295u32;
(*__t1).0[315].class = 196608u32;
(*__t1).0[315].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[315].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[316].vendor = 32902u32;
(*__t1).0[316].device = 18132u32;
(*__t1).0[316].subvendor = 4294967295u32;
(*__t1).0[316].subdevice = 4294967295u32;
(*__t1).0[316].class = 196608u32;
(*__t1).0[316].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[316].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[317].vendor = 32902u32;
(*__t1).0[317].device = 42880u32;
(*__t1).0[317].subvendor = 4294967295u32;
(*__t1).0[317].subdevice = 4294967295u32;
(*__t1).0[317].class = 196608u32;
(*__t1).0[317].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[317].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[318].vendor = 32902u32;
(*__t1).0[318].device = 42881u32;
(*__t1).0[318].subvendor = 4294967295u32;
(*__t1).0[318].subdevice = 4294967295u32;
(*__t1).0[318].class = 196608u32;
(*__t1).0[318].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[318].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[319].vendor = 32902u32;
(*__t1).0[319].device = 42882u32;
(*__t1).0[319].subvendor = 4294967295u32;
(*__t1).0[319].subdevice = 4294967295u32;
(*__t1).0[319].class = 196608u32;
(*__t1).0[319].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[319].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[320].vendor = 32902u32;
(*__t1).0[320].device = 42883u32;
(*__t1).0[320].subvendor = 4294967295u32;
(*__t1).0[320].subdevice = 4294967295u32;
(*__t1).0[320].class = 196608u32;
(*__t1).0[320].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[320].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[321].vendor = 32902u32;
(*__t1).0[321].device = 42888u32;
(*__t1).0[321].subvendor = 4294967295u32;
(*__t1).0[321].subdevice = 4294967295u32;
(*__t1).0[321].class = 196608u32;
(*__t1).0[321].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[321].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[322].vendor = 32902u32;
(*__t1).0[322].device = 42889u32;
(*__t1).0[322].subvendor = 4294967295u32;
(*__t1).0[322].subdevice = 4294967295u32;
(*__t1).0[322].class = 196608u32;
(*__t1).0[322].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[322].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[323].vendor = 32902u32;
(*__t1).0[323].device = 42890u32;
(*__t1).0[323].subvendor = 4294967295u32;
(*__t1).0[323].subdevice = 4294967295u32;
(*__t1).0[323].class = 196608u32;
(*__t1).0[323].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[323].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[324].vendor = 32902u32;
(*__t1).0[324].device = 42891u32;
(*__t1).0[324].subvendor = 4294967295u32;
(*__t1).0[324].subdevice = 4294967295u32;
(*__t1).0[324].class = 196608u32;
(*__t1).0[324].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[324].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[325].vendor = 32902u32;
(*__t1).0[325].device = 42785u32;
(*__t1).0[325].subvendor = 4294967295u32;
(*__t1).0[325].subdevice = 4294967295u32;
(*__t1).0[325].class = 196608u32;
(*__t1).0[325].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[325].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[326].vendor = 32902u32;
(*__t1).0[326].device = 42913u32;
(*__t1).0[326].subvendor = 4294967295u32;
(*__t1).0[326].subdevice = 4294967295u32;
(*__t1).0[326].class = 196608u32;
(*__t1).0[326].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[326].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[327].vendor = 32902u32;
(*__t1).0[327].device = 42921u32;
(*__t1).0[327].subvendor = 4294967295u32;
(*__t1).0[327].subdevice = 4294967295u32;
(*__t1).0[327].class = 196608u32;
(*__t1).0[327].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[327].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[328].vendor = 32902u32;
(*__t1).0[328].device = 42924u32;
(*__t1).0[328].subvendor = 4294967295u32;
(*__t1).0[328].subdevice = 4294967295u32;
(*__t1).0[328].class = 196608u32;
(*__t1).0[328].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[328].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[329].vendor = 32902u32;
(*__t1).0[329].device = 42925u32;
(*__t1).0[329].subvendor = 4294967295u32;
(*__t1).0[329].subdevice = 4294967295u32;
(*__t1).0[329].class = 196608u32;
(*__t1).0[329].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[329].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[330].vendor = 32902u32;
(*__t1).0[330].device = 42784u32;
(*__t1).0[330].subvendor = 4294967295u32;
(*__t1).0[330].subdevice = 4294967295u32;
(*__t1).0[330].class = 196608u32;
(*__t1).0[330].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[330].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[331].vendor = 32902u32;
(*__t1).0[331].device = 42912u32;
(*__t1).0[331].subvendor = 4294967295u32;
(*__t1).0[331].subdevice = 4294967295u32;
(*__t1).0[331].class = 196608u32;
(*__t1).0[331].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[331].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[332].vendor = 32902u32;
(*__t1).0[332].device = 42920u32;
(*__t1).0[332].subvendor = 4294967295u32;
(*__t1).0[332].subdevice = 4294967295u32;
(*__t1).0[332].class = 196608u32;
(*__t1).0[332].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[332].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[333].vendor = 32902u32;
(*__t1).0[333].device = 42922u32;
(*__t1).0[333].subvendor = 4294967295u32;
(*__t1).0[333].subdevice = 4294967295u32;
(*__t1).0[333].class = 196608u32;
(*__t1).0[333].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[333].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
(*__t1).0[334].vendor = 32902u32;
(*__t1).0[334].device = 42923u32;
(*__t1).0[334].subvendor = 4294967295u32;
(*__t1).0[334].subdevice = 4294967295u32;
(*__t1).0[334].class = 196608u32;
(*__t1).0[334].class_mask = 16711680u32;
(core::ptr::addr_of_mut!((*__t1).0[334].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const intel_early_ops, *const ()>(unsafe { core::ptr::addr_of!(gen11_early_ops.0) }) });
__t0
};

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut intel_graphics_stolen_res: resource = {
let mut __t0: resource = unsafe { core::mem::zeroed() };
__t0.start = 0u64;
__t0.end = 18446744073709551615u64;
__t0.name = core::ptr::null::<u8>();
__t0.flags = 512u64;
__t0.desc = 0u64;
__t0
};

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_graphics_stolen_res_555: *mut () = (unsafe { core::ptr::addr_of_mut!(intel_graphics_stolen_res) }) as *mut ();

#[unsafe(link_section = ".init.text")]
fn intel_graphics_stolen(num: i32, slot: i32, func: i32, early_ops: *const intel_early_ops) {
let mut base: u64 = 0;
let size: u64 = unsafe { unsafe { (*early_ops).stolen_size }.unwrap()(num, slot, func) };
let mut end: u64 = 0;
base = unsafe { unsafe { (*early_ops).stolen_base }.unwrap()(num, slot, func, size) };
if size == 0 || base == 0 {
return;
}
end = base.wrapping_add(size).wrapping_sub(1);
unsafe {
intel_graphics_stolen_res.start = base;
}
unsafe {
intel_graphics_stolen_res.end = end;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Reserving Intel graphics memory at %pR\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(intel_graphics_stolen_res)) };
// Mark this space as reserved
unsafe { e820__range_add(base, size, 2u32) };
unsafe { e820__update_table(e820_table) };
}

#[unsafe(link_section = ".init.text")]
extern "C" fn intel_graphics_quirks(num: i32, slot: i32, func: i32) {
let mut device: u16 = 0;
let mut i: i32 = 0;
// Reserve "stolen memory" for an integrated GPU.  If we've already
// found one, there's nothing to do for other (discrete) GPUs.
if resource_size((unsafe { core::ptr::addr_of_mut!(intel_graphics_stolen_res) }) as *const resource) != 0 {
return;
}
device = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 2u8) };
while (i as u64) < ARRAY_SIZE__25!() {
let driver_data: u64 = unsafe { (*(core::ptr::addr_of_mut!(intel_early_ids) as *mut __SlateImage_intel_early_ids)).0[i as usize].driver_data };
if (unsafe { (*(core::ptr::addr_of_mut!(intel_early_ids) as *mut __SlateImage_intel_early_ids)).0[i as usize].device }) == ((device as i32) as u32) {
let early_ops: *const intel_early_ops = driver_data as *const intel_early_ops;
intel_graphics_stolen(num, slot, func, early_ops);
return;
}
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn force_disable_hpet(num: i32, slot: i32, func: i32) {
unsafe {
boot_hpet_disable = true;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016x86/hpet: Will disable the HPET for this platform because it's not reliable\n".as_ptr() as *const u8) };
}

#[unsafe(link_section = ".init.text")]
extern "C" fn apple_airport_reset(bus: i32, slot: i32, func: i32) {
let mut mmio: *mut () = unsafe { core::mem::zeroed() };
let mut pmcsr: u16 = 0;
let mut addr: u64 = 0;
let mut i: i32 = 0;
if !unsafe { x86_apple_machine } {
return;
}
// Card may have been put into PCI_D3hot by grub quirk
pmcsr = unsafe { read_pci_config_16(bus as u8, slot as u8, func as u8, 68u8) };
if (pmcsr as i32) & 3 != 0 {
let __v1: u16 = ((pmcsr as i32) & !3) as u16;
pmcsr = __v1;
unsafe { write_pci_config_16(bus as u8, slot as u8, func as u8, 68u8, pmcsr) };
mdelay!(10i32);
pmcsr = unsafe { read_pci_config_16(bus as u8, slot as u8, func as u8, 68u8) };
if (pmcsr as i32) & 3 != 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013pci 0000:%02x:%02x.%d: Cannot power up Apple AirPort card\n".as_ptr() as *const u8, bus, slot, func) };
return;
}
}
addr = (unsafe { read_pci_config(bus as u8, slot as u8, func as u8, 16u8) }) as u64;
addr |= ((unsafe { read_pci_config(bus as u8, slot as u8, func as u8, 20u8) }) as u64) << 32i32;
let __v11: u64 = addr & !15;
addr = __v11;
mmio = unsafe { early_ioremap(addr, 16384u64) };
if mmio.is_null() {
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x013pci 0000:%02x:%02x.%d: Cannot iomap Apple AirPort card\n".as_ptr() as *const u8, bus, slot, func) };
return;
}
let mut __v14: i32;
while false {
}
__v14 = unsafe { _printk(c"\x016Resetting Apple AirPort card (left enabled by EFI)\n".as_ptr() as *const u8) };
while bcma_aread32__2!(2052i32, mmio) != 0 && i < 30 {
udelay(10u64);
i = i.wrapping_add(1);
}
bcma_awrite32!(2048i32, 1i32, mmio);
bcma_aread32!(2048i32, mmio);
udelay(1u64);
bcma_awrite32!(2048i32, 0i32, mmio);
bcma_aread32!(2048i32, mmio);
udelay(10u64);
unsafe { early_iounmap(mmio, 16384u64) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct chipset {
    vendor: u32,
    device: u32,
    class: u32,
    class_mask: u32,
    flags: u32,
    f: Option<unsafe extern "C" fn(i32, i32, i32)>,
}


/// HPET on the current version of the Baytrail platform has accuracy
/// problems: it will halt in deep idle state - so we disable it.
///
/// More details can be found in section 18.10.1.3 of the datasheet:
///
///    http://www.intel.com/content/dam/www/public/us/en/documents/datasheets/atom-z8000-datasheet-vol-1.pdf
#[unsafe(link_section = ".init.data")]
static mut early_qrk: __SlateAlign16<[chipset; 12]> = __SlateAlign16([{
let mut __t0: chipset = unsafe { core::mem::zeroed() };
__t0.vendor = 4318u32;
__t0.device = 4294967295u32;
__t0.class = 1540u32;
__t0.class_mask = 4294967295u32;
__t0.flags = 1u32;
__t0.f = Some(nvidia_bugs);
__t0
}, {
let mut __t1: chipset = unsafe { core::mem::zeroed() };
__t1.vendor = 4358u32;
__t1.device = 4294967295u32;
__t1.class = 1540u32;
__t1.class_mask = 4294967295u32;
__t1.flags = 1u32;
__t1.f = Some(via_bugs);
__t1
}, {
let mut __t2: chipset = unsafe { core::mem::zeroed() };
__t2.vendor = 4130u32;
__t2.device = 4352u32;
__t2.class = 1536u32;
__t2.class_mask = 4294967295u32;
__t2.flags = 0u32;
__t2.f = Some(fix_hypertransport_config);
__t2
}, {
let mut __t3: chipset = unsafe { core::mem::zeroed() };
__t3.vendor = 4098u32;
__t3.device = 17266u32;
__t3.class = 3077u32;
__t3.class_mask = 4294967295u32;
__t3.flags = 0u32;
__t3.f = Some(ati_bugs);
__t3
}, {
let mut __t4: chipset = unsafe { core::mem::zeroed() };
__t4.vendor = 4098u32;
__t4.device = 17285u32;
__t4.class = 3077u32;
__t4.class_mask = 4294967295u32;
__t4.flags = 0u32;
__t4.f = Some(ati_bugs_contd);
__t4
}, {
let mut __t5: chipset = unsafe { core::mem::zeroed() };
__t5.vendor = 32902u32;
__t5.device = 13315u32;
__t5.class = 1536u32;
__t5.class_mask = 6u32;
__t5.flags = 0u32;
__t5.f = Some(intel_remapping_check);
__t5
}, {
let mut __t6: chipset = unsafe { core::mem::zeroed() };
__t6.vendor = 32902u32;
__t6.device = 13317u32;
__t6.class = 1536u32;
__t6.class_mask = 6u32;
__t6.flags = 0u32;
__t6.f = Some(intel_remapping_check);
__t6
}, {
let mut __t7: chipset = unsafe { core::mem::zeroed() };
__t7.vendor = 32902u32;
__t7.device = 13318u32;
__t7.class = 1536u32;
__t7.class_mask = 6u32;
__t7.flags = 0u32;
__t7.f = Some(intel_remapping_check);
__t7
}, {
let mut __t8: chipset = unsafe { core::mem::zeroed() };
__t8.vendor = 32902u32;
__t8.device = 4294967295u32;
__t8.class = 768u32;
__t8.class_mask = 4294967295u32;
__t8.flags = 0u32;
__t8.f = Some(intel_graphics_quirks);
__t8
}, {
let mut __t9: chipset = unsafe { core::mem::zeroed() };
__t9.vendor = 32902u32;
__t9.device = 3840u32;
__t9.class = 1536u32;
__t9.class_mask = 4294967295u32;
__t9.flags = 0u32;
__t9.f = Some(force_disable_hpet);
__t9
}, {
let mut __t10: chipset = unsafe { core::mem::zeroed() };
__t10.vendor = 5348u32;
__t10.device = 17201u32;
__t10.class = 640u32;
__t10.class_mask = 4294967295u32;
__t10.flags = 0u32;
__t10.f = Some(apple_airport_reset);
__t10
}, {
let mut __t11: chipset = unsafe { core::mem::zeroed() };
__t11
}]);

/// check_dev_quirk - apply early quirks to a given PCI device
/// @num: bus number
/// @slot: slot number
/// @func: PCI function
///
/// Check the vendor & device ID against the early quirks table.
///
/// If the device is single function, let early_pci_scan_bus() know so we don't
/// poke at this device again.
#[unsafe(link_section = ".init.text")]
fn check_dev_quirk(num: i32, slot: i32, func: i32) -> i32 {
let class: u16 = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 10u8) };
let mut vendor: u16 = 0;
let mut device: u16 = 0;
let mut r#type: u8 = 0;
let mut sec: u8 = 0;
let mut i: i32 = 0;
if (class as i32) == 65535 {
return 1i32.wrapping_neg();
}
// no class, treat as single function
vendor = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 0u8) };
device = unsafe { read_pci_config_16(num as u8, slot as u8, func as u8, 2u8) };
while (unsafe { early_qrk.0[i as usize].f }) != None {
if ((unsafe { early_qrk.0[i as usize].vendor }) == 4294967295u32 || (unsafe { early_qrk.0[i as usize].vendor }) == ((vendor as i32) as u32)) && ((unsafe { early_qrk.0[i as usize].device }) == 4294967295u32 || (unsafe { early_qrk.0[i as usize].device }) == ((device as i32) as u32)) && ((unsafe { early_qrk.0[i as usize].class }) ^ ((class as i32) as u32)) & unsafe { early_qrk.0[i as usize].class_mask } == 0 {
if (unsafe { early_qrk.0[i as usize].flags }) & 3u32 != 3u32 {
unsafe { unsafe { early_qrk.0[i as usize].f }.unwrap()(num, slot, func) };
}
let __v0: *mut chipset = unsafe { core::ptr::addr_of_mut!(early_qrk.0) as *mut chipset }.wrapping_offset(i as isize);
unsafe {
(*__v0).flags = (*__v0).flags | 2;
}
}
i = i.wrapping_add(1);
}
r#type = unsafe { read_pci_config_byte(num as u8, slot as u8, func as u8, 14u8) };
if (r#type as i32) & 127 == 1 {
sec = unsafe { read_pci_config_byte(num as u8, slot as u8, func as u8, 25u8) };
if (sec as i32) > num {
early_pci_scan_bus(sec as i32);
}
}
if (r#type as i32) & 128 == 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".init.text")]
fn early_pci_scan_bus(bus: i32) {
// Poor man's PCI discovery
for slot in 0i32..32i32 {
for func in 0i32..8i32 {
// Only probe function 0 on single fn devices
if check_dev_quirk(bus, slot, func) != 0 {
break;
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn early_quirks() {
if (unsafe { early_pci_allowed() }) == 0 {
return;
}
early_pci_scan_bus(0i32);
}

static __slate_asm_string_gen3_stolen_size_3: [u8; 30] = *b"Unknown GMCH_CTRL value: %x!\n\0";

static __slate_asm_string_gen3_stolen_size_4: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/early-quirks.c\0";

static __slate_asm_string_i830_stolen_size_3: [u8; 30] = *b"Unknown GMCH_CTRL value: %x!\n\0";

static __slate_asm_string_i830_stolen_size_4: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/early-quirks.c\0";

static __slate_asm_string_i845_tseg_size_3: [u8; 28] = *b"Unknown ESMRAMC value: %x!\n\0";

static __slate_asm_string_i845_tseg_size_4: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/early-quirks.c\0";
