
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
//! This file contains work-arounds for many known PCI hardware bugs.
//! Devices present only on certain architectures (host bridges et cetera)
//! should be handled in arch-specific code.
//!
//! Note: any quirks for hotpluggable devices must _NOT_ be declared __init.
//!
//! Copyright (c) 1999 Martin Mares <mj@ucw.cz>
//!
//! Init/reset quirks for USB host controllers should be in the USB quirks
//! file, where their drivers can use them.
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! __struct_group() - Create a mirrored named and anonyomous struct
//!
//! @TAG: The tag name for the named sub-struct (usually empty)
//! @NAME: The identifier name of the mirrored sub-struct
//! @ATTRS: Any struct attributes (usually empty)
//! @MEMBERS: The member declarations for the mirrored structs
//!
//! Used to create an anonymous union of two structs with identical layout
//! and size: one anonymous and one named. The former's members can be used
//! normally without sub-struct naming, and the latter can be used to
//! reason about the start, end, and size of the group of struct members.
//! The named struct can also be explicitly tagged for layer reuse (C only),
//! as well as both having struct attributes appended.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! AER and DPC capabilities TLP Logging register sizes (PCIe r6.2, sec 7.8.4
//! & 7.9.14).
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! FIELD_PREP() - prepare a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_val:  value to put in the field
//!
//! FIELD_PREP() masks and shifts up the value.  The result should
//! be combined with other fields of the bitfield using logical OR.
//! # chars in a path name including nul
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
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
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
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
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! If !LOCKDEP, the value is meaningless
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! New user namespace
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
//! Each PCI slot may be a combo card with its own bus.  4 IRQ pins per slot.
//! Helper macro to avoid gfp flags if they are the default one
//! Keep unconverted code working
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
//! Conventional PCI and PCI-X Mode 1 devices have 256 bytes of
//! configuration space.  PCI-X Mode 2 and PCIe devices have 4096 bytes of
//! configuration space.
//! Vendors and devices.  Sort key: vendor first, device next.
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! 8237 DMA controllers
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! NQN names in commands fields specified one size
//! Runtime PM flag argument bits
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
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
// __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// struct_group() - Wrap a set of declarations in a mirrored struct
//
// @NAME: The identifier name of the mirrored sub-struct
// @MEMBERS: The member declarations for the mirrored structs
//
// Used to create an anonymous union of two structs with identical
// layout and size: one anonymous and one named. The former can be
// used normally without sub-struct naming, and the latter can be
// used to reason about the start, end, and size of the group of
// struct members.
// no tag
// no attrs
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// Force a compilation error if a constant expression is not a power of 2
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
// FIELD_GET() - extract a bitfield element
// @_mask: shifted mask defining the field's length and position
// @_reg:  value of entire bitfield
//
// FIELD_GET() extracts the field specified by @_mask from the
// bitfield passed in as @_reg by masking and shifting it down.
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
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
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
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// min_t - return minimum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// in_range - Determine if a value lies within a range.
// @val: Value to test.
// @start: First value in range.
// @len: Number of values in range.
//
// This is more efficient than "if (start <= val && val < (start + len))".
// It also gives a different answer if @start + @len overflows the size of
// the type by a sufficient amount to encompass @val.  Decide for yourself
// which behaviour you want, or prove that start + len never overflow.
// Do not blindly replace one form with the other.
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// SCHED_ISO: reserved but not implemented yet
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// See the comment in asm-generic/io.h about ioremap_np().
// Number of standard BARs
// 16 bits
// 16 bits
// 16 bits
// Enable response in Memory space
// Enable bus mastering
// Enable back-to-back writes
// High 24 bits are class, low 8 revision
// Reg. Level Programming Interface
// Device class
// 8 bits
// 8 bits
// Multi-Function Device (possible)
// Base addresses specify locations in memory or I/O space.
// Decoded size can be determined by writing a value of
// 0xffffffff to the register, and reading it back.  Only
// 1 bits are decoded.
// 32 bits
// 0x35-0x3b are reserved
// 8 bits
// 16-bit PC Card legacy mode base address (ExCa)
// Message Signalled Interrupts
// HyperTransport
// PCI Express
// Capability defined flags (16 bits)
// PME# from D0
// Start of the PME Mask in PMC
// PM control and status register
// Current power state (D0 to D3)
// MSI-X registers (in MSI-X capability)
// Message Control
// Mask all vectors for this function
// MSI-X enable
// Capabilities register
// Device/Port type
// Root Port
// Upstream Port
// Downstream Port
// PCIe to PCI/PCI-X Bridge
// PCI/PCI-X to PCIe Bridge
// Root Complex Integrated Endpoint
// Device capabilities
// Max_Payload_Size
// L1 Acceptable Latency
// Device Control
// Enable relaxed ordering
// Enable No Snoop
// ASPM L0s Support
// ASPM L1 Support
// Link Control
// Link Status
// Current Link Speed
// Negotiated Link Width
// Data Link Layer Link Active
// Link Bandwidth Management Status
// Slot Control
// Root Control
// Device Control 2
// Link Control 2
// Supported Speed 2.5GT/s
// Supported Speed 5GT/s
// Supported Speed 8GT/s
// Supported Speed 16GT/s
// Supported Speed 32GT/s
// Supported Speed 64GT/s
// Flit Mode Status
// Slot Control 2
// Slave/Primary link configuration
// MSI Mapping Capability
// Offset to flags
// Mapping enable
// Number of VF BARs
// Access Control Service
// ACS Capability Register
// Source Validation
// Translation Blocking
// P2P Request Redirect
// P2P Completion Redirect
// Upstream Forwarding
// P2P Egress Control
// Direct Translated P2P
// ACS Control Register
// Radeon
//
// RadeonIGP
// duplicate: ECTIVA
// duplicate: NXP
// duplicate: FREESCALE
// Tbt 1 Gen 1
// Tbt 1 Gen 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
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
macro_rules! ACPI_COMPANION {
($dev:expr) => {
to_acpi_device_node!(unsafe { (*$dev).fwnode })
};
}

macro_rules! ACPI_FAILURE {
($a:expr) => {
$a
};
}

macro_rules! ACPI_HANDLE {
($dev:expr) => {
acpi_device_handle(ACPI_COMPANION!($dev))
};
}

/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__7 {
() => {
((core::mem::size_of::<[u16; 14]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__5!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__10 {
() => {
((core::mem::size_of::<[u16; 119]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__3!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__19 {
() => {
((core::mem::size_of::<[pci_dev_acs_ops; 2]>() as u64) / (core::mem::size_of::<pci_dev_acs_ops>() as u64)).wrapping_add(__must_be_array__19!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__21 {
() => {
((core::mem::size_of::<[pci_dev_acs_ops; 2]>() as u64) / (core::mem::size_of::<pci_dev_acs_ops>() as u64)).wrapping_add(__must_be_array__14!() as u64)
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

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as u64).wrapping_sub(1) == 0
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

macro_rules! NVME_CAP_TIMEOUT {
($cap:expr) => {
($cap) >> 24i32 & 255
};
}

/// to align the pointer to the (next) page boundary
macro_rules! PAGE_ALIGN {
($addr:expr) => {
ALIGN__2!($addr, 4096u64)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PCIE_LNKCTL2_TLS2SPEED {
($lnkctl2:expr) => {
{
let mut __v: i32;
let __v2: u16 = ((($lnkctl2) as i32) & 15) as u16;
__v = if (__v2 as i32) == 6 { 25 } else { if (__v2 as i32) == 5 { 24 } else { if (__v2 as i32) == 4 { 23 } else { if (__v2 as i32) == 3 { 22 } else { if (__v2 as i32) == 2 { 21 } else { if (__v2 as i32) == 1 { 20 } else { 255 } } } } } };
__v
}
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN__3 {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ($func) & 7
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN__2 {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ((($func) & 7) as u32)
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ($func) & 7
};
}

macro_rules! PCI_FUNC {
($devfn:expr) => {
($devfn) & 7
};
}

macro_rules! PCI_FUNC__3 {
($devfn:expr) => {
(($devfn) as i32) & 7
};
}

macro_rules! PCI_POSSIBLE_ERROR {
($val:expr) => {
($val) == 4294967295u32
};
}

macro_rules! PCI_SLOT {
($devfn:expr) => {
($devfn) >> 3i32 & 31
};
}

macro_rules! PCI_SLOT__3 {
($devfn:expr) => {
(($devfn) as i32) >> 3i32 & 31
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

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as i32).wrapping_sub(1))
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut *mut hlist_node, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut hlist_node, $val) };
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u32
};
}

macro_rules! __bf_cast_unsigned__3 {
($x:expr) => {
($x) as u64
};
}

macro_rules! __bf_cast_unsigned__4 {
($x:expr) => {
($x) as u8
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
core::mem::size_of::<__SlateRecord_quirks_c_4855>() as i32
};
}

macro_rules! __must_be_array__5 {
() => {
core::mem::size_of::<__SlateRecord_quirks_c_3693>() as i32
};
}

macro_rules! __must_be_array__14 {
() => {
core::mem::size_of::<__SlateRecord_quirks_c_5492>() as i32
};
}

macro_rules! __must_be_array__19 {
() => {
core::mem::size_of::<__SlateRecord_quirks_c_5471>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of {
($ptr:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of!($ptr)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fwnode) as u64
};
}

macro_rules! offsetof__2 {
() => {
core::mem::offset_of!(pci_dev, bus_list) as u64
};
}

macro_rules! pci_resource_end {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).end }
};
}

macro_rules! pci_resource_len {
($dev:expr, $bar:expr) => {
{
let __v: u64 = if pci_resource_end!($dev, $bar) != 0 {
resource_size(pci_resource_n!($dev, $bar) as *const resource)
} else {
0
};
__v
}
};
}

/// These helpers provide future and backwards compatibility
/// for accessing popular PCI BAR info
macro_rules! pci_resource_n {
($dev:expr, $bar:expr) => {
unsafe { (*$dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(($bar) as isize)
};
}

macro_rules! pci_resource_start {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).start }
};
}

/// Convenience shorthand with allocation
macro_rules! request_region {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(ioport_resource), ($start) as u64, ($n) as u64, ($name) as *const u8, 0i32) };
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__3!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before {
($a:expr, $b:expr) => {
time_after!($b, $a)
};
}

macro_rules! to_acpi_device_node {
($__fwnode:expr) => {
{
let mut __v: *mut acpi_device;
let __v2: *mut fwnode_handle = $__fwnode;
let __v3: *mut acpi_device = if unsafe { is_acpi_device_node(__v2 as *const fwnode_handle) } {
let mut __v4: *mut acpi_device;
__v4 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__3!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v4
} else {
core::ptr::null_mut::<acpi_device>()
};
__v = __v3;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__3 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == (core::ptr::addr_of_mut!(__v3) as *mut $type);
__v = 1;
__v
}
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut initcall_debug: bool;
    static mut kmsg_fops: file_operations;
    static mut ioport_resource: resource;
    static mut iomem_resource: resource;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut pcie_bus_config: u32;
    static mut noioapicquirk: i32;
    static mut noioapicreroute: i32;
    static mut pci_pci_problems: i32;
    static mut pci_dfl_cache_line_size: u8;
    static mut pci_cache_line_size: u8;
    static mut x86_apple_machine: bool;
    static mut kstat: kernel_stat;
    static mut pm_suspend_global_flags: u32;
    static mut pci_pm_d3hot_delay: u32;
    static mut __start_pci_fixups_early: [pci_fixup; 0];
    static mut __end_pci_fixups_early: [pci_fixup; 0];
    static mut __start_pci_fixups_header: [pci_fixup; 0];
    static mut __end_pci_fixups_header: [pci_fixup; 0];
    static mut __start_pci_fixups_final: [pci_fixup; 0];
    static mut __end_pci_fixups_final: [pci_fixup; 0];
    static mut __start_pci_fixups_enable: [pci_fixup; 0];
    static mut __end_pci_fixups_enable: [pci_fixup; 0];
    static mut __start_pci_fixups_resume: [pci_fixup; 0];
    static mut __end_pci_fixups_resume: [pci_fixup; 0];
    static mut __start_pci_fixups_resume_early: [pci_fixup; 0];
    static mut __end_pci_fixups_resume_early: [pci_fixup; 0];
    static mut __start_pci_fixups_suspend: [pci_fixup; 0];
    static mut __end_pci_fixups_suspend: [pci_fixup; 0];
    static mut __start_pci_fixups_suspend_late: [pci_fixup; 0];
    static mut __end_pci_fixups_suspend_late: [pci_fixup; 0];
    static mut nr_ioapics: i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn insert_resource(arg0: *mut resource, arg1: *mut resource) -> i32;
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn iomem_get_mapping() -> *mut address_space;
    fn __compiletime_assert_34();
    fn __compiletime_assert_35();
    fn __compiletime_assert_36();
    fn __compiletime_assert_37();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_46();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strstr(arg0: *const u8, arg1: *const u8) -> *mut u8;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get() -> i64;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn device_link_add(arg0: *mut device, arg1: *mut device, arg2: u32) -> *mut device_link;
    fn ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn iounmap(arg0: *mut ());
    fn ioread8(arg0: *const ()) -> u32;
    fn ioread16(arg0: *const ()) -> u32;
    fn ioread32(arg0: *const ()) -> u32;
    fn ioread32be(arg0: *const ()) -> u32;
    fn iowrite32(arg0: u32, arg1: *mut ());
    fn iowrite32be(arg0: u32, arg1: *mut ());
    fn pci_iomap(arg0: *mut pci_dev, arg1: i32, arg2: u64) -> *mut ();
    fn pci_iounmap(arg0: *mut pci_dev, arg1: *mut ());
    fn pcie_get_speed_cap(arg0: *mut pci_dev) -> u32;
    fn pci_find_host_bridge(arg0: *mut pci_bus) -> *mut pci_host_bridge;
    fn pcibios_bus_to_resource(arg0: *mut pci_bus, arg1: *mut resource, arg2: *mut pci_bus_region);
    fn pci_dev_put(arg0: *mut pci_dev);
    fn pci_find_capability(arg0: *mut pci_dev, arg1: i32) -> u8;
    fn pci_find_ht_capability(arg0: *mut pci_dev, arg1: i32) -> u8;
    fn pci_find_next_ht_capability(arg0: *mut pci_dev, arg1: u8, arg2: i32) -> u8;
    fn pci_get_device(arg0: u32, arg1: u32, arg2: *mut pci_dev) -> *mut pci_dev;
    fn pci_get_slot(arg0: *mut pci_bus, arg1: u32) -> *mut pci_dev;
    fn pci_get_domain_bus_and_slot(arg0: i32, arg1: u32, arg2: u32) -> *mut pci_dev;
    fn pci_bus_read_config_dword(arg0: *mut pci_bus, arg1: u32, arg2: i32, arg3: *mut u32) -> i32;
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pci_read_config_word(arg0: *const pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_byte(arg0: *const pci_dev, arg1: i32, arg2: u8) -> i32;
    fn pci_write_config_word(arg0: *const pci_dev, arg1: i32, arg2: u16) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pcie_capability_read_word(arg0: *mut pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pcie_capability_clear_and_set_word_unlocked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pcie_capability_clear_and_set_word_locked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pci_enable_device(arg0: *mut pci_dev) -> i32;
    fn pci_enable_device_mem(arg0: *mut pci_dev) -> i32;
    fn pci_disable_device(arg0: *mut pci_dev);
    fn pci_disable_parity(arg0: *mut pci_dev);
    fn pcie_get_readrq(arg0: *mut pci_dev) -> i32;
    fn pcie_set_readrq(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn pcie_reset_flr(arg0: *mut pci_dev, arg1: bool) -> i32;
    fn pcie_flr(arg0: *mut pci_dev) -> i32;
    fn pci_reset_bus(arg0: *mut pci_dev) -> i32;
    fn pci_save_state(arg0: *mut pci_dev) -> i32;
    fn pci_restore_state(arg0: *mut pci_dev);
    fn pci_claim_resource(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn pci_match_id(arg0: *const pci_device_id, arg1: *mut pci_dev) -> *const pci_device_id;
    fn pci_walk_bus(arg0: *mut pci_bus, arg1: Option<unsafe extern "C" fn(*mut pci_dev, *mut ()) -> i32>, arg2: *mut ());
    fn pci_msi_enabled() -> bool;
    fn pcie_set_target_speed(arg0: *mut pci_dev, arg1: u32, arg2: bool) -> i32;
    fn pci_ats_disabled() -> bool;
    fn pci_add_dma_alias(arg0: *mut pci_dev, arg1: u8, arg2: u32);
    fn device_create_managed_software_node(arg0: *mut device, arg1: *const property_entry, arg2: *const software_node) -> i32;
    fn pci_dev_reset_iommu_prepare(arg0: *mut pci_dev) -> i32;
    fn pci_dev_reset_iommu_done(arg0: *mut pci_dev);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn acpi_get_table(arg0: *mut u8, arg1: u32, arg2: *mut *mut acpi_table_header) -> u32;
    fn acpi_put_table(arg0: *mut acpi_table_header);
    fn acpi_get_handle(arg0: *mut (), arg1: *const u8, arg2: *mut *mut ()) -> u32;
    fn acpi_execute_simple_method(arg0: *mut (), arg1: *mut u8, arg2: u64) -> u32;
    fn is_acpi_device_node(arg0: *const fwnode_handle) -> bool;
    fn dmi_check_system(arg0: *const dmi_system_id) -> i32;
    fn dmi_get_system_info(arg0: i32) -> *const u8;
    fn pm_runtime_allow(arg0: *mut device);
    fn pcie_cap_has_lnkctl2(arg0: *const pci_dev) -> bool;
    fn pci_reset_supported(arg0: *mut pci_dev) -> bool;
    fn pci_find_saved_cap(arg0: *mut pci_dev, arg1: u8) -> *mut pci_cap_saved_state;
    fn pci_no_msi();
    fn pci_configure_extended_tags(arg0: *mut pci_dev, arg1: *mut ()) -> i32;
    fn pci_resource_name(arg0: *mut pci_dev, arg1: u32) -> *const u8;
    fn pcie_aspm_remove_cap(arg0: *mut pci_dev, arg1: u32);
    fn __compiletime_assert_847();
    fn __compiletime_assert_848();
    fn __compiletime_assert_849();
    fn __compiletime_assert_850();
    fn __compiletime_assert_851();
    fn __compiletime_assert_1304();
    fn __compiletime_assert_1305();
    fn __compiletime_assert_1306();
    fn __compiletime_assert_1307();
    fn __compiletime_assert_1308();
    fn __compiletime_assert_1433();
    fn __compiletime_assert_1434();
    fn __compiletime_assert_1435();
    fn __compiletime_assert_1436();
    fn __compiletime_assert_1437();
    fn __compiletime_assert_1438();
    fn __compiletime_assert_1439();
    fn __compiletime_assert_1440();
    fn __compiletime_assert_1441();
    fn __compiletime_assert_1442();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
fn offset_to_ptr(off: *const i32) -> *mut () {
(off as u64).wrapping_add((unsafe { *off }) as u64) as *mut ()
}

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
fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
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
struct lock_class_key {
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
struct lockdep_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


// Resource type
// PCI/ISA I/O ports
// size indicates alignment
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// PCI control bits.  Shares IORESOURCE_BITS with above PCI ROM.
// Do not move resource
fn resource_set_size(res: *mut resource, size: u64) {
unsafe {
(*res).end = unsafe { (*res).start }.wrapping_add(size).wrapping_sub(1);
}
}

fn resource_set_range(res: *mut resource, start: u64, size: u64) {
unsafe {
(*res).start = start;
}
resource_set_size(res, size);
}

fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
fn hlist_add_head(n: *mut hlist_node, h: *mut hlist_head) {
let first: *mut hlist_node = unsafe { (*h).first };
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_34() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*n).next, first);
if true {
break;
}
}
if !first.is_null() {
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_35() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*first).pprev, unsafe { core::ptr::addr_of_mut!((*n).next) });
if true {
break;
}
}
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_36() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*h).first, n);
if true {
break;
}
}
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_37() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*n).pprev, unsafe { core::ptr::addr_of_mut!((*h).first) });
if true {
break;
}
}
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_46() };
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct pollfd {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
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
struct io_bitmap {
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
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits154U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits155U0,
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
    __slate_bits_0: __slate_bits::__SlateBits156U0,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
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
struct workqueue_struct {
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
struct rseq {
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
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
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
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
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits285U0,
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
    __slate_bits_0: __slate_bits::__SlateBits71U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits71U1,
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
    __slate_bits_2: __slate_bits::__SlateBits71U2,
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
struct rq_flags {
}


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// This thread should not be frozen
// I am kswapd
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


// Change timeval to jiffies, trying to avoid the
// most obvious overflows..
//
// And some not so obvious.
//
// Note that we don't want to return LONG_MAX, because
// for various timeout reasons we often end up having
// to wait "jiffies+1" in order to guarantee that we wait
// at _least_ "jiffies" - so "jiffies+1" had better still
// be positive.
fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
}
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
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
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct xol_area {
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits409U0,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
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
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
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
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
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
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
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
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
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
    __slate_bits_0: __slate_bits::__SlateBits459U0,
    vm_ops: *const vm_operations_struct,
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
struct irq_domain {
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
    __slate_bits_0: __slate_bits::__SlateBits461U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits486U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits486U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits486U2,
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
    __slate_bits_3: __slate_bits::__SlateBits486U3,
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
struct sock {
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table {
    procname: *const u8,
    data: *mut (),
    maxlen: i32,
    mode: u16,
    proc_handler: Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>,
    poll: *mut ctl_table_poll,
    extra1: *mut (),
    extra2: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_header {
    __slate_anon_0: __SlateRecord_sysctl_h_254,
    unregistering: *mut completion,
    ctl_table_arg: *const ctl_table,
    root: *mut ctl_table_root,
    set: *mut ctl_table_set,
    parent: *mut ctl_dir,
    node: *mut ctl_node,
    inodes: hlist_head,
    r#type: u32,
}


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
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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
union __SlateRecord_sysctl_h_254 {
    __slate_anon_0: __SlateRecord_sysctl_h_255,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sysctl_h_255 {
    ctl_table: *const ctl_table,
    ctl_table_size: i32,
    used: i32,
    count: i32,
    nreg: i32,
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
struct folio_queue {
}


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


fn task_pid_nr(tsk: *mut task_struct) -> i32 {
unsafe { (*tsk).pid }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits288U0,
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits631U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
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
unsafe { __compiletime_assert_415() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("416: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 416b - ., 3; .popsection\n# {0}", const 416 as i32, options(nostack, att_syntax));
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


// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
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
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
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
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
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
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


// Device link flags.
//
// STATELESS: The core will not remove this link automatically.
// AUTOREMOVE_CONSUMER: Remove the link automatically on consumer driver unbind.
// PM_RUNTIME: If set, the runtime PM framework will use this link.
// RPM_ACTIVE: Run pm_runtime_get_sync() on the supplier during link creation.
// AUTOREMOVE_SUPPLIER: Remove the link automatically on supplier driver unbind.
// AUTOPROBE_CONSUMER: Probe consumer driver automatically after supplier binds.
// MANAGED: The core tracks presence of supplier/consumer drivers (internal).
// SYNC_STATE_ONLY: Link only affects sync_state() behavior.
// INFERRED: Inferred from data (eg: firmware) and not from driver actions.
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
struct device_link {
    supplier: *mut device,
    s_node: list_head,
    consumer: *mut device,
    c_node: list_head,
    link_dev: device,
    status: i32,
    flags: u32,
    rpm_active: refcount_struct,
    kref: kref,
    rm_work: work_struct,
    supplier_preactivated: bool,
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
    __slate_bits_0: __slate_bits::__SlateBits483U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn device_disable_async_suspend(dev: *mut device) {
if ((unsafe { (*dev).power.__slate_bits_0.__get_is_prepared() }) as i32) == 0 {
unsafe {
let __t0 = false;
(*dev).power.__slate_bits_0.__set_async_suspend(__t0);
}
}
}

#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
}


#[inline(always)]
fn __outw(mut value: u16, mut port: u16) {
unsafe {
core::arch::asm!("outw %ax, %dx", in("ax") value as u16, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[inline(always)]
fn __inw(mut port: u16) -> u16 {
let mut value: u16 = 0;
unsafe {
let __t0: u16;
core::arch::asm!("inw %dx, %ax", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
value
}

fn readb(mut addr: *const ()) -> u8 {
let mut ret: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb ({1:r}),{0:l}", lateout(reg) __t0, in(reg) addr as *mut u8, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
ret
}

fn readl(mut addr: *const ()) -> u32 {
let mut ret: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl ({1:r}),{0:e}", lateout(reg) __t0, in(reg) addr as *mut u32, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


fn writeb(mut val: u8, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movb {0:l},({1:r})", in(reg) (val as u8) as u32, in(reg) addr as *mut u8, options(nostack, att_syntax));
}
}

fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
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
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits744U0,
    kobj: kobject,
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
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits72U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits72U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits72U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits72U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


fn pci_physfn(dev: *mut pci_dev) -> *mut pci_dev {
dev
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct pci_host_bridge {
    dev: device,
    bus: *mut pci_bus,
    ops: *mut pci_ops,
    child_ops: *mut pci_ops,
    sysdata: *mut (),
    busnr: i32,
    domain_nr: i32,
    windows: list_head,
    dma_ranges: list_head,
    ports: list_head,
    swizzle_irq: Option<unsafe extern "C" fn(*mut pci_dev, *mut u8) -> u8>,
    map_irq: Option<unsafe extern "C" fn(*const pci_dev, u8, u8) -> i32>,
    release_fn: Option<unsafe extern "C" fn(*mut pci_host_bridge)>,
    enable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    disable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev)>,
    reset_root_port: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    release_data: *mut (),
    __slate_bits_0: __slate_bits::__SlateBits763U0,
    align_resource: Option<unsafe extern "C" fn(*mut pci_dev, *const resource, u64, u64, u64) -> u64>,
    __slate_pad_19: [u8; 40],
    private: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits438U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_property_h_397 {
    pointer: *const (),
    value: __SlateRecord_property_h_399,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_property_h_399 {
    u8_data: [u8; 8],
    u16_data: [u16; 4],
    u32_data: [u32; 2],
    u64_data: [u64; 1],
    str: [*const u8; 1],
}


fn pci_is_root_bus(pbus: *mut pci_bus) -> bool {
unsafe { (*pbus).parent }.is_null()
}

fn pci_upstream_bridge(mut dev: *mut pci_dev) -> *mut pci_dev {
dev = pci_physfn(dev);
if pci_is_root_bus(unsafe { (*dev).bus }) {
return core::ptr::null_mut::<pci_dev>();
}
unsafe { (*(*dev).bus).self_ }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iova_bitmap {
}


// Error values that may be returned by PCI functions
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus_region {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_viommu {
}


// PCI_DEVICE_SUB - macro used to describe a specific PCI device with subsystem
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
// @subvend: the 16 bit PCI Subvendor ID
// @subdev: the 16 bit PCI Subdevice ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device with subsystem information.
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
fn pcie_capability_clear_and_set_word(dev: *mut pci_dev, pos: i32, clear: u16, set: u16) -> i32 {
match pos {
16 | 28 | 48 => {
return unsafe { pcie_capability_clear_and_set_word_locked(dev, pos, clear, set) };
}
_ => {
return unsafe { pcie_capability_clear_and_set_word_unlocked(dev, pos, clear, set) };
}
}
unsafe { core::mem::zeroed() }
}

fn pcie_capability_clear_word(dev: *mut pci_dev, pos: i32, clear: u16) -> i32 {
pcie_capability_clear_and_set_word(dev, pos, clear, 0u16)
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


// Bits set in the VMA until the stack is in its final location
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
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sysdata {
    domain: i32,
    node: i32,
    companion: *mut acpi_device,
    iommu: *mut (),
    fwnode: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_234 {
    iova_cookie: *mut iommu_dma_cookie,
    msi_cookie: *mut iommu_dma_msi_cookie,
    iommufd_hwpt: *mut iommufd_hw_pagetable,
    __slate_anon_3: __SlateRecord_iommu_h_238,
    __slate_anon_4: __SlateRecord_iommu_h_242,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_hw_pagetable {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_238 {
    handler: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32, *mut ()) -> i32>,
    handler_token: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_242 {
    mm: *mut mm_struct,
    users: i32,
    next: list_head,
}


fn to_pci_sysdata(bus: *const pci_bus) -> *mut pci_sysdata {
(unsafe { (*bus).sysdata }) as *mut pci_sysdata
}

fn pci_domain_nr(bus: *mut pci_bus) -> i32 {
unsafe { (*to_pci_sysdata(bus as *const pci_bus)).domain }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_364 {
    pgsize: u64,
    pt: __SlateRecord_iommu_h_370,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_370 {
    leaf_levels_bitmap: u8,
    table_levels_bitmap: u8,
}


fn pci_name(pdev: *const pci_dev) -> *const u8 {
dev_name(unsafe { core::ptr::addr_of!((*pdev).dev) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_fixup {
    vendor: u16,
    device: u16,
    class: u32,
    class_shift: u32,
    hook_offset: i32,
}


// No PCI PCI DMA
// Need low latency setting
// No PCI to AGP DMA
fn pci_pcie_cap(dev: *mut pci_dev) -> i32 {
(unsafe { (*dev).pcie_cap }) as i32
}

fn pci_is_pcie(dev: *mut pci_dev) -> bool {
pci_pcie_cap(dev) != 0
}

fn pcie_caps_reg(dev: *const pci_dev) -> u16 {
unsafe { (*dev).pcie_flags_reg }
}

fn pci_pcie_type(dev: *const pci_dev) -> i32 {
((pcie_caps_reg(dev) as i32) & 240) >> 4i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_910 {
    r#type: u32,
    value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_915 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_921 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_927 {
    r#type: u32,
    count: u32,
    elements: *mut acpi_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_933 {
    r#type: u32,
    actual_type: u32,
    handle: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_939 {
    r#type: u32,
    proc_id: u32,
    pblk_address: u64,
    pblk_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_946 {
    r#type: u32,
    system_level: u32,
    resource_order: u32,
}


fn pcie_find_root_port(mut dev: *mut pci_dev) -> *mut pci_dev {
while !dev.is_null() {
let __v0: bool = pci_is_pcie(dev) && pci_pcie_type(dev as *const pci_dev) == 4;
if __v0 {
return dev;
}
dev = pci_upstream_bridge(dev);
}
core::ptr::null_mut::<pci_dev>()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct software_node_ref_args {
    swnode: *const software_node,
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct property_entry {
    name: *const u8,
    length: u64,
    is_inline: bool,
    r#type: u32,
    __slate_anon_4: __SlateRecord_property_h_397,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct software_node {
    name: *const u8,
    parent: *const software_node,
    properties: *const property_entry,
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
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_page_request {
    flags: u32,
    pasid: u32,
    grpid: u32,
    perm: u32,
    addr: u64,
    private_data: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault {
    r#type: u32,
    prm: iommu_fault_page_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_page_response {
    pasid: u32,
    grpid: u32,
    code: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_fault {
    fault: iommu_fault,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits856U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_group {
    last_fault: iopf_fault,
    faults: list_head,
    fault_count: u64,
    pending_node: list_head,
    work: work_struct,
    attach_handle: *mut iommu_attach_handle,
    fault_param: *mut iommu_fault_param,
    node: list_head,
    cookie: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_queue {
    wq: *mut workqueue_struct,
    devices: list_head,
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_geometry {
    aperture_start: u64,
    aperture_end: u64,
    force_aperture: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits860U0,
}


// Domain for use in DMA-API
// implementation
#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain {
    r#type: u32,
    cookie_type: u32,
    is_iommupt: bool,
    ops: *const iommu_domain_ops,
    dirty_ops: *const iommu_dirty_ops,
    owner: *const iommu_ops,
    pgsize_bitmap: u64,
    geometry: iommu_domain_geometry,
    iopf_handler: Option<unsafe extern "C" fn(*mut iopf_group) -> i32>,
    __slate_anon_9: __SlateRecord_iommu_h_234,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_pages_list {
    pages: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_iotlb_gather {
    start: u64,
    end: u64,
    __slate_anon_2: __SlateRecord_iommu_h_364,
    freelist: iommu_pages_list,
    queued: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_bitmap {
    bitmap: *mut iova_bitmap,
    gather: *mut iommu_iotlb_gather,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_ops {
    set_dirty_tracking: Option<unsafe extern "C" fn(*mut iommu_domain, bool) -> i32>,
    read_and_clear_dirty: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_dirty_bitmap) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data {
    r#type: u32,
    uptr: *mut (),
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data_array {
    r#type: u32,
    uptr: *mut (),
    entry_len: u64,
    entry_num: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_ops {
    capable: Option<unsafe extern "C" fn(*mut device, u32) -> bool>,
    hw_info: Option<unsafe extern "C" fn(*mut device, *mut u32, *mut u32) -> *mut ()>,
    domain_alloc_identity: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_paging_flags: Option<unsafe extern "C" fn(*mut device, u32, *const iommu_user_data) -> *mut iommu_domain>,
    domain_alloc_paging: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_sva: Option<unsafe extern "C" fn(*mut device, *mut mm_struct) -> *mut iommu_domain>,
    domain_alloc_nested: Option<unsafe extern "C" fn(*mut device, *mut iommu_domain, u32, *const iommu_user_data) -> *mut iommu_domain>,
    probe_device: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_device>,
    release_device: Option<unsafe extern "C" fn(*mut device)>,
    probe_finalize: Option<unsafe extern "C" fn(*mut device)>,
    device_group: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>,
    get_resv_regions: Option<unsafe extern "C" fn(*mut device, *mut list_head)>,
    of_xlate: Option<unsafe extern "C" fn(*mut device, *const of_phandle_args) -> i32>,
    is_attach_deferred: Option<unsafe extern "C" fn(*mut device) -> bool>,
    page_response: Option<unsafe extern "C" fn(*mut device, *mut iopf_fault, *mut iommu_page_response)>,
    def_domain_type: Option<unsafe extern "C" fn(*mut device) -> i32>,
    get_viommu_size: Option<unsafe extern "C" fn(*mut device, u32) -> u64>,
    viommu_init: Option<unsafe extern "C" fn(*mut iommufd_viommu, *mut iommu_domain, *const iommu_user_data) -> i32>,
    default_domain_ops: *const iommu_domain_ops,
    owner: *mut module,
    identity_domain: *mut iommu_domain,
    blocked_domain: *mut iommu_domain,
    release_domain: *mut iommu_domain,
    default_domain: *mut iommu_domain,
    __slate_bits_0: __slate_bits::__SlateBits794U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_ops {
    attach_dev: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, *mut iommu_domain) -> i32>,
    set_dev_pasid: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32, *mut iommu_domain) -> i32>,
    map_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, u64, i32, u32, *mut u64) -> i32>,
    unmap_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_iotlb_gather) -> u64>,
    flush_iotlb_all: Option<unsafe extern "C" fn(*mut iommu_domain)>,
    iotlb_sync_map: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64) -> i32>,
    iotlb_sync: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_iotlb_gather)>,
    cache_invalidate_user: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_user_data_array) -> i32>,
    iova_to_phys: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> u64>,
    enforce_cache_coherency: Option<unsafe extern "C" fn(*mut iommu_domain) -> bool>,
    set_pgtable_quirks: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> i32>,
    free: Option<unsafe extern "C" fn(*mut iommu_domain)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_device {
    list: list_head,
    ops: *const iommu_ops,
    fwnode: *mut fwnode_handle,
    dev: *mut device,
    singleton_group: *mut iommu_group,
    max_pasids: u32,
    ready: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_param {
    lock: mutex,
    users: refcount_struct,
    rcu: callback_head,
    dev: *mut device,
    queue: *mut iopf_queue,
    queue_list: list_head,
    partial: list_head,
    faults: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
    lock: mutex,
    fault_param: *mut iommu_fault_param,
    fwspec: *mut iommu_fwspec,
    iommu_dev: *mut iommu_device,
    r#priv: *mut (),
    max_pasids: u32,
    __slate_bits_0: __slate_bits::__SlateBits716U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fwspec {
    iommu_fwnode: *mut fwnode_handle,
    flags: u32,
    num_ids: u32,
    ids: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_attach_handle {
    domain: *mut iommu_domain,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_user_namespace_h_26 {
    __slate_anon_0: __SlateRecord_user_namespace_h_27,
    __slate_anon_1: __SlateRecord_user_namespace_h_31,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_27 {
    extent: [uid_gid_extent; 5],
    nr_extents: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_31 {
    forward: *mut uid_gid_extent,
    reverse: *mut uid_gid_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct binfmt_misc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
    pasid: u32,
    mm: *mut mm_struct,
    sva_domains: list_head,
    mm_list_elm: list_head,
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

// Fixed by ACPI spec
#[repr(C)]
#[derive(Clone, Copy)]
union acpi_object {
    r#type: u32,
    integer: __SlateRecord_actypes_h_910,
    string: __SlateRecord_actypes_h_915,
    buffer: __SlateRecord_actypes_h_921,
    package: __SlateRecord_actypes_h_927,
    reference: __SlateRecord_actypes_h_933,
    processor: __SlateRecord_actypes_h_939,
    power_resource: __SlateRecord_actypes_h_946,
}


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


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits841U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_scan_handler {
    list_node: list_head,
    ids: *const acpi_device_id,
    r#match: Option<unsafe extern "C" fn(*const u8, *mut *const acpi_device_id) -> bool>,
    attach: Option<unsafe extern "C" fn(*mut acpi_device, *const acpi_device_id) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut acpi_device)>,
    post_eject: Option<unsafe extern "C" fn(*mut acpi_device)>,
    bind: Option<unsafe extern "C" fn(*mut device)>,
    unbind: Option<unsafe extern "C" fn(*mut device)>,
    hotplug: acpi_hotplug_profile,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_context {
    self_: *mut acpi_device,
    notify: Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>,
    uevent: Option<unsafe extern "C" fn(*mut acpi_device, u32)>,
    fixup: Option<unsafe extern "C" fn(*mut acpi_device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits847U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits848U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits852U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_pnp {
    bus_id: [u8; 8],
    instance_no: i32,
    r#type: acpi_pnp_type,
    bus_address: u64,
    unique_id: *mut u8,
    ids: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_flags {
    __slate_bits_0: __slate_bits::__SlateBits854U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_state {
    resources: list_head,
    flags: __SlateRecord_acpi_bus_h_249,
    power: i32,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power {
    state: i32,
    flags: acpi_device_power_flags,
    states: [acpi_device_power_state; 5],
    state_for_enumeration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits858U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nt_partition_info {
    xlink_enabled: u32,
    target_part_low: u32,
    target_part_high: u32,
    reserved: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_state {
    flags: __SlateRecord_acpi_bus_h_281,
    power: u8,
    performance: u8,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_switchtec_h_357 {
    ctl: u32,
    win_size: u32,
    xlate_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_switchtec_h_362 {
    win_size: u32,
    reserved: [u32; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf {
    state: i32,
    flags: acpi_device_perf_flags,
    state_count: i32,
    states: *mut acpi_device_perf_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits862U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_quirks_c_3693 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_quirks_c_4855 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup {
    gpe_device: *mut (),
    gpe_number: u64,
    sleep_state: u64,
    resources: list_head,
    flags: acpi_device_wakeup_flags,
    context: acpi_device_wakeup_context,
    ws: *mut wakeup_source,
    prepare_count: i32,
    enable_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_quirks_c_5471 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_quirks_c_5492 {
}


// The maximum expected number of CSI-2 data lanes.
//
// This number is not expected to ever have to be equal to or greater than the
// number of bits in an unsigned long variable, but if it needs to be increased
// above that limit, code will need to be adjusted accordingly.
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_node_port {
    port_name: [u8; 9],
    data_lanes: [u32; 8],
    lane_polarities: [u32; 9],
    link_frequencies: [u64; 8],
    port_nr: u32,
    crs_csi2_local: bool,
    port_props: [property_entry; 2],
    ep_props: [property_entry; 8],
    remote_ep: [software_node_ref_args; 1],
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits600U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits631U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits285U0 {
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
    pub struct __SlateBits71U0 {
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
    pub struct __SlateBits71U1 {
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
    pub struct __SlateBits459U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits901U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits288U0 {
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
    pub struct __SlateBits71U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits154U0 {
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
    pub struct __SlateBits156U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits486U0 {
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
    pub struct __SlateBits483U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits486U1 {
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
    pub struct __SlateBits486U2 {
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
    pub struct __SlateBits486U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits794U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits716U0 {
        #[bits(1)]
        pub attach_deferred: u32,
        #[bits(1)]
        pub pci_32bit_workaround: u32,
        #[bits(1)]
        pub require_direct: u32,
        #[bits(1)]
        pub shadow_on_flush: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits461U0 {
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
    pub struct __SlateBits155U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits409U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits438U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits744U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits873U0 {
        #[bits(7)]
        pub slot: u8,
        #[bits(1)]
        pub exact_match: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits763U0 {
        #[bits(1)]
        pub ignore_reset_delay: u32,
        #[bits(1)]
        pub no_ext_tags: u32,
        #[bits(1)]
        pub no_inc_mrrs: u32,
        #[bits(1)]
        pub native_aer: u32,
        #[bits(1)]
        pub native_pcie_hotplug: u32,
        #[bits(1)]
        pub native_shpc_hotplug: u32,
        #[bits(1)]
        pub native_pme: u32,
        #[bits(1)]
        pub native_ltr: u32,
        #[bits(1)]
        pub native_dpc: u32,
        #[bits(1)]
        pub native_cxl_error: u32,
        #[bits(1)]
        pub preserve_config: u32,
        #[bits(1)]
        pub size_windows: u32,
        #[bits(1)]
        pub msi_domain: u32,
        #[bits(1)]
        pub broken_l1ss_resume: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits847U0 {
        #[bits(1)]
        pub present: u32,
        #[bits(1)]
        pub enabled: u32,
        #[bits(1)]
        pub show_in_ui: u32,
        #[bits(1)]
        pub functional: u32,
        #[bits(1)]
        pub battery_present: u32,
        #[bits(27)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits848U0 {
        #[bits(1)]
        pub dynamic_status: u32,
        #[bits(1)]
        pub removable: u32,
        #[bits(1)]
        pub ejectable: u32,
        #[bits(1)]
        pub power_manageable: u32,
        #[bits(1)]
        pub initialized: u32,
        #[bits(1)]
        pub visited: u32,
        #[bits(1)]
        pub hotplug_notify: u32,
        #[bits(1)]
        pub is_dock_station: u32,
        #[bits(1)]
        pub of_compatible_ok: u32,
        #[bits(1)]
        pub coherent_dma: u32,
        #[bits(1)]
        pub cca_seen: u32,
        #[bits(1)]
        pub enumeration_by_parent: u32,
        #[bits(1)]
        pub honor_deps: u32,
        #[bits(19)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits852U0 {
        #[bits(1)]
        pub hardware_id: u32,
        #[bits(1)]
        pub bus_address: u32,
        #[bits(1)]
        pub platform_id: u32,
        #[bits(1)]
        pub backlight: u32,
        #[bits(28)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits854U0 {
        #[bits(1)]
        pub explicit_get: u32,
        #[bits(1)]
        pub power_resources: u32,
        #[bits(1)]
        pub inrush_current: u32,
        #[bits(1)]
        pub power_removed: u32,
        #[bits(1)]
        pub ignore_parent: u32,
        #[bits(1)]
        pub dsw_present: u32,
        #[bits(26)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits856U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits862U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits858U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits860U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_nodes {
    dev_props: [property_entry; 6],
    nodes: *mut software_node,
    nodeptrs: *mut *const software_node,
    ports: *mut acpi_device_software_node_port,
    num_ports: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device {
    pld_crc: u32,
    device_type: i32,
    handle: *mut (),
    fwnode: fwnode_handle,
    wakeup_list: list_head,
    del_list: list_head,
    status: acpi_device_status,
    flags: acpi_device_flags,
    pnp: acpi_device_pnp,
    power: acpi_device_power,
    wakeup: acpi_device_wakeup,
    performance: acpi_device_perf,
    dir: acpi_device_dir,
    data: acpi_device_data,
    handler: *mut acpi_scan_handler,
    hp: *mut acpi_hotplug_context,
    swnodes: *mut acpi_device_software_nodes,
    driver_gpios: *const acpi_gpio_mapping,
    driver_data: *mut (),
    dev: device,
    physical_node_count: u32,
    dep_unmet: u32,
    physical_node_list: list_head,
    physical_node_lock: mutex,
    remove: Option<unsafe extern "C" fn(*mut acpi_device)>,
}


fn acpi_device_handle(adev: *mut acpi_device) -> *mut () {
if !adev.is_null() { unsafe { (*adev).handle } } else { core::ptr::null_mut::<()>() }
}

// DWORD 3
// DWORD 5
#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_strmatch {
    __slate_bits_0: __slate_bits::__SlateBits873U0,
    substr: [u8; 79],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_system_id {
    callback: Option<unsafe extern "C" fn(*const dmi_system_id) -> i32>,
    ident: *const u8,
    matches: [dmi_strmatch; 4],
    driver_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree {
    ns_id: u64,
    __ns_ref_active: __SlateRecord_types_h_190,
    ns_unified_node: ns_tree_node,
    ns_tree_node: ns_tree_node,
    ns_owner_node: ns_tree_node,
    ns_owner_root: ns_tree_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
    __slate_anon_0: __SlateRecord_ns_common_types_h_112,
    ns_type: u32,
    stashed: *mut dentry,
    ops: *const proc_ns_operations,
    inum: u32,
    __slate_anon_5: ns_tree,
    ns_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
    uid_map: uid_gid_map,
    gid_map: uid_gid_map,
    projid_map: uid_gid_map,
    parent: *mut user_namespace,
    level: i32,
    owner: __SlateRecord_uidgid_types_h_9,
    group: __SlateRecord_uidgid_types_h_13,
    ns: ns_common,
    flags: u64,
    parent_could_setfcap: bool,
    keyring_name_list: list_head,
    user_keyring_register: *mut key,
    keyring_sem: rw_semaphore,
    work: work_struct,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    ucounts: *mut ucounts,
    ucount_max: [i64; 11],
    rlimit_max: [i64; 4],
    binfmt_misc: *mut binfmt_misc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
    node: hlist_nulls_node,
    ns: *mut user_namespace,
    uid: __SlateRecord_uidgid_types_h_9,
    rcu: callback_head,
    count: __SlateRecord_types_h_202,
    ucount: [__SlateRecord_types_h_197; 11],
    rlimit: [__SlateRecord_types_h_197; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits901U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits600U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


fn pm_suspend_via_firmware() -> bool {
((unsafe { pm_suspend_global_flags }) as u64) & BIT!(0i32) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ntb_info_regs {
    partition_count: u8,
    partition_id: u8,
    reserved1: u16,
    ep_map: u64,
    requester_id: u16,
    reserved2: u16,
    reserved3: [u32; 4],
    ntp_info: [nt_partition_info; 48],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ntb_ctrl_regs {
    partition_status: u32,
    partition_op: u32,
    partition_ctrl: u32,
    bar_setup: u32,
    bar_error: u32,
    lut_table_entries: u16,
    lut_table_offset: u16,
    lut_error: u32,
    req_id_table_size: u16,
    req_id_table_offset: u16,
    req_id_error: u32,
    reserved1: [u32; 7],
    bar_entry: [__SlateRecord_switchtec_h_357; 6],
    bar_ext_entry: [__SlateRecord_switchtec_h_362; 6],
    reserved2: [u32; 192],
    req_id_table: [u32; 512],
    reserved3: [u32; 256],
    lut_entry: [u64; 512],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// When a tracepoint is used, it's name is added to the __tracepoint_check
// section. This section is only used at build time to make sure all
// defined tracepoints are used. It is discarded after the build.
// Make sure the alignment of the structure in the __tracepoints section will
// not add unwanted padding between the beginning of the section and the
// structure. Force alignment to the same alignment as the section start.
//
// When lockdep is enabled, we make sure to always test if RCU is
// "watching" regardless if the tracepoint is enabled or not. Tracepoints
// require RCU to be active, and it should always warn at the tracepoint
// site if it is not watching, as it will need to be active when the
// tracepoint is enabled.
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_cap_saved_data {
    cap_nr: u16,
    cap_extended: bool,
    size: u32,
    data: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_cap_saved_state {
    next: hlist_node,
    cap: pci_cap_saved_data,
}


fn pcie_downstream_port(dev: *const pci_dev) -> bool {
let r#type: i32 = pci_pcie_type(dev);
r#type == 4 || r#type == 6 || r#type == 8
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
    pos: i32,
    nres: i32,
    cap: u32,
    ctrl: u16,
    total_VFs: u16,
    initial_VFs: u16,
    num_VFs: u16,
    offset: u16,
    stride: u16,
    vf_device: u16,
    pgsz: u32,
    link: u8,
    max_VF_buses: u8,
    driver_max_VFs: u16,
    dev: *mut pci_dev,
    self_: *mut pci_dev,
    class: u32,
    hdr_type: u8,
    subsystem_vendor: u16,
    subsystem_device: u16,
    barsz: [u64; 6],
    vf_rebar_cap: u16,
    drivers_autoprobe: bool,
}


// pci_dev priv_flags
#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
    nextbusn: u8,
    lastbusn: u8,
    bitmap: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev_reset_methods {
    vendor: u16,
    device: u16,
    reset: Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>,
}


#[unsafe(link_section = ".text.slate_distinct.quirks.of_pci_make_dev_node")]
extern "C" fn of_pci_make_dev_node(pdev: *mut pci_dev) {
}

// isa_dma_bridge_buggy
fn pcie_lbms_seen(dev: *mut pci_dev, lnksta: u16) -> bool {
if test_bit__4!(6i32, unsafe { core::ptr::addr_of_mut!((*dev).priv_flags) }) != 0 {
return true;
}
(lnksta as i32) & 16384 != 0
}

/// Retrain the link of a downstream PCIe port by hand if necessary.
///
/// This is needed at least where a downstream port of the ASMedia ASM2824
/// Gen 3 switch is wired to the upstream port of the Pericom PI7C9X2G304
/// Gen 2 switch, and observed with the Delock Riser Card PCI Express x1 >
/// 2 x PCIe x1 device, P/N 41433, plugged into the SiFive HiFive Unmatched
/// board.
///
/// In such a configuration the switches are supposed to negotiate the link
/// speed of preferably 5.0GT/s, falling back to 2.5GT/s.  However the link
/// continues switching between the two speeds indefinitely and the data
/// link layer never reaches the active state, with link training reported
/// repeatedly active ~84% of the time.  Forcing the target link speed to
/// 2.5GT/s with the upstream ASM2824 device makes the two switches talk to
/// each other correctly however.  And more interestingly retraining with a
/// higher target link speed afterwards lets the two successfully negotiate
/// 5.0GT/s.
///
/// With the ASM2824 we can rely on the otherwise optional Data Link Layer
/// Link Active status bit and in the failed link training scenario it will
/// be off along with the Link Bandwidth Management Status indicating that
/// hardware has changed the link speed or width in an attempt to correct
/// unreliable link operation.  For a port that has been left unconnected
/// both bits will be clear.  So use this information to detect the problem
/// rather than polling the Link Training bit and watching out for flips or
/// at least the active status.
///
/// Since the exact nature of the problem isn't known and in principle this
/// could trigger where an ASM2824 device is downstream rather upstream,
/// apply this erratum workaround to any downstream ports as long as they
/// support Link Active reporting and have the Link Control 2 register.
/// Restrict the speed to 2.5GT/s then with the Target Link Speed field,
/// request a retrain and check the result.
///
/// If this turns out successful, or where a 2.5GT/s speed restriction has
/// been previously arranged by the firmware and the port reports its link
/// already being up, lift the restriction, in a hope it is safe to do so,
/// letting the devices negotiate a higher speed.
///
/// Otherwise revert the speed to the original setting and request a retrain
/// again to remove any residual state, ignoring the result as it's supposed
/// to fail anyway.
///
/// Return 0 if the link has been successfully retrained.  Return an error
/// if retraining was not needed or we attempted a retrain and it failed.
#[unsafe(no_mangle)]
extern "C" fn pcie_failed_link_retrain(dev: *mut pci_dev) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut speed_cap: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(speed_cap) as *mut u32;
let mut oldlnkctl2: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u16 = core::ptr::addr_of_mut!(oldlnkctl2) as *mut u16;
let mut lnkctl2: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u16 = core::ptr::addr_of_mut!(lnkctl2) as *mut u16;
let mut lnksta: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u16 = core::ptr::addr_of_mut!(lnksta) as *mut u16;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 25i32.wrapping_neg();
if !pci_is_pcie(dev) {
__v2 = true;
} else {
__v2 = !pcie_downstream_port(dev as *const pci_dev);
}
if __v2 {
__v1 = true;
} else {
__v1 = !pcie_cap_has_lnkctl2(dev as *const pci_dev);
}
if __v1 || ((*dev).__slate_bits_3.__get_link_active_reporting() as i32) == 0 {
return ret;
} else {
*__slate_slot_4 = pcie_get_speed_cap(dev);
if (*__slate_slot_4 as u32) <= 20 {
return ret;
} else {
pcie_capability_read_word(dev, 18i32, core::ptr::addr_of_mut!(*__slate_slot_7));
pcie_capability_read_word(dev, 48i32, core::ptr::addr_of_mut!(*__slate_slot_5));
if (*__slate_slot_7 as i32) & 8192 == 0 {
__v0 = pcie_lbms_seen(dev, *__slate_slot_7);
} else {
__v0 = false;
}
'__join_1: {
if __v0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"broken device, retraining non-functional downstream link at 2.5GT/s\n".as_ptr() as *const u8);
ret = pcie_set_target_speed(dev, 20u32, false);
if ret != 0 {
break '__join_1;
}
}
pcie_capability_read_word(dev, 48i32, core::ptr::addr_of_mut!(*__slate_slot_6));
if (*__slate_slot_6 as i32) & 15 == 1 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"removing 2.5GT/s downstream link speed restriction\n".as_ptr() as *const u8);
ret = pcie_set_target_speed(dev, *__slate_slot_4, false);
if ret != 0 {
break '__join_1;
}
}
return ret;
}
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"retraining failed\n".as_ptr() as *const u8);
pcie_set_target_speed(dev, PCIE_LNKCTL2_TLS2SPEED!(*__slate_slot_5) as u32, true);
return ret;
}
}
}
}

fn fixup_debug_start(dev: *mut pci_dev, r#fn: Option<unsafe extern "C" fn(*mut pci_dev)>) -> i64 {
if unsafe { initcall_debug } {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"calling  %pS @ %i\n".as_ptr() as *const u8, unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *const u8>(r#fn) }, task_pid_nr(get_current())) };
}
unsafe { ktime_get() }
}

fn fixup_debug_report(dev: *mut pci_dev, calltime: i64, r#fn: Option<unsafe extern "C" fn(*mut pci_dev)>) {
let mut delta: i64 = 0;
let rettime: i64 = unsafe { ktime_get() };
let mut duration: u64 = 0;
delta = ktime_sub!(rettime, calltime);
duration = (ktime_to_ns(delta) as u64) >> 10i32;
if (unsafe { initcall_debug }) || duration > 10000 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%pS took %lld usecs\n".as_ptr() as *const u8, unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *const u8>(r#fn) }, duration) };
}
}

fn pci_do_fixups(dev: *mut pci_dev, mut f: *mut pci_fixup, end: *mut pci_fixup) {
let mut calltime: i64 = 0;
while f < end {
if ((unsafe { (*f).class }) == (unsafe { (*dev).class }) >> unsafe { (*f).class_shift } || (unsafe { (*f).class }) == 4294967295u32) && (((unsafe { (*f).vendor }) as i32) == ((unsafe { (*dev).vendor }) as i32) || ((unsafe { (*f).vendor }) as i32) == 65535i32) && (((unsafe { (*f).device }) as i32) == ((unsafe { (*dev).device }) as i32) || ((unsafe { (*f).device }) as i32) == 65535i32) {
let hook: Option<unsafe extern "C" fn(*mut pci_dev)> = unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut pci_dev)>>(offset_to_ptr((unsafe { core::ptr::addr_of_mut!((*f).hook_offset) }) as *const i32)) };
calltime = fixup_debug_start(dev, hook);
unsafe { hook.unwrap()(dev) };
fixup_debug_report(dev, calltime, hook);
}
let __v1: *mut pci_fixup = f;
f = __v1.wrapping_offset(1 as isize);
}
}

static mut pci_apply_fixup_final_quirks: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_fixup_device")]
extern "C" fn pci_fixup_device(pass: u32, dev: *mut pci_dev) {
let mut start: *mut pci_fixup = unsafe { core::mem::zeroed() };
let mut end: *mut pci_fixup = unsafe { core::mem::zeroed() };
match pass as u32 {
0 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_early) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_early) as *mut pci_fixup };
}
1 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_header) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_header) as *mut pci_fixup };
}
2 => {
if !unsafe { pci_apply_fixup_final_quirks } {
return;
}
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_final) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_final) as *mut pci_fixup };
}
3 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_enable) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_enable) as *mut pci_fixup };
}
4 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_resume) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_resume) as *mut pci_fixup };
}
6 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_resume_early) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_resume_early) as *mut pci_fixup };
}
5 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_suspend) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_suspend) as *mut pci_fixup };
}
7 => {
start = unsafe { core::ptr::addr_of_mut!(__start_pci_fixups_suspend_late) as *mut pci_fixup };
end = unsafe { core::ptr::addr_of_mut!(__end_pci_fixups_suspend_late) as *mut pci_fixup };
}
_ => {
return; // stupid compiler warning, you would think with an enum...
}
}
pci_do_fixups(dev, start, end);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_device_680: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *mut pci_dev)>, *mut ()>(Some(pci_fixup_device)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn pci_apply_final_quirks() -> i32 {
let mut dev: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let mut cls: u8 = 0;
let mut tmp: u8 = 0;
if (unsafe { pci_cache_line_size }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016PCI: CLS %u bytes\n".as_ptr() as *const u8, (pci_cache_line_size as i32) << 2i32) };
}
unsafe {
pci_apply_fixup_final_quirks = true;
}
loop {
let __v4: *mut pci_dev = unsafe { pci_get_device(4294967295u32, 4294967295u32, dev) };
dev = __v4;
if __v4.is_null() {
break;
}
pci_fixup_device(2u32, dev);
// If arch hasn't set it explicitly yet, use the CLS
// value shared by all PCI devices.  If there's a
// mismatch, fall back to the default value.
if (unsafe { pci_cache_line_size }) == 0 {
unsafe { pci_read_config_byte(dev as *const pci_dev, 12i32, core::ptr::addr_of_mut!(tmp)) };
if cls == 0 {
cls = tmp;
}
if !(tmp == 0 || (cls as i32) == (tmp as i32)) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"CLS mismatch (%u != %u), using %u bytes\n".as_ptr() as *const u8, (cls as i32) << 2i32, (tmp as i32) << 2i32, (pci_dfl_cache_line_size as i32) << 2i32) };
unsafe {
pci_cache_line_size = pci_dfl_cache_line_size;
}
}
}
}
if (unsafe { pci_cache_line_size }) == 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x016PCI: CLS %u bytes, default %u\n".as_ptr() as *const u8, (cls as i32) << 2i32, (pci_dfl_cache_line_size as i32) << 2i32) };
unsafe {
pci_cache_line_size = (if cls != 0 { cls as i32 } else { pci_dfl_cache_line_size as i32 }) as u8;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_apply_final_quirks_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(pci_apply_final_quirks)) };

/// Decoding should be disabled for a PCI device during BAR sizing to avoid
/// conflict. But doing so may cause problems on host bridge and perhaps other
/// key system devices. For devices that need to have mmio decoding always-on,
/// we need to set the dev->mmio_always_on bit.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_mmio_always_on")]
extern "C" fn quirk_mmio_always_on(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_1.__set_mmio_always_on(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_mmio_always_on_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_mmio_always_on)) };

/// The Mellanox Tavor device gives false positive parity errors.  Disable
/// parity error reporting.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_disable_parity_684: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev)>>(pci_disable_parity as *const ()) }) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_disable_parity_685: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev)>>(pci_disable_parity as *const ()) }) };

/// Deal with broken BIOSes that neglect to enable passive release,
/// which can cause problems in combination with the 82441FX/PPro MTRRs
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_passive_release")]
extern "C" fn quirk_passive_release(dev: *mut pci_dev) {
let mut d: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let mut dlc: u8 = 0;
// We have to make sure a particular bit is set in the PIIX3
// ISA bridge, so we have to go out and find it.
loop {
let __v3: *mut pci_dev = unsafe { pci_get_device(32902u32, 28672u32, d) };
d = __v3;
if __v3.is_null() {
break;
}
unsafe { pci_read_config_byte(d as *const pci_dev, 130i32, core::ptr::addr_of_mut!(dlc)) };
if (dlc as i32) & 2i32 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*d).dev) as *const device, c"PIIX3: Enabling Passive Release\n".as_ptr() as *const u8) };
dlc = ((dlc as i32) | 2i32) as u8;
unsafe { pci_write_config_byte(d as *const pci_dev, 130i32, dlc) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_passive_release_686: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_passive_release)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_passive_release_687: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_passive_release)) };

/// Intel NM10 "Tiger Point" LPC PM1a_STS.BM_STS must be clear
/// for some HT machines to use C4 w/o hanging.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_tigerpoint_bm_sts")]
extern "C" fn quirk_tigerpoint_bm_sts(dev: *mut pci_dev) {
let mut pmbase: u32 = 0;
let mut pm1a: u16 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(pmbase)) };
pmbase &= 65408;
pm1a = __inw(pmbase as u16);
if (pm1a as i32) & 16 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"[Firmware Bug]: Tiger Point LPC.BM_STS cleared\n".as_ptr() as *const u8) };
__outw(16u16, pmbase as u16);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tigerpoint_bm_sts_688: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tigerpoint_bm_sts)) };

/// Chipsets where PCI->PCI transfers vanish or hang
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nopcipci")]
extern "C" fn quirk_nopcipci(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 1 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 1;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nopcipci_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nopcipci)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nopcipci_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nopcipci)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nopciamd")]
extern "C" fn quirk_nopciamd(dev: *mut pci_dev) {
let mut rev: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 8i32, core::ptr::addr_of_mut!(rev)) };
if (rev as i32) == 19 {
// Erratum 24
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Chipset erratum: Disabling direct PCI/AGP transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 64;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nopciamd_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nopciamd)) };

/// Triton requires workarounds to be used by the drivers
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_triton")]
extern "C" fn quirk_triton(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 2 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Limiting direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 2;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_triton_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_triton)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_triton_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_triton)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_triton_694: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_triton)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_triton_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_triton)) };

/// VIA Apollo KT133 needs PCI latency patch
/// Made according to a Windows driver-based patch by George E. Breese;
/// see PCI Latency Adjust on http://www.viahardware.com/download/viatweak.shtm
/// Also see http://www.au-ja.org/review-kt133a-1-en.phtml for the info on
/// which Mr Breese based his work.
///
/// Updated based on further information from the site and also on
/// information provided by VIA
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vialatency")]
extern "C" fn quirk_vialatency(dev: *mut pci_dev) {
let mut busarb: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u8 = core::ptr::addr_of_mut!(busarb) as *mut u8;
let mut p: core::mem::MaybeUninit<*mut pci_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut pci_dev = core::ptr::addr_of_mut!(p) as *mut *mut pci_dev;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// Ok, we have a potential problem chipset here. Now see if we have
// a buggy southbridge.
*__slate_slot_5 = pci_get_device(4358u32, 1670u32, core::ptr::null_mut::<pci_dev>());
if !(*__slate_slot_5).is_null() {
// 0x40 - 0x4f == 686B, 0x10 - 0x2f == 686A;
// thanks Dan Hollis.
// Check for buggy part revisions
if ((*(*__slate_slot_5)).revision as i32) < 64 || ((*(*__slate_slot_5)).revision as i32) > 66 {
break '__join_0;
}
} else {
*__slate_slot_5 = pci_get_device(4358u32, 33329u32, core::ptr::null_mut::<pci_dev>());
if (*__slate_slot_5).is_null() {
break '__join_0;
} else {
// No problem parts
// Check for buggy part revisions
if ((*(*__slate_slot_5)).revision as i32) < 16 || ((*(*__slate_slot_5)).revision as i32) > 18 {
break '__join_0;
}
}
}
// Ok we have the problem. Now set the PCI master grant to occur
// every master grant. The apparent bug is that under high PCI load
// (quite common in Linux of course) you can get data loss when the
// CPU is held off the bus for 3 bus master requests.  This happens
// to include the IDE controllers....
//
// VIA only apply this fix when an SB Live! is present but under
// both Linux and Windows this isn't enough, and we have seen
// corruption without SB Live! but with things like 3 UDMA IDE
// controllers. So we ignore that bit of the VIA recommendation..
pci_read_config_byte(dev as *const pci_dev, 118i32, core::ptr::addr_of_mut!(*__slate_slot_4));
// Set bit 4 and bit 5 of byte 76 to 0x01
// "Master priority rotation on every PCI master grant"
let __v2: u8 = ((*__slate_slot_4 as i32) & -33i32) as u8;
*__slate_slot_4 = __v2;
*__slate_slot_4 = ((*__slate_slot_4 as i32) | 16i32) as u8;
pci_write_config_byte(dev as *const pci_dev, 118i32, *__slate_slot_4);
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Applying VIA southbridge workaround\n".as_ptr() as *const u8);
}
pci_dev_put(*__slate_slot_5);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

/// Must restore this on a resume from RAM
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_700: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vialatency_701: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vialatency)) };

/// VIA Apollo VP3 needs ETBF on BT848/878
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_viaetbf")]
extern "C" fn quirk_viaetbf(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 8 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Limiting direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 8;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_viaetbf_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_viaetbf)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vsfx")]
extern "C" fn quirk_vsfx(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 16 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Limiting direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 16;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vsfx_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vsfx)) };

/// ALi Magik requires workarounds to be used by the drivers that DMA to AGP
/// space. Latency must be set to 0xA and Triton workaround applied too.
/// [Info kindly provided by ALi]
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_alimagik")]
extern "C" fn quirk_alimagik(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 32 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Limiting direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | (32 | 2);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_alimagik_704: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_alimagik)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_alimagik_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_alimagik)) };

/// Natoma has some interesting boundary conditions with Zoran stuff at least
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_natoma")]
extern "C" fn quirk_natoma(dev: *mut pci_dev) {
if (unsafe { pci_pci_problems }) & 4 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Limiting direct PCI/PCI transfers\n".as_ptr() as *const u8) };
unsafe {
pci_pci_problems = pci_pci_problems | 4;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_710: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_natoma_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_natoma)) };

/// This chip can cause PCI parity errors if config register 0xA0 is read
/// while DMAs are occurring.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_citrine")]
extern "C" fn quirk_citrine(dev: *mut pci_dev) {
unsafe {
(*dev).cfg_size = 160;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_citrine_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_citrine)) };

/// This chip can cause bus lockups if config addresses above 0x600
/// are read or written.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nfp6000")]
extern "C" fn quirk_nfp6000(dev: *mut pci_dev) {
unsafe {
(*dev).cfg_size = 1536;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nfp6000_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nfp6000)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nfp6000_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nfp6000)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nfp6000_715: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nfp6000)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nfp6000_716: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nfp6000)) };

///  On IBM Crocodile ipr SAS adapters, expand BAR to system page size
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_extend_bar_to_page")]
extern "C" fn quirk_extend_bar_to_page(dev: *mut pci_dev) {
for i in 0i32..6i32 {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(i as isize);
let r_name: *const u8 = unsafe { pci_resource_name(dev, i as u32) };
let __v0: bool = (unsafe { (*r).flags }) & 512 != 0 && resource_size(r as *const resource) < _AC!() << 12i32;
if __v0 {
resource_set_range(r, 0u64, _AC!() << 12i32);
unsafe {
(*r).flags = (*r).flags | 536870912;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: expanded to page size\n".as_ptr() as *const u8, r_name, r) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_extend_bar_to_page_717: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_extend_bar_to_page)) };

/// S3 868 and 968 chips report region size equal to 32M, but they decode 64M.
/// If it's needed, re-allocate the region.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_s3_64M")]
extern "C" fn quirk_s3_64M(dev: *mut pci_dev) {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(0 as isize);
let __v0: bool = !IS_ALIGNED!(unsafe { (*r).start }, 67108864i32) || resource_size(r as *const resource) != 67108864;
if __v0 {
unsafe {
(*r).flags = (*r).flags | 536870912;
}
resource_set_range(r, 0u64, 67108864u64);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_s3_64M_718: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_s3_64M)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_s3_64M_719: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_s3_64M)) };

fn quirk_io(dev: *mut pci_dev, pos: i32, size: u32, name__0: *const u8) {
let mut region: u32 = 0;
let mut bus_region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bus_region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let res: *mut resource = pci_resource_n!(dev, pos);
let res_name: *const u8 = unsafe { pci_resource_name(dev, pos as u32) };
unsafe { pci_read_config_dword(dev as *const pci_dev, 16i32.wrapping_add(pos << 2i32), core::ptr::addr_of_mut!(region)) };
if region == 0 {
return;
}
unsafe {
(*res).name = pci_name(dev as *const pci_dev);
}
unsafe {
(*res).flags = (region as u64) & !(!3);
}
unsafe {
(*res).flags = (*res).flags | 262416u64;
}
let __v4: u32 = region & !size.wrapping_sub(1);
region = __v4;
// Convert from PCI bus to resource space
bus_region.start = region as u64;
bus_region.end = region.wrapping_add(size).wrapping_sub(1) as u64;
unsafe { pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(bus_region)) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"[Firmware Bug]: %s %pR: %s quirk\n".as_ptr() as *const u8, res_name, res, name__0) };
}

/// Some CS5536 BIOSes (for example, the Soekris NET5501 board w/ comBIOS
/// ver. 1.33  20070103) don't set the correct ISA PCI region header info.
/// BAR0 should be 8 bytes; instead, it may be set to something like 8k
/// (which conflicts w/ BAR1's memory range).
///
/// CS553x's ISA PCI BARs may also be read-only (ref:
/// https://bugzilla.kernel.org/show_bug.cgi?id=85991 - Comment #4 forward).
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_cs5536_vsa")]
extern "C" fn quirk_cs5536_vsa(dev: *mut pci_dev) {
if pci_resource_len!(dev, 0i32) != 8 {
quirk_io(dev, 0i32, 8u32, (unsafe { name }) as *const u8); // SMB
quirk_io(dev, 1i32, 256u32, (unsafe { name }) as *const u8); // GPIO
quirk_io(dev, 2i32, 64u32, (unsafe { name }) as *const u8); // MFGPT
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s bug detected (incorrect header); workaround applied\n".as_ptr() as *const u8, name) };
}
}

static mut name: *mut u8 = b"CS5536 ISA bridge\0".as_ptr() as *mut u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_cs5536_vsa_720: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_cs5536_vsa)) };

fn quirk_io_region(dev: *mut pci_dev, port: i32, size: u32, nr: i32, name__0: *const u8) {
let mut region: u16 = 0;
let mut bus_region: pci_bus_region = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bus_region) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let res: *mut resource = pci_resource_n!(dev, nr);
unsafe { pci_read_config_word(dev as *const pci_dev, port, core::ptr::addr_of_mut!(region)) };
let __v1: u16 = (((region as i32) as u32) & !size.wrapping_sub(1)) as u16;
region = __v1;
if region == 0 {
return;
}
unsafe {
(*res).name = pci_name(dev as *const pci_dev);
}
unsafe {
(*res).flags = 256;
}
// Convert from PCI bus to resource space
bus_region.start = region as u64;
bus_region.end = ((region as i32) as u32).wrapping_add(size).wrapping_sub(1) as u64;
unsafe { pcibios_bus_to_resource((*dev).bus, res, core::ptr::addr_of_mut!(bus_region)) };
// "res" is typically a bridge window resource that's not being
// used for a bridge window, so it's just a place to stash this
// non-standard resource.  Printing "nr" or pci_resource_name() of
// it doesn't really make sense.
if (unsafe { pci_claim_resource(dev, nr) }) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"quirk: %pR claimed by %s\n".as_ptr() as *const u8, res, name__0) };
}
}

/// ATI Northbridge setups MCE the processor if you even read somewhere
/// between 0x3b0->0x3bb or read 0x3d3
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ati_exploding_mce")]
extern "C" fn quirk_ati_exploding_mce(dev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ATI Northbridge, reserving I/O ports 0x3b0 to 0x3bb\n".as_ptr() as *const u8) };
// Mae rhaid i ni beidio ag edrych ar y lleoliadiau I/O hyn
request_region!(944i32, 12i32, c"RadeonIGP".as_ptr() as *mut u8);
request_region!(979i32, 1i32, c"RadeonIGP".as_ptr() as *mut u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ati_exploding_mce_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ati_exploding_mce)) };

/// In the AMD NL platform, this device ([1022:7912]) has a class code of
/// PCI_CLASS_SERIAL_USB_XHCI (0x0c0330), which means the xhci driver will
/// claim it. The same applies on the VanGogh platform device ([1022:163a]).
///
/// But the dwc3 driver is a more specific driver for this device, and we'd
/// prefer to use it instead of xhci. To prevent xhci from claiming the
/// device, change the class code to 0x0c03fe, which the PCI r3.0 spec
/// defines as "USB device (not host controller)". The dwc3 driver can then
/// claim it based on its Vendor and Device ID.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_dwc_class")]
extern "C" fn quirk_amd_dwc_class(pdev: *mut pci_dev) {
let class: u32 = unsafe { (*pdev).class };
if class != 787454 {
// Use "USB Device (not host controller)" class
unsafe {
(*pdev).class = 787454;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"PCI class overridden (%#08x -> %#08x) so dwc3 driver can claim this instead of xhci\n".as_ptr() as *const u8, class, (*pdev).class) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_dwc_class_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_dwc_class)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_dwc_class_723: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_dwc_class)) };

/// Synopsys USB 3.x host HAPS platform has a class code of
/// PCI_CLASS_SERIAL_USB_XHCI, and xhci driver can claim it.  However, these
/// devices should use dwc3-haps driver.  Change these devices' class code to
/// PCI_CLASS_SERIAL_USB_DEVICE to prevent the xhci-pci driver from claiming
/// them.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_synopsys_haps")]
extern "C" fn quirk_synopsys_haps(pdev: *mut pci_dev) {
let class: u32 = unsafe { (*pdev).class };
match (unsafe { (*pdev).device }) as i32 {
43981..=43983 => {
unsafe {
(*pdev).class = 787454;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"PCI class overridden (%#08x -> %#08x) so dwc3 driver can claim this instead of xhci\n".as_ptr() as *const u8, class, (*pdev).class) };
}
_ => {
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_synopsys_haps_724: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_synopsys_haps)) };

/// Let's make the southbridge information explicit instead of having to
/// worry about people probing the ACPI areas, for example.. (Yes, it
/// happens, and if you read the wrong ACPI register it will put the machine
/// to sleep with no way of waking it up again. Bummer).
///
/// ALI M7101: Two IO regions pointed to by words at
/// 	0xE0 (64 bytes of ACPI registers)
/// 	0xE2 (32 bytes of SMB registers)
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ali7101_acpi")]
extern "C" fn quirk_ali7101_acpi(dev: *mut pci_dev) {
quirk_io_region(dev, 224i32, 64u32, 7i32, c"ali7101 ACPI".as_ptr() as *const u8);
quirk_io_region(dev, 226i32, 32u32, 8i32, c"ali7101 SMB".as_ptr() as *const u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ali7101_acpi_725: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ali7101_acpi)) };

fn piix4_io_quirk(dev: *mut pci_dev, name__0: *const u8, port: u32, enable: u32) {
let mut devres: u32 = 0;
let mut mask: u32 = 0;
let mut size: u32 = 16;
let mut base: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, port as i32, core::ptr::addr_of_mut!(devres)) };
if devres & enable != enable {
return;
}
mask = devres >> 16i32 & 15;
base = devres & 65535;
loop {
let bit: u32 = size >> 1i32;
if bit & mask == bit {
break;
}
size = bit;
}
// For now we only print it out. Eventually we'll want to
// reserve it (at least if it's in the 0x1000+ range), but
// let's get enough confirmation reports first.
base &= size.wrapping_neg();
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s PIO at %04x-%04x\n".as_ptr() as *const u8, name__0, base, base.wrapping_add(size).wrapping_sub(1)) };
}

fn piix4_mem_quirk(dev: *mut pci_dev, name__0: *const u8, port: u32, enable: u32) {
let mut devres: u32 = 0;
let mut mask: u32 = 0;
let mut size: u32 = 8388608u32;
let mut base: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, port as i32, core::ptr::addr_of_mut!(devres)) };
if devres & enable != enable {
return;
}
base = devres & 4294901760;
mask = (devres & 63) << 16i32;
loop {
let bit: u32 = size >> 1i32;
if bit & mask == bit {
break;
}
size = bit;
}
// For now we only print it out. Eventually we'll want to
// reserve it, but let's get enough confirmation reports first.
base &= size.wrapping_neg();
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s MMIO at %04x-%04x\n".as_ptr() as *const u8, name__0, base, base.wrapping_add(size).wrapping_sub(1)) };
}

/// PIIX4 ACPI: Two IO regions pointed to by longwords at
/// 	0x40 (64 bytes of ACPI registers)
/// 	0x90 (16 bytes of SMB registers)
/// and a few strange programmable PIIX4 device resources.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_piix4_acpi")]
extern "C" fn quirk_piix4_acpi(dev: *mut pci_dev) {
let mut res_a: u32 = 0;
quirk_io_region(dev, 64i32, 64u32, 7i32, c"PIIX4 ACPI".as_ptr() as *const u8);
quirk_io_region(dev, 144i32, 16u32, 8i32, c"PIIX4 SMB".as_ptr() as *const u8);
// Device resource A has enables for some of the other ones
unsafe { pci_read_config_dword(dev as *const pci_dev, 92i32, core::ptr::addr_of_mut!(res_a)) };
piix4_io_quirk(dev, c"PIIX4 devres B".as_ptr() as *const u8, 96u32, 6291456u32);
piix4_io_quirk(dev, c"PIIX4 devres C".as_ptr() as *const u8, 100u32, 6291456u32);
// Device resource D is just bitfields for static resources
// Device 12 enabled?
if res_a & 536870912u32 != 0 {
piix4_io_quirk(dev, c"PIIX4 devres E".as_ptr() as *const u8, 104u32, 1048576u32);
piix4_mem_quirk(dev, c"PIIX4 devres F".as_ptr() as *const u8, 108u32, 128u32);
}
// Device 13 enabled?
if res_a & 1073741824u32 != 0 {
piix4_io_quirk(dev, c"PIIX4 devres G".as_ptr() as *const u8, 112u32, 1048576u32);
piix4_mem_quirk(dev, c"PIIX4 devres H".as_ptr() as *const u8, 116u32, 128u32);
}
piix4_io_quirk(dev, c"PIIX4 devres I".as_ptr() as *const u8, 120u32, 1048576u32);
piix4_io_quirk(dev, c"PIIX4 devres J".as_ptr() as *const u8, 124u32, 1048576u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_piix4_acpi_726: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_piix4_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_piix4_acpi_727: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_piix4_acpi)) };

/// ICH4, ICH4-M, ICH5, ICH5-M ACPI: Three IO regions pointed to by longwords at
/// 	0x40 (128 bytes of ACPI, GPIO & TCO registers)
/// 	0x58 (64 bytes of GPIO I/O space)
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ich4_lpc_acpi")]
extern "C" fn quirk_ich4_lpc_acpi(dev: *mut pci_dev) {
let mut enable: u8 = 0;
// The check for PCIBIOS_MIN_IO is to ensure we won't create a conflict
// with low legacy (and fixed) ports. We don't know the decoding
// priority and can't tell whether the legacy device or the one created
// here is really at that address.  This happens on boards with broken
// BIOSes.
unsafe { pci_read_config_byte(dev as *const pci_dev, 68i32, core::ptr::addr_of_mut!(enable)) };
if (enable as i32) & 16 != 0 {
quirk_io_region(dev, 64i32, 128u32, 7i32, c"ICH4 ACPI/GPIO/TCO".as_ptr() as *const u8);
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 92i32, core::ptr::addr_of_mut!(enable)) };
if (enable as i32) & 16 != 0 {
quirk_io_region(dev, 88i32, 64u32, 8i32, c"ICH4 GPIO".as_ptr() as *const u8);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_728: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_729: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_730: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_731: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_732: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_733: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_734: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_735: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_736: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich4_lpc_acpi_737: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich4_lpc_acpi)) };

fn ich6_lpc_acpi_gpio(dev: *mut pci_dev) {
let mut enable: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 68i32, core::ptr::addr_of_mut!(enable)) };
if (enable as i32) & 128 != 0 {
quirk_io_region(dev, 64i32, 128u32, 7i32, c"ICH6 ACPI/GPIO/TCO".as_ptr() as *const u8);
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 76i32, core::ptr::addr_of_mut!(enable)) };
if (enable as i32) & 16 != 0 {
quirk_io_region(dev, 72i32, 64u32, 8i32, c"ICH6 GPIO".as_ptr() as *const u8);
}
}

fn ich6_lpc_generic_decode(dev: *mut pci_dev, reg: u32, name__0: *const u8, dynsize: i32) {
let mut val: u32 = 0;
let mut size: u32 = 0;
let mut base: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, reg as i32, core::ptr::addr_of_mut!(val)) };
// Enabled?
if val & 1 == 0 {
return;
}
base = val & 65532;
if dynsize != 0 {
// This is not correct. It is 16, 32 or 64 bytes depending on
// register D31:F0:ADh bits 5:4.
//
// But this gets us at least _part_ of it.
size = 16;
} else {
size = 128;
}
let __v1: u32 = base & !size.wrapping_sub(1);
base = __v1;
// Just print it out for now. We should reserve it after more
// debugging.
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s PIO at %04x-%04x\n".as_ptr() as *const u8, name__0, base, base.wrapping_add(size).wrapping_sub(1)) };
}

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ich6_lpc")]
extern "C" fn quirk_ich6_lpc(dev: *mut pci_dev) {
// Shared ACPI/GPIO decode with all ICH6+
ich6_lpc_acpi_gpio(dev);
// ICH6-specific generic IO decode
ich6_lpc_generic_decode(dev, 132u32, c"LPC Generic IO decode 1".as_ptr() as *const u8, 0i32);
ich6_lpc_generic_decode(dev, 136u32, c"LPC Generic IO decode 2".as_ptr() as *const u8, 1i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich6_lpc_738: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich6_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich6_lpc_739: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich6_lpc)) };

fn ich7_lpc_generic_decode(dev: *mut pci_dev, reg: u32, name__0: *const u8) {
let mut val: u32 = 0;
let mut mask: u32 = 0;
let mut base: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, reg as i32, core::ptr::addr_of_mut!(val)) };
// Enabled?
if val & 1 == 0 {
return;
}
// IO base in bits 15:2, mask in bits 23:18, both are dword-based
base = val & 65532;
mask = val >> 16i32 & 252;
mask |= 3;
// Just print it out for now. We should reserve it after more
// debugging.
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s PIO at %04x (mask %04x)\n".as_ptr() as *const u8, name__0, base, mask) };
}

/// ICH7-10 has the same common LPC generic IO decode registers
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ich7_lpc")]
extern "C" fn quirk_ich7_lpc(dev: *mut pci_dev) {
// We share the common ACPI/GPIO decode with ICH6
ich6_lpc_acpi_gpio(dev);
// And have 4 ICH7+ generic decodes
ich7_lpc_generic_decode(dev, 132u32, c"ICH7 LPC Generic IO decode 1".as_ptr() as *const u8);
ich7_lpc_generic_decode(dev, 136u32, c"ICH7 LPC Generic IO decode 2".as_ptr() as *const u8);
ich7_lpc_generic_decode(dev, 140u32, c"ICH7 LPC Generic IO decode 3".as_ptr() as *const u8);
ich7_lpc_generic_decode(dev, 144u32, c"ICH7 LPC Generic IO decode 4".as_ptr() as *const u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_740: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_741: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_742: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_743: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_744: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_745: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_746: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_747: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_748: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_749: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_750: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_751: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ich7_lpc_752: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ich7_lpc)) };

/// VIA ACPI: One IO region pointed to by longword at
/// 	0x48 or 0x20 (256 bytes of ACPI registers)
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vt82c586_acpi")]
extern "C" fn quirk_vt82c586_acpi(dev: *mut pci_dev) {
if ((unsafe { (*dev).revision }) as i32) & 16 != 0 {
quirk_io_region(dev, 72i32, 256u32, 7i32, c"vt82c586 ACPI".as_ptr() as *const u8);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vt82c586_acpi_753: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vt82c586_acpi)) };

/// VIA VT82C686 ACPI: Three IO region pointed to by (long)words at
/// 	0x48 (256 bytes of ACPI registers)
/// 	0x70 (128 bytes of hardware monitoring register)
/// 	0x90 (16 bytes of SMB registers)
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vt82c686_acpi")]
extern "C" fn quirk_vt82c686_acpi(dev: *mut pci_dev) {
quirk_vt82c586_acpi(dev);
quirk_io_region(dev, 112i32, 128u32, 8i32, c"vt82c686 HW-mon".as_ptr() as *const u8);
quirk_io_region(dev, 144i32, 16u32, 9i32, c"vt82c686 SMB".as_ptr() as *const u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vt82c686_acpi_754: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vt82c686_acpi)) };

/// VIA VT8235 ISA Bridge: Two IO regions pointed to by words at
/// 	0x88 (128 bytes of power management registers)
/// 	0xd0 (16 bytes of SMB registers)
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vt8235_acpi")]
extern "C" fn quirk_vt8235_acpi(dev: *mut pci_dev) {
quirk_io_region(dev, 136i32, 128u32, 7i32, c"vt8235 PM".as_ptr() as *const u8);
quirk_io_region(dev, 208i32, 16u32, 8i32, c"vt8235 SMB".as_ptr() as *const u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vt8235_acpi_755: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vt8235_acpi)) };

/// TI XIO2000a PCIe-PCI Bridge erroneously reports it supports fast
/// back-to-back: Disable fast back-to-back on the secondary bus segment
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_xio2000a")]
extern "C" fn quirk_xio2000a(dev: *mut pci_dev) {
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut command: u16 = 0;
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"TI XIO2000a quirk detected; secondary bus fast back-to-back transfers disabled\n".as_ptr() as *const u8) };
pdev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*dev).subordinate).devices) });
while list_entry_is_head!(pdev, unsafe { core::ptr::addr_of_mut!((*(*dev).subordinate).devices) }, bus_list) == 0 {
unsafe { pci_read_config_word(pdev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(command)) };
if (command as i32) & 512 != 0 {
unsafe { pci_write_config_word(pdev as *const pci_dev, 4i32, ((command as i32) & !512) as u16) };
}
pdev = list_next_entry!(pdev, bus_list);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_xio2000a_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_xio2000a)) };

/// VIA 686A/B: If an IO-APIC is active, we need to route all on-chip
/// devices to the external APIC.
///
/// TODO: When we have device-specific interrupt routers, this code will go
/// away from quirks.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_ioapic")]
extern "C" fn quirk_via_ioapic(dev: *mut pci_dev) {
let tmp: u8 = if (unsafe { nr_ioapics }) < 1 {
0
} else {
31
};
// nothing routed to external APIC
// all known bits (4-0) routed to external APIC
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s VIA external APIC routing\n".as_ptr() as *const u8, if tmp != 0 { c"Enabling".as_ptr() as *mut u8 } else { c"Disabling".as_ptr() as *mut u8 }) };
// Offset 0x58: External APIC IRQ output control
unsafe { pci_write_config_byte(dev as *const pci_dev, 88i32, tmp) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_ioapic_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_ioapic)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_ioapic_758: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_ioapic)) };

/// VIA 8237: Some BIOSes don't set the 'Bypass APIC De-Assert Message' Bit.
/// This leads to doubled level interrupt rates.
/// Set this bit to get rid of cycle wastage.
/// Otherwise uncritical.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_vt8237_bypass_apic_deassert")]
extern "C" fn quirk_via_vt8237_bypass_apic_deassert(dev: *mut pci_dev) {
let mut misc_control2: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 91i32, core::ptr::addr_of_mut!(misc_control2)) };
if (misc_control2 as i32) & 8 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Bypassing VIA 8237 APIC De-Assert Message\n".as_ptr() as *const u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 91i32, ((misc_control2 as i32) | 8) as u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_vt8237_bypass_apic_deassert_759: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_vt8237_bypass_apic_deassert)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_vt8237_bypass_apic_deassert_760: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_vt8237_bypass_apic_deassert)) };

/// The AMD IO-APIC can hang the box when an APIC IRQ is masked.
/// We check all revs >= B0 (yet not in the pre production!) as the bug
/// is currently marked NoFix
///
/// We have multiple reports of hangs with this chipset that went away with
/// noapic specified. For the moment we assume it's the erratum. We may be wrong
/// of course. However the advice is demonstrably good even if so.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_ioapic")]
extern "C" fn quirk_amd_ioapic(dev: *mut pci_dev) {
if ((unsafe { (*dev).revision }) as i32) >= 2 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"I/O APIC: AMD Erratum #22 may be present. In the event of instability try\n".as_ptr() as *const u8) };
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"        : booting with the \"noapic\" option\n".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ioapic_761: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ioapic)) };

/// Some settings of MMRBC can lead to data corruption so block changes.
/// See AMD 8131 HyperTransport PCI-X Tunnel Revision Guide
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_8131_mmrbc")]
extern "C" fn quirk_amd_8131_mmrbc(dev: *mut pci_dev) {
if !unsafe { (*dev).subordinate }.is_null() && ((unsafe { (*dev).revision }) as i32) <= 18 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"AMD8131 rev %x detected; disabling PCI-X MMRBC\n".as_ptr() as *const u8, (*dev).revision as i32) };
let __v1: *mut pci_bus = unsafe { (*dev).subordinate };
unsafe {
(*__v1).bus_flags = (((*__v1).bus_flags as i32) | 2) as u16;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_8131_mmrbc_762: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_8131_mmrbc)) };

/// FIXME: it is questionable that quirk_via_acpi() is needed.  It shows up
/// as an ISA bridge, and does not support the PCI_INTERRUPT_LINE register
/// at all.  Therefore it seems like setting the pci_dev's IRQ to the value
/// of the ACPI SCI interrupt is only done for convenience.
/// 	-jgarzik
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_acpi")]
extern "C" fn quirk_via_acpi(d: *mut pci_dev) {
let mut irq: u8 = 0;
// VIA ACPI device: SCI IRQ line in PCI config byte 0x42
unsafe { pci_read_config_byte(d as *const pci_dev, 66i32, core::ptr::addr_of_mut!(irq)) };
irq = ((irq as i32) & 15) as u8;
if irq != 0 && (irq as i32) != 2 {
unsafe {
(*d).irq = irq as u32;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_acpi_763: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_acpi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_acpi_764: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_acpi)) };

/// VIA bridges which have VLink
static mut via_vlink_dev_lo: i32 = 1i32.wrapping_neg();

static mut via_vlink_dev_hi: i32 = 18;

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_bridge")]
extern "C" fn quirk_via_bridge(dev: *mut pci_dev) {
// See what bridge we have and find the device ranges
match (unsafe { (*dev).device }) as i32 {
1670 => {
unsafe {
via_vlink_dev_lo = PCI_SLOT!(unsafe { (*dev).devfn }) as i32;
}
// The VT82C686 is special; it attaches to PCI and can have
// any device number. All its subdevices are functions of
// that single device.
unsafe {
via_vlink_dev_hi = PCI_SLOT!(unsafe { (*dev).devfn }) as i32;
}
}
12839 | 13111 => {
unsafe {
via_vlink_dev_lo = 15;
}
}
12663 => {
unsafe {
via_vlink_dev_lo = 16;
}
}
12404 | 12553 | 12615 | 33329 => {
unsafe {
via_vlink_dev_lo = 17;
}
}
_ => {
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_765: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_766: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_768: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_769: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_770: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_771: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_bridge_772: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_bridge)) };

/// quirk_via_vlink		-	VIA VLink IRQ number update
/// @dev: PCI device
///
/// If the device we are dealing with is on a PIC IRQ we need to ensure that
/// the IRQ line register which usually is not relevant for PCI cards, is
/// actually written so that interrupts get sent to the right place.
///
/// We only do this on systems where a VIA south bridge was detected, and
/// only for VIA devices on the motherboard (see quirk_via_bridge above).
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_vlink")]
extern "C" fn quirk_via_vlink(dev: *mut pci_dev) {
let mut irq: u8 = 0;
let mut new_irq: u8 = 0;
// Check if we have VLink at all
if (unsafe { via_vlink_dev_lo }) == 1i32.wrapping_neg() {
return;
}
new_irq = (unsafe { (*dev).irq }) as u8;
// Don't quirk interrupts outside the legacy IRQ range
if new_irq == 0 || (new_irq as i32) > 15 {
return;
}
// Internal device ?
if ((unsafe { (*(*dev).bus).number }) as i32) != 0 || PCI_SLOT!(unsafe { (*dev).devfn }) > ((unsafe { via_vlink_dev_hi }) as u32) || PCI_SLOT!(unsafe { (*dev).devfn }) < ((unsafe { via_vlink_dev_lo }) as u32) {
return;
}
// This is an internal VLink device on a PIC interrupt. The BIOS
// ought to have set this but may not have, so we redo it.
unsafe { pci_read_config_byte(dev as *const pci_dev, 60i32, core::ptr::addr_of_mut!(irq)) };
if (new_irq as i32) != (irq as i32) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"VIA VLink IRQ fixup, from %d to %d\n".as_ptr() as *const u8, irq as i32, new_irq as i32) };
udelay(15u64); // unknown if delay really needed
unsafe { pci_write_config_byte(dev as *const pci_dev, 60i32, new_irq) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_vlink_773: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_vlink)) };

/// VIA VT82C598 has its device ID settable and many BIOSes set it to the ID
/// of VT82C597 for backward compatibility.  We need to switch it off to be
/// able to recognize the real type of the chip.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_vt82c598_id")]
extern "C" fn quirk_vt82c598_id(dev: *mut pci_dev) {
unsafe { pci_write_config_byte(dev as *const pci_dev, 252i32, 0u8) };
unsafe { pci_read_config_word(dev as *const pci_dev, 2i32, core::ptr::addr_of_mut!((*dev).device)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_vt82c598_id_774: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_vt82c598_id)) };

/// CardBus controllers have a legacy base address that enables them to
/// respond as i82365 PCMCIA controllers.  We don't want them to do this.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_cardbus_legacy")]
extern "C" fn quirk_cardbus_legacy(dev: *mut pci_dev) {
unsafe { pci_write_config_dword(dev as *const pci_dev, 68i32, 0u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_cardbus_legacy_775: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_cardbus_legacy)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_cardbus_legacy_776: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_cardbus_legacy)) };

/// Following the PCI ordering rules is optional on the AMD762. I'm not sure
/// what the designers were smoking but let's not inhale...
///
/// To be fair to AMD, it follows the spec by default, it's BIOS people who
/// turn it off!
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_ordering")]
extern "C" fn quirk_amd_ordering(dev: *mut pci_dev) {
let mut pcic: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 76i32, core::ptr::addr_of_mut!(pcic)) };
if pcic & 6 != 6 {
pcic |= 6;
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BIOS failed to enable PCI standards compliance; fixing this error\n".as_ptr() as *const u8) };
unsafe { pci_write_config_dword(dev as *const pci_dev, 76i32, pcic) };
unsafe { pci_read_config_dword(dev as *const pci_dev, 132i32, core::ptr::addr_of_mut!(pcic)) };
pcic |= 8388608u32; // Required in this mode
unsafe { pci_write_config_dword(dev as *const pci_dev, 132i32, pcic) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ordering_777: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ordering)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ordering_778: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ordering)) };

/// DreamWorks-provided workaround for Dunord I-3000 problem
///
/// This card decodes and responds to addresses not apparently assigned to
/// it.  We force a larger allocation to ensure that nothing gets put too
/// close to it.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_dunord")]
extern "C" fn quirk_dunord(dev: *mut pci_dev) {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(1 as isize);
unsafe {
(*r).flags = (*r).flags | 536870912;
}
resource_set_range(r, 0u64, 16777216u64);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dunord_779: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dunord)) };

/// i82380FB mobile docking controller: its PCI-to-PCI bridge is subtractive
/// decoding (transparent), and does indicate this in the ProgIf.
/// Unfortunately, the ProgIf value is wrong - 0x80 instead of 0x01.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_transparent_bridge")]
extern "C" fn quirk_transparent_bridge(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_transparent(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_transparent_bridge_780: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_transparent_bridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_transparent_bridge_781: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_transparent_bridge)) };

/// Enabling Link Bandwidth Management Interrupts (BW notifications) can cause
/// boot hangs on P45.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_p45_bw_notifications")]
extern "C" fn quirk_p45_bw_notifications(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_1.__set_no_bw_notif(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_p45_bw_notifications_782: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_p45_bw_notifications)) };

/// Common misconfiguration of the MediaGX/Geode PCI master that will reduce
/// PCI bandwidth from 70MB/s to 25MB/s.  See the GXM/GXLV/GX1 datasheets
/// found at http://www.national.com/analog for info on what these bits do.
/// <christer@weinigel.se>
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_mediagx_master")]
extern "C" fn quirk_mediagx_master(dev: *mut pci_dev) {
let mut reg: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 65i32, core::ptr::addr_of_mut!(reg)) };
if (reg as i32) & 2 != 0 {
let __v1: u8 = ((reg as i32) & !2) as u8;
reg = __v1;
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Fixup for MediaGX/Geode Slave Disconnect Boundary (0x41=0x%02x)\n".as_ptr() as *const u8, reg as i32) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 65i32, reg) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_mediagx_master_783: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_mediagx_master)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_mediagx_master_784: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_mediagx_master)) };

/// Ensure C0 rev restreaming is off. This is normally done by the BIOS but
/// in the odd case it is not the results are corruption hence the presence
/// of a Linux check.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_pxb")]
extern "C" fn quirk_disable_pxb(pdev: *mut pci_dev) {
let mut config: u16 = 0;
if ((unsafe { (*pdev).revision }) as i32) != 4 {
return;
}
// Only C0 requires this
unsafe { pci_read_config_word(pdev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(config)) };
if (config as i32) & 64i32 != 0 {
let __v1: u16 = ((config as i32) & -65i32) as u16;
config = __v1;
unsafe { pci_write_config_word(pdev as *const pci_dev, 64i32, config) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"C0 revision 450NX. Disabling PCI restreaming\n".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_pxb_785: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_pxb)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_pxb_786: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_pxb)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_ide_mode")]
extern "C" fn quirk_amd_ide_mode(pdev: *mut pci_dev) {
// set SBX00/Hudson-2 SATA in IDE mode to AHCI mode
let mut tmp: u8 = 0;
unsafe { pci_read_config_byte(pdev as *const pci_dev, 10i32, core::ptr::addr_of_mut!(tmp)) };
if (tmp as i32) == 1 {
unsafe { pci_read_config_byte(pdev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(tmp)) };
unsafe { pci_write_config_byte(pdev as *const pci_dev, 64i32, ((tmp as i32) | 1) as u8) };
unsafe { pci_write_config_byte(pdev as *const pci_dev, 9i32, 1u8) };
unsafe { pci_write_config_byte(pdev as *const pci_dev, 10i32, 6u8) };
unsafe { pci_write_config_byte(pdev as *const pci_dev, 64i32, tmp) };
unsafe {
(*pdev).class = 67073;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"set SATA to AHCI mode\n".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_787: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_788: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_789: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_790: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_791: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_793: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_ide_mode_794: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_ide_mode)) };

/// Serverworks CSB5 IDE does not fully support native mode
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_svwks_csb5ide")]
extern "C" fn quirk_svwks_csb5ide(pdev: *mut pci_dev) {
let mut prog: u8 = 0;
unsafe { pci_read_config_byte(pdev as *const pci_dev, 9i32, core::ptr::addr_of_mut!(prog)) };
if (prog as i32) & 5 != 0 {
let __v1: u8 = ((prog as i32) & !5) as u8;
prog = __v1;
let __v4: u32 = (unsafe { (*pdev).class }) & 4294967290u32;
unsafe {
(*pdev).class = __v4;
}
unsafe { pci_write_config_byte(pdev as *const pci_dev, 9i32, prog) };
// PCI layer will sort out resources
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_svwks_csb5ide_795: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_svwks_csb5ide)) };

/// Intel 82801CAM ICH3-M datasheet says IDE modes must be the same
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ide_samemode")]
extern "C" fn quirk_ide_samemode(pdev: *mut pci_dev) {
let mut prog: u8 = 0;
unsafe { pci_read_config_byte(pdev as *const pci_dev, 9i32, core::ptr::addr_of_mut!(prog)) };
if (prog as i32) & 1 != 0 && (prog as i32) & 4 == 0 || (prog as i32) & 4 != 0 && (prog as i32) & 1 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"IDE mode mismatch; forcing legacy mode\n".as_ptr() as *const u8) };
let __v2: u8 = ((prog as i32) & !5) as u8;
prog = __v2;
let __v5: u32 = (unsafe { (*pdev).class }) & 4294967290u32;
unsafe {
(*pdev).class = __v5;
}
unsafe { pci_write_config_byte(pdev as *const pci_dev, 9i32, prog) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ide_samemode_796: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ide_samemode)) };

/// Some ATA devices break if put into D3
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_ata_d3")]
extern "C" fn quirk_no_ata_d3(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 2) as u16;
}
}

/// Quirk the legacy ATA devices only. The AHCI ones are ok
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ata_d3_797: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ata_d3)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ata_d3_798: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ata_d3)) };

/// ALi loses some register settings that we cannot then restore
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ata_d3_799: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ata_d3)) };

/// VIA comes back fine but we need to keep it alive or ACPI GTM failures
/// occur when mode detecting
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ata_d3_800: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ata_d3)) };

/// This was originally an Alpha-specific thing, but it really fits here.
/// The i82375 PCI/EISA bridge appears as non-classified. Fix that.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_eisa_bridge")]
extern "C" fn quirk_eisa_bridge(dev: *mut pci_dev) {
unsafe {
(*dev).class = 393728u32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_eisa_bridge_801: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_eisa_bridge)) };

/// On ASUS P4B boards, the SMBus PCI Device within the ICH2/4 southbridge
/// is not activated. The myth is that Asus said that they do not want the
/// users to be irritated by just another PCI Device in the Win98 device
/// manager. (see the file prog/hotplug/README.p4b in the lm_sensors
/// package 2.7.0 for details)
///
/// The SMBus PCI Device can be activated by setting a bit in the ICH LPC
/// bridge. Unfortunately, this device has no subvendor/subdevice ID. So it
/// becomes necessary to do this tweak in two steps -- the chosen trigger
/// is either the Host bridge (preferred) or on-board VGA controller.
///
/// Note that we used to unhide the SMBus that way on Toshiba laptops
/// (Satellite A40 and Tecra M2) but then found that the thermal management
/// was done by SMM code, which could cause unsynchronized concurrent
/// accesses to the SMBus registers, with potentially bad effects. Thus you
/// should be very careful when adding new entries: if SMM is accessing the
/// Intel SMBus, this is a very good reason to leave it hidden.
///
/// Likewise, many recent laptops use ACPI for thermal management. If the
/// ACPI DSDT code accesses the SMBus, then Linux should not access it
/// natively, and keeping the SMBus hidden is the right thing to do. If you
/// are about to add an entry in the table below, please first disassemble
/// the DSDT and double-check that there is no code accessing the SMBus.
static mut asus_hides_smbus: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_hostbridge")]
extern "C" fn asus_hides_smbus_hostbridge(dev: *mut pci_dev) {
if unlikely__2!(((unsafe { (*dev).subsystem_vendor }) as i32) == 4163) != 0 {
if ((unsafe { (*dev).device }) as i32) == 6704 {
match (unsafe { (*dev).subsystem_device }) as i32 {
5670 | 32805 | 32880 | 32904 => {
unsafe {
asus_hides_smbus = 1;
}
// P4B-LX
// P4B
// P4B533
// L3C notebook
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9568 {
match (unsafe { (*dev).subsystem_device }) as i32 {
32915 | 32945 | 32946 => {
unsafe {
asus_hides_smbus = 1;
}
// P4GE-V
// P4PE
// P4B533-V
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9520 {
match (unsafe { (*dev).subsystem_device }) as i32 {
32816 => {
unsafe {
asus_hides_smbus = 1;
}
// P4T533
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9565 {
match (unsafe { (*dev).subsystem_device }) as i32 {
32880 => {
unsafe {
asus_hides_smbus = 1;
}
// P4G8X Deluxe
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9548 {
match (unsafe { (*dev).subsystem_device }) as i32 {
32969 => {
unsafe {
asus_hides_smbus = 1;
}
// PU-DLS
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 13696 {
match (unsafe { (*dev).subsystem_device }) as i32 {
5969 | 6177 | 6295 => {
unsafe {
asus_hides_smbus = 1;
}
// M2N notebook
// M5N notebook
// A6L notebook
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 13120 {
match (unsafe { (*dev).subsystem_device }) as i32 {
6219 | 6250 => {
unsafe {
asus_hides_smbus = 1;
}
// W1N notebook
// M6Ne notebook
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9584 {
match (unsafe { (*dev).subsystem_device }) as i32 {
33010 => {
unsafe {
asus_hides_smbus = 1;
}
// P4P800-X
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9616 {
match (unsafe { (*dev).subsystem_device }) as i32 {
6274 | 6519 => {
unsafe {
asus_hides_smbus = 1;
}
// M6V notebook
// A6VA notebook
}
_ => {
}
}
}
}
}
}
}
}
}
}
}
} else {
if unlikely__2!(((unsafe { (*dev).subsystem_vendor }) as i32) == 4156) != 0 {
if ((unsafe { (*dev).device }) as i32) == 13120 {
match (unsafe { (*dev).subsystem_device }) as i32 {
2188 | 2192 => {
unsafe {
asus_hides_smbus = 1;
}
// HP Compaq nc8000
// HP Compaq nc6000
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9584 {
match (unsafe { (*dev).subsystem_device }) as i32 {
106 | 4796 | 4797 => {
unsafe {
asus_hides_smbus = 1;
}
// HP D330L
// HP D530
// HP Compaq nx9500
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9592 {
match (unsafe { (*dev).subsystem_device }) as i32 {
4799 => {
unsafe {
asus_hides_smbus = 1;
}
// HP xw4100
}
_ => {
}
}
}
}
}
} else {
if unlikely__2!(((unsafe { (*dev).subsystem_vendor }) as i32) == 5197) != 0 {
if ((unsafe { (*dev).device }) as i32) == 13120 {
match (unsafe { (*dev).subsystem_device }) as i32 {
49164 => {
unsafe {
asus_hides_smbus = 1;
}
// Samsung P35 notebook
}
_ => {
}
}
}
} else {
if unlikely__2!(((unsafe { (*dev).subsystem_vendor }) as i32) == 3601) != 0 {
if ((unsafe { (*dev).device }) as i32) == 13120 {
match (unsafe { (*dev).subsystem_device }) as i32 {
88 => {
unsafe {
asus_hides_smbus = 1;
}
// Compaq Evo N620c
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 28963 {
match (unsafe { (*dev).subsystem_device }) as i32 {
45420 => {
unsafe {
asus_hides_smbus = 1;
}
// Compaq Deskpro EP 401963-001 (PCA# 010174)
// Motherboard doesn't have Host bridge
// subvendor/subdevice IDs, therefore checking
// its on-board VGA controller
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 9410 {
match (unsafe { (*dev).subsystem_device }) as i32 {
184..=186 => {
unsafe {
asus_hides_smbus = 1;
}
// Compaq Evo D510 CMT
// Compaq Evo D510 SFF
// Compaq Evo D510 USDT
// Motherboard doesn't have Host bridge
// subvendor/subdevice IDs and on-board VGA
// controller is disabled if an AGP card is
// inserted, therefore checking USB UHCI
// Controller #1
}
_ => {
}
}
} else {
if ((unsafe { (*dev).device }) as i32) == 4402 {
match (unsafe { (*dev).subsystem_device }) as i32 {
26 => {
unsafe {
asus_hides_smbus = 1;
}
// Compaq Deskpro EN SSF P667 815E
// Motherboard doesn't have host bridge
// subvendor/subdevice IDs, therefore checking
// its on-board VGA controller
}
_ => {
}
}
}
}
}
}
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_802: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_803: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_804: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_805: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_806: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_807: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_808: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_809: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_810: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_811: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_812: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_813: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_hostbridge_814: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_hostbridge)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_lpc")]
extern "C" fn asus_hides_smbus_lpc(dev: *mut pci_dev) {
let mut val: u16 = 0;
if likely!((unsafe { asus_hides_smbus }) == 0) != 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, 242i32, core::ptr::addr_of_mut!(val)) };
if (val as i32) & 8 != 0 {
unsafe { pci_write_config_word(dev as *const pci_dev, 242i32, ((val as i32) & !8) as u16) };
unsafe { pci_read_config_word(dev as *const pci_dev, 242i32, core::ptr::addr_of_mut!(val)) };
if (val as i32) & 8 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"i801 SMBus device continues to play 'hide and seek'! 0x%x\n".as_ptr() as *const u8, val as i32) };
} else {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabled i801 SMBus device\n".as_ptr() as *const u8) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_815: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_816: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_817: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_818: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_819: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_820: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_821: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_822: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_823: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_824: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_825: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_826: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_827: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_828: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc)) };

/// It appears we just have one such device. If not, we have a warning
static mut asus_rcba_base: *mut () = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_lpc_ich6_suspend")]
extern "C" fn asus_hides_smbus_lpc_ich6_suspend(dev: *mut pci_dev) {
let mut rcba: u32 = 0;
if likely!((unsafe { asus_hides_smbus }) == 0) != 0 {
return;
}
let mut __v0: i64;
let __ret_warn_on: i32 = !unsafe { asus_rcba_base }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("829: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 829b - ., 3; .popsection\n# {0}", const 829 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_asus_hides_smbus_lpc_ich6_suspend_1, sym __slate_asm_string_asus_hides_smbus_lpc_ich6_suspend_2, const 1685 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("830: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 830b - ., 4; .popsection\n# {0}", const 830 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { pci_read_config_dword(dev as *const pci_dev, 240i32, core::ptr::addr_of_mut!(rcba)) };
// use bits 31:14, 16 kB aligned
unsafe {
asus_rcba_base = ioremap((rcba & 4294950912) as u64, 16384u64);
}
if unsafe { asus_rcba_base }.is_null() {
return;
}
}

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_lpc_ich6_resume_early")]
extern "C" fn asus_hides_smbus_lpc_ich6_resume_early(dev: *mut pci_dev) {
let mut val: u32 = 0;
if likely!((unsafe { asus_hides_smbus }) == 0 || unsafe { asus_rcba_base }.is_null()) != 0 {
return;
}
// read the Function Disable register, dword mode only
val = readl(((unsafe { (asus_rcba_base as *const u8).wrapping_offset(13336 as isize) }) as *mut ()) as *const ());
// enable the SMBus device
writel(val & 4294967287, (unsafe { (asus_rcba_base as *const u8).wrapping_offset(13336 as isize) }) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_lpc_ich6_resume")]
extern "C" fn asus_hides_smbus_lpc_ich6_resume(dev: *mut pci_dev) {
if likely!((unsafe { asus_hides_smbus }) == 0 || unsafe { asus_rcba_base }.is_null()) != 0 {
return;
}
unsafe { iounmap(asus_rcba_base as *mut ()) };
unsafe {
asus_rcba_base = core::ptr::null_mut::<()>();
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabled ICH6/i801 SMBus device\n".as_ptr() as *const u8) };
}

#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_smbus_lpc_ich6")]
extern "C" fn asus_hides_smbus_lpc_ich6(dev: *mut pci_dev) {
asus_hides_smbus_lpc_ich6_suspend(dev);
asus_hides_smbus_lpc_ich6_resume_early(dev);
asus_hides_smbus_lpc_ich6_resume(dev);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_ich6_831: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc_ich6)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_ich6_suspend_832: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc_ich6_suspend)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_ich6_resume_833: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc_ich6_resume)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_smbus_lpc_ich6_resume_early_834: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_smbus_lpc_ich6_resume_early)) };

/// SiS 96x south bridge: BIOS typically hides SMBus device...
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_sis_96x_smbus")]
extern "C" fn quirk_sis_96x_smbus(dev: *mut pci_dev) {
let mut val: u8 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 119i32, core::ptr::addr_of_mut!(val)) };
if (val as i32) & 16 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling SiS 96x SMBus\n".as_ptr() as *const u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 119i32, ((val as i32) & !16) as u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_835: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_836: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_837: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_838: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_839: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_840: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_841: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_96x_smbus_842: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_96x_smbus)) };

// ... This is further complicated by the fact that some SiS96x south
// bridges pretend to be 85C503/5513 instead.  In that case see if we
// spotted a compatible north bridge to make sure.
// (pci_find_device() doesn't work yet)
//
// We can also enable the sis96x bit in the discovery register..
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_sis_503")]
extern "C" fn quirk_sis_503(dev: *mut pci_dev) {
let mut reg: u8 = 0;
let mut devid: u16 = 0;
unsafe { pci_read_config_byte(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(reg)) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 64i32, ((reg as i32) | 64i32) as u8) };
unsafe { pci_read_config_word(dev as *const pci_dev, 2i32, core::ptr::addr_of_mut!(devid)) };
if (devid as i32) & 65520 != 2400 && (devid as i32) != 24 {
unsafe { pci_write_config_byte(dev as *const pci_dev, 64i32, reg) };
return;
}
// Ok, it now shows up as a 96x.  Run the 96x quirk by hand in case
// it has already been processed.  (Depends on link order, which is
// apparently not guaranteed)
unsafe {
(*dev).device = devid;
}
quirk_sis_96x_smbus(dev);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_503_843: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_503)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_sis_503_844: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_sis_503)) };

/// On ASUS A8V and A8V Deluxe boards, the onboard AC97 audio controller
/// and MC97 modem controller are disabled when a second PCI soundcard is
/// present. This patch, tweaking the VT8237 ISA bridge, enables them.
/// -- bjd
#[unsafe(link_section = ".text.slate_distinct.quirks.asus_hides_ac97_lpc")]
extern "C" fn asus_hides_ac97_lpc(dev: *mut pci_dev) {
let mut val: u8 = 0;
let mut asus_hides_ac97: i32 = 0;
if likely!(((unsafe { (*dev).subsystem_vendor }) as i32) == 4163) != 0 {
if ((unsafe { (*dev).device }) as i32) == 12839 {
asus_hides_ac97 = 1;
}
}
if asus_hides_ac97 == 0 {
return;
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 80i32, core::ptr::addr_of_mut!(val)) };
if (val as i32) & 192 != 0 {
unsafe { pci_write_config_byte(dev as *const pci_dev, 80i32, ((val as i32) & !192) as u8) };
unsafe { pci_read_config_byte(dev as *const pci_dev, 80i32, core::ptr::addr_of_mut!(val)) };
if (val as i32) & 192 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Onboard AC97/MC97 devices continue to play 'hide and seek'! 0x%x\n".as_ptr() as *const u8, val as i32) };
} else {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabled onboard AC97/MC97 devices\n".as_ptr() as *const u8) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_ac97_lpc_845: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_ac97_lpc)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_asus_hides_ac97_lpc_846: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(asus_hides_ac97_lpc)) };

// If we are using libata we can drive this chip properly but must do this
// early on to make the additional device appear during the PCI scanning.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_jmicron_ata")]
extern "C" fn quirk_jmicron_ata(pdev: *mut pci_dev) {
let mut conf1: u32 = 0;
let mut conf5: u32 = 0;
let mut class: u32 = 0;
let mut hdr: u8 = 0;
// Only poke fn 0
if PCI_FUNC!(unsafe { (*pdev).devfn }) != 0 {
return;
}
unsafe { pci_read_config_dword(pdev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(conf1)) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 128i32, core::ptr::addr_of_mut!(conf5)) };
let __v1: u32 = conf1 & 4281339133u32;
conf1 = __v1; // Clear bit 1, 8, 9, 12-19, 22, 23
let __v3: u32 = conf5 & 4278190079u32;
conf5 = __v3; // Clear bit 24
// SATA single port
// SATA dual ports
// SATA dual ports
// The controller should be in single function ahci mode
// Set 8, 13, 15, 17
// Redirect IDE second PATA port to the right spot
// Enable dual function mode, AHCI on fn 0, IDE fn1
//
// Set the class codes correctly and then direct IDE 0
// Set 0, 1, 4, 5, 7, 8, 13, 15, 17, 22, 23
// The controller should be in single function IDE mode
// Set 22, 23
match (unsafe { (*pdev).device }) as i32 {
9056 | 9058 | 9060 => {
conf1 |= 172288; // SATA single port
// SATA dual ports
// SATA dual ports
// The controller should be in single function ahci mode
// Set 8, 13, 15, 17
}
9061 | 9062 => {
conf5 |= 16777216u32; // Redirect IDE second PATA port to the right spot
let _v16: u32 = conf1;
let _v17: u32 = _v16 | 12755379;
conf1 = _v17;
// Enable dual function mode, AHCI on fn 0, IDE fn1
//
// Set the class codes correctly and then direct IDE 0
// Set 0, 1, 4, 5, 7, 8, 13, 15, 17, 22, 23
}
9057 | 9059 | 9065 => {
conf1 |= 12755379;
// Enable dual function mode, AHCI on fn 0, IDE fn1
//
// Set the class codes correctly and then direct IDE 0
// Set 0, 1, 4, 5, 7, 8, 13, 15, 17, 22, 23
}
9064 => {
conf1 |= 12582912; // The controller should be in single function IDE mode
// Set 22, 23
}
_ => {
}
}
unsafe { pci_write_config_dword(pdev as *const pci_dev, 64i32, conf1) };
unsafe { pci_write_config_dword(pdev as *const pci_dev, 128i32, conf5) };
// Update pdev accordingly
unsafe { pci_read_config_byte(pdev as *const pci_dev, 14i32, core::ptr::addr_of_mut!(hdr)) };
unsafe {
(*pdev).hdr_type = ((hdr as i32) & 127) as u8;
}
let mut __v13: i32;
loop {
if __bf_cast_unsigned!(128i32) > ((__bf_cast_unsigned__4!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_847() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v15: i32;
loop {
if false {
unsafe { __compiletime_assert_848() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_849() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(128i32 >> (128u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_850() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)) & 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_851() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = ((hdr as i32) & 128) >> (128u64.trailing_zeros() as i32);
__v13 = __v15;
unsafe {
(*pdev).__slate_bits_3.__set_multifunction(__v13 as u32);
}
unsafe { pci_read_config_dword(pdev as *const pci_dev, 8i32, core::ptr::addr_of_mut!(class)) };
unsafe {
(*pdev).class = class >> 8i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_852: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_853: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_854: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_855: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_856: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_857: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_858: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_859: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_860: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_861: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_862: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_863: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_864: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_865: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_866: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_867: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_868: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_ata_869: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_ata)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_jmicron_async_suspend")]
extern "C" fn quirk_jmicron_async_suspend(dev: *mut pci_dev) {
if ((unsafe { (*dev).__slate_bits_3.__get_multifunction() }) as i32) != 0 {
device_disable_async_suspend(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"async suspend disabled to avoid multi-function power-on ordering issue\n".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_async_suspend_870: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_async_suspend)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_async_suspend_871: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_async_suspend)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_async_suspend_872: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_async_suspend)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_jmicron_async_suspend_873: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_jmicron_async_suspend)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_alder_ioapic")]
extern "C" fn quirk_alder_ioapic(pdev: *mut pci_dev) {
if (unsafe { (*pdev).class }) >> 8i32 != 65280 {
return;
}
// The first BAR is the location of the IO-APIC... we must
// not touch this (and it's already covered by the fixmap), so
// forcibly insert it into the resource tree.
let __v0: bool = pci_resource_start!(pdev, 0i32) != 0 && pci_resource_len!(pdev, 0i32) != 0;
if __v0 {
unsafe { insert_resource(core::ptr::addr_of_mut!(iomem_resource), ((*pdev).resource.as_mut_ptr() as *mut resource).wrapping_offset(0 as isize)) };
}
// The next five BARs all seem to be rubbish, so just clean
// them out.
for i in 1i32..6i32 {
unsafe { memset(((*pdev).resource.as_mut_ptr() as *mut resource).wrapping_offset(i as isize) as *mut (), 0i32, core::mem::size_of::<resource>() as u64) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_alder_ioapic_874: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_alder_ioapic)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_msi")]
extern "C" fn quirk_no_msi(dev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"avoiding MSI to work around a hardware defect\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_875: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_876: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_877: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_878: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_879: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_msi_880: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_msi)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_pcie_mch")]
extern "C" fn quirk_pcie_mch(pdev: *mut pci_dev) {
unsafe {
(*pdev).__slate_bits_3.__set_no_msi(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_mch_881: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_mch)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_mch_882: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_mch)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_mch_883: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_mch)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_mch_884: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_mch)) };

/// HiSilicon KunPeng920 and KunPeng930 have devices appear as PCI but are
/// actually on the AMBA bus. These fake PCI devices can support SVA via
/// SMMU stall feature, by setting dma-can-stall for ACPI platforms.
///
/// Normally stalling must not be enabled for PCI devices, since it would
/// break the PCI requirement for free-flowing writes and may lead to
/// deadlock.  We expect PCI devices to support ATS and PRI if they want to
/// be fault-tolerant, so there's no ACPI binding to describe anything else,
/// even when a "PCI" device turns out to be a regular old SoC device
/// dressed up as a RCiEP and normal rules don't apply.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_huawei_pcie_sva")]
extern "C" fn quirk_huawei_pcie_sva(pdev: *mut pci_dev) {
let mut __slate_compound_0: property_entry = unsafe { core::mem::zeroed() };
let mut properties: __SlateAlign16<[property_entry; 2]> = __SlateAlign16([unsafe { *{
__slate_compound_0 = {
let mut __t0: property_entry = core::mem::zeroed();
__t0.name = c"dma-can-stall".as_ptr() as *const u8;
__t0.is_inline = true;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, {
let mut __t1: property_entry = unsafe { core::mem::zeroed() };
__t1
}]);
if ((unsafe { (*pdev).revision }) as i32) != 33 && ((unsafe { (*pdev).revision }) as i32) != 48 {
return;
}
unsafe {
(*pdev).__slate_bits_2.__set_pasid_no_tlp(1u32);
}
// Set the dma-can-stall property on ACPI platforms. Device tree
// can set it directly.
let __v1: bool = unsafe { (*pdev).dev.of_node }.is_null() && (unsafe { device_create_managed_software_node(core::ptr::addr_of_mut!((*pdev).dev), properties.0.as_mut_ptr() as *const property_entry, core::ptr::null::<software_node>()) }) != 0;
if __v1 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"could not add stall property".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_885: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_886: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_887: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_888: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_889: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_huawei_pcie_sva_890: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_huawei_pcie_sva)) };

/// It's possible for the MSI to get corrupted if SHPC and ACPI are used
/// together on certain PXH-based systems.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_pcie_pxh")]
extern "C" fn quirk_pcie_pxh(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PXH quirk detected; SHPC device MSI disabled\n".as_ptr() as *const u8) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_pxh_891: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_pxh)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_pxh_892: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_pxh)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_pxh_893: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_pxh)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_pxh_894: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_pxh)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pcie_pxh_895: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pcie_pxh)) };

/// Some Intel PCI Express chipsets have trouble with downstream device
/// power management.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_intel_pcie_pm")]
extern "C" fn quirk_intel_pcie_pm(dev: *mut pci_dev) {
unsafe {
pci_pm_d3hot_delay = 120;
}
unsafe {
(*dev).__slate_bits_1.__set_no_d1d2(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_896: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_897: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_898: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_899: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_900: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_901: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_902: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_903: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_904: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_905: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_906: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_907: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_908: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_909: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_910: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_911: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_912: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_913: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_914: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_915: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_pcie_pm_916: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_pcie_pm)) };

fn quirk_d3hot_delay(dev: *mut pci_dev, delay: u32) {
if (unsafe { (*dev).d3hot_delay }) >= delay {
return;
}
unsafe {
(*dev).d3hot_delay = delay;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"extending delay after power-on from D3hot to %d msec\n".as_ptr() as *const u8, (*dev).d3hot_delay) };
}

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_radeon_pm")]
extern "C" fn quirk_radeon_pm(dev: *mut pci_dev) {
if ((unsafe { (*dev).subsystem_vendor }) as i32) == 4203 && ((unsafe { (*dev).subsystem_device }) as i32) == 226 {
quirk_d3hot_delay(dev, 20u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_radeon_pm_917: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_radeon_pm)) };

/// NVIDIA Ampere-based HDA controllers can wedge the whole device if a bus
/// reset is performed too soon after transition to D0, extend d3hot_delay
/// to previous effective default for all NVIDIA HDA controllers.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_hda_pm")]
extern "C" fn quirk_nvidia_hda_pm(dev: *mut pci_dev) {
quirk_d3hot_delay(dev, 20u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_hda_pm_918: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_hda_pm)) };

/// Ryzen5/7 XHCI controllers fail upon resume from runtime suspend or s2idle.
/// https://bugzilla.kernel.org/show_bug.cgi?id=205587
///
/// The kernel attempts to transition these devices to D3cold, but that seems
/// to be ineffective on the platforms in question; the PCI device appears to
/// remain on in D3hot state. The D3hot-to-D0 transition then requires an
/// extended delay in order to succeed.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_ryzen_xhci_d3hot")]
extern "C" fn quirk_ryzen_xhci_d3hot(dev: *mut pci_dev) {
quirk_d3hot_delay(dev, 20u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ryzen_xhci_d3hot_919: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ryzen_xhci_d3hot)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ryzen_xhci_d3hot_920: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ryzen_xhci_d3hot)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_ryzen_xhci_d3hot_921: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_ryzen_xhci_d3hot)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.dmi_disable_ioapicreroute")]
extern "C" fn dmi_disable_ioapicreroute(d: *const dmi_system_id) -> i32 {
unsafe {
noioapicreroute = 1;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016%s detected: disable boot interrupt reroute\n".as_ptr() as *const u8, (*d).ident) };
0
}

/// Systems to exclude from boot interrupt reroute quirks
static boot_interrupt_dmi_table: __SlateAlign16<[dmi_system_id; 2]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.callback = Some(dmi_disable_ioapicreroute);
__t0.ident = (b"ASUSTek Computer INC. M2N-LR\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 101u8, 107u8, 32u8, 67u8, 111u8, 109u8, 112u8, 117u8, 116u8, 101u8, 114u8, 32u8, 73u8, 78u8, 67u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [77u8, 50u8, 78u8, 45u8, 76u8, 82u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4
}]);

/// Boot interrupts on some chipsets cannot be turned off. For these chipsets,
/// remap the original interrupt in the Linux kernel to the boot interrupt, so
/// that a PCI device's interrupt handler is installed on the boot interrupt
/// line instead.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_reroute_to_boot_interrupts_intel")]
extern "C" fn quirk_reroute_to_boot_interrupts_intel(dev: *mut pci_dev) {
unsafe { dmi_check_system(core::ptr::addr_of!(boot_interrupt_dmi_table.0) as *const dmi_system_id) };
if (unsafe { noioapicquirk }) != 0 || (unsafe { noioapicreroute }) != 0 {
return;
}
unsafe {
(*dev).__slate_bits_3.__set_irq_reroute_variant(1u32);
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"rerouting interrupts for [%04x:%04x]\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_922: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_923: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_924: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_925: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_926: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_927: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_928: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_929: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_930: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_931: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_932: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_933: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_934: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_935: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_936: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reroute_to_boot_interrupts_intel_937: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reroute_to_boot_interrupts_intel)) };

// On some chipsets we can disable the generation of legacy INTx boot
// interrupts.
// IO-APIC1 on 6300ESB generates boot interrupts, see Intel order no
// 300641-004US, section 5.7.3.
//
// Core IO on Xeon E5 1600/2600/4600, see Intel order no 326509-003.
// Core IO on Xeon E5 v2, see Intel order no 329188-003.
// Core IO on Xeon E7 v2, see Intel order no 329595-002.
// Core IO on Xeon E5 v3, see Intel order no 330784-003.
// Core IO on Xeon E7 v3, see Intel order no 332315-001US.
// Core IO on Xeon E5 v4, see Intel order no 333810-002US.
// Core IO on Xeon E7 v4, see Intel order no 332315-001US.
// Core IO on Xeon D-1500, see Intel order no 332051-001.
// Core IO on Xeon Scalable, see Intel order no 610950.
// Bus 0, Dev 29, Func 5
// Bus 0, Dev 5, Func 0
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_intel_boot_interrupt")]
extern "C" fn quirk_disable_intel_boot_interrupt(dev: *mut pci_dev) {
let mut pci_config_word: u16 = 0;
let mut pci_config_dword: u32 = 0;
if (unsafe { noioapicquirk }) != 0 {
return;
}
match (unsafe { (*dev).device }) as i32 {
9644 => {
unsafe { pci_read_config_word(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(pci_config_word)) };
pci_config_word = ((pci_config_word as i32) | 16384i32) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, 64i32, pci_config_word) };
}
3624 | 8244 | 12072 | 15400 | 28456 => {
unsafe { pci_read_config_dword(dev as *const pci_dev, 332i32, core::ptr::addr_of_mut!(pci_config_dword)) }; // Xeon E5 1600/2600/4600
// Xeon E5/E7 V2
// Xeon E5/E7 V3,V4
// Xeon D-1500
// Xeon Scalable Family
pci_config_dword |= 33554432u32;
unsafe { pci_write_config_dword(dev as *const pci_dev, 332i32, pci_config_dword) };
}
_ => {
return;
}
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disabled boot interrupts on device [%04x:%04x]\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
}

/// Device 29 Func 5 Device IDs of IO-APIC
/// containing ABAR—APIC1 Alternate Base Address Register
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_938: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_939: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

/// Device 5 Func 0 Device IDs of Core IO modules/hubs
/// containing Coherent Interface Protocol Interrupt Control
///
/// Device IDs obtained from volume 2 datasheets of commented
/// families above.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_940: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_941: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_942: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_943: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_944: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_945: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_946: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_947: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_948: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_intel_boot_interrupt_949: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_intel_boot_interrupt)) };

// Disable boot interrupts on HT-1000
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_broadcom_boot_interrupt")]
extern "C" fn quirk_disable_broadcom_boot_interrupt(dev: *mut pci_dev) {
let mut pci_config_dword: u32 = 0;
let mut irq: u8 = 16;
if (unsafe { noioapicquirk }) != 0 {
return;
}
unsafe { pci_read_config_dword(dev as *const pci_dev, 100i32, core::ptr::addr_of_mut!(pci_config_dword)) };
unsafe { pci_write_config_dword(dev as *const pci_dev, 100i32, pci_config_dword | 1u32) };
while (irq as i32) < 48i32 {
__outb(irq, 3072u16);
__outb(0u8, 3073u16);
irq = (irq as i32).wrapping_add(1) as u8;
}
unsafe { pci_write_config_dword(dev as *const pci_dev, 100i32, pci_config_dword) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disabled boot interrupts on device [%04x:%04x]\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_broadcom_boot_interrupt_950: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_broadcom_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_broadcom_boot_interrupt_951: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_broadcom_boot_interrupt)) };

// Disable boot interrupts on AMD and ATI chipsets
// NOIOAMODE needs to be disabled to disable "boot interrupts". For AMD 8131
// rev. A0 and B0, NOIOAMODE needs to be disabled anyway to fix IO-APIC mode
// (due to an erratum).
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_amd_813x_boot_interrupt")]
extern "C" fn quirk_disable_amd_813x_boot_interrupt(dev: *mut pci_dev) {
let mut pci_config_dword: u32 = 0;
if (unsafe { noioapicquirk }) != 0 {
return;
}
if ((unsafe { (*dev).revision }) as i32) == 18 || ((unsafe { (*dev).revision }) as i32) == 19 {
return;
}
unsafe { pci_read_config_dword(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(pci_config_dword)) };
let __v1: u32 = pci_config_dword & 4294967294u32;
pci_config_dword = __v1;
unsafe { pci_write_config_dword(dev as *const pci_dev, 64i32, pci_config_dword) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disabled boot interrupts on device [%04x:%04x]\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_813x_boot_interrupt_952: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_813x_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_813x_boot_interrupt_953: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_813x_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_813x_boot_interrupt_954: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_813x_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_813x_boot_interrupt_955: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_813x_boot_interrupt)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_amd_8111_boot_interrupt")]
extern "C" fn quirk_disable_amd_8111_boot_interrupt(dev: *mut pci_dev) {
let mut pci_config_word: u16 = 0;
if (unsafe { noioapicquirk }) != 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, 86i32, core::ptr::addr_of_mut!(pci_config_word)) };
if pci_config_word == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"boot interrupts on device [%04x:%04x] already disabled\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
return;
}
unsafe { pci_write_config_word(dev as *const pci_dev, 86i32, 0u16) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disabled boot interrupts on device [%04x:%04x]\n".as_ptr() as *const u8, (*dev).vendor as i32, (*dev).device as i32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_8111_boot_interrupt_956: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_8111_boot_interrupt)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_amd_8111_boot_interrupt_957: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_amd_8111_boot_interrupt)) };

/// Toshiba TC86C001 IDE controller reports the standard 8-byte BAR0 size
/// but the PIO transfers won't work if BAR0 falls at the odd 8 bytes.
/// Re-allocate the region if needed...
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_tc86c001_ide")]
extern "C" fn quirk_tc86c001_ide(dev: *mut pci_dev) {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(0 as isize);
if (unsafe { (*r).start }) & 8 != 0 {
unsafe {
(*r).flags = (*r).flags | 536870912;
}
resource_set_range(r, 0u64, 16u64);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tc86c001_ide_958: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tc86c001_ide)) };

/// PLX PCI 9050 PCI Target bridge controller has an erratum that prevents the
/// local configuration registers accessible via BAR0 (memory) or BAR1 (i/o)
/// being read correctly if bit 7 of the base address is set.
/// The BAR0 or BAR1 region may be disabled (size 0) or enabled (size 128).
/// Re-allocate the regions to a 256-byte boundary if necessary.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_plx_pci9050")]
extern "C" fn quirk_plx_pci9050(dev: *mut pci_dev) {
let mut bar: u32 = 0;
// Fixed in revision 2 (PCI 9052).
if ((unsafe { (*dev).revision }) as i32) >= 2 {
return;
}
while bar <= 1 {
if pci_resource_len!(dev, bar) == 128 && pci_resource_start!(dev, bar) & 128 != 0 {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(bar as isize);
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Re-allocating PLX PCI 9050 BAR %u to length 256 to avoid bit 7 bug\n".as_ptr() as *const u8, bar) };
unsafe {
(*r).flags = (*r).flags | 536870912;
}
resource_set_range(r, 0u64, 256u64);
}
bar = bar.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_plx_pci9050_959: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_plx_pci9050)) };

/// The following Meilhaus (vendor ID 0x1402) device IDs (amongst others)
/// may be using the PLX PCI 9050: 0x0630, 0x0940, 0x0950, 0x0960, 0x100b,
/// 0x1400, 0x140a, 0x140b, 0x14e0, 0x14ea, 0x14eb, 0x1604, 0x1608, 0x160c,
/// 0x168f, 0x2000, 0x2600, 0x3000, 0x810a, 0x810b.
///
/// Currently, device IDs 0x2000 and 0x2600 are used by the Comedi "me_daq"
/// driver.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_plx_pci9050_960: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_plx_pci9050)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_plx_pci9050_961: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_plx_pci9050)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_netmos")]
extern "C" fn quirk_netmos(dev: *mut pci_dev) {
let num_parallel: u32 = ((((unsafe { (*dev).subsystem_device }) as i32) & 240) >> 4i32) as u32;
let num_serial: u32 = (((unsafe { (*dev).subsystem_device }) as i32) & 15) as u32;
// These Netmos parts are multiport serial devices with optional
// parallel ports.  Even when parallel ports are present, they
// are identified as class SERIAL, which means the serial driver
// will claim them.  To prevent this, mark them as class OTHER.
// These combo devices should be claimed by parport_serial.
//
// The subdevice ID is of the form 0x00PS, where <P> is the number
// of parallel ports and <S> is the number of serial ports.
// Well, this rule doesn't hold for the following 9835 device
match (unsafe { (*dev).device }) as i32 {
38965 => {
if ((unsafe { (*dev).subsystem_vendor }) as i32) == 4116 && ((unsafe { (*dev).subsystem_device }) as i32) == 665 {
return;
}
// Well, this rule doesn't hold for the following 9835 device
if num_parallel != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Netmos %04x (%u parallel, %u serial); changing class SERIAL to OTHER (use parport_serial)\n".as_ptr() as *const u8, (*dev).device as i32, num_parallel, num_serial) };
unsafe {
(*dev).class = 491520u32 | (*dev).class & 255;
}
}
}
38709 | 38725 | 38981 | 38997 => {
if num_parallel != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Netmos %04x (%u parallel, %u serial); changing class SERIAL to OTHER (use parport_serial)\n".as_ptr() as *const u8, (*dev).device as i32, num_parallel, num_serial) };
unsafe {
(*dev).class = 491520u32 | (*dev).class & 255;
}
}
}
_ => {
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_netmos_962: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_netmos)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_e100_interrupt")]
extern "C" fn quirk_e100_interrupt(dev: *mut pci_dev) {
let mut command: u16 = 0;
let mut pmcsr: u16 = 0;
let mut cmd_hi: u8 = 0;
match (unsafe { (*dev).device }) as i32 {
4137 | 4144..=4148 | 4152..=4158 | 4176..=4183 | 4185 | 4196..=4203 | 4241..=4245 | 4617 | 4649 | 9289 | 9305 | 9309 | 10204 => {
// PCI IDs taken from drivers/net/e100.c
}
_ => {
return;
}
}
// Some firmware hands off the e100 with interrupts enabled,
// which can cause a flood of interrupts if packets are
// received before the driver attaches to the device.  So
// disable all e100 interrupts here.  The driver will
// re-enable them when it's ready.
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(command)) };
if (command as i32) & 2 == 0 || pci_resource_start!(dev, 0i32) == 0 {
return;
}
// Check that the device is in the D0 power state. If it's not,
// there is no point to look any further.
if (unsafe { (*dev).pm_cap }) != 0 {
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
if (pmcsr as i32) & 3 != 0 {
return;
}
}
// Convert from PCI bus to resource space.
let csr: *mut u8 = (unsafe { ioremap(pci_resource_start!(dev, 0i32), 8u64) }) as *mut u8;
if csr.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Can't map e100 registers\n".as_ptr() as *const u8) };
return;
}
cmd_hi = readb(csr.wrapping_offset(3 as isize) as *const ());
if (cmd_hi as i32) == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Firmware left e100 interrupts enabled; disabling\n".as_ptr() as *const u8) };
writeb(1u8, csr.wrapping_offset(3 as isize) as *mut ());
}
unsafe { iounmap(csr as *mut ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_e100_interrupt_963: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_e100_interrupt)) };

/// The 82575 and 82598 may experience data corruption issues when transitioning
/// out of L0S.  To prevent this we need to disable L0S on the PCIe link.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_aspm_l0s")]
extern "C" fn quirk_disable_aspm_l0s(dev: *mut pci_dev) {
unsafe { pcie_aspm_remove_cap(dev, 1024u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_964: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_965: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_966: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_967: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_968: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_969: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_970: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_971: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_972: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_973: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_974: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_975: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_976: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_977: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

/// Realtek RTS525A generates a Replay Timer Timeout storm when L0s is enabled.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_978: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_aspm_l0s_l1")]
extern "C" fn quirk_disable_aspm_l0s_l1(dev: *mut pci_dev) {
unsafe { pcie_aspm_remove_cap(dev, 3072u32) };
}

/// ASM1083/1085 PCIe-PCI bridge devices cause AER timeout errors on the
/// upstream PCIe root port when ASPM is enabled. At least L0s mode is affected;
/// disable both L0s and L1 for now to be safe.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_l1_979: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s_l1)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_l1_980: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s_l1)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_l1_981: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s_l1)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_aspm_l0s_l1_982: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_aspm_l0s_l1)) };

/// Some Pericom PCIe-to-PCI bridges in reverse mode need the PCIe Retrain
/// Link bit cleared after starting the link retrain process to allow this
/// process to finish.
///
/// Affected devices: PI7C9X110, PI7C9X111SL, PI7C9X130.  See also the
/// Pericom Errata Sheet PI7C9X111SLB_errata_rev1.2_102711.pdf.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_enable_clear_retrain_link")]
extern "C" fn quirk_enable_clear_retrain_link(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_1.__set_clear_retrain_link(1u32);
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enable PCIe Retrain Link quirk\n".as_ptr() as *const u8) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_enable_clear_retrain_link_983: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_enable_clear_retrain_link)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_enable_clear_retrain_link_984: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_enable_clear_retrain_link)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_enable_clear_retrain_link_985: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_enable_clear_retrain_link)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.fixup_rev1_53c810")]
extern "C" fn fixup_rev1_53c810(dev: *mut pci_dev) {
let class: u32 = unsafe { (*dev).class };
// rev 1 ncr53c810 chips don't set the class at all which means
// they don't get their resources remapped. Fix that here.
if class != 0 {
return;
}
unsafe {
(*dev).class = 65536u32;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"NCR 53c810 rev 1 PCI class overridden (%#08x -> %#08x)\n".as_ptr() as *const u8, class, (*dev).class) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_rev1_53c810_986: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_rev1_53c810)) };

/// Enable 1k I/O space granularity on the Intel P64H2
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_p64h2_1k_io")]
extern "C" fn quirk_p64h2_1k_io(dev: *mut pci_dev) {
let mut en1k: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 64i32, core::ptr::addr_of_mut!(en1k)) };
if (en1k as i32) & 512 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enable I/O Space to 1KB granularity\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_3.__set_io_window_1k(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_p64h2_1k_io_987: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_p64h2_1k_io)) };

/// Under some circumstances, AER is not linked with extended capabilities.
/// Force it to be linked by setting the corresponding control bit in the
/// config space.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_ck804_pcie_aer_ext_cap")]
extern "C" fn quirk_nvidia_ck804_pcie_aer_ext_cap(dev: *mut pci_dev) {
let mut b: u8 = 0;
if (unsafe { pci_read_config_byte(dev as *const pci_dev, 3905i32, core::ptr::addr_of_mut!(b)) }) == 0 {
if (b as i32) & 32 == 0 {
unsafe { pci_write_config_byte(dev as *const pci_dev, 3905i32, ((b as i32) | 32) as u8) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Linking AER extended capability\n".as_ptr() as *const u8) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_ck804_pcie_aer_ext_cap_988: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_ck804_pcie_aer_ext_cap)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_ck804_pcie_aer_ext_cap_989: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_ck804_pcie_aer_ext_cap)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_via_cx700_pci_parking_caching")]
extern "C" fn quirk_via_cx700_pci_parking_caching(dev: *mut pci_dev) {
// Disable PCI Bus Parking and PCI Master read caching on CX700
// which causes unspecified timing errors with a VT6212L on the PCI
// bus leading to USB2.0 packet loss.
//
// This quirk is only enabled if a second (on the external PCI bus)
// VT6212L is found -- the CX700 core itself also contains a USB
// host controller with the same PCI ID as the VT6212L.
// Count VT6212L instances
let mut p: *mut pci_dev = unsafe { pci_get_device(4358u32, 12548u32, core::ptr::null_mut::<pci_dev>()) };
let mut b: u8 = 0;
// p should contain the first (internal) VT6212L -- see if we have
// an external one by searching again.
p = unsafe { pci_get_device(4358u32, 12548u32, p) };
if p.is_null() {
return;
}
unsafe { pci_dev_put(p) };
if (unsafe { pci_read_config_byte(dev as *const pci_dev, 118i32, core::ptr::addr_of_mut!(b)) }) == 0 {
if (b as i32) & 64 != 0 {
// Turn off PCI Bus Parking
unsafe { pci_write_config_byte(dev as *const pci_dev, 118i32, ((b as i32) ^ 64) as u8) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling VIA CX700 PCI parking\n".as_ptr() as *const u8) };
}
}
if (unsafe { pci_read_config_byte(dev as *const pci_dev, 114i32, core::ptr::addr_of_mut!(b)) }) == 0 {
if (b as i32) != 0 {
// Turn off PCI Master read caching
unsafe { pci_write_config_byte(dev as *const pci_dev, 114i32, 0u8) };
// Set PCI Master Bus time-out to "1x16 PCLK"
unsafe { pci_write_config_byte(dev as *const pci_dev, 117i32, 1u8) };
// Disable "Read FIFO Timer"
unsafe { pci_write_config_byte(dev as *const pci_dev, 119i32, 0u8) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling VIA CX700 PCI caching\n".as_ptr() as *const u8) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_via_cx700_pci_parking_caching_990: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_via_cx700_pci_parking_caching)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_brcm_5719_limit_mrrs")]
extern "C" fn quirk_brcm_5719_limit_mrrs(dev: *mut pci_dev) {
let mut rev: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 244i32, core::ptr::addr_of_mut!(rev)) };
// Only CAP the MRRS if the device is a 5719 A0
if rev == 91328512 {
let readrq: i32 = unsafe { pcie_get_readrq(dev) };
if readrq > 2048 {
unsafe { pcie_set_readrq(dev, 2048i32) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_brcm_5719_limit_mrrs_991: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_brcm_5719_limit_mrrs)) };

/// Originally in EDAC sources for i82875P: Intel tells BIOS developers to
/// hide device 6 which configures the overflow device access containing the
/// DRBs - this is where we expose device 6.
/// http://www.x86-secret.com/articles/tweak/pat/patsecrets-2.htm
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_unhide_mch_dev6")]
extern "C" fn quirk_unhide_mch_dev6(dev: *mut pci_dev) {
let mut reg: u8 = 0;
if (unsafe { pci_read_config_byte(dev as *const pci_dev, 244i32, core::ptr::addr_of_mut!(reg)) }) == 0 && (reg as i32) & 2 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling MCH 'Overflow' Device\n".as_ptr() as *const u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, 244i32, ((reg as i32) | 2) as u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_unhide_mch_dev6_992: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_unhide_mch_dev6)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_unhide_mch_dev6_993: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_unhide_mch_dev6)) };

/// Some chipsets do not support MSI. We cannot easily rely on setting
/// PCI_BUS_FLAGS_NO_MSI in its bus flags because there are actually some
/// other buses controlled by the chipset even if Linux is not aware of it.
/// Instead of setting the flag on all buses in the machine, simply disable
/// MSI globally.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_all_msi")]
extern "C" fn quirk_disable_all_msi(dev: *mut pci_dev) {
unsafe { pci_no_msi() };
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"MSI quirk detected; MSI disabled\n".as_ptr() as *const u8) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_994: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_995: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_996: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_997: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_998: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_999: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_1000: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_1001: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_all_msi_1002: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_all_msi)) };

/// Disable MSI on chipsets that are known to not support it
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_disable_msi")]
extern "C" fn quirk_disable_msi(dev: *mut pci_dev) {
if !unsafe { (*dev).subordinate }.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"MSI quirk detected; subordinate MSI disabled\n".as_ptr() as *const u8) };
let __v1: *mut pci_bus = unsafe { (*dev).subordinate };
unsafe {
(*__v1).bus_flags = (((*__v1).bus_flags as i32) | 1) as u16;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_msi_1003: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_msi_1004: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_msi_1005: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_disable_msi_1006: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_disable_msi)) };

/// The APC bridge device in AMD 780 family northbridges has some random
/// OEM subsystem ID in its vendor ID register (erratum 18), so instead
/// we use the possible vendor/device IDs of the host bridge for the
/// declared quirk, and search for the APC bridge by slot number.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_780_apc_msi")]
extern "C" fn quirk_amd_780_apc_msi(host_bridge: *mut pci_dev) {
let apc_bridge: *mut pci_dev = unsafe { pci_get_slot((*host_bridge).bus, PCI_DEVFN!(1i32, 0i32) as u32) };
if !apc_bridge.is_null() {
if ((unsafe { (*apc_bridge).device }) as i32) == 38402 {
quirk_disable_msi(apc_bridge);
}
unsafe { pci_dev_put(apc_bridge) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_780_apc_msi_1007: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_780_apc_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_780_apc_msi_1008: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_780_apc_msi)) };

/// Go through the list of HyperTransport capabilities and return 1 if a HT
/// MSI capability is found and enabled.
fn msi_ht_cap_enabled(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = (unsafe { pci_find_ht_capability(dev, 168i32) }) as i32;
let mut ttl: i32 = 48;
loop {
let __v1: bool = if pos != 0 {
let __v2: i32 = ttl;
ttl = __v2.wrapping_sub(1);
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
let mut flags: u8 = 0;
if (unsafe { pci_read_config_byte(dev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(flags)) }) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Found %s HT MSI Mapping\n".as_ptr() as *const u8, if (flags as i32) & 1 != 0 { c"enabled".as_ptr() as *mut u8 } else { c"disabled".as_ptr() as *mut u8 }) };
return ((flags as i32) & 1 != 0) as i32;
}
pos = (unsafe { pci_find_next_ht_capability(dev, pos as u8, 168i32) }) as i32;
}
0
}

/// Check the HyperTransport MSI mapping to know whether MSI is enabled or not
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_msi_ht_cap")]
extern "C" fn quirk_msi_ht_cap(dev: *mut pci_dev) {
if msi_ht_cap_enabled(dev) == 0 {
quirk_disable_msi(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_ht_cap_1009: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_ht_cap)) };

/// The nVidia CK804 chipset may have 2 HT MSI mappings.  MSI is supported
/// if the MSI capability is set in any of these mappings.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_ck804_msi_ht_cap")]
extern "C" fn quirk_nvidia_ck804_msi_ht_cap(dev: *mut pci_dev) {
// Check HT MSI cap on this chipset and the root one.  A single one
// having MSI is enough to be sure that MSI is supported.
let pdev: *mut pci_dev = unsafe { pci_get_slot((*dev).bus, 0u32) };
if pdev.is_null() {
return;
}
if msi_ht_cap_enabled(pdev) == 0 {
quirk_msi_ht_cap(dev);
}
unsafe { pci_dev_put(pdev) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_ck804_msi_ht_cap_1010: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_ck804_msi_ht_cap)) };

/// Force enable MSI mapping capability on HT bridges
#[unsafe(link_section = ".text.slate_distinct.quirks.ht_enable_msi_mapping")]
extern "C" fn ht_enable_msi_mapping(dev: *mut pci_dev) {
let mut pos: i32 = (unsafe { pci_find_ht_capability(dev, 168i32) }) as i32;
let mut ttl: i32 = 48;
loop {
let __v1: bool = if pos != 0 {
let __v2: i32 = ttl;
ttl = __v2.wrapping_sub(1);
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
let mut flags: u8 = 0;
if (unsafe { pci_read_config_byte(dev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(flags)) }) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling HT MSI Mapping\n".as_ptr() as *const u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, pos.wrapping_add(2), ((flags as i32) | 1) as u8) };
}
pos = (unsafe { pci_find_next_ht_capability(dev, pos as u8, 168i32) }) as i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ht_enable_msi_mapping_1011: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(ht_enable_msi_mapping)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ht_enable_msi_mapping_1012: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(ht_enable_msi_mapping)) };

/// The P5N32-SLI motherboards from Asus have a problem with MSI
/// for the MCP55 NIC. It is not yet determined whether the MSI problem
/// also affects other devices. As for now, turn off MSI for this device.
#[unsafe(link_section = ".text.slate_distinct.quirks.nvenet_msi_disable")]
extern "C" fn nvenet_msi_disable(dev: *mut pci_dev) {
let board_name: *const u8 = unsafe { dmi_get_system_info(14i32) };
let __v0: bool = !board_name.is_null() && (!unsafe { strstr(board_name, c"P5N32-SLI PREMIUM".as_ptr() as *const u8) }.is_null() || !unsafe { strstr(board_name, c"P5N32-E SLI".as_ptr() as *const u8) }.is_null());
if __v0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling MSI for MCP55 NIC on P5N32-SLI\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nvenet_msi_disable_1013: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nvenet_msi_disable)) };

/// PCIe spec r6.0 sec 6.1.4.3 says that if MSI/MSI-X is enabled, the device
/// can't use INTx interrupts. Tegra's PCIe Root Ports don't generate MSI
/// interrupts for PME and AER events; instead only INTx interrupts are
/// generated. Though Tegra's PCIe Root Ports can generate MSI interrupts
/// for other events, since PCIe specification doesn't support using a mix of
/// INTx and MSI/MSI-X, it is required to disable MSI interrupts to avoid port
/// service drivers registering their respective ISRs for MSIs.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_nvidia_tegra_disable_rp_msi")]
extern "C" fn pci_quirk_nvidia_tegra_disable_rp_msi(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1014: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1015: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1016: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1017: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1018: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1019: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1020: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1021: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1022: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1023: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1024: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1025: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1026: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1027: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1028: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_quirk_nvidia_tegra_disable_rp_msi_1029: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_quirk_nvidia_tegra_disable_rp_msi)) };

/// Some versions of the MCP55 bridge from Nvidia have a legacy IRQ routing
/// config register.  This register controls the routing of legacy
/// interrupts from devices that route through the MCP55.  If this register
/// is misprogrammed, interrupts are only sent to the BSP, unlike
/// conventional systems where the IRQ is broadcast to all online CPUs.  Not
/// having this register set properly prevents kdump from booting up
/// properly, so let's make sure that we have it set correctly.
/// Note that this is an undocumented register.
#[unsafe(link_section = ".text.slate_distinct.quirks.nvbridge_check_legacy_irq_routing")]
extern "C" fn nvbridge_check_legacy_irq_routing(dev: *mut pci_dev) {
let mut cfg: u32 = 0;
if (unsafe { pci_find_capability(dev, 8i32) }) == 0 {
return;
}
unsafe { pci_read_config_dword(dev as *const pci_dev, 116i32, core::ptr::addr_of_mut!(cfg)) };
if cfg & 32772u32 != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Rewriting IRQ routing register on MCP55\n".as_ptr() as *const u8) };
let __v3: u32 = cfg & 4294934523u32;
cfg = __v3;
unsafe { pci_write_config_dword(dev as *const pci_dev, 116i32, cfg) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nvbridge_check_legacy_irq_routing_1030: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nvbridge_check_legacy_irq_routing)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nvbridge_check_legacy_irq_routing_1031: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nvbridge_check_legacy_irq_routing)) };

fn ht_check_msi_mapping(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let mut ttl: i32 = 48;
let mut found: i32 = 0;
// Check if there is HT MSI cap or enabled on this device
pos = (unsafe { pci_find_ht_capability(dev, 168i32) }) as i32;
loop {
let __v1: bool = if pos != 0 {
let __v2: i32 = ttl;
ttl = __v2.wrapping_sub(1);
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
let mut flags: u8 = 0;
if found < 1 {
found = 1;
}
if (unsafe { pci_read_config_byte(dev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(flags)) }) == 0 {
if (flags as i32) & 1 != 0 {
if found < 2 {
found = 2;
break;
}
}
}
pos = (unsafe { pci_find_next_ht_capability(dev, pos as u8, 168i32) }) as i32;
}
found
}

fn host_bridge_with_leaf(host_bridge: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let dev_no: i32 = ((unsafe { (*host_bridge).devfn }) >> 3i32) as i32;
let mut found: i32 = 0;
for i in dev_no.wrapping_add(1)..32 {
let dev: *mut pci_dev = unsafe { pci_get_slot((*host_bridge).bus, PCI_DEVFN!(i, 0i32) as u32) };
if !dev.is_null() {
// found next host bridge?
pos = (unsafe { pci_find_ht_capability(dev, 0i32) }) as i32;
if pos != 0 {
unsafe { pci_dev_put(dev) };
break;
}
if ht_check_msi_mapping(dev) != 0 {
found = 1;
unsafe { pci_dev_put(dev) };
break;
}
unsafe { pci_dev_put(dev) };
}
}
found
}

// link control
// link control to
fn is_end_of_ht_chain(dev: *mut pci_dev) -> i32 {
let mut ctrl: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u16 = core::ptr::addr_of_mut!(ctrl) as *mut u16;
let mut flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u16 = core::ptr::addr_of_mut!(flags) as *mut u16;
let mut end: i32 = unsafe { core::mem::zeroed() };
let mut ctrl_off: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ctrl_off) as *mut i32;
let mut pos: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(pos) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
end = 0;
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_4 = pci_find_ht_capability(dev, 0i32) as i32;
if *__slate_slot_4 != 0 {
pci_read_config_word(dev as *const pci_dev, (*__slate_slot_4).wrapping_add(2), core::ptr::addr_of_mut!(*__slate_slot_1));
*__slate_slot_3 = if (*__slate_slot_1 as i32) >> 10i32 & 1 != 0 { 4 } else { 8 };
pci_read_config_word(dev as *const pci_dev, (*__slate_slot_4).wrapping_add(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_0));
if (*__slate_slot_0 as i32) & 64i32 != 0 {
end = 1;
}
}
return end;
}
}

fn nv_ht_enable_msi_mapping(dev: *mut pci_dev) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// don't enable end_device/host_bridge with leaf directly here
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut found: i32 = unsafe { core::mem::zeroed() };
let mut dev_no: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(dev_no) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut pos: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(pos) as *mut i32;
let mut host_bridge: core::mem::MaybeUninit<*mut pci_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut pci_dev = core::ptr::addr_of_mut!(host_bridge) as *mut *mut pci_dev;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
found = 0;
*__slate_slot_5 = ((*dev).devfn >> 3i32) as i32;
*__slate_slot_6 = *__slate_slot_5;
'__join_11: {
loop {
if *__slate_slot_6 >= 0 {
*__slate_slot_8 = pci_get_slot((*dev).bus, PCI_DEVFN!(*__slate_slot_6, 0i32) as u32);
if !(*__slate_slot_8).is_null() {
*__slate_slot_7 = pci_find_ht_capability(*__slate_slot_8, 0i32) as i32;
if *__slate_slot_7 != 0 {
break;
} else {
pci_dev_put(*__slate_slot_8);
}
}
let __v3: i32 = *__slate_slot_6;
*__slate_slot_6 = __v3.wrapping_sub(1);
} else {
break '__join_11;
}
}
found = 1;
}
if found == 0 {
return;
} else {
if *__slate_slot_8 == dev {
__v1 = is_end_of_ht_chain(*__slate_slot_8) != 0;
} else {
__v1 = false;
}
if __v1 {
__v0 = host_bridge_with_leaf(*__slate_slot_8) != 0;
} else {
__v0 = false;
}
if !__v0 {
// root did that !
if msi_ht_cap_enabled(*__slate_slot_8) == 0 {
ht_enable_msi_mapping(dev);
}
}
pci_dev_put(*__slate_slot_8);
}
}
}

fn ht_disable_msi_mapping(dev: *mut pci_dev) {
let mut pos: i32 = (unsafe { pci_find_ht_capability(dev, 168i32) }) as i32;
let mut ttl: i32 = 48;
loop {
let __v1: bool = if pos != 0 {
let __v2: i32 = ttl;
ttl = __v2.wrapping_sub(1);
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
let mut flags: u8 = 0;
if (unsafe { pci_read_config_byte(dev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(flags)) }) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling HT MSI Mapping\n".as_ptr() as *const u8) };
unsafe { pci_write_config_byte(dev as *const pci_dev, pos.wrapping_add(2), ((flags as i32) & !1) as u8) };
}
pos = (unsafe { pci_find_next_ht_capability(dev, pos as u8, 168i32) }) as i32;
}
}

fn __nv_msi_ht_cap_quirk(dev: *mut pci_dev, all: i32) {
let mut found: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(found) as *mut i32;
let mut pos: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(pos) as *mut i32;
let mut host_bridge: core::mem::MaybeUninit<*mut pci_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut pci_dev = core::ptr::addr_of_mut!(host_bridge) as *mut *mut pci_dev;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !pci_msi_enabled() {
return;
} else {
// check if there is HT MSI cap or enabled on this device
*__slate_slot_0 = ht_check_msi_mapping(dev);
// no HT MSI CAP
if *__slate_slot_0 == 0 {
return;
} else {
// HT MSI mapping should be disabled on devices that are below
// a non-HyperTransport host bridge. Locate the host bridge.
*__slate_slot_2 = pci_get_domain_bus_and_slot(pci_domain_nr((*dev).bus), 0u32, PCI_DEVFN!(0i32, 0i32) as u32);
if (*__slate_slot_2).is_null() {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"nv_msi_ht_cap_quirk didn't locate host bridge\n".as_ptr() as *const u8);
return;
} else {
*__slate_slot_1 = pci_find_ht_capability(*__slate_slot_2, 0i32) as i32;
if *__slate_slot_1 != 0 {
// Host bridge is to HT
if *__slate_slot_0 == 1 {
// it is not enabled, try to enable it
if all != 0 {
ht_enable_msi_mapping(dev);
} else {
nv_ht_enable_msi_mapping(dev);
}
}
} else {
// HT MSI is not enabled
if *__slate_slot_0 != 1 {
// Host bridge is not to HT, disable HT MSI mapping on this device
ht_disable_msi_mapping(dev);
}
}
pci_dev_put(*__slate_slot_2);
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.quirks.nv_msi_ht_cap_quirk_all")]
extern "C" fn nv_msi_ht_cap_quirk_all(dev: *mut pci_dev) {
__nv_msi_ht_cap_quirk(dev, 1i32);
return;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nv_msi_ht_cap_quirk_all_1032: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nv_msi_ht_cap_quirk_all)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nv_msi_ht_cap_quirk_all_1033: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nv_msi_ht_cap_quirk_all)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.nv_msi_ht_cap_quirk_leaf")]
extern "C" fn nv_msi_ht_cap_quirk_leaf(dev: *mut pci_dev) {
__nv_msi_ht_cap_quirk(dev, 0i32);
return;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nv_msi_ht_cap_quirk_leaf_1034: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nv_msi_ht_cap_quirk_leaf)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nv_msi_ht_cap_quirk_leaf_1035: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nv_msi_ht_cap_quirk_leaf)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_msi_intx_disable_bug")]
extern "C" fn quirk_msi_intx_disable_bug(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 1) as u16;
}
}

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_msi_intx_disable_ati_bug")]
extern "C" fn quirk_msi_intx_disable_ati_bug(dev: *mut pci_dev) {
// SB700 MSI issue will be fixed at HW level from revision A21;
// we need check PCI REVISION ID of SMBus controller to get SB700
// revision.
let p: *mut pci_dev = unsafe { pci_get_device(4098u32, 17285u32, core::ptr::null_mut::<pci_dev>()) };
if p.is_null() {
return;
}
if ((unsafe { (*p).revision }) as i32) < 59 && ((unsafe { (*p).revision }) as i32) >= 48 {
unsafe {
(*dev).dev_flags = (((*dev).dev_flags as i32) | 1) as u16;
}
}
unsafe { pci_dev_put(p) };
}

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_msi_intx_disable_qca_bug")]
extern "C" fn quirk_msi_intx_disable_qca_bug(dev: *mut pci_dev) {
// AR816X/AR817X/E210X MSI is fixed at HW level from revision 0x18
if ((unsafe { (*dev).revision }) as i32) < 24 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"set MSI_INTX_DISABLE_BUG flag\n".as_ptr() as *const u8) };
unsafe {
(*dev).dev_flags = (((*dev).dev_flags as i32) | 1) as u16;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1036: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1037: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1038: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1039: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1040: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1041: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_ati_bug_1042: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_ati_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_ati_bug_1043: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_ati_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_ati_bug_1044: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_ati_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_ati_bug_1045: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_ati_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_ati_bug_1046: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_ati_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1047: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1048: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1049: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1050: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1051: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1052: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1053: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1054: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_bug_1055: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_qca_bug_1056: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_qca_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_qca_bug_1057: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_qca_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_qca_bug_1058: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_qca_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_qca_bug_1059: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_qca_bug)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_msi_intx_disable_qca_bug_1060: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_msi_intx_disable_qca_bug)) };

/// Amazon's Annapurna Labs 1c36:0031 Root Ports don't support MSI-X, so it
/// should be disabled on platforms where the device (mistakenly) advertises it.
///
/// Notice that this quirk also disables MSI (which may work, but hasn't been
/// tested), since currently there is no standard way to disable only MSI-X.
///
/// The 0031 device id is reused for other non Root Port device types,
/// therefore the quirk is registered for the PCI_CLASS_BRIDGE_PCI class.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_al_msi_disable")]
extern "C" fn quirk_al_msi_disable(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling MSI/MSI-X\n".as_ptr() as *const u8) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_al_msi_disable_1061: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_al_msi_disable)) };

/// Allow manual resource allocation for PCI hotplug bridges via
/// pci=hpmemsize=nnM and pci=hpiosize=nnM parameters. For some PCI-PCI
/// hotplug bridges, like PLX 6254 (former HINT HB6), kernel fails to
/// allocate resources when hotplug device is inserted and PCI bus is
/// rescanned.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_hotplug_bridge")]
extern "C" fn quirk_hotplug_bridge(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_is_hotplug_bridge(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_hotplug_bridge_1062: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_hotplug_bridge)) };

/// This is a quirk for the Ricoh MMC controller found as a part of some
/// multifunction chips.
///
/// This is very similar and based on the ricoh_mmc driver written by
/// Philip Langdale. Thank you for these magic sequences.
///
/// These chips implement the four main memory card controllers (SD, MMC,
/// MS, xD) and one or both of CardBus or FireWire.
///
/// It happens that they implement SD and MMC support as separate
/// controllers (and PCI functions). The Linux SDHCI driver supports MMC
/// cards but the chip detects MMC cards in hardware and directs them to the
/// MMC controller - so the SDHCI driver never sees them.
///
/// To get around this, we must disable the useless MMC controller.  At that
/// point, the SDHCI controller will start seeing them.  It seems to be the
/// case that the relevant PCI registers to deactivate the MMC controller
/// live on PCI function 0, which might be the CardBus controller or the
/// FireWire controller, depending on the particular chip in question
///
/// This has to be done early, because as soon as we disable the MMC controller
/// other PCI functions shift up one level, e.g. function #2 becomes function
/// #1, and this will confuse the PCI core.
/// This is a quirk for masking VT-d spec-defined errors to platform error
/// handling logic. Without this, platforms using Intel 7500, 5500 chipsets
/// (and the derivative chipsets like X58 etc) seem to generate NMI/SMI (based
/// on the RAS config settings of the platform) when a VT-d fault happens.
/// The resulting SMI caused the system to hang.
///
/// VT-d spec-related errors are already handled by the VT-d OS code, so no
/// need to report the same error through other channels.
#[unsafe(link_section = ".text.slate_distinct.quirks.vtd_mask_spec_errors")]
extern "C" fn vtd_mask_spec_errors(dev: *mut pci_dev) {
let mut word: u32 = 0;
unsafe { pci_read_config_dword(dev as *const pci_dev, 428i32, core::ptr::addr_of_mut!(word)) };
unsafe { pci_write_config_dword(dev as *const pci_dev, 428i32, word | 2147483648u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vtd_mask_spec_errors_1063: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(vtd_mask_spec_errors)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vtd_mask_spec_errors_1064: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(vtd_mask_spec_errors)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.fixup_ti816x_class")]
extern "C" fn fixup_ti816x_class(dev: *mut pci_dev) {
let class: u32 = unsafe { (*dev).class };
// TI 816x devices do not have class code set when in PCIe boot mode
unsafe {
(*dev).class = 262144u32;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PCI class overridden (%#08x -> %#08x)\n".as_ptr() as *const u8, class, (*dev).class) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_ti816x_class_1065: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_ti816x_class)) };

/// Some PCIe devices do not work reliably with the claimed maximum
/// payload size supported.
#[unsafe(link_section = ".text.slate_distinct.quirks.fixup_mpss_256")]
extern "C" fn fixup_mpss_256(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_0.__set_pcie_mpss(1u8);
}
// 256 bytes
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_mpss_256_1066: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_mpss_256)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_mpss_256_1067: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_mpss_256)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_mpss_256_1068: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_mpss_256)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fixup_mpss_256_1069: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(fixup_mpss_256)) };

/// Intel 5000 and 5100 Memory controllers have an erratum with read completion
/// coalescing (which is enabled by default on some BIOSes) and MPS of 256B.
/// Since there is no way of knowing what the PCIe MPS on each fabric will be
/// until all of the devices are discovered and buses walked, read completion
/// coalescing must be disabled.  Unfortunately, it cannot be re-enabled because
/// it is possible to hotplug a device with MPS of 256B.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_intel_mc_errata")]
extern "C" fn quirk_intel_mc_errata(dev: *mut pci_dev) {
let mut err: i32 = 0;
let mut rcc: u16 = 0;
if ((unsafe { pcie_bus_config }) as u32) == 0 || ((unsafe { pcie_bus_config }) as u32) == 1 {
return;
}
// Intel erratum specifies bits to change but does not say what
// they are.  Keeping them magical until such time as the registers
// and values can be explained.
err = unsafe { pci_read_config_word(dev as *const pci_dev, 72i32, core::ptr::addr_of_mut!(rcc)) };
if err != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Error attempting to read the read completion coalescing register\n".as_ptr() as *const u8) };
return;
}
if (rcc as i32) & 1024i32 == 0 {
return;
}
let __v2: u16 = ((rcc as i32) & -1025i32) as u16;
rcc = __v2;
err = unsafe { pci_write_config_word(dev as *const pci_dev, 72i32, rcc) };
if err != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Error attempting to write the read completion coalescing register\n".as_ptr() as *const u8) };
return;
}
let mut __v4: i64;
let __ret_do_once: bool = true;
let mut __v5: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely__2!(__ret_cond) != 0 && unlikely__2!(!unsafe { __already_done }) != 0 {
unsafe {
__already_done = true;
}
__ret_once = true;
}
__v5 = __ret_once as i64;
if __v5 != 0 {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x016Read completion coalescing disabled due to hardware erratum relating to 256B MPS\n".as_ptr() as *const u8) };
}
__v4 = __ret_do_once as i64;
}

#[unsafe(link_section = ".data..once")]
static mut __already_done: bool = unsafe { core::mem::zeroed() };

/// Intel 5000 series memory controllers and ports 2-7
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1070: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1071: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1072: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1073: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1074: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1075: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1076: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1077: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1078: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1079: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1080: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1081: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1082: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1083: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

/// Intel 5100 series memory controllers and ports 2-7
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1084: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1085: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1086: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1087: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1088: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1089: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1090: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1091: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1092: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1093: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_mc_errata_1094: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_mc_errata)) };

/// Ivytown NTB BAR sizes are misreported by the hardware due to an erratum.
/// To work around this, query the size it should be configured to by the
/// device and modify the resource end to correspond to this new size.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_intel_ntb")]
extern "C" fn quirk_intel_ntb(dev: *mut pci_dev) {
let mut rc: i32 = 0;
let mut val: u8 = 0;
rc = unsafe { pci_read_config_byte(dev as *const pci_dev, 208i32, core::ptr::addr_of_mut!(val)) };
if rc != 0 {
return;
}
resource_set_size(unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(2 as isize), 1u64 << (val as i32));
rc = unsafe { pci_read_config_byte(dev as *const pci_dev, 209i32, core::ptr::addr_of_mut!(val)) };
if rc != 0 {
return;
}
resource_set_size(unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(4 as isize), 1u64 << (val as i32));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_ntb_1095: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_ntb)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_ntb_1096: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_ntb)) };

// Some BIOS implementations leave the Intel GPU interrupts enabled, even
// though no one is handling them (e.g., if the i915 driver is never
// loaded).  Additionally the interrupt destination is not set up properly
// and the interrupt ends up -somewhere-.
//
// These spurious interrupts are "sticky" and the kernel disables the
// (shared) interrupt line after 100,000+ generated interrupts.
//
// Fix it by disabling the still enabled interrupts.  This resolves crashes
// often seen on monitor unplug.
#[unsafe(link_section = ".text.slate_distinct.quirks.disable_igfx_irq")]
extern "C" fn disable_igfx_irq(dev: *mut pci_dev) {
let regs: *mut () = unsafe { pci_iomap(dev, 0i32, 0u64) };
if regs.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"igfx quirk: Can't iomap PCI device\n".as_ptr() as *const u8) };
return;
}
// Check if any interrupt line is still enabled
if readl(((unsafe { (regs as *const u8).wrapping_offset(278540 as isize) }) as *mut ()) as *const ()) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BIOS left Intel GPU interrupts enabled; disabling\n".as_ptr() as *const u8) };
writel(0u32, (unsafe { (regs as *const u8).wrapping_offset(278540 as isize) }) as *mut ());
}
unsafe { pci_iounmap(dev, regs) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1097: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1098: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1099: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1100: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1101: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1102: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_disable_igfx_irq_1103: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(disable_igfx_irq)) };

/// PCI devices which are on Intel chips can skip the 10ms delay
/// before entering D3 mode.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_remove_d3hot_delay")]
extern "C" fn quirk_remove_d3hot_delay(dev: *mut pci_dev) {
unsafe {
(*dev).d3hot_delay = 0;
}
}

/// C600 Series devices do not need 10ms d3hot_delay
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1104: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1105: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1106: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

/// Lynxpoint-H PCH devices do not need 10ms d3hot_delay
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1107: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1108: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1109: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1110: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1111: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1112: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1113: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1114: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1115: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1116: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1117: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

/// Intel Cherrytrail devices do not need 10ms d3hot_delay
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1118: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1119: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1120: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1121: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1122: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1123: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1124: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1125: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_remove_d3hot_delay_1126: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_remove_d3hot_delay)) };

/// Some devices may pass our check in pci_intx_mask_supported() if
/// PCI_COMMAND_INTX_DISABLE works though they actually do not properly
/// support this feature.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_broken_intx_masking")]
extern "C" fn quirk_broken_intx_masking(dev: *mut pci_dev) {
unsafe {
(*dev).__slate_bits_3.__set_broken_intx_masking(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1127: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1128: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

// Ralink RT2800 802.11n PCI
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1129: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

// Ceton InfiniTV4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1130: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

/// Realtek RTL8169 PCI Gigabit Ethernet Controller (rev 10)
/// Subsystem: Realtek RTL8169/8110 Family PCI Gigabit Ethernet NIC
///
/// RTL8110SC - Fails under PCI device assignment using DisINTx masking.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1131: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

/// Intel i40e (XL710/X710) 10/20/40GbE NICs all have broken INTx masking,
/// DisINTx can be set but the interrupt status bit is non-functional.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1132: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1133: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1134: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1135: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1136: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1137: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1138: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1139: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1140: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1141: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1142: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1143: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1144: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1145: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1146: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_broken_intx_masking_1147: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_broken_intx_masking)) };

static mut mellanox_broken_intx_devs: __SlateAlign16<[u16; 14]> = __SlateAlign16([25408u16, 25418u16, 25428u16, 26418u16, 26428u16, 25448u16, 26448u16, 25458u16, 26458u16, 26468u16, 26438u16, 26478u16, 4099u16, 4103u16]);

/// Check ConnectX-4/LX FW version to see if it supports legacy interrupts.
/// If so, don't mark it as broken.
/// FW minor > 99 means older FW version format and no INTx masking support.
/// FW minor < 14 means new FW version format and no INTx masking support.
#[unsafe(link_section = ".text.slate_distinct.quirks.mellanox_check_broken_intx_masking")]
extern "C" fn mellanox_check_broken_intx_masking(pdev: *mut pci_dev) {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut fw_sub_min: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(fw_sub_min) as *mut u32;
let mut fw_maj_min: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(fw_maj_min) as *mut u32;
let mut fw_subminor: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u16 = core::ptr::addr_of_mut!(fw_subminor) as *mut u16;
let mut fw_minor: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u16 = core::ptr::addr_of_mut!(fw_minor) as *mut u16;
let mut fw_major: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u16 = core::ptr::addr_of_mut!(fw_major) as *mut u16;
let mut fw_ver: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut u32 = core::ptr::addr_of_mut!(fw_ver) as *mut *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_4 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_2 = 0;
while (*__slate_slot_2 as u64) < ARRAY_SIZE__7!() {
if ((*pdev).device as i32) == (mellanox_broken_intx_devs.0[*__slate_slot_2 as usize] as i32) {
(*pdev).__slate_bits_3.__set_broken_intx_masking(1u32);
return;
} else {
let __v1: i32 = *__slate_slot_2;
*__slate_slot_2 = __v1.wrapping_add(1);
}
}
// Getting here means Connect-IB cards and up. Connect-IB has no INTx
// support so shouldn't be checked further
if ((*pdev).device as i32) == 4113 {
return;
} else {
if ((*pdev).device as i32) != 4115 && ((*pdev).device as i32) != 4117 {
return;
} else {
// For ConnectX-4 and ConnectX-4LX, need to check FW support
if pci_enable_device_mem(pdev) != 0 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Can't enable device memory\n".as_ptr() as *const u8);
return;
} else {
*__slate_slot_8 = ioremap(pci_resource_start!(pdev, 0i32), 4u64) as *mut u32;
if (*__slate_slot_8).is_null() {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Can't map ConnectX-4 initialization segment\n".as_ptr() as *const u8);
} else {
// Reading from resource space should be 32b aligned
*__slate_slot_4 = ioread32be(*__slate_slot_8 as *const ());
*__slate_slot_3 = ioread32be((*__slate_slot_8).wrapping_offset(1 as isize) as *const ());
*__slate_slot_7 = (*__slate_slot_4 & 65535) as u16;
*__slate_slot_6 = (*__slate_slot_4 >> 16i32) as u16;
*__slate_slot_5 = (*__slate_slot_3 & 65535) as u16;
if (*__slate_slot_6 as i32) > 99 || (*__slate_slot_6 as i32) < 14 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"ConnectX-4: FW %u.%u.%u doesn't support INTx masking, disabling. Please upgrade FW to %d.14.1100 and up for INTx support\n".as_ptr() as *const u8, *__slate_slot_7 as i32, *__slate_slot_6 as i32, *__slate_slot_5 as i32, if ((*pdev).device as i32) == 4115 { 12i32 } else { 14i32 });
(*pdev).__slate_bits_3.__set_broken_intx_masking(1u32);
}
iounmap(*__slate_slot_8 as *mut ());
}
pci_disable_device(pdev);
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mellanox_check_broken_intx_masking_1148: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(mellanox_check_broken_intx_masking)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_bus_reset")]
extern "C" fn quirk_no_bus_reset(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 64) as u16;
}
}

/// After asserting Secondary Bus Reset to downstream devices via a GB10
/// Root Port, the link may not retrain correctly.
/// https://lore.kernel.org/r/20251113084441.2124737-1-Johnny-CC.Chang@mediatek.com
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1149: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1150: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

/// Some NVIDIA GPU devices do not work with bus reset, SBR needs to be
/// prevented for those affected devices.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_no_bus_reset")]
extern "C" fn quirk_nvidia_no_bus_reset(dev: *mut pci_dev) {
if ((unsafe { (*dev).device }) as i32) & 65472 == 9024 {
quirk_no_bus_reset(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_no_bus_reset_1151: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_no_bus_reset)) };

/// Some Atheros AR9xxx and QCA988x chips do not behave after a bus reset.
/// The device will throw a Link Down error on AER-capable systems and
/// regardless of AER, config space of the device is never accessible again
/// and typically causes the system to hang or reset when access is attempted.
/// https://lore.kernel.org/r/20140923210318.498dacbd@dualc.maya.org/
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1152: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1153: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1154: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1155: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1156: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1157: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1158: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

// WCN6855
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1159: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

// WCN7850
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1160: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

// SDX62/SDX65
/// Root port on some Cavium CN8xxx chips do not successfully complete a bus
/// reset when used with certain child devices.  After the reset, config
/// accesses to the child may fail.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1161: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

/// Some TI KeyStone C667X devices do not support bus/hot reset.  The PCIESS
/// automatically disables LTSSM when Secondary Bus Reset is received and
/// the device stops working.  Prevent bus reset for these devices.  With
/// this change, the device can be assigned to VMs with VFIO, but it will
/// leak state between VMs.  Reference
/// https://e2e.ti.com/support/processors/f/791/t/954382
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1162: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

/// Reports from users making use of PCI device assignment with ASM1164
/// controllers indicate an issue with bus reset where the device fails to
/// retrain.  The issue appears more common in configurations with multiple
/// controllers.  The device does indicate PM reset support (NoSoftRst-),
/// therefore this still leaves a viable reset method.
/// https://forum.proxmox.com/threads/problems-with-pcie-passthrough-with-two-identical-devices.149003/
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_bus_reset_1163: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_bus_reset)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_pm_reset")]
extern "C" fn quirk_no_pm_reset(dev: *mut pci_dev) {
// We can't do a bus reset on root bus devices, but an ineffective
// PM reset may be better than nothing.
if !pci_is_root_bus(unsafe { (*dev).bus }) {
unsafe {
(*dev).dev_flags = (((*dev).dev_flags as i32) | 128) as u16;
}
}
}

/// Some AMD/ATI GPUS (HD8570 - Oland) report that a D3hot->D0 transition
/// causes a reset (i.e., they advertise NoSoftRst-).  This transition seems
/// to have no effect on the device: it retains the framebuffer contents and
/// monitor sync.  Advertising this support makes other layers, like VFIO,
/// assume pci_reset_function() is viable for this device.  Mark it as
/// unavailable to skip it when testing reset methods.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_pm_reset_1164: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_pm_reset)) };

/// Spectrum-{1,2,3,4} devices report that a D3hot->D0 transition causes a reset
/// (i.e., they advertise NoSoftRst-). However, this transition does not have
/// any effect on the device: It continues to be operational and network ports
/// remain up. Advertising this support makes it seem as if a PM reset is viable
/// for these devices. Mark it as unavailable to skip it when testing reset
/// methods.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_pm_reset_1165: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_pm_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_pm_reset_1166: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_pm_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_pm_reset_1167: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_pm_reset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_pm_reset_1168: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_pm_reset)) };

/// Thunderbolt controllers with broken MSI hotplug signaling:
/// Entire 1st generation (Light Ridge, Eagle Ridge, Light Peak) and part
/// of the 2nd generation (Cactus Ridge 4C up to revision 1, Port Ridge).
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_thunderbolt_hotplug_msi")]
extern "C" fn quirk_thunderbolt_hotplug_msi(pdev: *mut pci_dev) {
if ((unsafe { (*pdev).__slate_bits_3.__get_is_pciehp() }) as i32) != 0 && (((unsafe { (*pdev).device }) as i32) != 5447 || ((unsafe { (*pdev).revision }) as i32) <= 1) {
unsafe {
(*pdev).__slate_bits_3.__set_no_msi(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_thunderbolt_hotplug_msi_1169: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_thunderbolt_hotplug_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_thunderbolt_hotplug_msi_1170: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_thunderbolt_hotplug_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_thunderbolt_hotplug_msi_1171: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_thunderbolt_hotplug_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_thunderbolt_hotplug_msi_1172: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_thunderbolt_hotplug_msi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_thunderbolt_hotplug_msi_1173: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_thunderbolt_hotplug_msi)) };

/// Apple: Shutdown Cactus Ridge Thunderbolt controller.
///
/// On Apple hardware the Cactus Ridge Thunderbolt controller needs to be
/// shutdown before suspend. Otherwise the native host interface (NHI) will not
/// be present after resume if a device was plugged in before suspend.
///
/// The Thunderbolt controller consists of a PCIe switch with downstream
/// bridges leading to the NHI and to the tunnel PCI bridges.
///
/// This quirk cuts power to the whole chip. Therefore we have to apply it
/// during suspend_noirq of the upstream bridge.
///
/// Power is automagically restored before resume. No action is needed.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_apple_poweroff_thunderbolt")]
extern "C" fn quirk_apple_poweroff_thunderbolt(dev: *mut pci_dev) {
let mut SXIO: *mut () = unsafe { core::mem::zeroed() };
let mut SXFP: *mut () = unsafe { core::mem::zeroed() };
let mut SXLV: *mut () = unsafe { core::mem::zeroed() };
if !unsafe { x86_apple_machine } {
return;
}
if pci_pcie_type(dev as *const pci_dev) != 5 {
return;
}
// SXIO/SXFP/SXLF turns off power to the Thunderbolt controller.
// We don't know how to turn it back on again, but firmware does,
// so we can only use SXIO/SXFP/SXLF if we're suspending via
// firmware.
if !pm_suspend_via_firmware() {
return;
}
let bridge: *mut () = ACPI_HANDLE!(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
if bridge.is_null() {
return;
}
// SXIO and SXLV are present only on machines requiring this quirk.
// Thunderbolt bridges in external devices might have the same
// device ID as those on the host, but they will not have the
// associated ACPI methods. This implicitly checks that we are at
// the right bridge.
let __v4: bool = ACPI_FAILURE!(unsafe { acpi_get_handle(bridge, c"DSB0.NHI0.SXIO".as_ptr() as *const u8, core::ptr::addr_of_mut!(SXIO)) }) != 0 || ACPI_FAILURE!(unsafe { acpi_get_handle(bridge, c"DSB0.NHI0.SXFP".as_ptr() as *const u8, core::ptr::addr_of_mut!(SXFP)) }) != 0;
let __v5: bool = __v4 || ACPI_FAILURE!(unsafe { acpi_get_handle(bridge, c"DSB0.NHI0.SXLV".as_ptr() as *const u8, core::ptr::addr_of_mut!(SXLV)) }) != 0;
if __v5 {
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"quirk: cutting power to Thunderbolt controller...\n".as_ptr() as *const u8) };
// magic sequence
unsafe { acpi_execute_simple_method(SXIO, core::ptr::null_mut::<u8>(), 1u64) };
unsafe { acpi_execute_simple_method(SXFP, core::ptr::null_mut::<u8>(), 0u64) };
unsafe { msleep(300u32) };
unsafe { acpi_execute_simple_method(SXLV, core::ptr::null_mut::<u8>(), 0u64) };
unsafe { acpi_execute_simple_method(SXIO, core::ptr::null_mut::<u8>(), 0u64) };
unsafe { acpi_execute_simple_method(SXLV, core::ptr::null_mut::<u8>(), 0u64) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_apple_poweroff_thunderbolt_1174: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_apple_poweroff_thunderbolt)) };

/// Following are device-specific reset methods which can be used to
/// reset a single function if other methods (e.g. FLR, PM D0->D3) are
/// not available.
#[unsafe(link_section = ".text.slate_distinct.quirks.reset_intel_82599_sfp_virtfn")]
extern "C" fn reset_intel_82599_sfp_virtfn(dev: *mut pci_dev, probe: bool) -> i32 {
// http://www.intel.com/content/dam/doc/datasheet/82599-10-gbe-controller-datasheet.pdf
//
// The 82599 supports FLR on VFs, but FLR support is reported only
// in the PF DEVCAP (sec 9.3.10.4), not in the VF DEVCAP (sec 9.5).
// Thus we must call pcie_flr() directly without first checking if it is
// supported.
if !probe {
unsafe { pcie_flr(dev) };
}
0
}

// set timeout 10 seconds
#[unsafe(link_section = ".text.slate_distinct.quirks.reset_ivb_igd")]
extern "C" fn reset_ivb_igd(dev: *mut pci_dev, probe: bool) -> i32 {
let mut val: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(val) as *mut u32;
let mut timeout: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(timeout) as *mut u64;
let mut mmio_base: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(mmio_base) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if probe {
return 0;
} else {
*__slate_slot_2 = pci_iomap(dev, 0i32, 0u64);
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
iowrite32(2u32, (*__slate_slot_2 as *const u8).wrapping_offset(282640 as isize) as *mut ());
// Clobbering SOUTH_CHICKEN2 register is fine only if the next
// driver loaded sets the right bits. However, this's a reset and
// the bits have been set by i915 previously, so we clobber
// SOUTH_CHICKEN2 register directly here.
iowrite32(5u32, (*__slate_slot_2 as *const u8).wrapping_offset(794628 as isize) as *mut ());
*__slate_slot_0 = ioread32(((*__slate_slot_2 as *const u8).wrapping_offset(815620 as isize) as *mut ()) as *const ()) & 4294967294;
iowrite32(*__slate_slot_0, (*__slate_slot_2 as *const u8).wrapping_offset(815620 as isize) as *mut ());
*__slate_slot_1 = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(10000u32));
'__join_0: {
loop {
*__slate_slot_0 = ioread32(((*__slate_slot_2 as *const u8).wrapping_offset(815616 as isize) as *mut ()) as *const ());
if *__slate_slot_0 & 2952790016 == 0 {
break '__join_0;
} else {
msleep(10u32);
if !time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), *__slate_slot_1) {
break;
}
}
}
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"timeout during reset\n".as_ptr() as *const u8);
}
iowrite32(2u32, (*__slate_slot_2 as *const u8).wrapping_offset(852224 as isize) as *mut ());
pci_iounmap(dev, *__slate_slot_2);
return 0;
}
}
}
}

/// Device-specific reset method for Chelsio T4-based adapters
#[unsafe(link_section = ".text.slate_distinct.quirks.reset_chelsio_generic_dev")]
extern "C" fn reset_chelsio_generic_dev(dev: *mut pci_dev, probe: bool) -> i32 {
let mut old_command: u16 = 0;
let mut msix_flags: u16 = 0;
// If this isn't a Chelsio T4-based device, return -ENOTTY indicating
// that we have no device-specific reset method.
if ((unsafe { (*dev).device }) as i32) & 61440 != 16384 {
return 25i32.wrapping_neg();
}
// If this is the "probe" phase, return 0 indicating that we can
// reset this device.
if probe {
return 0;
}
// T4 can wedge if there are DMAs in flight within the chip and Bus
// Master has been disabled.  We need to have it on till the Function
// Level Reset completes.  (BUS_MASTER is disabled in
// pci_reset_function()).
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(old_command)) };
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, ((old_command as i32) | 4) as u16) };
// Perform the actual device function reset, saving and restoring
// configuration information around the reset.
unsafe { pci_save_state(dev) };
// T4 also suffers a Head-Of-Line blocking problem if MSI-X interrupts
// are disabled when an MSI-X interrupt message needs to be delivered.
// So we briefly re-enable MSI-X interrupts for the duration of the
// FLR.  The pci_restore_state() below will restore the original
// MSI-X state.
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).msix_cap as i32).wrapping_add(2), core::ptr::addr_of_mut!(msix_flags)) };
if (msix_flags as i32) & 32768 == 0 {
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).msix_cap as i32).wrapping_add(2), ((msix_flags as i32) | 32768 | 16384) as u16) };
}
unsafe { pcie_flr(dev) };
// Restore the configuration information (BAR values, etc.) including
// the original PCI Configuration Space Command word, and return
// success.
unsafe { pci_restore_state(dev) };
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, old_command) };
0
}

/// The Samsung SM961/PM961 controller can sometimes enter a fatal state after
/// FLR where config space reads from the device return -1.  We seem to be
/// able to avoid this condition if we disable the NVMe controller prior to
/// FLR.  This quirk is generic for any NVMe class device requiring similar
/// assistance to quiesce the device prior to FLR.
///
/// NVMe specification: https://nvmexpress.org/resources/specifications/
/// Revision 1.0e:
///    Chapter 2: Required and optional PCI config registers
///    Chapter 3: NVMe control registers
///    Chapter 7.3: Reset behavior
#[unsafe(link_section = ".text.slate_distinct.quirks.nvme_disable_and_flr")]
extern "C" fn nvme_disable_and_flr(dev: *mut pci_dev, probe: bool) -> i32 {
let mut cmd: u16 = 0;
let mut cfg: u32 = 0;
let __v0: bool = (unsafe { (*dev).class }) != 67586 || (unsafe { pcie_reset_flr(dev, true) }) != 0;
if __v0 || pci_resource_start!(dev, 0i32) == 0 {
return 25i32.wrapping_neg();
}
if probe {
return 0;
}
let bar: *mut () = unsafe { pci_iomap(dev, 0i32, 20u64.wrapping_add(core::mem::size_of::<u32>() as u64)) };
if bar.is_null() {
return 25i32.wrapping_neg();
}
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(cmd)) };
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, ((cmd as i32) | 2) as u16) };
cfg = readl(((unsafe { (bar as *const u8).wrapping_offset(20 as isize) }) as *mut ()) as *const ());
// Disable controller if enabled
if cfg & 1 != 0 {
let cap: u32 = readl(((unsafe { (bar as *const u8).wrapping_offset(0 as isize) }) as *mut ()) as *const ());
let mut timeout: u64 = 0;
// Per nvme_disable_ctrl() skip shutdown notification as it
// could complete commands to the admin queue.  We only intend
// to quiesce the device before reset.
let __v2: u32 = cfg & 4294918142u32;
cfg = __v2;
writel(cfg, (unsafe { (bar as *const u8).wrapping_offset(20 as isize) }) as *mut ());
// Some controllers require an additional delay here, see
// NVME_QUIRK_DELAY_BEFORE_CHK_RDY.  None of those are yet
// supported by this quirk.
// Cap register provides max timeout in 500ms increments
timeout = ((NVME_CAP_TIMEOUT!(cap).wrapping_add(1).wrapping_mul(1000) / 2) as u64).wrapping_add(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
loop {
let status: u32 = readl(((unsafe { (bar as *const u8).wrapping_offset(28 as isize) }) as *mut ()) as *const ());
// Ready status becomes zero on disable complete
if status & 1 == 0 {
break;
}
unsafe { msleep(100u32) };
if time_after__2!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, timeout) {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Timeout waiting for NVMe ready status to clear after disable\n".as_ptr() as *const u8) };
break;
}
}
}
unsafe { pci_iounmap(dev, bar) };
unsafe { pcie_flr(dev) };
0
}

/// Some NVMe controllers such as Intel DC P3700 and Solidigm P44 Pro will
/// timeout waiting for ready status to change after NVMe enable if the driver
/// starts interacting with the device too soon after FLR.  A 250ms delay after
/// FLR has heuristically proven to produce reliably working results for device
/// assignment cases.
#[unsafe(link_section = ".text.slate_distinct.quirks.delay_250ms_after_flr")]
extern "C" fn delay_250ms_after_flr(dev: *mut pci_dev, probe: bool) -> i32 {
if probe {
return unsafe { pcie_reset_flr(dev, true) };
}
unsafe { pcie_reset_flr(dev, false) };
unsafe { msleep(250u32) };
0
}

// 15 seconds
/// Device-specific reset method for Huawei Intelligent NIC virtual functions
#[unsafe(link_section = ".text.slate_distinct.quirks.reset_hinic_vf_dev")]
extern "C" fn reset_hinic_vf_dev(pdev: *mut pci_dev, probe: bool) -> i32 {
let mut val: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(val) as *mut u32;
let mut bar: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(bar) as *mut *mut ();
let mut timeout: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(timeout) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if probe {
return 0;
} else {
*__slate_slot_1 = pci_iomap(pdev, 0i32, 0u64);
if (*__slate_slot_1).is_null() {
return 25i32.wrapping_neg();
} else {
// Get and check firmware capabilities
*__slate_slot_0 = ioread32be(((*__slate_slot_1 as *const u8).wrapping_offset(4096 as isize) as *mut ()) as *const ());
if (*__slate_slot_0 as u64) & 1073741824u64 == 0 {
pci_iounmap(pdev, *__slate_slot_1);
return 25i32.wrapping_neg();
} else {
// Set HINIC_VF_FLR_PROC_BIT for the start of FLR
*__slate_slot_0 = ioread32be(((*__slate_slot_1 as *const u8).wrapping_offset(3712 as isize) as *mut ()) as *const ());
*__slate_slot_0 = ((*__slate_slot_0 as u64) | 262144u64) as u32;
iowrite32be(*__slate_slot_0, (*__slate_slot_1 as *const u8).wrapping_offset(3712 as isize) as *mut ());
pcie_flr(pdev);
// The device must recapture its Bus and Device Numbers after FLR
// in order generate Completions.  Issue a config write to let the
// device capture this information.
pci_write_config_word(pdev as *const pci_dev, 0i32, 0u16);
// Firmware clears HINIC_VF_FLR_PROC_BIT when reset is complete
*__slate_slot_2 = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(15000u32));
'__join_0: {
loop {
*__slate_slot_0 = ioread32be(((*__slate_slot_1 as *const u8).wrapping_offset(3712 as isize) as *mut ()) as *const ());
if (*__slate_slot_0 as u64) & 262144u64 == 0 {
break '__join_0;
} else {
msleep(20u32);
if !time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), *__slate_slot_2) {
break;
}
}
}
*__slate_slot_0 = ioread32be(((*__slate_slot_1 as *const u8).wrapping_offset(3712 as isize) as *mut ()) as *const ());
if (*__slate_slot_0 as u64) & 262144u64 != 0 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Reset dev timeout, FLR ack reg: %#010x\n".as_ptr() as *const u8, *__slate_slot_0);
}
}
pci_iounmap(pdev, *__slate_slot_1);
return 0;
}
}
}
}
}

static pci_dev_reset_methods: __SlateAlign16<[pci_dev_reset_methods; 10]> = __SlateAlign16([{
let mut __t0: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t0.vendor = 32902u16;
__t0.device = 4333u16;
__t0.reset = Some(reset_intel_82599_sfp_virtfn);
__t0
}, {
let mut __t1: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t1.vendor = 32902u16;
__t1.device = 342u16;
__t1.reset = Some(reset_ivb_igd);
__t1
}, {
let mut __t2: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t2.vendor = 32902u16;
__t2.device = 358u16;
__t2.reset = Some(reset_ivb_igd);
__t2
}, {
let mut __t3: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t3.vendor = 5197u16;
__t3.device = 43012u16;
__t3.reset = Some(nvme_disable_and_flr);
__t3
}, {
let mut __t4: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t4.vendor = 32902u16;
__t4.device = 2387u16;
__t4.reset = Some(delay_250ms_after_flr);
__t4
}, {
let mut __t5: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t5.vendor = 32902u16;
__t5.device = 2644u16;
__t5.reset = Some(delay_250ms_after_flr);
__t5
}, {
let mut __t6: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t6.vendor = 606u16;
__t6.device = 61868u16;
__t6.reset = Some(delay_250ms_after_flr);
__t6
}, {
let mut __t7: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t7.vendor = 5157u16;
__t7.device = 65535u16;
__t7.reset = Some(reset_chelsio_generic_dev);
__t7
}, {
let mut __t8: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t8.vendor = 6629u16;
__t8.device = 14174u16;
__t8.reset = Some(reset_hinic_vf_dev);
__t8
}, {
let mut __t9: pci_dev_reset_methods = unsafe { core::mem::zeroed() };
__t9.vendor = 0u16;
__t9
}]);

fn __pci_dev_specific_reset(dev: *mut pci_dev, probe: bool, i: *const pci_dev_reset_methods) -> i32 {
let mut ret: i32 = unsafe { pci_dev_reset_iommu_prepare(dev) };
if ret != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, ret) };
return ret;
}
ret = unsafe { unsafe { (*i).reset }.unwrap()(dev, probe) };
unsafe { pci_dev_reset_iommu_done(dev) };
ret
}

/// These device-specific reset methods are here rather than in a driver
/// because when a host assigns a device to a guest VM, the host may need
/// to reset the device but probably doesn't have a driver for it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_dev_specific_reset")]
extern "C" fn pci_dev_specific_reset(dev: *mut pci_dev, probe: bool) -> i32 {
let mut i: *const pci_dev_reset_methods = unsafe { core::mem::zeroed() };
i = unsafe { core::ptr::addr_of!(pci_dev_reset_methods.0) as *const pci_dev_reset_methods };
while (unsafe { (*i).reset }) != None {
if (((unsafe { (*i).vendor }) as i32) == ((unsafe { (*dev).vendor }) as i32) || ((unsafe { (*i).vendor }) as i32) == 65535i32) && (((unsafe { (*i).device }) as i32) == ((unsafe { (*dev).device }) as i32) || ((unsafe { (*i).device }) as i32) == 65535i32) {
return __pci_dev_specific_reset(dev, probe, i);
}
let __v1: *const pci_dev_reset_methods = i;
i = __v1.wrapping_offset(1 as isize);
}
25i32.wrapping_neg()
}

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_dma_func0_alias")]
extern "C" fn quirk_dma_func0_alias(dev: *mut pci_dev) {
if PCI_FUNC!(unsafe { (*dev).devfn }) != 0 {
unsafe { pci_add_dma_alias(dev, PCI_DEVFN__2!(PCI_SLOT!(unsafe { (*dev).devfn }), 0i32) as u8, 1u32) };
}
}

/// https://bugzilla.redhat.com/show_bug.cgi?id=605888
///
/// Some Ricoh devices use function 0 as the PCIe requester ID for DMA.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func0_alias_1175: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func0_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func0_alias_1176: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func0_alias)) };

/// Some Glenfly chips use function 0 as the PCIe Requester ID for DMA
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func0_alias_1177: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func0_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func0_alias_1178: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func0_alias)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_dma_func1_alias")]
extern "C" fn quirk_dma_func1_alias(dev: *mut pci_dev) {
if PCI_FUNC!(unsafe { (*dev).devfn }) != 1 {
unsafe { pci_add_dma_alias(dev, PCI_DEVFN__2!(PCI_SLOT!(unsafe { (*dev).devfn }), 1i32) as u8, 1u32) };
}
}

/// Marvell 88SE9123 uses function 1 as the requester ID for DMA.  In some
/// SKUs function 1 is present and is a legacy IDE controller, in other
/// SKUs this function is not present, making this a ghost requester.
/// https://bugzilla.kernel.org/show_bug.cgi?id=42679
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1179: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1180: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c136
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1181: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1182: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c14
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1183: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1184: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c47 + c57
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1185: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c59
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1186: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c78
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1187: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c134
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1188: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c46
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1189: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c135
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1190: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c127
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1191: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c49
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1192: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1193: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1194: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1195: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugs.gentoo.org/show_bug.cgi?id=497630
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1196: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

/// https://bugzilla.kernel.org/show_bug.cgi?id=42679#c117
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_dma_func1_alias_1197: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_dma_func1_alias)) };

// Lite-On
// Plextor M6E (Marvell 88SS9183)
/// Some devices DMA with the wrong devfn, not just the wrong function.
/// quirk_fixed_dma_alias() uses this table to create fixed aliases, where
/// the alias is "fixed" and independent of the device devfn.
///
/// For example, the Adaptec 3405 is a PCIe card with an Intel 80333 I/O
/// processor.  To software, this appears as a PCIe-to-PCI/X bridge with a
/// single device on the secondary bus.  In reality, the single exposed
/// device at 0e.0 is the Address Translation Unit (ATU) of the controller
/// that provides a bridge to the internal bus of the I/O processor.  The
/// controller supports private devices, which can be hidden from PCI config
/// space.  In the case of the Adaptec 3405, a private device at 01.0
/// appears to be the DMA engine, which therefore needs to become a DMA
/// alias for the device.
/// Adaptec 3405
/// Adaptec 3805
static fixed_dma_alias_tbl: __SlateAlign16<[pci_device_id; 3]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 36869u32;
__t0.device = 645u32;
__t0.subvendor = 36869u32;
__t0.subdevice = 699u32;
__t0.driver_data = 8u64;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1.vendor = 36869u32;
__t1.device = 645u32;
__t1.subvendor = 36869u32;
__t1.subdevice = 700u32;
__t1.driver_data = 8u64;
__t1
}, {
let mut __t2: pci_device_id = unsafe { core::mem::zeroed() };
__t2.vendor = 0u32;
__t2
}]);

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_fixed_dma_alias")]
extern "C" fn quirk_fixed_dma_alias(dev: *mut pci_dev) {
let id: *const pci_device_id = unsafe { pci_match_id(core::ptr::addr_of!(fixed_dma_alias_tbl.0) as *const pci_device_id, dev) };
if !id.is_null() {
unsafe { pci_add_dma_alias(dev, (*id).driver_data as u8, 1u32) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_fixed_dma_alias_1198: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_fixed_dma_alias)) };

/// A few PCIe-to-PCI bridges fail to expose a PCIe capability, resulting in
/// using the wrong DMA alias for the device.  Some of these devices can be
/// used as either forward or reverse bridges, so we need to test whether the
/// device is operating in the correct mode.  We could probably apply this
/// quirk to PCI_ANY_ID, but for now we'll just use known offenders.  The test
/// is for a non-root, non-PCIe bridge where the upstream device is PCIe and
/// is not a PCIe-to-PCI bridge, then @pdev is actually a PCIe-to-PCI bridge.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_use_pcie_bridge_dma_alias")]
extern "C" fn quirk_use_pcie_bridge_dma_alias(pdev: *mut pci_dev) {
let __v0: bool = !pci_is_root_bus(unsafe { (*pdev).bus }) && ((unsafe { (*pdev).hdr_type }) as i32) == 1 && !pci_is_pcie(pdev);
let __v1: bool = __v0 && pci_is_pcie(unsafe { (*(*pdev).bus).self_ });
let __v2: bool = __v1 && pci_pcie_type((unsafe { (*(*pdev).bus).self_ }) as *const pci_dev) != 7;
if __v2 {
unsafe {
(*pdev).dev_flags = (((*pdev).dev_flags as i32) | 32) as u16;
}
}
}

/// ASM1083/1085, https://bugzilla.kernel.org/show_bug.cgi?id=44881#c46
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_use_pcie_bridge_dma_alias_1199: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_use_pcie_bridge_dma_alias)) };

/// Tundra 8113, https://bugzilla.kernel.org/show_bug.cgi?id=44881#c43
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_use_pcie_bridge_dma_alias_1200: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_use_pcie_bridge_dma_alias)) };

/// ITE 8892, https://bugzilla.kernel.org/show_bug.cgi?id=73551
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_use_pcie_bridge_dma_alias_1201: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_use_pcie_bridge_dma_alias)) };

/// ITE 8893 has the same problem as the 8892
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_use_pcie_bridge_dma_alias_1202: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_use_pcie_bridge_dma_alias)) };

/// Intel 82801, https://bugzilla.kernel.org/show_bug.cgi?id=44881#c49
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_use_pcie_bridge_dma_alias_1203: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_use_pcie_bridge_dma_alias)) };

/// MIC x200 NTB forwards PCIe traffic using multiple alien RIDs. They have to
/// be added as aliases to the DMA device in order to allow buffer access
/// when IOMMU is enabled. Following devfns have to match RIT-LUT table
/// programmed in the EEPROM.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_mic_x200_dma_alias")]
extern "C" fn quirk_mic_x200_dma_alias(pdev: *mut pci_dev) {
unsafe { pci_add_dma_alias(pdev, PCI_DEVFN!(16i32, 0i32) as u8, 1u32) };
unsafe { pci_add_dma_alias(pdev, PCI_DEVFN!(17i32, 0i32) as u8, 1u32) };
unsafe { pci_add_dma_alias(pdev, PCI_DEVFN!(18i32, 3i32) as u8, 1u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_mic_x200_dma_alias_1204: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_mic_x200_dma_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_mic_x200_dma_alias_1205: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_mic_x200_dma_alias)) };

/// Intel Visual Compute Accelerator (VCA) is a family of PCIe add-in devices
/// exposing computational units via Non Transparent Bridges (NTB, PEX 87xx).
///
/// Similarly to MIC x200, we need to add DMA aliases to allow buffer access
/// when IOMMU is enabled.  These aliases allow computational unit access to
/// host memory.  These aliases mark the whole VCA device as one IOMMU
/// group.
///
/// All possible slot numbers (0x20) are used, since we are unable to tell
/// what slot is used on other side.  This quirk is intended for both host
/// and computational unit sides.  The VCA devices have up to five functions
/// (four for DMA channels and one additional).
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_pex_vca_alias")]
extern "C" fn quirk_pex_vca_alias(pdev: *mut pci_dev) {
let num_pci_slots: u32 = 32;
for slot in 0..num_pci_slots {
unsafe { pci_add_dma_alias(pdev, PCI_DEVFN__2!(slot, 0i32) as u8, 5u32) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1206: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1207: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1208: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1209: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1210: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_pex_vca_alias_1211: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_pex_vca_alias)) };

/// The IOMMU and interrupt controller on Broadcom Vulcan/Cavium ThunderX2 are
/// associated not at the root bus, but at a bridge below. This quirk avoids
/// generating invalid DMA aliases.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_bridge_cavm_thrx2_pcie_root")]
extern "C" fn quirk_bridge_cavm_thrx2_pcie_root(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 512) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_bridge_cavm_thrx2_pcie_root_1212: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_bridge_cavm_thrx2_pcie_root)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_bridge_cavm_thrx2_pcie_root_1213: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_bridge_cavm_thrx2_pcie_root)) };

/// AST1150 doesn't use a real PCI bus and always forwards the requester ID
/// from downstream devices.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_aspeed_pci_bridge_no_alias")]
extern "C" fn quirk_aspeed_pci_bridge_no_alias(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 16384) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_aspeed_pci_bridge_no_alias_1214: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_aspeed_pci_bridge_no_alias)) };

/// Intersil/Techwell TW686[4589]-based video capture cards have an empty (zero)
/// class code.  Fix it.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_tw686x_class")]
extern "C" fn quirk_tw686x_class(pdev: *mut pci_dev) {
let class: u32 = unsafe { (*pdev).class };
// Use "Multimedia controller" class
unsafe {
(*pdev).class = 294913u32;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"TW686x PCI class overridden (%#08x -> %#08x)\n".as_ptr() as *const u8, class, (*pdev).class) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tw686x_class_1215: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tw686x_class)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tw686x_class_1216: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tw686x_class)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tw686x_class_1217: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tw686x_class)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_tw686x_class_1218: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_tw686x_class)) };

/// Some devices have problems with Transaction Layer Packets with the Relaxed
/// Ordering Attribute set.  Such devices should mark themselves and other
/// device drivers should check before sending TLPs with RO set.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_relaxedordering_disable")]
extern "C" fn quirk_relaxedordering_disable(dev: *mut pci_dev) {
unsafe {
(*dev).dev_flags = (((*dev).dev_flags as i32) | 2048) as u16;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disable Relaxed Ordering Attributes to avoid PCIe Completion erratum\n".as_ptr() as *const u8) };
}

/// Intel Xeon processors based on Broadwell/Haswell microarchitecture Root
/// Complex have a Flow Control Credit issue which can cause performance
/// problems with Upstream Transaction Layer Packets with Relaxed Ordering set.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1219: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1220: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1221: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1222: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1223: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1224: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1225: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1226: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1227: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1228: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1229: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1230: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1231: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1232: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1233: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1234: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1235: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1236: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1237: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1238: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1239: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1240: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1241: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1242: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1243: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1244: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1245: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1246: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

/// The AMD ARM A1100 (aka "SEATTLE") SoC has a bug in its PCIe Root Complex
/// where Upstream Transaction Layer Packets with the Relaxed Ordering
/// Attribute clear are allowed to bypass earlier TLPs with Relaxed Ordering
/// set.  This is a violation of the PCIe 3.0 Transaction Ordering Rules
/// outlined in Section 2.4.1 (PCI Express(r) Base Specification Revision 3.0
/// November 10, 2010).  As a result, on this platform we can't use Relaxed
/// Ordering for Upstream TLPs.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1247: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1248: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_pci_fixup_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section\t\".initcall5s.init\", \"a\"\t\t\n__initcall__kmod_quirks__681_292_pci_apply_final_quirks5s:\t\t\t\n.long\t{1} - .\t\n.previous\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short (~0), (~0)\t\t\t\n.long 0x0600, 8\t\t\n.long {2} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0x5a44\t\t\t\n.long (~0), 0\t\t\n.long pci_disable_parity - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0x5a46\t\t\t\n.long (~0), 0\t\t\n.long pci_disable_parity - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1237\t\t\t\n.long (~0), 0\t\t\n.long {3} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1237\t\t\t\n.long (~0), 0\t\t\n.long {3} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x27bc\t\t\t\n.long (~0), 0\t\t\n.long {4} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x5597\t\t\t\n.long (~0), 0\t\t\n.long {5} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0496\t\t\t\n.long (~0), 0\t\t\n.long {5} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7454\t\t\t\n.long (~0), 0\t\t\n.long {6} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x122d\t\t\t\n.long (~0), 0\t\t\n.long {7} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7030\t\t\t\n.long (~0), 0\t\t\n.long {7} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1250\t\t\t\n.long (~0), 0\t\t\n.long {7} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7100\t\t\t\n.long (~0), 0\t\t\n.long {7} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0305\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x8391\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3112\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0305\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x8391\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3112\t\t\t\n.long (~0), 0\t\t\n.long {8} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0597\t\t\t\n.long (~0), 0\t\t\n.long {9} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0576\t\t\t\n.long (~0), 0\t\t\n.long {10} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, 0x1647\t\t\t\n.long (~0), 0\t\t\n.long {11} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, 0x1651\t\t\t\n.long (~0), 0\t\t\n.long {11} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1237\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7180\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7181\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7190\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7191\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7192\t\t\t\n.long (~0), 0\t\t\n.long {12} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1014, 0x028C\t\t\t\n.long (~0), 0\t\t\n.long {13} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19ee, 0x4000\t\t\t\n.long (~0), 0\t\t\n.long {14} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19ee, 0x6000\t\t\t\n.long (~0), 0\t\t\n.long {14} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19ee, 0x5000\t\t\t\n.long (~0), 0\t\t\n.long {14} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19ee, 0x6003\t\t\t\n.long (~0), 0\t\t\n.long {14} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1014, 0x034a\t\t\t\n.long (~0), 0\t\t\n.long {15} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x5333, 0x8880\t\t\t\n.long (~0), 0\t\t\n.long {16} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x5333, 0x88f0\t\t\t\n.long (~0), 0\t\t\n.long {16} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x2090\t\t\t\n.long (~0), 0\t\t\n.long {17} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0xcab0\t\t\t\n.long (~0), 0\t\t\n.long {18} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7912\t\t\t\n.long (~0), 0\t\t\n.long {19} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x163a\t\t\t\n.long (~0), 0\t\t\n.long {19} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x16c3, (~0)\t\t\t\n.long 0x0c0330, 0\t\t\n.long {20} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, 0x7101\t\t\t\n.long (~0), 0\t\t\n.long {21} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7113\t\t\t\n.long (~0), 0\t\t\n.long {22} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x719b\t\t\t\n.long (~0), 0\t\t\n.long {22} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2410\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2420\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2440\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x244c\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2480\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x248c\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24c0\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24cc\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24d0\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25a1\t\t\t\n.long (~0), 0\t\t\n.long {23} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2640\t\t\t\n.long (~0), 0\t\t\n.long {24} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2641\t\t\t\n.long (~0), 0\t\t\n.long {24} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x27b8\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x27b9\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x27bd\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2810\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2812\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2814\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2811\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2815\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2912\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2914\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2916\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2918\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3a16\t\t\t\n.long (~0), 0\t\t\n.long {25} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3040\t\t\t\n.long (~0), 0\t\t\n.long {26} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3057\t\t\t\n.long (~0), 0\t\t\n.long {27} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3177\t\t\t\n.long (~0), 0\t\t\n.long {28} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x104c, 0x8231\t\t\t\n.long (~0), 0\t\t\n.long {29} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0686\t\t\t\n.long (~0), 0\t\t\n.long {30} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0686\t\t\t\n.long (~0), 0\t\t\n.long {30} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3227\t\t\t\n.long (~0), 0\t\t\n.long {31} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3227\t\t\t\n.long (~0), 0\t\t\n.long {31} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7410\t\t\t\n.long (~0), 0\t\t\n.long {32} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7450\t\t\t\n.long (~0), 0\t\t\n.long {33} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3040\t\t\t\n.long (~0), 0\t\t\n.long {34} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3057\t\t\t\n.long (~0), 0\t\t\n.long {34} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0686\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x8231\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3074\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3147\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3109\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3177\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3227\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3337\t\t\t\n.long (~0), 0\t\t\n.long {35} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {36} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0597\t\t\t\n.long (~0), 0\t\t\n.long {37} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short (~0), (~0)\t\t\t\n.long 0x0607, 8\t\t\n.long {38} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short (~0), (~0)\t\t\t\n.long 0x0607, 8\t\t\n.long {38} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x700C\t\t\t\n.long (~0), 0\t\t\n.long {39} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x700C\t\t\t\n.long (~0), 0\t\t\n.long {39} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x5544, 0x0001\t\t\t\n.long (~0), 0\t\t\n.long {40} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x124b\t\t\t\n.long (~0), 0\t\t\n.long {41} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1179, 0x605\t\t\t\n.long (~0), 0\t\t\n.long {41} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2e21\t\t\t\n.long (~0), 0\t\t\n.long {42} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1078, 0x0001\t\t\t\n.long (~0), 0\t\t\n.long {43} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1078, 0x0001\t\t\t\n.long (~0), 0\t\t\n.long {43} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x84cb\t\t\t\n.long (~0), 0\t\t\n.long {44} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x84cb\t\t\t\n.long (~0), 0\t\t\n.long {44} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4380\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4380\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4390\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4390\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7800\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7800\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7900\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7900\t\t\t\n.long (~0), 0\t\t\n.long {45} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0212\t\t\t\n.long (~0), 0\t\t\n.long {46} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x248a\t\t\t\n.long (~0), 0\t\t\n.long {47} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, (~0)\t\t\t\n.long 0x0101, 8\t\t\n.long {48} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, (~0)\t\t\t\n.long 0x0101, 8\t\t\n.long {48} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, (~0)\t\t\t\n.long 0x0101, 8\t\t\n.long {48} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, (~0)\t\t\t\n.long 0x0101, 8\t\t\n.long {48} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0482\t\t\t\n.long (~0), 0\t\t\n.long {49} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1a30\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2560\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2530\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2570\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2578\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x255d\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x254c\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3340\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3580\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2590\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x7123\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24c2\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1132\t\t\t\n.long (~0), 0\t\t\n.long {50} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2410\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24c0\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2440\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2480\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x248c\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24cc\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24d0\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2410\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24c0\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2440\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2480\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x248c\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24cc\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x24d0\t\t\t\n.long (~0), 0\t\t\n.long {51} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2641\t\t\t\n.long (~0), 0\t\t\n.long {52} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_suspend, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2641\t\t\t\n.long (~0), 0\t\t\n.long {53} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2641\t\t\t\n.long (~0), 0\t\t\n.long {54} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2641\t\t\t\n.long (~0), 0\t\t\n.long {55} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0961\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0962\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0963\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0018\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0961\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0962\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0963\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0018\t\t\t\n.long (~0), 0\t\t\n.long {56} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0008\t\t\t\n.long (~0), 0\t\t\n.long {57} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0008\t\t\t\n.long (~0), 0\t\t\n.long {57} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3227\t\t\t\n.long (~0), 0\t\t\n.long {58} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x3227\t\t\t\n.long (~0), 0\t\t\n.long {58} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2360\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2361\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2362\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2363\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2364\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2365\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2366\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2368\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2369\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2360\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2361\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2362\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2363\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2364\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2365\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2366\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2368\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2369\t\t\t\n.long (~0), 0\t\t\n.long {59} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, (~0)\t\t\t\n.long 0x0101, 8\t\t\n.long {60} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, (~0)\t\t\t\n.long 0x010601, 0\t\t\n.long {60} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2362\t\t\t\n.long (~0), 0\t\t\n.long {60} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x236f\t\t\t\n.long (~0), 0\t\t\n.long {60} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0008\t\t\t\n.long (~0), 0\t\t\n.long {61} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4386\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4387\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4388\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4389\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x438a\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x438b\t\t\t\n.long (~0), 0\t\t\n.long {62} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3590\t\t\t\n.long (~0), 0\t\t\n.long {63} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3592\t\t\t\n.long (~0), 0\t\t\n.long {63} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x359e\t\t\t\n.long (~0), 0\t\t\n.long {63} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0x1610\t\t\t\n.long 0x0604, 8\t\t\n.long {63} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa250\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa251\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa255\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa256\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa258\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0xa259\t\t\t\n.long (~0), 0\t\t\n.long {64} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0320\t\t\t\n.long (~0), 0\t\t\n.long {65} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0321\t\t\t\n.long (~0), 0\t\t\n.long {65} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0329\t\t\t\n.long (~0), 0\t\t\n.long {65} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032a\t\t\t\n.long (~0), 0\t\t\n.long {65} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032c\t\t\t\n.long (~0), 0\t\t\n.long {65} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e2\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e3\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e4\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e5\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e6\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e7\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f7\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f8\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f9\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25fa\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2601\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2602\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2603\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2604\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2605\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2606\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2607\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2608\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2609\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x260a\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x260b\t\t\t\n.long (~0), 0\t\t\n.long {66} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x6741\t\t\t\n.long (~0), 0\t\t\n.long {67} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x0403, 8\t\t\n.long {68} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x15e0\t\t\t\n.long (~0), 0\t\t\n.long {69} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x15e1\t\t\t\n.long (~0), 0\t\t\n.long {69} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1639\t\t\t\n.long (~0), 0\t\t\n.long {69} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0370\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0372\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2670\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0329\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032a\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032c\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0330\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0332\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0370\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0372\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2670\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0329\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032a\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x032c\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0330\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0332\t\t\t\n.long (~0), 0\t\t\n.long {70} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25ac\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25ac\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3c28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0e28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2034\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3c28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0e28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f28\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2034\t\t\t\n.long (~0), 0\t\t\n.long {71} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0205\t\t\t\n.long (~0), 0\t\t\n.long {72} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0205\t\t\t\n.long (~0), 0\t\t\n.long {72} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7450\t\t\t\n.long (~0), 0\t\t\n.long {73} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7450\t\t\t\n.long (~0), 0\t\t\n.long {73} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7458\t\t\t\n.long (~0), 0\t\t\n.long {73} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7458\t\t\t\n.long (~0), 0\t\t\n.long {73} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x746b\t\t\t\n.long (~0), 0\t\t\n.long {74} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x746b\t\t\t\n.long (~0), 0\t\t\n.long {74} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x102f, 0x0105\t\t\t\n.long (~0), 0\t\t\n.long {75} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b5, 0x9050\t\t\t\n.long (~0), 0\t\t\n.long {76} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1402, 0x2000\t\t\t\n.long (~0), 0\t\t\n.long {76} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1402, 0x2600\t\t\t\n.long (~0), 0\t\t\n.long {76} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x9710, (~0)\t\t\t\n.long 0x0700, 8\t\t\n.long {77} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, (~0)\t\t\t\n.long 0x0200, 8\t\t\n.long {78} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10a7\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10a9\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10b6\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10c6\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10c7\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10c8\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10d6\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10db\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10dd\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10e1\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10ec\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10f1\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x10f4\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1508\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10ec, 0x525a\t\t\t\n.long (~0), 0\t\t\n.long {79} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b21, 0x1080\t\t\t\n.long (~0), 0\t\t\n.long {80} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1957, 0x0451\t\t\t\n.long (~0), 0\t\t\n.long {80} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1959, 0xa002\t\t\t\n.long (~0), 0\t\t\n.long {80} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x19e5, 0x1105\t\t\t\n.long (~0), 0\t\t\n.long {80} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0xe110\t\t\t\n.long (~0), 0\t\t\n.long {81} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0xe111\t\t\t\n.long (~0), 0\t\t\n.long {81} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0xe130\t\t\t\n.long (~0), 0\t\t\n.long {81} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1000, 0x0001\t\t\t\n.long (~0), 0\t\t\n.long {82} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1460\t\t\t\n.long (~0), 0\t\t\n.long {83} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x005d\t\t\t\n.long (~0), 0\t\t\n.long {84} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x005d\t\t\t\n.long (~0), 0\t\t\n.long {84} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x324e\t\t\t\n.long (~0), 0\t\t\n.long {85} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x1657\t\t\t\n.long (~0), 0\t\t\n.long {86} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2570\t\t\t\n.long (~0), 0\t\t\n.long {87} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2578\t\t\t\n.long (~0), 0\t\t\n.long {87} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0017\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x5a33\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x5950\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0336\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0351\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0364\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0x0204\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1039, 0x0761\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x144d, 0xa5e3\t\t\t\n.long (~0), 0\t\t\n.long {88} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7450\t\t\t\n.long (~0), 0\t\t\n.long {89} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1106, 0xa238\t\t\t\n.long (~0), 0\t\t\n.long {89} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x5a3f\t\t\t\n.long (~0), 0\t\t\n.long {89} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x17f3, 0x1031\t\t\t\n.long (~0), 0\t\t\n.long {89} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x9600\t\t\t\n.long (~0), 0\t\t\n.long {90} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x9601\t\t\t\n.long (~0), 0\t\t\n.long {90} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0132\t\t\t\n.long (~0), 0\t\t\n.long {91} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x005d\t\t\t\n.long (~0), 0\t\t\n.long {92} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0036\t\t\t\n.long (~0), 0\t\t\n.long {93} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7458\t\t\t\n.long (~0), 0\t\t\n.long {93} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0373\t\t\t\n.long (~0), 0\t\t\n.long {94} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x1ad0\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x1ad1\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x1ad2\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0bf0\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0bf1\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0e1c\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0e1d\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0e12\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0e13\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0fae\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0faf\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x10e5\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x10e6\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x229a\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x229c\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x229e\t\t\t\n.long 0x0604, 8\t\t\n.long {95} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0360\t\t\t\n.long (~0), 0\t\t\n.long {96} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0364\t\t\t\n.long (~0), 0\t\t\n.long {96} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {97} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b9, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {97} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {98} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {98} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x166a\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x166b\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x1668\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x1669\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x1678\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x1679\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4390\t\t\t\n.long (~0), 0\t\t\n.long {100} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4391\t\t\t\n.long (~0), 0\t\t\n.long {100} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4392\t\t\t\n.long (~0), 0\t\t\n.long {100} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4393\t\t\t\n.long (~0), 0\t\t\n.long {100} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4394\t\t\t\n.long (~0), 0\t\t\n.long {100} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4373\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4374\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x4375\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1062\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1063\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x2060\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x2062\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1073\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1083\t\t\t\n.long (~0), 0\t\t\n.long {99} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1090\t\t\t\n.long (~0), 0\t\t\n.long {101} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x1091\t\t\t\n.long (~0), 0\t\t\n.long {101} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x10a0\t\t\t\n.long (~0), 0\t\t\n.long {101} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0x10a1\t\t\t\n.long (~0), 0\t\t\n.long {101} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1969, 0xe091\t\t\t\n.long (~0), 0\t\t\n.long {101} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1c36, 0x0031\t\t\t\n.long 0x0604, 8\t\t\n.long {102} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x3388, 0x0020\t\t\t\n.long (~0), 0\t\t\n.long {103} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x342e\t\t\t\n.long (~0), 0\t\t\n.long {104} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x3c28\t\t\t\n.long (~0), 0\t\t\n.long {104} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x104c, 0xb800\t\t\t\n.long 0x0000, 8\t\t\n.long {105} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1924, 0x0703\t\t\t\n.long (~0), 0\t\t\n.long {106} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1924, 0x6703\t\t\t\n.long (~0), 0\t\t\n.long {106} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1924, 0x0710\t\t\t\n.long (~0), 0\t\t\n.long {106} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b21, 0x0612\t\t\t\n.long (~0), 0\t\t\n.long {106} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25c0\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25d0\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25d4\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25d8\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e2\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e3\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e4\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e5\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e6\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25e7\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f7\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f8\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25f9\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x25fa\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65c0\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e2\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e3\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e4\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e5\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e6\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65e7\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65f7\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65f8\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65f9\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x65fa\t\t\t\n.long (~0), 0\t\t\n.long {107} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0e08\t\t\t\n.long (~0), 0\t\t\n.long {108} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0e0d\t\t\t\n.long (~0), 0\t\t\n.long {108} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0042\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0046\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x004a\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0102\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0106\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x010a\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0152\t\t\t\n.long (~0), 0\t\t\n.long {109} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0412\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0c00\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x0c0c\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c02\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c18\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c1c\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c20\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c22\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c26\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c2d\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c31\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c3a\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c3d\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x8c4e\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2280\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2298\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x229c\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22b0\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22b5\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22b7\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22b8\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22d8\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x22dc\t\t\t\n.long (~0), 0\t\t\n.long {110} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1425, 0x0030\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1814, 0x0601\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b7c, 0x0004\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1102, 0x000b\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10ec, 0x8169\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1572\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1574\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1580\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1581\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1583\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1584\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1585\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1586\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1587\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1588\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1589\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x158a\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x158b\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x37d0\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x37d1\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x37d2\t\t\t\n.long (~0), 0\t\t\n.long {111} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {112} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x22CE\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x22D0\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {114} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x0030\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x0032\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x003c\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x0033\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x0034\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x168c, 0x003e\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x17cb, 0x1103\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x17cb, 0x1107\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x17cb, 0x0308\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x177d, 0xa100\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x104c, 0xb005\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b21, 0x1164\t\t\t\n.long (~0), 0\t\t\n.long {113} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, (~0)\t\t\t\n.long 0x0300, 8\t\t\n.long {115} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0xcb84\t\t\t\n.long (~0), 0\t\t\n.long {115} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0xcf6c\t\t\t\n.long (~0), 0\t\t\n.long {115} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0xcf70\t\t\t\n.long (~0), 0\t\t\n.long {115} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x15b3, 0xcf80\t\t\t\n.long (~0), 0\t\t\n.long {115} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1513\t\t\t\n.long (~0), 0\t\t\n.long {116} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x151a\t\t\t\n.long (~0), 0\t\t\n.long {116} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x151b\t\t\t\n.long (~0), 0\t\t\n.long {116} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1547\t\t\t\n.long (~0), 0\t\t\n.long {116} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1549\t\t\t\n.long (~0), 0\t\t\n.long {116} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_suspend_late, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1547\t\t\t\n.long (~0), 0\t\t\n.long {117} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1180, 0xe832\t\t\t\n.long (~0), 0\t\t\n.long {118} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1180, 0xe476\t\t\t\n.long (~0), 0\t\t\n.long {118} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x6766, 0x3d40\t\t\t\n.long (~0), 0\t\t\n.long {118} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x6766, 0x3d41\t\t\t\n.long (~0), 0\t\t\n.long {118} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9120\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9123\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9125\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9128\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9130\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9170\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9172\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x917a\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9182\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9183\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x91a0\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9215\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9220\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9230\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b4b, 0x9235\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1103, 0x0642\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1103, 0x0645\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x197B, 0x2392\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1c28, 0x0122\t\t\t\n.long (~0), 0\t\t\n.long {119} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x9005, 0x0285\t\t\t\n.long (~0), 0\t\t\n.long {120} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b21, 0x1080\t\t\t\n.long (~0), 0\t\t\n.long {121} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10e3, 0x8113\t\t\t\n.long (~0), 0\t\t\n.long {121} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1283, 0x8892\t\t\t\n.long (~0), 0\t\t\n.long {121} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1283, 0x8893\t\t\t\n.long (~0), 0\t\t\n.long {121} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x244e\t\t\t\n.long (~0), 0\t\t\n.long {121} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2260\t\t\t\n.long (~0), 0\t\t\n.long {122} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2264\t\t\t\n.long (~0), 0\t\t\n.long {122} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2954\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2955\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2956\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2958\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2959\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x295A\t\t\t\n.long (~0), 0\t\t\n.long {123} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x9000\t\t\t\n.long (~0), 0\t\t\n.long {124} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14e4, 0x9084\t\t\t\n.long (~0), 0\t\t\n.long {124} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1a03, 0x1150\t\t\t\n.long (~0), 0\t\t\n.long {125} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1797, 0x6864\t\t\t\n.long 0x0000, 8\t\t\n.long {126} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1797, 0x6865\t\t\t\n.long 0x0000, 8\t\t\n.long {126} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1797, 0x6868\t\t\t\n.long 0x0000, 8\t\t\n.long {126} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1797, 0x6869\t\t\t\n.long 0x0000, 8\t\t\n.long {126} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f01\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f02\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f03\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f04\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f05\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f06\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f07\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f08\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f09\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f0a\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f0b\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f0c\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f0d\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x6f0e\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f01\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f02\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f03\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f04\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f05\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f06\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f07\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f08\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f09\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f0a\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f0b\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f0c\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f0d\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x2f0e\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1a00\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1a01\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1a02\t\t\t\n.long 0x0000, 8\t\t\n.long {127} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1425, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {128} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x443\t\t\t\n.long (~0), 0\t\t\n.long {129} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1487\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x148c\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x149c\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x7901\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x1502\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, 0x17f0\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1502\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1503\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14c3, 0x0616\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x14c3, 0x7925\t\t\t\n.long (~0), 0\t\t\n.long {130} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0xd063, 0x1000\t\t\t\n.long (~0), 0\t\t\n.long {131} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x13C1, 0x1004\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x13C1, 0x1005\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0132\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0140\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0141\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0142\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0144\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0420\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1166, 0x0422\t\t\t\n.long (~0), 0\t\t\n.long {132} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x98e4\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x6900\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7310\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7312\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7318\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7319\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x731a\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x731b\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x731e\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x731f\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7340\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7341\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x7347\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x734f\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, 0x15d8\t\t\t\n.long (~0), 0\t\t\n.long {133} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1451\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1452\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1453\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1454\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1455\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1457\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1459\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x145a\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x145c\t\t\t\n.long (~0), 0\t\t\n.long {134} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1957, (~0)\t\t\t\n.long (~0), 0\t\t\n.long {135} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, (~0)\t\t\t\n.long 0x0403, 8\t\t\n.long {136} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1022, (~0)\t\t\t\n.long 0x0403, 8\t\t\n.long {136} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x0403, 8\t\t\n.long {136} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x0c03, 8\t\t\n.long {137} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, (~0)\t\t\t\n.long 0x0c03, 8\t\t\n.long {137} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x0c80, 8\t\t\n.long {138} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1002, (~0)\t\t\t\n.long 0x0c80, 8\t\t\n.long {138} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x03, 16\t\t\n.long {139} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, (~0)\t\t\t\n.long 0x03, 16\t\t\n.long {139} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8531\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8532\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8533\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8534\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8535\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8536\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8541\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8542\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8543\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8544\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8545\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8546\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8551\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8552\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8553\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8554\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8555\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8556\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8561\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8562\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8563\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8564\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8565\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8566\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8571\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8572\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8573\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8574\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8575\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x8576\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4000\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4084\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4068\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4052\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4036\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4028\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4100\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4184\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4168\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4152\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4136\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4128\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4200\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4284\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4268\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4252\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4236\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4228\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4352\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4336\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4328\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4452\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4436\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4428\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4552\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4536\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x4528\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5000\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5084\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5068\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5052\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5036\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5028\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5100\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5184\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5168\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5152\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5136\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5128\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5200\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5284\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5268\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5252\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5236\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5228\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5300\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5384\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5368\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5352\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5336\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5328\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5400\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5484\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5468\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5452\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5436\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5428\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5500\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5584\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5568\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5552\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5536\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x11f8, 0x5528\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1001\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1002\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1003\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1004\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1005\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1006\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x1008\t\t\t\n.long 0x0680, 8\t\t\n.long {140} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b5, 0x87b0\t\t\t\n.long (~0), 0\t\t\n.long {141} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10b5, 0x87b1\t\t\t\n.long (~0), 0\t\t\n.long {141} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x13b1\t\t\t\n.long 0x0300, 8\t\t\n.long {142} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b21, 0x2142\t\t\t\n.long (~0), 0\t\t\n.long {143} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x400e\t\t\t\n.long (~0), 0\t\t\n.long {144} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x400f\t\t\t\n.long (~0), 0\t\t\n.long {144} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1ac1, 0x089a\t\t\t\n.long 0x0000, 8\t\t\n.long {145} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2404\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2404\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2304\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2304\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2303\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2303\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0xb404\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0xb404\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_enable, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2608\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_resume, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x12D8, 0x2608\t\t\t\n.long (~0), 0\t\t\n.long {146} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10de, 0x0ab8\t\t\t\n.long (~0), 0\t\t\n.long {147} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1533\t\t\t\n.long (~0), 0\t\t\n.long {148} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1536\t\t\t\n.long (~0), 0\t\t\n.long {148} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1537\t\t\t\n.long (~0), 0\t\t\n.long {148} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_early, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x1538\t\t\t\n.long (~0), 0\t\t\n.long {148} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f80\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f81\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f82\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f83\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f84\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f85\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f86\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f87\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x4f88\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5690\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5691\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5692\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5693\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5694\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x5695\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a0\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a1\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a2\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a3\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a4\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a5\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56a6\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56b0\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56b1\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56c0\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_header, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x8086, 0x56c1\t\t\t\n.long (~0), 0\t\t\n.long {149} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10ee, 0x5020\t\t\t\n.long (~0), 0\t\t\n.long {150} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x10ee, 0x5021\t\t\t\n.long (~0), 0\t\t\n.long {150} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1b36, 0x0005\t\t\t\n.long (~0), 0\t\t\n.long {150} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x1055, 0x9660\t\t\t\n.long (~0), 0\t\t\n.long {150} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n\n.section .pci_fixup_final, \"a\"\t\t\t\t\n.balign\t16\t\t\t\t\t\n.short 0x5555, 0x0004\t\t\t\n.long (~0), 0\t\t\n.long {151} - .\t\t\t\t\n.previous\t\t\t\t\t\t\n", sym pci_fixup_device, sym pci_apply_final_quirks, sym quirk_mmio_always_on, sym quirk_passive_release, sym quirk_tigerpoint_bm_sts, sym quirk_nopcipci, sym quirk_nopciamd, sym quirk_triton, sym quirk_vialatency, sym quirk_viaetbf, sym quirk_vsfx, sym quirk_alimagik, sym quirk_natoma, sym quirk_citrine, sym quirk_nfp6000, sym quirk_extend_bar_to_page, sym quirk_s3_64M, sym quirk_cs5536_vsa, sym quirk_ati_exploding_mce, sym quirk_amd_dwc_class, sym quirk_synopsys_haps, sym quirk_ali7101_acpi, sym quirk_piix4_acpi, sym quirk_ich4_lpc_acpi, sym quirk_ich6_lpc, sym quirk_ich7_lpc, sym quirk_vt82c586_acpi, sym quirk_vt82c686_acpi, sym quirk_vt8235_acpi, sym quirk_xio2000a, sym quirk_via_ioapic, sym quirk_via_vt8237_bypass_apic_deassert, sym quirk_amd_ioapic, sym quirk_amd_8131_mmrbc, sym quirk_via_acpi, sym quirk_via_bridge, sym quirk_via_vlink, sym quirk_vt82c598_id, sym quirk_cardbus_legacy, sym quirk_amd_ordering, sym quirk_dunord, sym quirk_transparent_bridge, sym quirk_p45_bw_notifications, sym quirk_mediagx_master, sym quirk_disable_pxb, sym quirk_amd_ide_mode, sym quirk_svwks_csb5ide, sym quirk_ide_samemode, sym quirk_no_ata_d3, sym quirk_eisa_bridge, sym asus_hides_smbus_hostbridge, sym asus_hides_smbus_lpc, sym asus_hides_smbus_lpc_ich6, sym asus_hides_smbus_lpc_ich6_suspend, sym asus_hides_smbus_lpc_ich6_resume, sym asus_hides_smbus_lpc_ich6_resume_early, sym quirk_sis_96x_smbus, sym quirk_sis_503, sym asus_hides_ac97_lpc, sym quirk_jmicron_ata, sym quirk_jmicron_async_suspend, sym quirk_alder_ioapic, sym quirk_no_msi, sym quirk_pcie_mch, sym quirk_huawei_pcie_sva, sym quirk_pcie_pxh, sym quirk_intel_pcie_pm, sym quirk_radeon_pm, sym quirk_nvidia_hda_pm, sym quirk_ryzen_xhci_d3hot, sym quirk_reroute_to_boot_interrupts_intel, sym quirk_disable_intel_boot_interrupt, sym quirk_disable_broadcom_boot_interrupt, sym quirk_disable_amd_813x_boot_interrupt, sym quirk_disable_amd_8111_boot_interrupt, sym quirk_tc86c001_ide, sym quirk_plx_pci9050, sym quirk_netmos, sym quirk_e100_interrupt, sym quirk_disable_aspm_l0s, sym quirk_disable_aspm_l0s_l1, sym quirk_enable_clear_retrain_link, sym fixup_rev1_53c810, sym quirk_p64h2_1k_io, sym quirk_nvidia_ck804_pcie_aer_ext_cap, sym quirk_via_cx700_pci_parking_caching, sym quirk_brcm_5719_limit_mrrs, sym quirk_unhide_mch_dev6, sym quirk_disable_all_msi, sym quirk_disable_msi, sym quirk_amd_780_apc_msi, sym quirk_msi_ht_cap, sym quirk_nvidia_ck804_msi_ht_cap, sym ht_enable_msi_mapping, sym nvenet_msi_disable, sym pci_quirk_nvidia_tegra_disable_rp_msi, sym nvbridge_check_legacy_irq_routing, sym nv_msi_ht_cap_quirk_all, sym nv_msi_ht_cap_quirk_leaf, sym quirk_msi_intx_disable_bug, sym quirk_msi_intx_disable_ati_bug, sym quirk_msi_intx_disable_qca_bug, sym quirk_al_msi_disable, sym quirk_hotplug_bridge, sym vtd_mask_spec_errors, sym fixup_ti816x_class, sym fixup_mpss_256, sym quirk_intel_mc_errata, sym quirk_intel_ntb, sym disable_igfx_irq, sym quirk_remove_d3hot_delay, sym quirk_broken_intx_masking, sym mellanox_check_broken_intx_masking, sym quirk_no_bus_reset, sym quirk_nvidia_no_bus_reset, sym quirk_no_pm_reset, sym quirk_thunderbolt_hotplug_msi, sym quirk_apple_poweroff_thunderbolt, sym quirk_dma_func0_alias, sym quirk_dma_func1_alias, sym quirk_fixed_dma_alias, sym quirk_use_pcie_bridge_dma_alias, sym quirk_mic_x200_dma_alias, sym quirk_pex_vca_alias, sym quirk_bridge_cavm_thrx2_pcie_root, sym quirk_aspeed_pci_bridge_no_alias, sym quirk_tw686x_class, sym quirk_relaxedordering_disable, sym quirk_chelsio_T5_disable_root_port_attributes, sym quirk_intel_qat_vf_cap, sym quirk_no_flr, sym quirk_no_flr_snet, sym quirk_no_ext_tags, sym quirk_amd_harvest_no_ats, sym quirk_intel_e2000_no_ats, sym quirk_fsl_no_msi, sym quirk_gpu_hda, sym quirk_gpu_usb, sym quirk_gpu_usb_typec_ucsi, sym quirk_nvidia_hda, sym quirk_switchtec_ntb_dma_alias, sym quirk_plx_ntb_dma_alias, sym quirk_reset_lenovo_thinkpad_p50_nvgpu, sym pci_fixup_no_d0_pme, sym pci_fixup_no_msi_no_pme, sym apex_pci_fixup_class, sym pci_fixup_pericom_acs_store_forward, sym nvidia_ion_ahci_fixup, sym rom_bar_overlap_defect, sym aspm_l1_acceptable_latency, sym of_pci_make_dev_node, sym pci_fixup_d3cold_delay_1sec, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_relaxedordering_disable_1249: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_relaxedordering_disable)) };

/// Per PCIe r3.0, sec 2.2.9, "Completion headers must supply the same
/// values for the Attribute as were supplied in the header of the
/// corresponding Request, except as explicitly allowed when IDO is used."
///
/// If a non-compliant device generates a completion with a different
/// attribute than the request, the receiver may accept it (which itself
/// seems non-compliant based on sec 2.3.2), or it may handle it as a
/// Malformed TLP or an Unexpected Completion, which will probably lead to a
/// device access timeout.
///
/// If the non-compliant device generates completions with zero attributes
/// (instead of copying the attributes from the request), we can work around
/// this by disabling the "Relaxed Ordering" and "No Snoop" attributes in
/// upstream devices so they always generate requests with zero attributes.
///
/// This affects other devices under the same Root Port, but since these
/// attributes are performance hints, there should be no functional problem.
///
/// Note that Configuration Space accesses are never supposed to have TLP
/// Attributes, so we're safe waiting till after any Configuration Space
/// accesses to do the Root Port fixup.
fn quirk_disable_root_port_attributes(pdev: *mut pci_dev) {
let root_port: *mut pci_dev = pcie_find_root_port(pdev);
if root_port.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"PCIe Completion erratum may cause device errors\n".as_ptr() as *const u8) };
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*root_port).dev) as *const device, c"Disabling No Snoop/Relaxed Ordering Attributes to avoid PCIe Completion erratum in %s\n".as_ptr() as *const u8, dev_name(core::ptr::addr_of_mut!((*pdev).dev) as *const device)) };
pcie_capability_clear_word(root_port, 8i32, 2064u16);
}

/// The Chelsio T5 chip fails to copy TLP Attributes from a Request to the
/// Completion it generates.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_chelsio_T5_disable_root_port_attributes")]
extern "C" fn quirk_chelsio_T5_disable_root_port_attributes(pdev: *mut pci_dev) {
// This mask/compare operation selects for Physical Function 4 on a
// T5.  We only need to fix up the Root Port once for any of the
// PFs.  PF[0..3] have PCI Device IDs of 0x50xx, but PF4 is uniquely
// 0x54xx so we use that one.
if ((unsafe { (*pdev).device }) as i32) & 65280 == 21504 {
quirk_disable_root_port_attributes(pdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_chelsio_T5_disable_root_port_attributes_1250: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_chelsio_T5_disable_root_port_attributes)) };

/// pci_acs_ctrl_enabled - compare desired ACS controls with those provided
/// 			  by a device
/// @acs_ctrl_req: Bitmask of desired ACS controls
/// @acs_ctrl_ena: Bitmask of ACS controls enabled or provided implicitly by
/// 		  the hardware design
///
/// Return 1 if all ACS controls in the @acs_ctrl_req bitmask are included
/// in @acs_ctrl_ena, i.e., the device provides all the access controls the
/// caller desires.  Return 0 otherwise.
fn pci_acs_ctrl_enabled(acs_ctrl_req: u16, acs_ctrl_ena: u16) -> i32 {
if (acs_ctrl_req as i32) & (acs_ctrl_ena as i32) == (acs_ctrl_req as i32) {
return 1;
}
0
}

/// AMD has indicated that the devices below do not support peer-to-peer
/// in any system where they are found in the southbridge with an AMD
/// IOMMU in the system.  Multifunction devices that do not support
/// peer-to-peer between functions can claim to support a subset of ACS.
/// Such devices effectively enable request redirect (RR) and completion
/// redirect (CR) since all transactions are redirected to the upstream
/// root complex.
///
/// https://lore.kernel.org/r/201207111426.q6BEQTbh002928@mail.maya.org/
/// https://lore.kernel.org/r/20120711165854.GM25282@amd.com/
/// https://lore.kernel.org/r/20121005130857.GX4009@amd.com/
///
/// 1002:4385 SBx00 SMBus Controller
/// 1002:439c SB7x0/SB8x0/SB9x0 IDE Controller
/// 1002:4383 SBx00 Azalia (Intel HDA)
/// 1002:439d SB7x0/SB8x0/SB9x0 LPC host controller
/// 1002:4384 SBx00 PCI to PCI Bridge
/// 1002:4399 SB7x0/SB8x0/SB9x0 USB OHCI2 Controller
///
/// https://bugzilla.kernel.org/show_bug.cgi?id=81841#c15
///
/// 1022:780f [AMD] FCH PCI Bridge
/// 1022:7809 [AMD] FCH USB OHCI Controller
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_amd_sb_acs")]
extern "C" fn pci_quirk_amd_sb_acs(dev: *mut pci_dev, mut acs_flags: u16) -> i32 {
let mut header: *mut acpi_table_header = core::ptr::null_mut::<acpi_table_header>();
let mut status: u32 = 0;
// Targeting multifunction devices on the SB (appears on root bus)
let __v0: bool = ((unsafe { (*dev).__slate_bits_3.__get_multifunction() }) as i32) == 0 || !pci_is_root_bus(unsafe { (*dev).bus });
if __v0 {
return 19i32.wrapping_neg();
}
// The IVRS table describes the AMD IOMMU
status = unsafe { acpi_get_table(c"IVRS".as_ptr() as *mut u8, 0u32, core::ptr::addr_of_mut!(header)) };
if ACPI_FAILURE!(status) != 0 {
return 19i32.wrapping_neg();
}
unsafe { acpi_put_table(header) };
// Filter out flags not applicable to multifunction
acs_flags = ((acs_flags as i32) & (4 | 8 | 32 | 64)) as u16;
pci_acs_ctrl_enabled(acs_flags, 12u16)
}

fn pci_quirk_cavium_acs_match(dev: *mut pci_dev) -> bool {
let __v0: bool = !pci_is_pcie(dev) || pci_pcie_type(dev as *const pci_dev) != 4;
if __v0 {
return false;
}
match (unsafe { (*dev).device }) as i32 {
40960..=43007 | 44932 | 47236 => {
// Effectively selects all downstream ports for whole ThunderX1
// (which represents 8 SoCs).
return true; // ThunderX1
// ThunderX2
// ThunderX3
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_cavium_acs")]
extern "C" fn pci_quirk_cavium_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
if !pci_quirk_cavium_acs_match(dev) {
return 25i32.wrapping_neg();
}
// Cavium Root Ports don't advertise an ACS capability.  However,
// the RTL internally implements similar protection as if ACS had
// Source Validation, Request Redirection, Completion Redirection,
// and Upstream Forwarding features enabled.  Assert that the
// hardware implements and enables equivalent ACS functionality for
// these flags.
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_xgene_acs")]
extern "C" fn pci_quirk_xgene_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
// X-Gene Root Ports matching this quirk do not allow peer-to-peer
// transactions with others, allowing masking out these bits as if they
// were unimplemented in the ACS capability.
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

/// Many Zhaoxin Root Ports and Switch Downstream Ports have no ACS capability.
/// But the implementation could block peer-to-peer transactions between them
/// and provide ACS-like functionality.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_zhaoxin_pcie_ports_acs")]
extern "C" fn pci_quirk_zhaoxin_pcie_ports_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
let __v0: bool = !pci_is_pcie(dev) || pci_pcie_type(dev as *const pci_dev) != 4 && pci_pcie_type(dev as *const pci_dev) != 6;
if __v0 {
return 25i32.wrapping_neg();
}
// Future Zhaoxin Root Ports and Switch Downstream Ports will
// implement ACS capability in accordance with the PCIe Spec.
match (unsafe { (*dev).device }) as i32 {
1808..=1822 | 1825 | 1827..=1874 => {
return pci_acs_ctrl_enabled(acs_flags, 29u16);
}
_ => {
}
}
0
}

/// Many Intel PCH Root Ports do provide ACS-like features to disable peer
/// transactions and validate bus numbers in requests, but do not provide an
/// actual PCIe ACS capability.  This is the list of device IDs known to fall
/// into that category as provided by Intel in Red Hat bugzilla 1037684.
/// Ibexpeak PCH
/// Cougarpoint PCH
/// Pantherpoint PCH
/// Lynxpoint-H PCH
/// Lynxpoint-LP PCH
/// Wildcat PCH
/// Patsburg (X79) PCH
/// Wellsburg (X99) PCH
/// Lynx Point (9 series) PCH
static pci_quirk_intel_pch_acs_ids: __SlateAlign16<[u16; 119]> = __SlateAlign16([15170u16, 15171u16, 15172u16, 15173u16, 15174u16, 15175u16, 15176u16, 15177u16, 15178u16, 15179u16, 15180u16, 15181u16, 15182u16, 15183u16, 15184u16, 15185u16, 7184u16, 7185u16, 7186u16, 7187u16, 7188u16, 7189u16, 7190u16, 7191u16, 7192u16, 7193u16, 7194u16, 7195u16, 7196u16, 7197u16, 7198u16, 7199u16, 7696u16, 7697u16, 7698u16, 7699u16, 7700u16, 7701u16, 7702u16, 7703u16, 7704u16, 7705u16, 7706u16, 7707u16, 7708u16, 7709u16, 7710u16, 7711u16, 35856u16, 35857u16, 35858u16, 35859u16, 35860u16, 35861u16, 35862u16, 35863u16, 35864u16, 35865u16, 35866u16, 35867u16, 35868u16, 35869u16, 35870u16, 35871u16, 39952u16, 39953u16, 39954u16, 39955u16, 39956u16, 39957u16, 39958u16, 39959u16, 39960u16, 39961u16, 39962u16, 39963u16, 40080u16, 40081u16, 40082u16, 40083u16, 40084u16, 40085u16, 40086u16, 40087u16, 40088u16, 40089u16, 40090u16, 40091u16, 7440u16, 7442u16, 7444u16, 7446u16, 7448u16, 7450u16, 7452u16, 7454u16, 36112u16, 36113u16, 36114u16, 36115u16, 36116u16, 36117u16, 36118u16, 36119u16, 36120u16, 36121u16, 36122u16, 36123u16, 36124u16, 36125u16, 36126u16, 35984u16, 35986u16, 35988u16, 35990u16, 35992u16, 35994u16, 35996u16, 35998u16]);

fn pci_quirk_intel_pch_acs_match(dev: *mut pci_dev) -> bool {
let mut i: i32 = 0;
// Filter out a few obvious non-matches first
let __v0: bool = !pci_is_pcie(dev) || pci_pcie_type(dev as *const pci_dev) != 4;
if __v0 {
return false;
}
while (i as u64) < ARRAY_SIZE__10!() {
if ((unsafe { pci_quirk_intel_pch_acs_ids.0[i as usize] }) as i32) == ((unsafe { (*dev).device }) as i32) {
return true;
}
i = i.wrapping_add(1);
}
false
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_intel_pch_acs")]
extern "C" fn pci_quirk_intel_pch_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
if !pci_quirk_intel_pch_acs_match(dev) {
return 25i32.wrapping_neg();
}
if ((unsafe { (*dev).dev_flags }) as i32) & 8 != 0 {
return pci_acs_ctrl_enabled(acs_flags, 29u16);
}
pci_acs_ctrl_enabled(acs_flags, 0u16)
}

/// These QCOM Root Ports do provide ACS-like features to disable peer
/// transactions and validate bus numbers in requests, but do not provide an
/// actual PCIe ACS capability.  Hardware supports source validation but it
/// will report the issue as Completer Abort instead of ACS Violation.
/// Hardware doesn't support peer-to-peer and each Root Port is a Root
/// Complex with unique segment numbers.  It is not possible for one Root
/// Port to pass traffic to another Root Port.  All PCIe transactions are
/// terminated inside the Root Port.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_qcom_rp_acs")]
extern "C" fn pci_quirk_qcom_rp_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

/// Each of these NXP Root Ports is in a Root Complex with a unique segment
/// number and does provide isolation features to disable peer transactions
/// and validate bus numbers in requests, but does not provide an ACS
/// capability.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_nxp_rp_acs")]
extern "C" fn pci_quirk_nxp_rp_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_al_acs")]
extern "C" fn pci_quirk_al_acs(dev: *mut pci_dev, mut acs_flags: u16) -> i32 {
if pci_pcie_type(dev as *const pci_dev) != 4 {
return 25i32.wrapping_neg();
}
// Amazon's Annapurna Labs root ports don't include an ACS capability,
// but do include ACS-like functionality. The hardware doesn't support
// peer-to-peer transactions via the root port and each has a unique
// segment number.
//
// Additionally, the root ports cannot send traffic to each other.
let __v1: u16 = ((acs_flags as i32) & !(1 | 4 | 8 | 16)) as u16;
acs_flags = __v1;
if acs_flags != 0 { 0 } else { 1 }
}

/// Sunrise Point PCH root ports implement ACS, but unfortunately as shown in
/// the datasheet (Intel 100 Series Chipset Family PCH Datasheet, Vol. 2,
/// 12.1.46, 12.1.47)[1] this chipset uses dwords for the ACS capability and
/// control registers whereas the PCIe spec packs them into words (Rev 3.0,
/// 7.16 ACS Extended Capability).  The bit definitions are correct, but the
/// control register is at offset 8 instead of 6 and we should probably use
/// dword accesses to them.  This applies to the following PCI Device IDs, as
/// found in volume 1 of the datasheet[2]:
///
/// 0xa110-0xa11f Sunrise Point-H PCI Express Root Port #{0-16}
/// 0xa167-0xa16a Sunrise Point-H PCI Express Root Port #{17-20}
///
/// N.B. This doesn't fix what lspci shows.
///
/// The 100 series chipset specification update includes this as errata #23[3].
///
/// The 200 series chipset (Union Point) has the same bug according to the
/// specification update (Intel 200 Series Chipset Family Platform Controller
/// Hub, Specification Update, January 2017, Revision 001, Document# 335194-001,
/// Errata 22)[4].  Per the datasheet[5], root port PCI Device IDs for this
/// chipset include:
///
/// 0xa290-0xa29f PCI Express Root port #{0-16}
/// 0xa2e7-0xa2ee PCI Express Root port #{17-24}
///
/// Mobile chipsets are also affected, 7th & 8th Generation
/// Specification update confirms ACS errata 22, status no fix: (7th Generation
/// Intel Processor Family I/O for U/Y Platforms and 8th Generation Intel
/// Processor Family I/O for U Quad Core Platforms Specification Update,
/// August 2017, Revision 002, Document#: 334660-002)[6]
/// Device IDs from I/O datasheet: (7th Generation Intel Processor Family I/O
/// for U/Y Platforms and 8th Generation Intel ® Processor Family I/O for U
/// Quad Core Platforms, Vol 1 of 2, August 2017, Document#: 334658-003)[7]
///
/// 0x9d10-0x9d1b PCI Express Root port #{1-12}
///
/// [1] https://www.intel.com/content/www/us/en/chipsets/100-series-chipset-datasheet-vol-2.html
/// [2] https://www.intel.com/content/www/us/en/chipsets/100-series-chipset-datasheet-vol-1.html
/// [3] https://www.intel.com/content/www/us/en/chipsets/100-series-chipset-spec-update.html
/// [4] https://www.intel.com/content/www/us/en/chipsets/200-series-chipset-pch-spec-update.html
/// [5] https://www.intel.com/content/www/us/en/chipsets/200-series-chipset-pch-datasheet-vol-1.html
/// [6] https://www.intel.com/content/www/us/en/processors/core/7th-gen-core-family-mobile-u-y-processor-lines-i-o-spec-update.html
/// [7] https://www.intel.com/content/www/us/en/processors/core/7th-gen-core-family-mobile-u-y-processor-lines-i-o-datasheet-vol-1.html
fn pci_quirk_intel_spt_pch_acs_match(dev: *mut pci_dev) -> bool {
let __v0: bool = !pci_is_pcie(dev) || pci_pcie_type(dev as *const pci_dev) != 4;
if __v0 {
return false;
}
match (unsafe { (*dev).device }) as i32 {
41232..=41247 | 41319..=41322 | 41616..=41631 | 41703..=41710 | 40208..=40219 => {
return true; // Sunrise Point
// Union Point
// 7th & 8th Gen Mobile
}
_ => {
}
}
false
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_intel_spt_pch_acs")]
extern "C" fn pci_quirk_intel_spt_pch_acs(dev: *mut pci_dev, mut acs_flags: u16) -> i32 {
let mut pos: i32 = 0;
let mut cap: u32 = 0;
let mut ctrl: u32 = 0;
if !pci_quirk_intel_spt_pch_acs_match(dev) {
return 25i32.wrapping_neg();
}
pos = (unsafe { (*dev).acs_cap }) as i32;
if pos == 0 {
return 25i32.wrapping_neg();
}
// see pci_acs_flags_enabled()
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(cap)) };
acs_flags = (((acs_flags as i32) as u32) & (cap | 32)) as u16;
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(8i32), core::ptr::addr_of_mut!(ctrl)) };
pci_acs_ctrl_enabled(acs_flags, ctrl as u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_mf_endpoint_acs")]
extern "C" fn pci_quirk_mf_endpoint_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
// SV, TB, and UF are not relevant to multifunction endpoints.
//
// Multifunction devices are only required to implement RR, CR, and DT
// in their ACS capability if they support peer-to-peer transactions.
// Devices matching this quirk have been verified by the vendor to not
// perform peer-to-peer with other functions, allowing us to mask out
// these bits as if they were unimplemented in the ACS capability.
pci_acs_ctrl_enabled(acs_flags, 95u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_rciep_acs")]
extern "C" fn pci_quirk_rciep_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
// Intel RCiEP's are required to allow p2p only on translated
// addresses.  Refer to Intel VT-d specification, r3.1, sec 3.16,
// "Root-Complex Peer to Peer Considerations".
if pci_pcie_type(dev as *const pci_dev) != 9 {
return 25i32.wrapping_neg();
}
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_brcm_acs")]
extern "C" fn pci_quirk_brcm_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
// iProc PAXB Root Ports don't advertise an ACS capability, but
// they do not allow peer-to-peer transactions between Root Ports.
// Allow each Root Port to be in a separate IOMMU group by masking
// SV/RR/CR/UF bits.
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_loongson_acs")]
extern "C" fn pci_quirk_loongson_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
// Loongson PCIe Root Ports don't advertise an ACS capability, but
// they do not allow peer-to-peer transactions between Root Ports.
// Allow each Root Port to be in a separate IOMMU group by masking
// SV/RR/CR/UF bits.
pci_acs_ctrl_enabled(acs_flags, 29u16)
}

/// Wangxun 40G/25G/10G/1G NICs have no ACS capability, but on
/// multi-function devices, the hardware isolates the functions by
/// directing all peer-to-peer traffic upstream as though PCI_ACS_RR and
/// PCI_ACS_CR were set.
/// SFxxx 1G NICs(em).
/// RP1000/RP2000 10G NICs(sp).
/// FF5xxx 40G/25G/10G NICs(aml).
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_wangxun_nic_acs")]
extern "C" fn pci_quirk_wangxun_nic_acs(dev: *mut pci_dev, acs_flags: u16) -> i32 {
match (unsafe { (*dev).device }) as i32 {
256..=271 | 4097 | 8193 | 20496 | 20517 | 20544 | 20752 | 20773 | 20800 => {
return pci_acs_ctrl_enabled(acs_flags, 29u16); // EM
// SP
// AML
// AML
}
_ => {
}
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev_acs_enabled {
    vendor: u16,
    device: u16,
    acs_enabled: Option<unsafe extern "C" fn(*mut pci_dev, u16) -> i32>,
}


/// 82580
/// 82576
/// 82575
/// I350
/// 82571 (Quads omitted due to non-ACS switch)
/// I219
/// QCOM QDF2xxx root ports
/// QCOM SA8775P root port
/// QCOM Hamoa root port
/// QCOM Glymur root port
/// HXT SD4800 root ports. The ACS design is same as QCOM QDF2xxx
/// Intel PCH root ports
/// Emulex BE3-R
/// Emulex Skyhawk-R
/// Cavium ThunderX
/// Cavium multi-function devices
/// APM X-Gene
/// Ampere Computing
/// Broadcom multi-function device
/// Loongson PCIe Root Ports
/// Amazon Annapurna Labs
/// Zhaoxin multi-function devices
/// NXP root ports, xx=16, 12, or 08 cores
///
/// LX2xx0A : without security features + CAN-FD
/// LX2xx0C : security features + CAN-FD
/// LX2xx0E : security features + CAN
/// LX2xx0N : without security features + CAN
/// LX2xx2A : without security features + CAN-FD
/// LX2xx2C : security features + CAN-FD
/// LX2xx2E : security features + CAN
/// LX2xx2N : without security features + CAN
/// Zhaoxin Root/Downstream Ports
/// Wangxun nics
static pci_dev_acs_enabled: __SlateAlign16<[pci_dev_acs_enabled; 133]> = __SlateAlign16([{
let mut __t0: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t0.vendor = 4098u16;
__t0.device = 17285u16;
__t0.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t0
}, {
let mut __t1: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t1.vendor = 4098u16;
__t1.device = 17308u16;
__t1.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t1
}, {
let mut __t2: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t2.vendor = 4098u16;
__t2.device = 17283u16;
__t2.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t2
}, {
let mut __t3: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t3.vendor = 4098u16;
__t3.device = 17309u16;
__t3.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t3
}, {
let mut __t4: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t4.vendor = 4098u16;
__t4.device = 17284u16;
__t4.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t4
}, {
let mut __t5: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t5.vendor = 4098u16;
__t5.device = 17305u16;
__t5.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t5
}, {
let mut __t6: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t6.vendor = 4130u16;
__t6.device = 30735u16;
__t6.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t6
}, {
let mut __t7: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t7.vendor = 4130u16;
__t7.device = 30729u16;
__t7.acs_enabled = Some(pci_quirk_amd_sb_acs);
__t7
}, {
let mut __t8: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t8.vendor = 6436u16;
__t8.device = 2307u16;
__t8.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t8
}, {
let mut __t9: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t9.vendor = 6436u16;
__t9.device = 2339u16;
__t9.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t9
}, {
let mut __t10: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t10.vendor = 6436u16;
__t10.device = 2563u16;
__t10.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t10
}, {
let mut __t11: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t11.vendor = 32902u16;
__t11.device = 4294u16;
__t11.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t11
}, {
let mut __t12: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t12.vendor = 32902u16;
__t12.device = 4315u16;
__t12.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t12
}, {
let mut __t13: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t13.vendor = 32902u16;
__t13.device = 4317u16;
__t13.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t13
}, {
let mut __t14: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t14.vendor = 32902u16;
__t14.device = 4321u16;
__t14.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t14
}, {
let mut __t15: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t15.vendor = 32902u16;
__t15.device = 4337u16;
__t15.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t15
}, {
let mut __t16: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t16.vendor = 32902u16;
__t16.device = 4343u16;
__t16.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t16
}, {
let mut __t17: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t17.vendor = 32902u16;
__t17.device = 4344u16;
__t17.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t17
}, {
let mut __t18: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t18.vendor = 32902u16;
__t18.device = 4345u16;
__t18.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t18
}, {
let mut __t19: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t19.vendor = 32902u16;
__t19.device = 4346u16;
__t19.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t19
}, {
let mut __t20: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t20.vendor = 32902u16;
__t20.device = 4347u16;
__t20.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t20
}, {
let mut __t21: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t21.vendor = 32902u16;
__t21.device = 4348u16;
__t21.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t21
}, {
let mut __t22: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t22.vendor = 32902u16;
__t22.device = 5383u16;
__t22.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t22
}, {
let mut __t23: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t23.vendor = 32902u16;
__t23.device = 5396u16;
__t23.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t23
}, {
let mut __t24: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t24.vendor = 32902u16;
__t24.device = 5404u16;
__t24.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t24
}, {
let mut __t25: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t25.vendor = 32902u16;
__t25.device = 5417u16;
__t25.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t25
}, {
let mut __t26: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t26.vendor = 32902u16;
__t26.device = 5418u16;
__t26.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t26
}, {
let mut __t27: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t27.vendor = 32902u16;
__t27.device = 5453u16;
__t27.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t27
}, {
let mut __t28: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t28.vendor = 32902u16;
__t28.device = 5455u16;
__t28.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t28
}, {
let mut __t29: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t29.vendor = 32902u16;
__t29.device = 5457u16;
__t29.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t29
}, {
let mut __t30: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t30.vendor = 32902u16;
__t30.device = 5464u16;
__t30.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t30
}, {
let mut __t31: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t31.vendor = 32902u16;
__t31.device = 5385u16;
__t31.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t31
}, {
let mut __t32: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t32.vendor = 32902u16;
__t32.device = 5390u16;
__t32.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t32
}, {
let mut __t33: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t33.vendor = 32902u16;
__t33.device = 5391u16;
__t33.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t33
}, {
let mut __t34: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t34.vendor = 32902u16;
__t34.device = 5392u16;
__t34.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t34
}, {
let mut __t35: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t35.vendor = 32902u16;
__t35.device = 5393u16;
__t35.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t35
}, {
let mut __t36: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t36.vendor = 32902u16;
__t36.device = 5398u16;
__t36.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t36
}, {
let mut __t37: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t37.vendor = 32902u16;
__t37.device = 5415u16;
__t37.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t37
}, {
let mut __t38: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t38.vendor = 32902u16;
__t38.device = 4297u16;
__t38.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t38
}, {
let mut __t39: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t39.vendor = 32902u16;
__t39.device = 4326u16;
__t39.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t39
}, {
let mut __t40: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t40.vendor = 32902u16;
__t40.device = 4327u16;
__t40.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t40
}, {
let mut __t41: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t41.vendor = 32902u16;
__t41.device = 4328u16;
__t41.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t41
}, {
let mut __t42: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t42.vendor = 32902u16;
__t42.device = 5386u16;
__t42.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t42
}, {
let mut __t43: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t43.vendor = 32902u16;
__t43.device = 5389u16;
__t43.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t43
}, {
let mut __t44: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t44.vendor = 32902u16;
__t44.device = 5400u16;
__t44.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t44
}, {
let mut __t45: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t45.vendor = 32902u16;
__t45.device = 5414u16;
__t45.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t45
}, {
let mut __t46: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t46.vendor = 32902u16;
__t46.device = 4263u16;
__t46.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t46
}, {
let mut __t47: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t47.vendor = 32902u16;
__t47.device = 4265u16;
__t47.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t47
}, {
let mut __t48: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t48.vendor = 32902u16;
__t48.device = 4310u16;
__t48.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t48
}, {
let mut __t49: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t49.vendor = 32902u16;
__t49.device = 5409u16;
__t49.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t49
}, {
let mut __t50: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t50.vendor = 32902u16;
__t50.device = 5410u16;
__t50.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t50
}, {
let mut __t51: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t51.vendor = 32902u16;
__t51.device = 5411u16;
__t51.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t51
}, {
let mut __t52: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t52.vendor = 32902u16;
__t52.device = 5412u16;
__t52.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t52
}, {
let mut __t53: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t53.vendor = 32902u16;
__t53.device = 4190u16;
__t53.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t53
}, {
let mut __t54: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t54.vendor = 32902u16;
__t54.device = 4191u16;
__t54.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t54
}, {
let mut __t55: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t55.vendor = 32902u16;
__t55.device = 4192u16;
__t55.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t55
}, {
let mut __t56: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t56.vendor = 32902u16;
__t56.device = 4313u16;
__t56.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t56
}, {
let mut __t57: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t57.vendor = 32902u16;
__t57.device = 5559u16;
__t57.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t57
}, {
let mut __t58: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t58.vendor = 32902u16;
__t58.device = 5560u16;
__t58.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t58
}, {
let mut __t59: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t59.vendor = 32902u16;
__t59.device = 65535u16;
__t59.acs_enabled = Some(pci_quirk_rciep_acs);
__t59
}, {
let mut __t60: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t60.vendor = 6091u16;
__t60.device = 1024u16;
__t60.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t60
}, {
let mut __t61: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t61.vendor = 6091u16;
__t61.device = 1025u16;
__t61.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t61
}, {
let mut __t62: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t62.vendor = 6091u16;
__t62.device = 277u16;
__t62.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t62
}, {
let mut __t63: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t63.vendor = 6091u16;
__t63.device = 273u16;
__t63.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t63
}, {
let mut __t64: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t64.vendor = 6091u16;
__t64.device = 288u16;
__t64.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t64
}, {
let mut __t65: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t65.vendor = 7615u16;
__t65.device = 1025u16;
__t65.acs_enabled = Some(pci_quirk_qcom_rp_acs);
__t65
}, {
let mut __t66: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t66.vendor = 32902u16;
__t66.device = 65535u16;
__t66.acs_enabled = Some(pci_quirk_intel_pch_acs);
__t66
}, {
let mut __t67: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t67.vendor = 32902u16;
__t67.device = 65535u16;
__t67.acs_enabled = Some(pci_quirk_intel_spt_pch_acs);
__t67
}, {
let mut __t68: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t68.vendor = 6562u16;
__t68.device = 1808u16;
__t68.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t68
}, {
let mut __t69: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t69.vendor = 4319u16;
__t69.device = 1824u16;
__t69.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t69
}, {
let mut __t70: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t70.vendor = 6013u16;
__t70.device = 65535u16;
__t70.acs_enabled = Some(pci_quirk_cavium_acs);
__t70
}, {
let mut __t71: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t71.vendor = 6013u16;
__t71.device = 40998u16;
__t71.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t71
}, {
let mut __t72: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t72.vendor = 6013u16;
__t72.device = 41049u16;
__t72.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t72
}, {
let mut __t73: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t73.vendor = 6013u16;
__t73.device = 41056u16;
__t73.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t73
}, {
let mut __t74: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t74.vendor = 4328u16;
__t74.device = 57348u16;
__t74.acs_enabled = Some(pci_quirk_xgene_acs);
__t74
}, {
let mut __t75: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t75.vendor = 7663u16;
__t75.device = 57349u16;
__t75.acs_enabled = Some(pci_quirk_xgene_acs);
__t75
}, {
let mut __t76: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t76.vendor = 7663u16;
__t76.device = 57350u16;
__t76.acs_enabled = Some(pci_quirk_xgene_acs);
__t76
}, {
let mut __t77: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t77.vendor = 7663u16;
__t77.device = 57351u16;
__t77.acs_enabled = Some(pci_quirk_xgene_acs);
__t77
}, {
let mut __t78: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t78.vendor = 7663u16;
__t78.device = 57352u16;
__t78.acs_enabled = Some(pci_quirk_xgene_acs);
__t78
}, {
let mut __t79: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t79.vendor = 7663u16;
__t79.device = 57353u16;
__t79.acs_enabled = Some(pci_quirk_xgene_acs);
__t79
}, {
let mut __t80: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t80.vendor = 7663u16;
__t80.device = 57354u16;
__t80.acs_enabled = Some(pci_quirk_xgene_acs);
__t80
}, {
let mut __t81: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t81.vendor = 7663u16;
__t81.device = 57355u16;
__t81.acs_enabled = Some(pci_quirk_xgene_acs);
__t81
}, {
let mut __t82: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t82.vendor = 7663u16;
__t82.device = 57356u16;
__t82.acs_enabled = Some(pci_quirk_xgene_acs);
__t82
}, {
let mut __t83: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t83.vendor = 5348u16;
__t83.device = 5847u16;
__t83.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t83
}, {
let mut __t84: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t84.vendor = 5348u16;
__t84.device = 5968u16;
__t84.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t84
}, {
let mut __t85: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t85.vendor = 5348u16;
__t85.device = 5969u16;
__t85.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t85
}, {
let mut __t86: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t86.vendor = 5348u16;
__t86.device = 5970u16;
__t86.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t86
}, {
let mut __t87: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t87.vendor = 5348u16;
__t87.device = 5984u16;
__t87.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t87
}, {
let mut __t88: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t88.vendor = 5348u16;
__t88.device = 5985u16;
__t88.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t88
}, {
let mut __t89: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t89.vendor = 5348u16;
__t89.device = 5986u16;
__t89.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t89
}, {
let mut __t90: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t90.vendor = 5348u16;
__t90.device = 5987u16;
__t90.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t90
}, {
let mut __t91: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t91.vendor = 5348u16;
__t91.device = 55060u16;
__t91.acs_enabled = Some(pci_quirk_brcm_acs);
__t91
}, {
let mut __t92: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t92.vendor = 20u16;
__t92.device = 15369u16;
__t92.acs_enabled = Some(pci_quirk_loongson_acs);
__t92
}, {
let mut __t93: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t93.vendor = 20u16;
__t93.device = 15385u16;
__t93.acs_enabled = Some(pci_quirk_loongson_acs);
__t93
}, {
let mut __t94: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t94.vendor = 20u16;
__t94.device = 15401u16;
__t94.acs_enabled = Some(pci_quirk_loongson_acs);
__t94
}, {
let mut __t95: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t95.vendor = 20u16;
__t95.device = 31241u16;
__t95.acs_enabled = Some(pci_quirk_loongson_acs);
__t95
}, {
let mut __t96: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t96.vendor = 20u16;
__t96.device = 31257u16;
__t96.acs_enabled = Some(pci_quirk_loongson_acs);
__t96
}, {
let mut __t97: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t97.vendor = 20u16;
__t97.device = 31273u16;
__t97.acs_enabled = Some(pci_quirk_loongson_acs);
__t97
}, {
let mut __t98: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t98.vendor = 20u16;
__t98.device = 31289u16;
__t98.acs_enabled = Some(pci_quirk_loongson_acs);
__t98
}, {
let mut __t99: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t99.vendor = 20u16;
__t99.device = 31305u16;
__t99.acs_enabled = Some(pci_quirk_loongson_acs);
__t99
}, {
let mut __t100: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t100.vendor = 20u16;
__t100.device = 31321u16;
__t100.acs_enabled = Some(pci_quirk_loongson_acs);
__t100
}, {
let mut __t101: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t101.vendor = 20u16;
__t101.device = 31337u16;
__t101.acs_enabled = Some(pci_quirk_loongson_acs);
__t101
}, {
let mut __t102: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t102.vendor = 7222u16;
__t102.device = 49u16;
__t102.acs_enabled = Some(pci_quirk_al_acs);
__t102
}, {
let mut __t103: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t103.vendor = 7447u16;
__t103.device = 12344u16;
__t103.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t103
}, {
let mut __t104: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t104.vendor = 7447u16;
__t104.device = 12548u16;
__t104.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t104
}, {
let mut __t105: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t105.vendor = 7447u16;
__t105.device = 36995u16;
__t105.acs_enabled = Some(pci_quirk_mf_endpoint_acs);
__t105
}, {
let mut __t106: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t106.vendor = 6487u16;
__t106.device = 36225u16;
__t106.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t106
}, {
let mut __t107: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t107.vendor = 6487u16;
__t107.device = 36257u16;
__t107.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t107
}, {
let mut __t108: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t108.vendor = 6487u16;
__t108.device = 36227u16;
__t108.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t108
}, {
let mut __t109: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t109.vendor = 6487u16;
__t109.device = 36224u16;
__t109.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t109
}, {
let mut __t110: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t110.vendor = 6487u16;
__t110.device = 36256u16;
__t110.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t110
}, {
let mut __t111: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t111.vendor = 6487u16;
__t111.device = 36226u16;
__t111.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t111
}, {
let mut __t112: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t112.vendor = 6487u16;
__t112.device = 36240u16;
__t112.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t112
}, {
let mut __t113: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t113.vendor = 6487u16;
__t113.device = 36272u16;
__t113.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t113
}, {
let mut __t114: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t114.vendor = 6487u16;
__t114.device = 36242u16;
__t114.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t114
}, {
let mut __t115: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t115.vendor = 6487u16;
__t115.device = 36241u16;
__t115.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t115
}, {
let mut __t116: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t116.vendor = 6487u16;
__t116.device = 36273u16;
__t116.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t116
}, {
let mut __t117: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t117.vendor = 6487u16;
__t117.device = 36243u16;
__t117.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t117
}, {
let mut __t118: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t118.vendor = 6487u16;
__t118.device = 36233u16;
__t118.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t118
}, {
let mut __t119: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t119.vendor = 6487u16;
__t119.device = 36265u16;
__t119.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t119
}, {
let mut __t120: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t120.vendor = 6487u16;
__t120.device = 36235u16;
__t120.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t120
}, {
let mut __t121: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t121.vendor = 6487u16;
__t121.device = 36232u16;
__t121.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t121
}, {
let mut __t122: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t122.vendor = 6487u16;
__t122.device = 36264u16;
__t122.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t122
}, {
let mut __t123: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t123.vendor = 6487u16;
__t123.device = 36234u16;
__t123.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t123
}, {
let mut __t124: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t124.vendor = 6487u16;
__t124.device = 36248u16;
__t124.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t124
}, {
let mut __t125: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t125.vendor = 6487u16;
__t125.device = 36280u16;
__t125.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t125
}, {
let mut __t126: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t126.vendor = 6487u16;
__t126.device = 36250u16;
__t126.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t126
}, {
let mut __t127: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t127.vendor = 6487u16;
__t127.device = 36249u16;
__t127.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t127
}, {
let mut __t128: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t128.vendor = 6487u16;
__t128.device = 36281u16;
__t128.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t128
}, {
let mut __t129: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t129.vendor = 6487u16;
__t129.device = 36251u16;
__t129.acs_enabled = Some(pci_quirk_nxp_rp_acs);
__t129
}, {
let mut __t130: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t130.vendor = 7447u16;
__t130.device = 65535u16;
__t130.acs_enabled = Some(pci_quirk_zhaoxin_pcie_ports_acs);
__t130
}, {
let mut __t131: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t131.vendor = 32904u16;
__t131.device = 65535u16;
__t131.acs_enabled = Some(pci_quirk_wangxun_nic_acs);
__t131
}, {
let mut __t132: pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
__t132.vendor = 0u16;
__t132
}]);

/// pci_dev_specific_acs_enabled - check whether device provides ACS controls
/// @dev:	PCI device
/// @acs_flags:	Bitmask of desired ACS controls
///
/// Returns:
///   -ENOTTY:	No quirk applies to this device; we can't tell whether the
/// 		device provides the desired controls
///   0:		Device does not provide all the desired controls
///   >0:	Device provides all the controls in @acs_flags
#[unsafe(no_mangle)]
extern "C" fn pci_dev_specific_acs_enabled(dev: *mut pci_dev, acs_flags: u16) -> i32 {
let mut i: *const pci_dev_acs_enabled = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
// Allow devices that do not expose standard PCIe ACS capabilities
// or control to indicate their support here.  Multi-function express
// devices which do not allow internal peer-to-peer between functions,
// but do not implement PCIe ACS may wish to return true here.
i = unsafe { core::ptr::addr_of!(pci_dev_acs_enabled.0) as *const pci_dev_acs_enabled };
while (unsafe { (*i).acs_enabled }) != None {
if (((unsafe { (*i).vendor }) as i32) == ((unsafe { (*dev).vendor }) as i32) || ((unsafe { (*i).vendor }) as i32) == 65535i32) && (((unsafe { (*i).device }) as i32) == ((unsafe { (*dev).device }) as i32) || ((unsafe { (*i).device }) as i32) == 65535i32) {
ret = unsafe { unsafe { (*i).acs_enabled }.unwrap()(dev, acs_flags) };
if ret >= 0 {
return ret;
}
}
let __v1: *const pci_dev_acs_enabled = i;
i = __v1.wrapping_offset(1 as isize);
}
25i32.wrapping_neg()
}

// Config space offset of Root Complex Base Address register
// 31:14 RCBA address
// RCBA Enable
// Backbone Scratch Pad Register
// Backbone Peer Non-Posted Disable
// Backbone Peer Posted Disable
// Upstream Peer Decode Configuration Register
// 5:0 Peer Decode Enable bits
fn pci_quirk_enable_intel_lpc_acs(dev: *mut pci_dev) -> i32 {
let mut rcba: u32 = 0;
let mut bspr: u32 = 0;
let mut updcr: u32 = 0;
// Read the RCBA register from the LPC (D31:F0).  PCH root ports
// are D28:F* and therefore get probed before LPC, thus we can't
// use pci_get_slot()/pci_read_config_dword() here.
unsafe { pci_bus_read_config_dword((*dev).bus, PCI_DEVFN!(31i32, 0i32) as u32, 240i32, core::ptr::addr_of_mut!(rcba)) };
if rcba & 1u32 == 0 {
return 22i32.wrapping_neg();
}
let rcba_mem: *mut () = unsafe { ioremap((rcba & 4294950912) as u64, PAGE_ALIGN!(4116i32) as u64) };
if rcba_mem.is_null() {
return 12i32.wrapping_neg();
}
// The BSPR can disallow peer cycles, but it's set by soft strap and
// therefore read-only.  If both posted and non-posted peer cycles are
// disallowed, we're ok.  If either are allowed, then we need to use
// the UPDCR to disable peer decodes for each port.  This provides the
// PCIe ACS equivalent of PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF
bspr = readl(((unsafe { (rcba_mem as *const u8).wrapping_offset(4356 as isize) }) as *mut ()) as *const ());
bspr &= 768u32;
if bspr != 768u32 {
updcr = readl(((unsafe { (rcba_mem as *const u8).wrapping_offset(4116 as isize) }) as *mut ()) as *const ());
if updcr & 63 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Disabling UPDCR peer decodes\n".as_ptr() as *const u8) };
let __v4: u32 = updcr & 4294967232u32;
updcr = __v4;
writel(updcr, (unsafe { (rcba_mem as *const u8).wrapping_offset(4116 as isize) }) as *mut ());
}
}
unsafe { iounmap(rcba_mem as *mut ()) };
0
}

// Miscellaneous Port Configuration register
// MPC: Invalid Receive Bus Number Check Enable
fn pci_quirk_enable_intel_rp_mpc_acs(dev: *mut pci_dev) {
let mut mpc: u32 = 0;
// When enabled, the IRBNCE bit of the MPC register enables the
// equivalent of PCI ACS Source Validation (PCI_ACS_SV), which
// ensures that requester IDs fall within the bus number range
// of the bridge.  Enable if not already.
unsafe { pci_read_config_dword(dev as *const pci_dev, 216i32, core::ptr::addr_of_mut!(mpc)) };
if mpc & 67108864u32 == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling MPC IRBNCE\n".as_ptr() as *const u8) };
mpc |= 67108864u32;
unsafe { pci_write_config_dword(dev as *const pci_dev, 216i32, mpc) };
}
}

/// Currently this quirk does the equivalent of
/// PCI_ACS_SV | PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF
///
/// TODO: This quirk also needs to do equivalent of PCI_ACS_TB,
/// if dev->external_facing || dev->untrusted
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_enable_intel_pch_acs")]
extern "C" fn pci_quirk_enable_intel_pch_acs(dev: *mut pci_dev) -> i32 {
if !pci_quirk_intel_pch_acs_match(dev) {
return 25i32.wrapping_neg();
}
if pci_quirk_enable_intel_lpc_acs(dev) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Failed to enable Intel PCH ACS quirk\n".as_ptr() as *const u8) };
return 0;
}
pci_quirk_enable_intel_rp_mpc_acs(dev);
unsafe {
(*dev).dev_flags = (((*dev).dev_flags as i32) | 8) as u16;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Intel PCH root port ACS workaround enabled\n".as_ptr() as *const u8) };
0
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_enable_intel_spt_pch_acs")]
extern "C" fn pci_quirk_enable_intel_spt_pch_acs(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let mut cap: u32 = 0;
let mut ctrl: u32 = 0;
if !pci_quirk_intel_spt_pch_acs_match(dev) {
return 25i32.wrapping_neg();
}
pos = (unsafe { (*dev).acs_cap }) as i32;
if pos == 0 {
return 25i32.wrapping_neg();
}
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(cap)) };
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(8i32), core::ptr::addr_of_mut!(ctrl)) };
ctrl |= cap & 1;
ctrl |= cap & 4;
ctrl |= cap & 8;
ctrl |= cap & 16;
if (unsafe { pci_ats_disabled() }) || ((unsafe { (*dev).__slate_bits_3.__get_external_facing() }) as i32) != 0 || ((unsafe { (*dev).__slate_bits_3.__get_untrusted() }) as i32) != 0 {
ctrl |= cap & 2;
}
unsafe { pci_write_config_dword(dev as *const pci_dev, pos.wrapping_add(8i32), ctrl) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Intel SPT PCH root port ACS workaround enabled\n".as_ptr() as *const u8) };
0
}

#[unsafe(link_section = ".text.slate_distinct.quirks.pci_quirk_disable_intel_spt_pch_acs_redir")]
extern "C" fn pci_quirk_disable_intel_spt_pch_acs_redir(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let mut cap: u32 = 0;
let mut ctrl: u32 = 0;
if !pci_quirk_intel_spt_pch_acs_match(dev) {
return 25i32.wrapping_neg();
}
pos = (unsafe { (*dev).acs_cap }) as i32;
if pos == 0 {
return 25i32.wrapping_neg();
}
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(cap)) };
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(8i32), core::ptr::addr_of_mut!(ctrl)) };
let __v1: u32 = ctrl & 4294967251u32;
ctrl = __v1;
unsafe { pci_write_config_dword(dev as *const pci_dev, pos.wrapping_add(8i32), ctrl) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Intel SPT PCH root port workaround: disabled ACS redirect\n".as_ptr() as *const u8) };
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev_acs_ops {
    vendor: u16,
    device: u16,
    enable_acs: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    disable_acs_redir: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
}


static pci_dev_acs_ops: __SlateAlign16<[pci_dev_acs_ops; 2]> = __SlateAlign16([{
let mut __t0: pci_dev_acs_ops = unsafe { core::mem::zeroed() };
__t0.vendor = 32902u16;
__t0.device = 65535u16;
__t0.enable_acs = Some(pci_quirk_enable_intel_pch_acs);
__t0
}, {
let mut __t1: pci_dev_acs_ops = unsafe { core::mem::zeroed() };
__t1.vendor = 32902u16;
__t1.device = 65535u16;
__t1.enable_acs = Some(pci_quirk_enable_intel_spt_pch_acs);
__t1.disable_acs_redir = Some(pci_quirk_disable_intel_spt_pch_acs_redir);
__t1
}]);

#[unsafe(no_mangle)]
extern "C" fn pci_dev_specific_enable_acs(dev: *mut pci_dev) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
while (i as u64) < ARRAY_SIZE__19!() {
let p: *const pci_dev_acs_ops = unsafe { core::ptr::addr_of!(pci_dev_acs_ops.0) as *const pci_dev_acs_ops }.wrapping_offset(i as isize);
if (((unsafe { (*p).vendor }) as i32) == ((unsafe { (*dev).vendor }) as i32) || ((unsafe { (*p).vendor }) as i32) == 65535i32) && (((unsafe { (*p).device }) as i32) == ((unsafe { (*dev).device }) as i32) || ((unsafe { (*p).device }) as i32) == 65535i32) && (unsafe { (*p).enable_acs }) != None {
ret = unsafe { unsafe { (*p).enable_acs }.unwrap()(dev) };
if ret >= 0 {
return ret;
}
}
i = i.wrapping_add(1);
}
25i32.wrapping_neg()
}

#[unsafe(no_mangle)]
extern "C" fn pci_dev_specific_disable_acs_redir(dev: *mut pci_dev) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
while (i as u64) < ARRAY_SIZE__21!() {
let p: *const pci_dev_acs_ops = unsafe { core::ptr::addr_of!(pci_dev_acs_ops.0) as *const pci_dev_acs_ops }.wrapping_offset(i as isize);
if (((unsafe { (*p).vendor }) as i32) == ((unsafe { (*dev).vendor }) as i32) || ((unsafe { (*p).vendor }) as i32) == 65535i32) && (((unsafe { (*p).device }) as i32) == ((unsafe { (*dev).device }) as i32) || ((unsafe { (*p).device }) as i32) == 65535i32) && (unsafe { (*p).disable_acs_redir }) != None {
ret = unsafe { unsafe { (*p).disable_acs_redir }.unwrap()(dev) };
if ret >= 0 {
return ret;
}
}
i = i.wrapping_add(1);
}
25i32.wrapping_neg()
}

/// The PCI capabilities list for Intel DH895xCC VFs (device ID 0x0443) with
/// QuickAssist Technology (QAT) is prematurely terminated in hardware.  The
/// Next Capability pointer in the MSI Capability Structure should point to
/// the PCIe Capability Structure but is incorrectly hardwired as 0 terminating
/// the list.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_intel_qat_vf_cap")]
extern "C" fn quirk_intel_qat_vf_cap(pdev: *mut pci_dev) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut pos: i32 = 0;
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut next_cap: u8 = 0;
let mut reg16: u16 = 0;
let mut state: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
// Bail if the hardware bug is fixed
let __v0: bool = (unsafe { (*pdev).pcie_cap }) != 0 || (unsafe { pci_find_capability(pdev, 16i32) }) != 0;
if __v0 {
return;
}
// Bail if MSI Capability Structure is not found for some reason
pos = (unsafe { pci_find_capability(pdev, 5i32) }) as i32;
if pos == 0 {
return;
}
// Bail if Next Capability pointer in the MSI Capability Structure
// is not the expected incorrect 0x00.
unsafe { pci_read_config_byte(pdev as *const pci_dev, pos.wrapping_add(1), core::ptr::addr_of_mut!(next_cap)) };
if next_cap != 0 {
return;
}
// PCIe Capability Structure is expected to be at 0x50 and should
// terminate the list (Next Capability pointer is 0x00).  Verify
// Capability Id and Next Capability pointer is as expected.
// Open-code some of set_pcie_port_type() and pci_cfg_space_size_ext()
// to correctly set kernel data structures which have already been
// set incorrectly due to the hardware bug.
pos = 80;
unsafe { pci_read_config_word(pdev as *const pci_dev, pos, core::ptr::addr_of_mut!(reg16)) };
if (reg16 as i32) == 0 | 16 {
let mut status: u32 = 0;
let size: i32 = 7u64.wrapping_mul(core::mem::size_of::<u16>() as u64) as i32;
unsafe {
(*pdev).pcie_cap = pos as u8;
}
unsafe { pci_read_config_word(pdev as *const pci_dev, pos.wrapping_add(2), core::ptr::addr_of_mut!(reg16)) };
unsafe {
(*pdev).pcie_flags_reg = reg16;
}
unsafe { pci_read_config_word(pdev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(reg16)) };
unsafe {
let __t0 = ((reg16 as i32) & 7) as u8;
(*pdev).__slate_bits_0.__set_pcie_mpss(__t0);
}
unsafe {
(*pdev).cfg_size = 4096;
}
ret = unsafe { pci_read_config_dword(pdev as *const pci_dev, 256i32, core::ptr::addr_of_mut!(status)) };
if ret != 0 || PCI_POSSIBLE_ERROR!(status) {
unsafe {
(*pdev).cfg_size = 256;
}
}
if !unsafe { pci_find_saved_cap(pdev, 16u8) }.is_null() {
return;
}
// Save PCIe cap
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(24u64.wrapping_add(size as u64), 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(24u64.wrapping_add(size as u64), 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
state = __v1 as *mut pci_cap_saved_state;
if state.is_null() {
return;
}
unsafe {
(*state).cap.cap_nr = 16;
}
unsafe {
(*state).cap.cap_extended = false;
}
unsafe {
(*state).cap.size = size as u32;
}
let cap: *mut u16 = unsafe { core::ptr::addr_of_mut!((*state).cap.data) as *mut u32 }.wrapping_offset(0 as isize) as *mut u16;
let __v6: i32 = i;
i = __v6.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 8i32, cap.wrapping_offset(__v6 as isize)) };
let __v8: i32 = i;
i = __v8.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 16i32, cap.wrapping_offset(__v8 as isize)) };
let __v10: i32 = i;
i = __v10.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 24i32, cap.wrapping_offset(__v10 as isize)) };
let __v12: i32 = i;
i = __v12.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 28i32, cap.wrapping_offset(__v12 as isize)) };
let __v14: i32 = i;
i = __v14.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 40i32, cap.wrapping_offset(__v14 as isize)) };
let __v16: i32 = i;
i = __v16.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 48i32, cap.wrapping_offset(__v16 as isize)) };
let __v18: i32 = i;
i = __v18.wrapping_add(1);
unsafe { pcie_capability_read_word(pdev, 56i32, cap.wrapping_offset(__v18 as isize)) };
hlist_add_head(unsafe { core::ptr::addr_of_mut!((*state).next) }, unsafe { core::ptr::addr_of_mut!((*pdev).saved_cap_space) });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_qat_vf_cap_1251: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_qat_vf_cap)) };

/// FLR may cause the following to devices to hang:
///
/// AMD Starship/Matisse HD Audio Controller 0x1487
/// AMD Starship USB 3.0 Host Controller 0x148c
/// AMD Matisse USB 3.0 Host Controller 0x149c
/// AMD Neural Processing Unit 0x1502 0x17f0
/// Intel 82579LM Gigabit Ethernet Controller 0x1502
/// Intel 82579V Gigabit Ethernet Controller 0x1503
/// Mediatek MT7922 802.11ax PCI Express Wireless Network Adapter
/// Mediatek MT7925 802.11be PCI Express Wireless Network Adapter
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_flr")]
extern "C" fn quirk_no_flr(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 1024) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1252: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1253: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1254: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1255: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1256: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1257: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1258: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1259: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1260: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_1261: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr)) };

/// FLR may cause the SolidRun SNET DPU (rev 0x1) to hang
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_flr_snet")]
extern "C" fn quirk_no_flr_snet(dev: *mut pci_dev) {
if ((unsafe { (*dev).revision }) as i32) == 1 {
quirk_no_flr(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_flr_snet_1262: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_flr_snet)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_no_ext_tags")]
extern "C" fn quirk_no_ext_tags(pdev: *mut pci_dev) {
let bridge: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*pdev).bus) };
if bridge.is_null() {
return;
}
unsafe {
(*bridge).__slate_bits_0.__set_no_ext_tags(1u32);
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"disabling Extended Tags (this device can't handle them)\n".as_ptr() as *const u8) };
unsafe { pci_walk_bus((*bridge).bus, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev, *mut ()) -> i32>>(pci_configure_extended_tags as *const ()) }, core::ptr::null_mut::<()>()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1263: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1264: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1265: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1266: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1267: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1268: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1269: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1270: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_no_ext_tags_1271: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_no_ext_tags)) };

fn quirk_no_ats(pdev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"disabling ATS\n".as_ptr() as *const u8) };
unsafe {
(*pdev).ats_cap = 0;
}
}

/// Some devices require additional driver setup to enable ATS.  Don't use
/// ATS for those devices as ATS will be enabled before the driver has had a
/// chance to load and configure the device.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_amd_harvest_no_ats")]
extern "C" fn quirk_amd_harvest_no_ats(pdev: *mut pci_dev) {
if ((unsafe { (*pdev).device }) as i32) == 5592 {
if ((unsafe { (*pdev).revision }) as i32) == 207 && ((unsafe { (*pdev).subsystem_vendor }) as i32) == 59984 && (((unsafe { (*pdev).subsystem_device }) as i32) == 52761 || ((unsafe { (*pdev).subsystem_device }) as i32) == 52240 || ((unsafe { (*pdev).subsystem_device }) as i32) == 52232) {
quirk_no_ats(pdev);
}
} else {
quirk_no_ats(pdev);
}
}

/// AMD Stoney platform GPU
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1272: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

/// AMD Iceland dGPU
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1273: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

/// AMD Navi10 dGPU
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1274: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1275: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1276: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1277: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1278: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1279: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1280: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1281: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

/// AMD Navi14 dGPU
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1282: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1283: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1284: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1285: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

/// AMD Raven platform iGPU
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_amd_harvest_no_ats_1286: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_amd_harvest_no_ats)) };

/// Intel IPU E2000 revisions before C0 implement incorrect endianness
/// in ATS Invalidate Request message body. Disable ATS for those devices.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_intel_e2000_no_ats")]
extern "C" fn quirk_intel_e2000_no_ats(pdev: *mut pci_dev) {
if ((unsafe { (*pdev).revision }) as i32) < 32 {
quirk_no_ats(pdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1287: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1288: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1289: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1290: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1291: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1292: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1293: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1294: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_intel_e2000_no_ats_1295: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_intel_e2000_no_ats)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_gpu_ats_required")]
extern "C" fn quirk_nvidia_gpu_ats_required(pdev: *mut pci_dev) -> bool {
match (unsafe { (*pdev).device }) as i32 {
11776..=11839 => {
return true; // GB20B
}
_ => {
}
}
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev_ats_required {
    vendor: u16,
    device: u16,
    ats_required: Option<unsafe extern "C" fn(*mut pci_dev) -> bool>,
}


/// NVIDIA GPUs
/// NVIDIA CX10 Family NVlink-C2C
static pci_dev_ats_required: __SlateAlign16<[pci_dev_ats_required; 3]> = __SlateAlign16([{
let mut __t0: pci_dev_ats_required = unsafe { core::mem::zeroed() };
__t0.vendor = 4318u16;
__t0.device = 65535u16;
__t0.ats_required = Some(quirk_nvidia_gpu_ats_required);
__t0
}, {
let mut __t1: pci_dev_ats_required = unsafe { core::mem::zeroed() };
__t1.vendor = 5555u16;
__t1.device = 8449u16;
__t1.ats_required = None;
__t1
}, {
let mut __t2: pci_dev_ats_required = unsafe { core::mem::zeroed() };
__t2.vendor = 0u16;
__t2
}]);

/// Some NVIDIA devices do not implement CXL config space, but present as PCIe
/// devices that can issue CXL-like cache operations like CXL.cache. Thus, they
/// require ATS to obtain host physical addresses, like pci_cxl_ats_required().
#[unsafe(no_mangle)]
extern "C" fn pci_dev_specific_ats_required(pdev: *mut pci_dev) -> bool {
let mut i: *const pci_dev_ats_required = unsafe { core::mem::zeroed() };
i = unsafe { core::ptr::addr_of!(pci_dev_ats_required.0) as *const pci_dev_ats_required };
while (unsafe { (*i).vendor }) != 0 {
if ((unsafe { (*i).vendor }) as i32) == ((unsafe { (*pdev).vendor }) as i32) {
let __v1: bool = (unsafe { (*i).ats_required }) != None && unsafe { unsafe { (*i).ats_required }.unwrap()(pdev) };
if __v1 {
return true;
}
if (unsafe { (*i).ats_required }) == None && ((unsafe { (*i).device }) as i32) == ((unsafe { (*pdev).device }) as i32) {
return true;
}
}
let __v2: *const pci_dev_ats_required = i;
i = __v2.wrapping_offset(1 as isize);
}
false
}

/// Freescale PCIe doesn't support MSI in RC mode
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_fsl_no_msi")]
extern "C" fn quirk_fsl_no_msi(pdev: *mut pci_dev) {
if pci_pcie_type(pdev as *const pci_dev) == 4 {
unsafe {
(*pdev).__slate_bits_3.__set_no_msi(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_fsl_no_msi_1296: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_fsl_no_msi)) };

/// Although not allowed by the spec, some multi-function devices have
/// dependencies of one function (consumer) on another (supplier).  For the
/// consumer to work in D0, the supplier must also be in D0.  Create a
/// device link from the consumer to the supplier to enforce this
/// dependency.  Runtime PM is allowed by default on the consumer to prevent
/// it from permanently keeping the supplier awake.
fn pci_create_device_link(pdev: *mut pci_dev, consumer: u32, supplier: u32, class: u32, class_shift: u32) {
let mut supplier_pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
if PCI_FUNC!(unsafe { (*pdev).devfn }) != consumer {
return;
}
supplier_pdev = unsafe { pci_get_domain_bus_and_slot(pci_domain_nr((*pdev).bus), (*(*pdev).bus).number as u32, PCI_DEVFN__3!(PCI_SLOT!(unsafe { (*pdev).devfn }), supplier)) };
if supplier_pdev.is_null() || (unsafe { (*supplier_pdev).class }) >> class_shift != class {
unsafe { pci_dev_put(supplier_pdev) };
return;
}
if !unsafe { device_link_add(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::addr_of_mut!((*supplier_pdev).dev), (BIT!(0i32) | BIT!(2i32)) as u32) }.is_null() {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"D0 power state depends on %s\n".as_ptr() as *const u8, pci_name(supplier_pdev as *const pci_dev)) };
} else {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Cannot enforce power dependency on %s\n".as_ptr() as *const u8, pci_name(supplier_pdev as *const pci_dev)) };
}
unsafe { pm_runtime_allow(core::ptr::addr_of_mut!((*pdev).dev)) };
unsafe { pci_dev_put(supplier_pdev) };
}

/// Create device link for GPUs with integrated HDA controller for streaming
/// audio to attached displays.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_gpu_hda")]
extern "C" fn quirk_gpu_hda(hda: *mut pci_dev) {
pci_create_device_link(hda, 1u32, 0u32, 3u32, 16u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_hda_1297: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_hda)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_hda_1298: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_hda)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_hda_1299: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_hda)) };

/// Create device link for GPUs with integrated USB xHCI Host
/// controller to VGA.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_gpu_usb")]
extern "C" fn quirk_gpu_usb(usb: *mut pci_dev) {
pci_create_device_link(usb, 2u32, 0u32, 3u32, 16u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_usb_1300: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_usb)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_usb_1301: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_usb)) };

// Create device link for GPUs with integrated Type-C UCSI controller
// to VGA. Currently there is no class code defined for UCSI device over PCI
// so using UNKNOWN class for now and it will be updated when UCSI
// over PCI gets a class code.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_gpu_usb_typec_ucsi")]
extern "C" fn quirk_gpu_usb_typec_ucsi(ucsi: *mut pci_dev) {
pci_create_device_link(ucsi, 3u32, 0u32, 3u32, 16u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_usb_typec_ucsi_1302: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_usb_typec_ucsi)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_gpu_usb_typec_ucsi_1303: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_gpu_usb_typec_ucsi)) };

// Enable the NVIDIA GPU integrated HDA controller if the BIOS left it
// disabled.  https://devtalk.nvidia.com/default/topic/1024022
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_nvidia_hda")]
extern "C" fn quirk_nvidia_hda(gpu: *mut pci_dev) {
let mut hdr_type: u8 = 0;
let mut val: u32 = 0;
// There was no integrated HDA controller before MCP89
if ((unsafe { (*gpu).device }) as i32) < 2208 {
return;
}
// Bit 25 at offset 0x488 enables the HDA controller
unsafe { pci_read_config_dword(gpu as *const pci_dev, 1160i32, core::ptr::addr_of_mut!(val)) };
if (val as u64) & BIT!(25i32) != 0 {
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*gpu).dev) as *const device, c"Enabling HDA controller\n".as_ptr() as *const u8) };
unsafe { pci_write_config_dword(gpu as *const pci_dev, 1160i32, ((val as u64) | BIT!(25i32)) as u32) };
// The GPU becomes a multi-function device when the HDA is enabled
unsafe { pci_read_config_byte(gpu as *const pci_dev, 14i32, core::ptr::addr_of_mut!(hdr_type)) };
let mut __v1: i32;
loop {
if __bf_cast_unsigned!(128i32) > ((__bf_cast_unsigned__4!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_1304() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v3: i32;
loop {
if false {
unsafe { __compiletime_assert_1305() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_1306() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(128i32 >> (128u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_1307() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)) & 128u64.wrapping_add(1u64 << (128u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_1308() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = ((hdr_type as i32) & 128) >> (128u64.trailing_zeros() as i32);
__v1 = __v3;
unsafe {
(*gpu).__slate_bits_3.__set_multifunction(__v1 as u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_hda_1309: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_hda)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_nvidia_hda_1310: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_nvidia_hda)) };

/// Some IDT switches incorrectly flag an ACS Source Validation error on
/// completions for config read requests even though PCIe r7.0, sec
/// 6.12.1.1, says that completions are never affected by ACS Source
/// Validation.  Here's the text of IDT 89H32H8G3-YC, erratum #36:
///
///   Item #36 - Downstream port applies ACS Source Validation to Completions
///   Section 6.12.1.1 of the PCI Express Base Specification 3.1 states that
///   completions are never affected by ACS Source Validation.  However,
///   completions received by a downstream port of the PCIe switch from a
///   device that has not yet captured a PCIe bus number are incorrectly
///   dropped by ACS Source Validation by the switch downstream port.
///
/// The workaround suggested by IDT is to issue a config write to the
/// downstream device before issuing the first config read.  This allows the
/// downstream device to capture its bus and device numbers (see PCIe r7.0,
/// sec 2.2.9.1), thus avoiding the ACS error on the completion.
///
/// However, we don't know when the device is ready to accept the config
/// write, and the issue affects resets of the switch as well as enumeration,
/// so disable use of ACS SV for these devices altogether.
#[unsafe(no_mangle)]
extern "C" fn pci_disable_broken_acs_cap(pdev: *mut pci_dev) {
if ((unsafe { (*pdev).vendor }) as i32) == 4381 && (((unsafe { (*pdev).device }) as i32) == 32949 || ((unsafe { (*pdev).device }) as i32) == 32912) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Disabling broken ACS SV; downstream device isolation reduced\n".as_ptr() as *const u8) };
let __v3: u16 = (((unsafe { (*pdev).acs_capabilities }) as i32) & !1) as u16;
unsafe {
(*pdev).acs_capabilities = __v3;
}
}
}

/// Microsemi Switchtec NTB uses devfn proxy IDs to move TLPs between
/// NT endpoints via the internal switch fabric. These IDs replace the
/// originating Requester ID TLPs which access host memory on peer NTB
/// ports. Therefore, all proxy IDs must be aliased to the NTB device
/// to permit access when the IOMMU is turned on.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_switchtec_ntb_dma_alias")]
extern "C" fn quirk_switchtec_ntb_dma_alias(pdev: *mut pci_dev) {
let mut mmio_ntb: *mut ntb_info_regs = unsafe { core::mem::zeroed() };
let mut partition_map: u64 = 0;
let mut partition: u8 = 0;
let mut pp: i32 = 0;
if (unsafe { pci_enable_device(pdev) }) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Cannot enable Switchtec device\n".as_ptr() as *const u8) };
return;
}
let mmio: *mut () = unsafe { pci_iomap(pdev, 0i32, 0u64) };
if mmio.is_null() {
unsafe { pci_disable_device(pdev) };
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Cannot iomap Switchtec device\n".as_ptr() as *const u8) };
return;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Setting Switchtec proxy ID aliases\n".as_ptr() as *const u8) };
mmio_ntb = ((unsafe { (mmio as *const u8).wrapping_offset(65536 as isize) }) as *mut ()) as *mut ntb_info_regs;
let mmio_ctrl: *mut ntb_ctrl_regs = ((unsafe { (mmio_ntb as *const u8).wrapping_offset(16384 as isize) }) as *mut ()) as *mut ntb_ctrl_regs;
partition = (unsafe { ioread8(core::ptr::addr_of_mut!((*mmio_ntb).partition_id) as *const ()) }) as u8;
partition_map = (unsafe { ioread32(core::ptr::addr_of_mut!((*mmio_ntb).ep_map) as *const ()) }) as u64;
partition_map |= ((unsafe { ioread32(core::ptr::addr_of_mut!((*mmio_ntb).ep_map).wrapping_offset(4 as isize) as *const ()) }) as u64) << 32i32;
let __v7: u64 = partition_map & !(1u64 << (partition as i32));
partition_map = __v7;
while (pp as u64) < (core::mem::size_of::<u64>() as u64).wrapping_mul(8) {
let mut mmio_peer_ctrl: *mut ntb_ctrl_regs = unsafe { core::mem::zeroed() };
let mut table_sz: u32 = 0;
let mut te: i32 = 0;
if partition_map & 1u64 << pp != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Processing partition %d\n".as_ptr() as *const u8, pp) };
}
mmio_peer_ctrl = mmio_ctrl.wrapping_offset(pp as isize);
table_sz = unsafe { ioread16(core::ptr::addr_of_mut!((*mmio_peer_ctrl).req_id_table_size) as *const ()) };
if table_sz == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Partition %d table_sz 0\n".as_ptr() as *const u8, pp) };
} else {
if table_sz > 512 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Invalid Switchtec partition %d table_sz %d\n".as_ptr() as *const u8, pp, table_sz) };
} else {
te = 0;
while (te as u32) < table_sz {
let rid_entry: u32 = unsafe { ioread32((core::ptr::addr_of_mut!((*mmio_peer_ctrl).req_id_table) as *mut u32).wrapping_offset(te as isize) as *const ()) };
let devfn: u8 = (rid_entry >> 1i32 & 255) as u8;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Aliasing Partition %d Proxy ID %02x.%d\n".as_ptr() as *const u8, pp, PCI_SLOT__3!(devfn), PCI_FUNC__3!(devfn)) };
}
unsafe { pci_add_dma_alias(pdev, devfn, 1u32) };
te = te.wrapping_add(1);
}
}
}
}
pp = pp.wrapping_add(1);
}
unsafe { pci_iounmap(pdev, mmio) };
unsafe { pci_disable_device(pdev) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1311: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 24xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1312: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 32xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1313: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 48xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1314: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 64xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1315: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 80xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1316: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 96xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1317: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 24xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1318: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 32xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1319: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 48xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1320: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 64xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1321: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 80xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1322: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 96xG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1323: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 24XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1324: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 32XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1325: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 48XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1326: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 64XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1327: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 80XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1328: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 96XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1329: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 24XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1330: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 32XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1331: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 48XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1332: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 64XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1333: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 80XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1334: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXL 96XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1335: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 24XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1336: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 32XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1337: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 48XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1338: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 64XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1339: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 80XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1340: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXI 96XG3
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1341: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 100XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1342: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 84XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1343: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 68XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1344: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1345: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1346: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1347: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 100XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1348: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 84XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1349: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 68XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1350: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1351: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1352: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1353: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 100XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1354: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 84XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1355: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 68XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1356: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1357: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1358: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1359: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1360: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1361: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1362: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1363: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1364: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1365: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 52XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1366: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 36XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1367: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 28XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1368: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1369: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1370: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1371: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1372: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1373: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFX 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1374: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1375: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1376: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1377: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1378: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1379: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSX 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1380: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1381: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1382: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1383: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1384: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1385: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAX 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1386: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1387: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1388: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1389: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1390: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1391: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PFXA 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1392: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1393: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1394: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1395: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1396: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1397: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PSXA 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1398: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 100XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1399: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 84XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1400: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 68XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1401: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 52XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1402: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 36XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1403: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PAXA 28XG5
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1404: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1001XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1405: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1002XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1406: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1003XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1407: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1004XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1408: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1005XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1409: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1006XG4
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_switchtec_ntb_dma_alias_1410: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_switchtec_ntb_dma_alias)) };

// PCI1008XG4
/// The PLX NTB uses devfn proxy IDs to move TLPs between NT endpoints.
/// These IDs are used to forward responses to the originator on the other
/// side of the NTB.  Alias all possible IDs to the NTB to permit access when
/// the IOMMU is turned on.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_plx_ntb_dma_alias")]
extern "C" fn quirk_plx_ntb_dma_alias(pdev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Setting PLX NTB proxy ID aliases\n".as_ptr() as *const u8) };
// PLX NTB may use all 256 devfns
unsafe { pci_add_dma_alias(pdev, 0u8, 256u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_plx_ntb_dma_alias_1411: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_plx_ntb_dma_alias)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_plx_ntb_dma_alias_1412: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_plx_ntb_dma_alias)) };

/// On Lenovo Thinkpad P50 SKUs with a Nvidia Quadro M1000M, the BIOS does
/// not always reset the secondary Nvidia GPU between reboots if the system
/// is configured to use Hybrid Graphics mode.  This results in the GPU
/// being left in whatever state it was in during the *previous* boot, which
/// causes spurious interrupts from the GPU, which in turn causes us to
/// disable the wrong IRQ and end up breaking the touchpad.  Unsurprisingly,
/// this also completely breaks nouveau.
///
/// Luckily, it seems a simple reset of the Nvidia GPU brings it back to a
/// clean state and fixes all these issues.
///
/// When the machine is configured in Dedicated display mode, the issue
/// doesn't occur.  Fortunately the GPU advertises NoReset+ when in this
/// mode, so we can detect that and avoid resetting it.
#[unsafe(link_section = ".text.slate_distinct.quirks.quirk_reset_lenovo_thinkpad_p50_nvgpu")]
extern "C" fn quirk_reset_lenovo_thinkpad_p50_nvgpu(pdev: *mut pci_dev) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut map: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(map) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ((*pdev).subsystem_vendor as i32) != 6058 || ((*pdev).subsystem_device as i32) != 8750 {
__v0 = true;
} else {
__v0 = !pci_reset_supported(pdev);
}
if __v0 {
return;
} else {
if pci_enable_device_mem(pdev) != 0 {
return;
} else {
// Based on nvkm_device_ctor() in
// drivers/gpu/drm/nouveau/nvkm/engine/device/base.c
*__slate_slot_2 = pci_iomap(pdev, 0i32, 143360u64);
if (*__slate_slot_2).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Can't map MMIO space\n".as_ptr() as *const u8);
} else {
// Make sure the GPU looks like it's been POSTed before resetting
// it.
if ioread32(((*__slate_slot_2 as *const u8).wrapping_offset(140300 as isize) as *mut ()) as *const ()) & 2 != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"[Firmware Bug]: GPU left initialized by EFI, resetting\n".as_ptr() as *const u8);
*__slate_slot_1 = pci_reset_bus(pdev);
if *__slate_slot_1 < 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Failed to reset GPU: %d\n".as_ptr() as *const u8, *__slate_slot_1);
}
}
iounmap(*__slate_slot_2 as *mut ());
}
pci_disable_device(pdev);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_quirk_reset_lenovo_thinkpad_p50_nvgpu_1413: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(quirk_reset_lenovo_thinkpad_p50_nvgpu)) };

/// Device [1b21:2142]
/// When in D0, PME# doesn't get asserted when plugging USB 3.0 device.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_fixup_no_d0_pme")]
extern "C" fn pci_fixup_no_d0_pme(dev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PME# does not work under D0, disabling it\n".as_ptr() as *const u8) };
let __v2: u32 = unsafe { (*dev).__slate_bits_1.__get_pme_support() };
let __v3: u32 = ((__v2 as i32) & -2i32) as u32;
unsafe {
(*dev).__slate_bits_1.__set_pme_support(__v3);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_no_d0_pme_1414: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_no_d0_pme)) };

/// Device 12d8:0x400e [OHCI] and 12d8:0x400f [EHCI]
///
/// These devices advertise PME# support in all power states but don't
/// reliably assert it.
///
/// These devices also advertise MSI, but documentation (PI7C9X440SL.pdf)
/// says "The MSI Function is not implemented on this device" in chapters
/// 7.3.27, 7.3.29-7.3.31.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_fixup_no_msi_no_pme")]
extern "C" fn pci_fixup_no_msi_no_pme(dev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"MSI is not implemented on this device, disabling it\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_3.__set_no_msi(1u32);
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PME# is unreliable, disabling it\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_1.__set_pme_support(0u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_no_msi_no_pme_1415: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_no_msi_no_pme)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_no_msi_no_pme_1416: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_no_msi_no_pme)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.apex_pci_fixup_class")]
extern "C" fn apex_pci_fixup_class(pdev: *mut pci_dev) {
unsafe {
(*pdev).class = 557056u32 | (*pdev).class;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_apex_pci_fixup_class_1417: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(apex_pci_fixup_class)) };

// Pericom PI7C9X2G404/PI7C9X2G304/PI7C9X2G303 switch erratum E5 -
// ACS P2P Request Redirect is not functional
//
// When ACS P2P Request Redirect is enabled and bandwidth is not balanced
// between upstream and downstream ports, packets are queued in an internal
// buffer until CPLD packet. The workaround is to use the switch in store and
// forward mode.
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_fixup_pericom_acs_store_forward")]
extern "C" fn pci_fixup_pericom_acs_store_forward(pdev: *mut pci_dev) {
let mut upstream: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut val: u16 = 0;
// Downstream ports only
if pci_pcie_type(pdev as *const pci_dev) != 6 {
return;
}
// Check for ACS P2P Request Redirect use
if (unsafe { (*pdev).acs_cap }) == 0 {
return;
}
unsafe { pci_read_config_word(pdev as *const pci_dev, ((*pdev).acs_cap as i32).wrapping_add(6), core::ptr::addr_of_mut!(val)) };
if (val as i32) & 4 == 0 {
return;
}
upstream = pci_upstream_bridge(pdev);
if upstream.is_null() {
return;
}
unsafe { pci_read_config_word(upstream as *const pci_dev, 116i32, core::ptr::addr_of_mut!(val)) };
if ((val as i32) as u64) & BIT!(0i32) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*upstream).dev) as *const device, c"Setting PI7C9X2Gxxx store-forward mode to avoid ACS erratum\n".as_ptr() as *const u8) };
unsafe { pci_write_config_word(upstream as *const pci_dev, 116i32, (((val as i32) as u64) | BIT!(0i32)) as u16) };
}
}

/// Apply fixup on enable and on resume, in order to apply the fix up whenever
/// ACS configuration changes or switch mode is reset
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1418: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1419: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1420: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1421: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1422: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1423: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1424: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1425: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1426: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_pericom_acs_store_forward_1427: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_pericom_acs_store_forward)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.nvidia_ion_ahci_fixup")]
extern "C" fn nvidia_ion_ahci_fixup(__v0: *mut pci_dev) {
unsafe {
(*__v0).dev_flags = (((*__v0).dev_flags as i32) | 4096) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nvidia_ion_ahci_fixup_1428: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(nvidia_ion_ahci_fixup)) };

#[unsafe(link_section = ".text.slate_distinct.quirks.rom_bar_overlap_defect")]
extern "C" fn rom_bar_overlap_defect(dev: *mut pci_dev) {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"working around ROM BAR overlap defect\n".as_ptr() as *const u8) };
unsafe {
(*dev).__slate_bits_3.__set_rom_bar_overlap(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_rom_bar_overlap_defect_1429: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(rom_bar_overlap_defect)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_rom_bar_overlap_defect_1430: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(rom_bar_overlap_defect)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_rom_bar_overlap_defect_1431: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(rom_bar_overlap_defect)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_rom_bar_overlap_defect_1432: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(rom_bar_overlap_defect)) };

// Several Intel DG2 graphics devices advertise that they can only tolerate
// 1us latency when transitioning from L1 to L0, which may prevent ASPM L1
// from being enabled.  But in fact these devices can tolerate unlimited
// latency.  Override their Device Capabilities value to allow ASPM L1 to
// be enabled.
#[unsafe(link_section = ".text.slate_distinct.quirks.aspm_l1_acceptable_latency")]
extern "C" fn aspm_l1_acceptable_latency(dev: *mut pci_dev) {
let mut l1_lat: u32 = 0;
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(3584i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_1433() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_1434() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_1435() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(3584i32 >> (3584u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_1436() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 3584u64.wrapping_add(1u64 << (3584u64.trailing_zeros() as i32)) & 3584u64.wrapping_add(1u64 << (3584u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_1437() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*dev).devcap }) & 3584) >> (3584u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
l1_lat = __v0 as u32;
if l1_lat < 7 {
let mut __v7: i32;
loop {
if (__bf_cast_unsigned!(3584i32) as u64) > __bf_cast_unsigned__3!(18446744073709551615u64) {
unsafe { __compiletime_assert_1438() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: i32;
loop {
if false {
unsafe { __compiletime_assert_1439() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_1440() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(3584i32 >> (3584u64.trailing_zeros() as i32)) & 7i32 } else { 0i32 }) != 0 {
unsafe { __compiletime_assert_1441() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 3584u64.wrapping_add(1u64 << (3584u64.trailing_zeros() as i32)) & 3584u64.wrapping_add(1u64 << (3584u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_1442() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = 7i32 << (3584u64.trailing_zeros() as i32) & 3584;
__v7 = __v9;
unsafe {
(*dev).devcap = (*dev).devcap | (__v7 as u32);
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ASPM: overriding L1 acceptable latency from %#x to 0x7\n".as_ptr() as *const u8, l1_lat) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1443: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1444: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1445: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1446: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1447: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1448: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1449: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1450: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1451: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1452: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1453: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1454: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1455: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1456: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1457: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1458: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1459: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1460: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1461: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1462: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1463: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1464: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1465: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1466: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1467: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_aspm_l1_acceptable_latency_1468: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(aspm_l1_acceptable_latency)) };

/// For a PCI device with multiple downstream devices, its driver may use
/// a flattened device tree to describe the downstream devices.
/// To overlay the flattened device tree, the PCI device and all its ancestor
/// devices need to have device tree nodes on system base device tree. Thus,
/// before driver probing, it might need to add a device tree node as the final
/// fixup.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_of_pci_make_dev_node_1469: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(of_pci_make_dev_node)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_of_pci_make_dev_node_1470: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(of_pci_make_dev_node)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_of_pci_make_dev_node_1471: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(of_pci_make_dev_node)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_of_pci_make_dev_node_1472: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(of_pci_make_dev_node)) };

/// Devices known to require a longer delay before first config space access
/// after reset recovery or resume from D3cold:
///
/// VideoPropulsion (aka Genroco) Torrent QN16e MPEG QAM Modulator
#[unsafe(link_section = ".text.slate_distinct.quirks.pci_fixup_d3cold_delay_1sec")]
extern "C" fn pci_fixup_d3cold_delay_1sec(pdev: *mut pci_dev) {
unsafe {
(*pdev).d3cold_delay = 1000;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_fixup_d3cold_delay_1sec_1473: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_fixup_d3cold_delay_1sec)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_asus_hides_smbus_lpc_ich6_suspend_1: [u8; 1] = *b"\0";

static __slate_asm_string_asus_hides_smbus_lpc_ich6_suspend_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/drivers/pci/quirks.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
