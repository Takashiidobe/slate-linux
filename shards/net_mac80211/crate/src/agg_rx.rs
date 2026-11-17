
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
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! HT handling
//!
//! Copyright 2003, Jouni Malinen <jkmaline@cc.hut.fi>
//! Copyright 2002-2005, Instant802 Networks, Inc.
//! Copyright 2005-2006, Devicescape Software, Inc.
//! Copyright 2006-2007	Jiri Benc <jbenc@suse.cz>
//! Copyright 2007, Michael Wu <flamingice@sourmilk.net>
//! Copyright 2007-2010, Intel Corporation
//! Copyright(c) 2015-2017 Intel Deutschland GmbH
//! Copyright (C) 2018-2026 Intel Corporation
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
//! @a is a power of 2 value
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! ARRAY_SIZE - get the number of elements in array @arr
//! @arr: array to be sized
//! # chars in a path name including nul
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
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
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
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
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
//! __cacheline_group_begin_aligned - declare an aligned group start
//! @GROUP: name of the group
//! @...: optional group alignment
//!
//! The following block inside a struct:
//!
//! 	__cacheline_group_begin_aligned(grp);
//! 	field a;
//! 	field b;
//! 	__cacheline_group_end_aligned(grp);
//!
//! will always be aligned to either the specified alignment or
//! ``SMP_CACHE_BYTES``.
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! The IDs of the various system clocks (for POSIX.1b interval timers):
//! Sign Flag
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! writeable
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! used to decode ioctl numbers..
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
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! We handle most unaligned accesses in hardware.  On the other hand
//! unaligned DMA can be quite expensive on some Nehalem processors.
//!
//! Based on this we disable the IP header alignment in network drivers.
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
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
//! Helper macro to avoid gfp flags if they are the default one
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
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! number of env pointers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Desired design of maximum size and alignment (see RFC2553)
//! Generate SD flag indexes
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
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! copy'n'paste compression ;)
//! Octets in one ethernet addr
//! MQPRIO
//! These macros are put here for binary compatibility with userspace apps that
//! make use of them. For kernel code and new userspace apps, use the TCA_ID_*
//! versions.
//! struct flow_keys_digest:
//!
//! This structure is used to hold a digest of the full flow keys. This is a
//! larger "hash" of a flow to allow definitively matching specific flows where
//! the 32 bit skb->hash is not large enough. The size is limited to 16 bytes so
//! that it can be used in CB of skb (see sch_choke for an example).
//! To all
//!  We overload the LSB of the struct page pointer to indicate whether it's
//! a page or net_iov.
//! Don't change this without changing skb_csum_unnecessary!
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! IPstats
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! 31 pointers + header align the folio_batch structure to a power of two
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! Well, we should have at least one descriptor open
//! to accept passed FDs 8)
//! this can be increased when necessary - don't expose to userland
//! Largest hardware address length
//! packet dropped
//! Helpers for hash table generation of ethernet nics:
//!
//! Ethernet sends the least significant bit of a byte first, thus crc32_le
//! is used. The output of crc32_le is bit reversed [most significant bit
//! is in bit nr 0], thus it must be reversed before use. Except for
//! nics that bit swap the result internally...
//! __get_unaligned_t - read an unaligned value from memory.
//! @type:	the type to load from the pointer.
//! @ptr:	the pointer to load from.
//!
//! Use memcpy to affect an unaligned type sized load avoiding undefined behavior
//! from approaches like type punning that require -fno-strict-aliasing in order
//! to be correct. As type may be const, use __unqual_scalar_typeof to map to a
//! non-const type - you can't memcpy into a const type. The
//! __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
//! expression rather than type, a pointer is used to avoid warnings about mixing
//! the use of 0 and NULL. The void* cast silences ubsan warnings.
//! FIELD_PREP() - prepare a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_val:  value to put in the field
//!
//! FIELD_PREP() masks and shifts up the value.  The result should
//! be combined with other fields of the bitfield using logical OR.
//! management
//! HWMP IE processing macros
//! bits unique to S1G beacon frame control
//! Notice of Absence attribute - described in P2P spec 4.1.14
//!
//! Typical max value used here
//! define userspace visible states
//! __jhash_mix - mix 3 32-bit values reversibly.
//! Miscellaneous stuff.
//! 									 * Even if you want to feed it just 0/1 you should have			 * some bits for the non-fractional part...
//! 0x00 == ___RX_DROP_UNUSABLE
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
// Used by functions that use '__builtin_return_address'. These function
// don't want to be splited or made inline, which can make
// the '__builtin_return_address' get unexpected address.
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
// __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// sizeof_field() - Report the size of a struct field in bytes
//
// @TYPE: The structure containing the field of interest
// @MEMBER: The field to return the size of
// offsetofend() - Report the offset of a struct field within the struct
//
// @TYPE: The type of the structure
// @MEMBER: The member within the structure to get the end offset of
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
// struct_group_tagged() - Create a struct_group with a reusable tag
//
// @TAG: The tag name for the named sub-struct
// @NAME: The identifier name of the mirrored sub-struct
// @MEMBERS: The member declarations for the mirrored structs
//
// Used to create an anonymous union of two structs with identical
// layout and size: one anonymous and one named. The former can be
// used normally without sub-struct naming, and the latter can be
// used to reason about the start, end, and size of the group of
// struct members. Includes struct tag argument for the named copy,
// so the specified layout can be reused later.
// no attrs
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
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
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Generic interface.
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
// max_t - return maximum of two values, using the specified type
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
// swap - swap values of @a and @b
// @a: first value
// @b: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// AUX clock support. AUXiliary clocks are dynamically configured by
// enabling a clock ID. These clock can be steered independently of the
// core timekeeper. The kernel can support up to 8 auxiliary clocks, but
// the actual limit depends on eventual architecture constraints vs. VDSO.
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// File exists
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Operation now in progress
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// memset_after - Set a value after a struct member to the end of a struct
//
// @obj: Address of target struct instance
// @v: Byte value to repeatedly write
// @member: after which struct member to start writing bytes
//
// This is good for clearing padding following the given member.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// list_for_each_entry_rcu	-	iterate over rcu list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// @cond:	optional lockdep expression if called from non-RCU protection.
//
// This list-traversal primitive may safely run concurrently with
// the _rcu list-mutation primitives such as list_add_rcu()
// as long as the traversal is guarded by rcu_read_lock().
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Implementation specific max size
// Generate SD flag bits
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
// Segmentation offload feature mask
// If one device supports one of these features, then enable them
// for all in netdev_increment_features.
// If one device doesn't support one of these features, then disable it
// for all in netdev_increment_features.
// obsoleted, can be reused
// To group
// To someone else
// XXX: How to extract netmem_desc from page must be changed, once
// netmem_desc no longer overlays on page and will be allocated through
// slab.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
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
// PT_TRACE_* event enable flags
// address is generated randomly
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
// __jhash_final - final mixing of 3 32-bit values (a,b,c) into c
// An arbitrary initial parameter
// 0x10
// 0x20
// 0x30
// 0x40
// 0x50
// this line for the trailing \ - add before this
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

macro_rules! GENMASK_INPUT_CHECK__7 {
() => {
core::mem::size_of::<__SlateRecord_agg_rx_c_192>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__10 {
() => {
core::mem::size_of::<__SlateRecord_agg_rx_c_221>() as i32
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
(GENMASK_INPUT_CHECK__7!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__8 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__10!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! GENMASK__7 {
($h:expr, $l:expr) => {
GENMASK_TYPE__8!(u64, $h, $l)
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

macro_rules! TU_TO_EXP_TIME {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(TU_TO_JIFFIES!($x))
};
}

/// convert time units
macro_rules! TU_TO_JIFFIES {
($x:expr) => {
usecs_to_jiffies((($x) as i32).wrapping_mul(1024) as u32)
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

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__39 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut tid_ampdu_rx, ($val) as *mut tid_ampdu_rx) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__41 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut tid_ampdu_rx, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut sk_buff, $val) };
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

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
}
};
}

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_b52e897e44daaa8($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
}
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

macro_rules! __must_be_array__30 {
() => {
core::mem::size_of::<__SlateRecord_agg_rx_c_413_4>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__36 {
() => {
((const { core::mem::size_of::<*mut tid_ampdu_rx>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tid_ampdu_rx>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tid_ampdu_rx>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tid_ampdu_rx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__12 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __rcu_dereference_protected__24 {
($p:expr, $c:expr) => {
{
let mut __v: *mut tid_ampdu_rx;
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
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
macro_rules! __set_bit__4 {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_b52e897e44daaa8($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__4 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
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
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut $type;
__v
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
macro_rules! container_of__9 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut $type;
__v
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
macro_rules! container_of__55 {
($ptr:expr) => {
{
let mut __v: *mut tid_ampdu_rx;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__12!() as isize)) }) as *mut ()) as *mut tid_ampdu_rx;
__v
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
macro_rules! container_of__57 {
($ptr:expr) => {
{
let mut __v: *mut tid_ampdu_rx;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!() as isize)) }) as *mut ()) as *mut tid_ampdu_rx;
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
macro_rules! flex_array_size__11 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<__SlateRecord_sta_info_h_259>() as u64).wrapping_add(__must_be_array__30!() as u64))
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

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
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

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

/// this is a define for better error reporting (file/line)
macro_rules! lockdep_assert_wiphy {
($wiphy:expr) => {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*$wiphy).mtx) });
};
}

macro_rules! offsetof__12 {
() => {
40u64
};
}

macro_rules! offsetof__36 {
() => {
33u64
};
}

macro_rules! offsetof__15 {
() => {
80u64
};
}

macro_rules! offsetof {
() => {
0u64
};
}

macro_rules! offsetof__4 {
() => {
5072u64
};
}

macro_rules! raw_spin_lock_bh {
($lock:expr) => {
unsafe { _raw_spin_lock_bh($lock) };
};
}

macro_rules! raw_spin_unlock_bh {
($lock:expr) => {
unsafe { _raw_spin_unlock_bh($lock) };
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
macro_rules! rcu_dereference_protected__16 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__24!($p, $c)
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

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__7 {
($count:expr) => {
size_add(144u64, flex_array_size__11!($count))
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
struct_size__7!($count)
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_b52e897e44daaa8($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(($nr) as u32) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit((($nr) as u32) as u64, ($addr) as *const u64) as i32
} else {
_test_bit((($nr) as u32) as u64, ($addr) as *const u64) as i32
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
macro_rules! time_after__3 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__3!(u64) != 0;
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
time_after__3!($b, $a)
};
}

/// time_is_after_jiffies - return true if a is after jiffies
/// @a: time (unsigned long) to compare to jiffies
///
/// Return: %true is time a is after jiffies, otherwise %false.
macro_rules! time_is_after_jiffies {
($a:expr) => {
time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, $a)
};
}

macro_rules! timer_container_of__2 {
($callback_timer:expr) => {
container_of__57!($callback_timer)
};
}

macro_rules! timer_container_of__4 {
($callback_timer:expr) => {
container_of__55!($callback_timer)
};
}

/// timer_setup - prepare a timer for first use
/// @timer: the timer in question
/// @callback: the function to call when timer expires
/// @flags: any TIMER_* flags
///
/// Regular timer initialization should use either DEFINE_TIMER() above,
/// or timer_setup(). For timers on the stack, timer_setup_on_stack() must
/// be used and must be balanced with a call to timer_destroy_on_stack().
macro_rules! timer_setup {
($timer:expr, $callback:expr, $flags:expr) => {
__timer_init!($timer, $callback, $flags);
};
}

macro_rules! type_max {
() => {
__type_max!()
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
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    static mut rfc1042_header: [u8; 6];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn _raw_spin_lock_bh(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_bh(arg0: *mut raw_spinlock);
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn __compiletime_assert_158();
    fn __usecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn timer_shutdown_sync(arg0: *mut timer_list) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn sk_skb_reason_drop(arg0: *const sock, arg1: *mut sk_buff, arg2: u32);
    fn __compiletime_assert_581();
    fn __compiletime_assert_582();
    fn __compiletime_assert_583();
    fn skb_put(arg0: *mut sk_buff, arg1: u32) -> *mut ();
    fn __netdev_alloc_skb(arg0: *mut net_device, arg1: u32, arg2: u32) -> *mut sk_buff;
    fn net_ratelimit() -> i32;
    fn __field_overflow();
    fn __bad_mask();
    fn wiphy_work_queue(arg0: *mut wiphy, arg1: *mut wiphy_work);
    fn sta_info_get_bss(arg0: *mut ieee80211_sub_if_data, arg1: *const u8) -> *mut sta_info;
    fn ieee80211_channel_switch(arg0: *mut wiphy, arg1: *mut net_device, arg2: *mut cfg80211_csa_settings) -> i32;
    fn ieee80211_send_delba(arg0: *mut ieee80211_sub_if_data, arg1: *const u8, arg2: u16, arg3: u16, arg4: u16, arg5: bool);
    fn ieee80211_release_reorder_timeout(arg0: *mut sta_info, arg1: i32);
    fn ieee80211_s1g_use_ndp_ba(arg0: *const ieee80211_sub_if_data, arg1: *const sta_info) -> bool;
    fn ieee80211_tx_skb_tid(arg0: *mut ieee80211_sub_if_data, arg1: *mut sk_buff, arg2: i32, arg3: i32);
    fn ieee802_11_parse_elems_full(arg0: *mut ieee80211_elems_parse_params) -> *mut ieee802_11_elems;
    fn drv_ampdu_action(arg0: *mut ieee80211_local, arg1: *mut ieee80211_sub_if_data, arg2: *mut ieee80211_ampdu_params) -> i32;
    fn __compiletime_assert_2222();
    fn __compiletime_assert_2225();
    fn __compiletime_assert_2226();
    fn __compiletime_assert_2227();
    fn __compiletime_assert_2228();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
    #[link_name = "llvm.is.constant.i16"]
    fn __slate_intrinsic_b52e897e44daaa8(arg0: u16) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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
struct kcov_common_handle_id {
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


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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
struct lock_class_key {
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
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

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
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

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
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

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

// __cacheline_group_end_aligned - declare an aligned group end
// @GROUP: name of the group
// @...: optional alignment (same as was in __cacheline_group_begin_aligned())
//
// Note that the end marker is aligned to sizeof(long) to allow more precise
// size assertion. It also declares a padding at the end to avoid next field
// falling into this cacheline.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_5() };
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
__verify_pcpu_ptr__4!();
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
struct perf_event {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
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
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits124U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits129U0,
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
    __slate_bits_0: __slate_bits::__SlateBits130U0,
    pkru: u32,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
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
struct futex_private_hash {
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
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct timerqueue_head {
    rb_root: rb_root_cached,
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
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn spin_lock_bh(lock: *mut spinlock) {
raw_spin_lock_bh!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn spin_unlock_bh(lock: *mut spinlock) {
raw_spin_unlock_bh!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
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
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct bpf_local_storage {
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits287U0,
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
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
    __slate_bits_0: __slate_bits::__SlateBits76U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits76U1,
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
    __slate_bits_2: __slate_bits::__SlateBits76U2,
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
struct vm_struct {
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
struct mem_cgroup {
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


fn rcu_read_lock_held() -> i32 {
1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


fn rcu_read_lock_bh_held() -> i32 {
1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
// rcu_dereference_all_check() - rcu_dereference_all with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// This is similar to rcu_dereference_check(), but allows protection
// by all forms of vanilla RCU readers, including preemption disabled,
// bh-disabled, and interrupt-disabled regions of code.  Note that "vanilla
// RCU" excludes SRCU and the various Tasks RCU flavors.  Please note
// that this macro should not be backported to any Linux-kernel version
// preceding v5.0 due to changes in synchronize_rcu() semantics prior
// to that version.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
// rcu_dereference_all() - fetch RCU-all-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// Makes rcu_dereference_check() do the dirty work.
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


// time_after_eq - returns true if the time a is after or the same as time b.
// @a: first comparable as unsigned long
// @b: second comparable as unsigned long
//
// Return: %true is time a is after or the same as time b, otherwise %false.
// time_before_eq - returns true if the time a is before or the same as time b.
// @a: first comparable as unsigned long
// @b: second comparable as unsigned long
//
// Return: %true is time a is before or the same as time b, otherwise %false.
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
fn jiffies_to_usecs(j: u64) -> u32 {
// Hz usually doesn't go much further MSEC_PER_SEC.
// jiffies_to_usecs() and usecs_to_jiffies() depend on that.
loop {
if false {
unsafe { __compiletime_assert_158() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
1000u64.wrapping_mul(j) as u32
}

fn _usecs_to_jiffies(u: u32) -> u64 {
((u as i64).wrapping_add(1000000 / 1000).wrapping_sub(1) / (1000000 / 1000)) as u64
}

#[inline(always)]
fn usecs_to_jiffies(u: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(u) } {
if u > jiffies_to_usecs(4611686018427387902u64) {
return 4611686018427387902u64;
}
return _usecs_to_jiffies(u);
} else {
return unsafe { __usecs_to_jiffies(u) };
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
struct uprobe_xol_ops {
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
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
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
struct xol_area {
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blocking_notifier_head {
    rwsem: rw_semaphore,
    head: *mut notifier_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_notifier_head {
    head: *mut notifier_block,
}


// Suits me
// Don't call further
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
struct return_consumer {
    cookie: u64,
    id: u64,
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
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits418U0,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
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
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
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
struct rhlist_head {
    rhead: rhash_head,
    next: *mut rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhashtable_compare_arg {
    ht: *mut rhashtable,
    key: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhashtable_params {
    nelem_hint: u16,
    key_len: u16,
    key_offset: u16,
    head_offset: u16,
    max_size: u32,
    min_size: u16,
    insecure_elasticity: bool,
    automatic_shrinking: bool,
    hashfn: Option<unsafe extern "C" fn(*const (), u32, u32) -> u32>,
    obj_hashfn: Option<unsafe extern "C" fn(*const (), u32, u32) -> u32>,
    obj_cmpfn: Option<unsafe extern "C" fn(*mut rhashtable_compare_arg, *const ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhashtable {
    tbl: *mut bucket_table,
    key_len: u32,
    max_elems: u32,
    p: rhashtable_params,
    rhlist: bool,
    run_work: work_struct,
    run_irq_work: irq_work,
    mutex: mutex,
    lock: spinlock,
    nelems: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhltable {
    ht: rhashtable,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct fserror_event {
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
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
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
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
    __slate_bits_0: __slate_bits::__SlateBits585U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct io_uring_cmd {
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
unsafe { __compiletime_assert_390() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 3; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
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
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct proto_accept_arg {
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
union __SlateRecord_socket_h_84 {
    msg_control: *mut (),
    msg_control_user: *mut (),
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
union __SlateRecord_in6_h_34 {
    u6_addr8: [u8; 16],
    u6_addr16: [u16; 8],
    u6_addr32: [u32; 4],
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
    __slate_bits_0: __slate_bits::__SlateBits659U0,
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits597U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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


// == WRITE
// == READ
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
struct sockaddr {
    sa_family: u16,
    sa_data: [u8; 14],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sockaddr_unsized {
    sa_family: u16,
    sa_data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msghdr {
    msg_name: *mut (),
    msg_namelen: i32,
    msg_inq: i32,
    msg_iter: iov_iter,
    __slate_anon_4: __SlateRecord_socket_h_84,
    __slate_bits_0: __slate_bits::__SlateBits686U0,
    msg_flags: u32,
    msg_controllen: u64,
    msg_ubuf: *mut ubuf_info,
    sg_from_iter: Option<unsafe extern "C" fn(*mut sk_buff, *mut iov_iter, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct in6_addr {
    in6_u: __SlateRecord_in6_h_34,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_if_h_210 {
    raw_hdlc: *mut __SlateRecord_ioctl_h_56,
    cisco: *mut __SlateRecord_ioctl_h_80,
    fr: *mut __SlateRecord_ioctl_h_66,
    fr_pvc: *mut __SlateRecord_ioctl_h_70,
    fr_pvc_info: *mut __SlateRecord_ioctl_h_75,
    x25: *mut __SlateRecord_ioctl_h_89,
    sync: *mut __SlateRecord_ioctl_h_44,
    te1: *mut __SlateRecord_ioctl_h_51,
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
union __SlateRecord_if_h_236 {
    ifrn_name: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_if_h_241 {
    ifru_addr: sockaddr,
    ifru_dstaddr: sockaddr,
    ifru_broadaddr: sockaddr,
    ifru_netmask: sockaddr,
    ifru_hwaddr: sockaddr,
    ifru_flags: i16,
    ifru_ivalue: i32,
    ifru_mtu: i32,
    ifru_map: ifmap,
    ifru_slave: [u8; 16],
    ifru_newname: [u8; 16],
    ifru_data: *mut (),
    ifru_settings: if_settings,
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
    __slate_bits_0: __slate_bits::__SlateBits702U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits702U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits702U2,
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
    __slate_bits_3: __slate_bits::__SlateBits702U3,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct __SlateRecord_ioctl_h_44 {
    clock_rate: u32,
    clock_type: u32,
    loopback: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_51 {
    clock_rate: u32,
    clock_type: u32,
    loopback: u16,
    slot_map: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_56 {
    encoding: u16,
    parity: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_66 {
    t391: u32,
    t392: u32,
    n391: u32,
    n392: u32,
    n393: u32,
    lmi: u16,
    dce: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_70 {
    dlci: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_75 {
    dlci: u32,
    master: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_80 {
    interval: u32,
    timeout: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ioctl_h_89 {
    dce: u16,
    modulo: u32,
    window: u32,
    t1: u32,
    t2: u32,
    n2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ifmap {
    mem_start: u64,
    mem_end: u64,
    base_addr: u16,
    irq: u8,
    dma: u8,
    port: u8,
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
struct if_settings {
    r#type: u32,
    size: u32,
    ifs_ifsu: __SlateRecord_if_h_210,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ifreq {
    ifr_ifrn: __SlateRecord_if_h_236,
    ifr_ifru: __SlateRecord_if_h_241,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
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
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
union __SlateRecord_skbuff_h_339 {
    __slate_anon_0: __SlateRecord_skbuff_h_339_2,
    list: sk_buff_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_339_2 {
    next: *mut sk_buff,
    prev: *mut sk_buff,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sk_buff_list {
    next: *mut sk_buff,
    prev: *mut sk_buff,
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
union __SlateRecord_skbuff_h_463 {
    hwtstamp: i64,
    netdev_data: *mut (),
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
union __SlateRecord_skbuff_h_887 {
    __slate_anon_0: __SlateRecord_skbuff_h_888,
    rbnode: rb_node,
    list: list_head,
    ll_node: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_888 {
    next: *mut sk_buff,
    prev: *mut sk_buff,
    __slate_anon_2: __SlateRecord_skbuff_h_893,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_893 {
    dev: *mut net_device,
    dev_scratch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_909 {
    tstamp: i64,
    skb_mstamp_ns: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_921 {
    __slate_anon_0: __SlateRecord_skbuff_h_922,
    tcp_tsorted_anchor: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_922 {
    _skb_refdst: u64,
    destructor: Option<unsafe extern "C" fn(*mut sk_buff)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_970 {
    __slate_anon_0: __SlateRecord_skbuff_h_970_2,
    headers: __SlateRecord_skbuff_h_970_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_970_2 {
    __pkt_type_offset: [u8; 0],
    __slate_bits_0: __slate_bits::__SlateBits817U0,
    __mono_tc_offset: [u8; 0],
    __slate_bits_1: __slate_bits::__SlateBits817U1,
    __slate_bits_2: __slate_bits::__SlateBits817U2,
    tc_index: u16,
    alloc_cpu: u16,
    __slate_anon_30: __SlateRecord_skbuff_h_1041,
    priority: u32,
    skb_iif: i32,
    hash: u32,
    __slate_anon_34: __SlateRecord_skbuff_h_1051,
    __slate_anon_35: __SlateRecord_skbuff_h_1059,
    secmark: u32,
    __slate_anon_37: __SlateRecord_skbuff_h_1068,
    __slate_anon_38: __SlateRecord_skbuff_h_1073,
    inner_transport_header: u16,
    inner_network_header: u16,
    inner_mac_header: u16,
    protocol: u16,
    transport_header: u16,
    network_header: u16,
    mac_header: u16,
    kcov_handle: kcov_common_handle_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1041 {
    csum: u32,
    __slate_anon_1: __SlateRecord_skbuff_h_1043,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_1043 {
    csum_start: u16,
    csum_offset: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1051 {
    vlan_all: u32,
    __slate_anon_1: __SlateRecord_skbuff_h_1053,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_1053 {
    vlan_proto: u16,
    vlan_tci: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1059 {
    napi_id: u32,
    sender_cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1068 {
    mark: u32,
    reserved_tailroom: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1073 {
    inner_protocol: u16,
    inner_ipproto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_970_3 {
    __pkt_type_offset: [u8; 0],
    __slate_bits_0: __slate_bits::__SlateBits825U0,
    __mono_tc_offset: [u8; 0],
    __slate_bits_1: __slate_bits::__SlateBits825U1,
    __slate_bits_2: __slate_bits::__SlateBits825U2,
    tc_index: u16,
    alloc_cpu: u16,
    __slate_anon_30: __SlateRecord_skbuff_h_1041_2,
    priority: u32,
    skb_iif: i32,
    hash: u32,
    __slate_anon_34: __SlateRecord_skbuff_h_1051_2,
    __slate_anon_35: __SlateRecord_skbuff_h_1059_2,
    secmark: u32,
    __slate_anon_37: __SlateRecord_skbuff_h_1068_2,
    __slate_anon_38: __SlateRecord_skbuff_h_1073_2,
    inner_transport_header: u16,
    inner_network_header: u16,
    inner_mac_header: u16,
    protocol: u16,
    transport_header: u16,
    network_header: u16,
    mac_header: u16,
    kcov_handle: kcov_common_handle_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1041_2 {
    csum: u32,
    __slate_anon_1: __SlateRecord_skbuff_h_1043_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_1043_2 {
    csum_start: u16,
    csum_offset: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1051_2 {
    vlan_all: u32,
    __slate_anon_1: __SlateRecord_skbuff_h_1053_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_skbuff_h_1053_2 {
    vlan_proto: u16,
    vlan_tci: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1059_2 {
    napi_id: u32,
    sender_cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1068_2 {
    mark: u32,
    reserved_tailroom: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_skbuff_h_1073_2 {
    inner_protocol: u16,
    inner_ipproto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dst_entry {
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
struct prot_inuse {
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
    __slate_bits_0: __slate_bits::__SlateBits699U0,
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
struct __SlateRecord_siphash_h_21 {
    key: [u64; 2],
}


// Total octets in header.
// Min. octets in frame sans FCS
// Octets in the FCS
// Internet Protocol packet
// IPv6 over bluebook
// Fibre Channel over Ethernet
// TDLS
// If the value in the ethernet type is more than this value
// then the frame is Ethernet II. Else it is 802.3
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ethhdr {
    h_dest: [u8; 6],
    h_source: [u8; 6],
    h_proto: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ip_conntrack_stat {
    found: u32,
    invalid: u32,
    insert: u32,
    insert_failed: u32,
    clash_resolve: u32,
    drop: u32,
    early_drop: u32,
    error: u32,
    expect_new: u32,
    expect_create: u32,
    expect_delete: u32,
    search_restart: u32,
    chaintoolong: u32,
}


// Maximum value in skb->csum_level
#[repr(C)]
#[derive(Clone, Copy)]
struct sk_buff_head {
    __slate_anon_0: __SlateRecord_skbuff_h_339,
    qlen: u32,
    lock: spinlock,
}


// Set skb_shinfo(skb)->gso_size to this in case you want skb_segment to
// segment using its current segmentation instead.
#[repr(C)]
#[derive(Clone, Copy)]
struct skb_shared_hwtstamps {
    __slate_anon_0: __SlateRecord_skbuff_h_463,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ubuf_info_ops {
    complete: Option<unsafe extern "C" fn(*mut sk_buff, *mut ubuf_info, bool)>,
    link_skb: Option<unsafe extern "C" fn(*mut sk_buff, *mut ubuf_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ubuf_info {
    ops: *const ubuf_info_ops,
    refcnt: refcount_struct,
    flags: u8,
}


// DOC: dataref and headerless skbs
//
// Transport layers send out clones of payload skbs they hold for
// retransmissions. To allow lower layers of the stack to prepend their headers
// we split &skb_shared_info.dataref into two halves.
// The lower 16 bits count the overall number of references.
// The higher 16 bits indicate how many of the references are payload-only.
// skb_header_cloned() checks if skb is allowed to add / write the headers.
//
// The creator of the skb (e.g. TCP) marks its skb as &sk_buff.nohdr
// (via __skb_header_release()). Any clone created from marked skb will get
// &sk_buff.hdr_len populated with the available headroom.
// If there's the only clone in existence it's able to modify the headroom
// at will. The sequence of calls inside the transport layer is::
//
//  <alloc skb>
//  skb_reserve()
//  __skb_header_release()
//  skb_clone()
//  // send the clone down the stack
//
// This is not a very generic construct and it depends on the transport layers
// doing the right thing. In practice there's usually only one payload-only skb.
// Having multiple payload-only skbs with different lengths of hdr_len is not
// possible. The payload-only skbs should never leave their owner.
#[repr(C)]
#[derive(Clone, Copy)]
struct sk_buff {
    __slate_anon_0: __SlateRecord_skbuff_h_887,
    sk: *mut sock,
    __slate_anon_2: __SlateRecord_skbuff_h_909,
    cb: [u8; 48],
    __slate_anon_4: __SlateRecord_skbuff_h_921,
    _nfct: u64,
    len: u32,
    data_len: u32,
    mac_len: u16,
    hdr_len: u16,
    queue_mapping: u16,
    __cloned_offset: [u8; 0],
    __slate_bits_0: __slate_bits::__SlateBits681U0,
    active_extensions: u8,
    __slate_anon_20: __SlateRecord_skbuff_h_970,
    tail: u32,
    end: u32,
    head: *mut u8,
    data: *mut u8,
    truesize: u32,
    users: refcount_struct,
    extensions: *mut skb_ext,
}


// skb might have a dst pointer attached, refcounted or not.
// _skb_refdst low order bit is set if refcount was _not_ taken
fn skb_dst(skb: &sk_buff) -> *mut dst_entry {
// If refdst was not refcounted, check we still are in a
// rcu_read_lock section
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = (unsafe { (*(skb as *const sk_buff)).__slate_anon_4.__slate_anon_0._skb_refdst }) & 1 != 0 && rcu_read_lock_held() == 0;
let __v2: bool = __v1 && rcu_read_lock_bh_held() == 0;
__ret_warn_on = __v2 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("562: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 562b - ., 3; .popsection\n# {0}", const 562 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_skb_dst_3, sym __slate_asm_string_skb_dst_4, const 1164 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("563: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 563b - ., 4; .popsection\n# {0}", const 563 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
((unsafe { (*(skb as *const sk_buff)).__slate_anon_4.__slate_anon_0._skb_refdst }) & !1) as *mut dst_entry
}

fn kfree_skb_reason(skb: *mut sk_buff, reason: u32) {
unsafe { sk_skb_reason_drop(core::ptr::null::<sock>(), skb, reason) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_inet_frag_h_87 {
    v4: frag_v4_compare_key,
    v6: frag_v6_compare_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipv4_devconf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fib_rules_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fib_table {
}


// Internal
fn skb_peek(list_: *const sk_buff_head) -> *mut sk_buff {
let mut skb: *mut sk_buff = unsafe { (*list_).__slate_anon_0.__slate_anon_0.next };
if skb == (list_ as *mut sk_buff) {
skb = core::ptr::null_mut::<sk_buff>();
}
skb
}

fn __skb_queue_head_init(list: *mut sk_buff_head) {
let __v0: *mut sk_buff = list as *mut sk_buff;
unsafe {
(*list).__slate_anon_0.__slate_anon_0.next = __v0;
}
unsafe {
(*list).__slate_anon_0.__slate_anon_0.prev = __v0;
}
unsafe {
(*list).qlen = 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct inet_hashinfo {
}


fn __skb_unlink(skb: *mut sk_buff, list: *mut sk_buff_head) {
let mut next: *mut sk_buff = unsafe { core::mem::zeroed() };
let mut prev: *mut sk_buff = unsafe { core::mem::zeroed() };
loop {
loop {
if !(__native_word__12!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_581() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*list).qlen, unsafe { (*list).qlen }.wrapping_sub(1));
if true {
break;
}
}
next = unsafe { (*skb).__slate_anon_0.__slate_anon_0.next };
prev = unsafe { (*skb).__slate_anon_0.__slate_anon_0.prev };
unsafe {
(*skb).__slate_anon_0.__slate_anon_0.prev = core::ptr::null_mut::<sk_buff>();
}
unsafe {
(*skb).__slate_anon_0.__slate_anon_0.next = core::ptr::null_mut::<sk_buff>();
}
loop {
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_582() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*next).__slate_anon_0.__slate_anon_0.prev, prev);
if true {
break;
}
}
loop {
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut sk_buff>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_583() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*prev).__slate_anon_0.__slate_anon_0.next, next);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tcp_fastopen_context {
}


fn __skb_dequeue(list: *mut sk_buff_head) -> *mut sk_buff {
let skb: *mut sk_buff = skb_peek(list as *const sk_buff_head);
if !skb.is_null() {
__skb_unlink(skb, list);
}
skb
}

fn skb_put_zero(skb: *mut sk_buff, len: u32) -> *mut () {
let tmp: *mut () = unsafe { skb_put(skb, len) };
unsafe { memset(tmp, 0i32, len as u64) };
tmp
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ipv4_h_96 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct udp_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ip_ra_chain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inet_peer_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tcp_congestion_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mr_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fib_notifier_ops {
}


fn skb_reserve(skb: *mut sk_buff, len: i32) {
let __v1: *mut u8 = unsafe { (*skb).data };
unsafe {
(*skb).data = __v1.wrapping_offset(len as isize);
}
let __v4: u32 = unsafe { (*skb).tail };
unsafe {
(*skb).tail = __v4.wrapping_add(len as u32);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct neighbour {
}


fn __skb_queue_purge_reason(list: *mut sk_buff_head, reason: u32) {
loop {
let __v0: *mut sk_buff = __skb_dequeue(list);
let skb: *mut sk_buff = __v0;
if __v0.is_null() {
break;
}
kfree_skb_reason(skb, reason);
}
}

fn __skb_queue_purge(list: *mut sk_buff_head) {
__skb_queue_purge_reason(list, 103u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ipv6_devconf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fib6_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt6_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt6_statistics {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fib6_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seg6_pernet_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ipv6_h_124 {
    head: hlist_head,
    lock: spinlock,
    seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioam6_pernet_data {
}


fn netdev_alloc_skb(dev: *mut net_device, length: u32) -> *mut sk_buff {
unsafe { __netdev_alloc_skb(dev, length, (BIT!(5i32) as u32) | (BIT!(11i32) as u32)) }
}

fn dev_alloc_skb(length: u32) -> *mut sk_buff {
netdev_alloc_skb(core::ptr::null_mut::<net_device>(), length)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct nf_logger {
}


// For small packets <= CHECKSUM_BREAK perform checksum complete directly
// in checksum_init.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct skb_ext {
    refcnt: refcount_struct,
    offset: [u8; 1],
    chunks: u8,
    __slate_pad_3: [u8; 2],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_hook_entries {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local_h_12 {
    a: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dql {
    num_queued: u32,
    adj_limit: u32,
    last_obj_cnt: u32,
    stall_thrs: u16,
    history_head: u64,
    history: [u64; 4],
    __slate_pad_6: [u8; 8],
    limit: u32,
    num_completed: u32,
    prev_ovlimit: u32,
    prev_num_queued: u32,
    prev_last_obj_cnt: u32,
    lowest_slack: u32,
    slack_start_time: u64,
    max_limit: u32,
    min_limit: u32,
    slack_hold_time: u32,
    stall_max: u16,
    last_reap: u64,
    stall_cnt: u64,
}


// Set some static maximums
#[repr(C)]
#[derive(Clone, Copy)]
struct netns_core {
    sysctl_hdr: *mut ctl_table_header,
    sysctl_somaxconn: i32,
    sysctl_txq_reselection: i32,
    sysctl_optmem_max: i32,
    sysctl_txrehash: u8,
    sysctl_tstamp_allow_data: u8,
    sysctl_bypass_prot_mem: u8,
    prot_inuse: *mut prot_inuse,
    rps_default_mask: *mut cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local64_h_26 {
    a: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_u64_stats_sync_h_75 {
    v: __SlateRecord_local64_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_ct_event_notifier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipstats_mib {
    mibs: [u64; 38],
    syncp: u64_stats_sync,
}


// ICMP
#[repr(C)]
#[derive(Clone, Copy)]
struct icmp_mib {
    mibs: [u64; 30],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icmpmsg_mib {
    mibs: [__SlateRecord_types_h_197; 512],
}


// ICMP6 (IPv6-ICMP)
#[repr(C)]
#[derive(Clone, Copy)]
struct icmpv6_mib {
    mibs: [u64; 7],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icmpv6msg_mib {
    mibs: [__SlateRecord_types_h_197; 512],
}


// TCP
#[repr(C)]
#[derive(Clone, Copy)]
struct tcp_mib {
    mibs: [u64; 16],
}


// UDP
#[repr(C)]
#[derive(Clone, Copy)]
struct udp_mib {
    mibs: [u64; 10],
}


// Linux
#[repr(C)]
#[derive(Clone, Copy)]
struct linux_mib {
    mibs: [u64; 136],
}


// Linux Xfrm
// Linux TLS
#[repr(C)]
#[derive(Clone, Copy)]
struct netns_mib {
    ip_statistics: *mut ipstats_mib,
    ipv6_statistics: *mut ipstats_mib,
    tcp_statistics: *mut tcp_mib,
    net_statistics: *mut linux_mib,
    udp_statistics: *mut udp_mib,
    udp_stats_in6: *mut udp_mib,
    icmp_statistics: *mut icmp_mib,
    icmpmsg_statistics: *mut icmpmsg_mib,
    icmpv6_statistics: *mut icmpv6_mib,
    icmpv6msg_statistics: *mut icmpv6msg_mib,
    proc_net_devsnmp6: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unix_table {
    locks: *mut spinlock,
    buckets: *mut hlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_unix {
    table: unix_table,
    sysctl_max_dgram_qlen: i32,
    ctl: *mut ctl_table_header,
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
struct net_generic {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uevent_sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_packet {
    sklist_lock: mutex,
    sklist: hlist_head,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct fqdir {
    high_thresh: i64,
    low_thresh: i64,
    timeout: i32,
    max_dist: i32,
    f: *mut inet_frags,
    net: *mut net,
    dead: bool,
    __slate_pad_7: [u8; 23],
    rhashtable: rhashtable,
    __slate_pad_9: [u8; 32],
    mem: __SlateRecord_types_h_197,
    destroy_work: work_struct,
    free_list: llist_node,
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
struct frag_v4_compare_key {
    saddr: u32,
    daddr: u32,
    user: u32,
    vif: u32,
    id: u16,
    protocol: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct frag_v6_compare_key {
    saddr: in6_addr,
    daddr: in6_addr,
    user: u32,
    id: u32,
    iif: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_697 {
    func: Option<unsafe extern "C" fn(u64)>,
    callback: Option<unsafe extern "C" fn(*mut tasklet_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inet_frag_queue {
    node: rhash_head,
    key: __SlateRecord_inet_frag_h_87,
    timer: timer_list,
    lock: spinlock,
    refcnt: refcount_struct,
    rb_fragments: rb_root,
    fragments_tail: *mut sk_buff,
    last_run_head: *mut sk_buff,
    stamp: i64,
    len: i32,
    meat: i32,
    tstamp_type: u8,
    flags: u8,
    max_size: u16,
    fqdir: *mut fqdir,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inet_frags {
    qsize: u32,
    constructor: Option<unsafe extern "C" fn(*mut inet_frag_queue, *const ())>,
    destructor: Option<unsafe extern "C" fn(*mut inet_frag_queue)>,
    frag_expire: Option<unsafe extern "C" fn(*mut timer_list)>,
    frags_cachep: *mut kmem_cache,
    frags_cache_name: *const u8,
    rhash_params: rhashtable_params,
    refcnt: refcount_struct,
    completion: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct local_ports {
    range: u32,
    warned: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ping_group_range {
    lock: seqlock,
    range: [__SlateRecord_uidgid_types_h_13; 2],
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inet_timewait_death_row {
    tw_refcount: refcount_struct,
    __slate_pad_1: [u8; 60],
    hashinfo: *mut inet_hashinfo,
    sysctl_max_tw_buckets: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysctl_fib_multipath_hash_seed {
    user_seed: u32,
    mp_seed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_ipv4 {
    __cacheline_group_begin__netns_ipv4_read_tx: [u8; 0],
    sysctl_tcp_early_retrans: u8,
    sysctl_tcp_tso_win_divisor: u8,
    sysctl_tcp_tso_rtt_log: u8,
    sysctl_tcp_autocorking: u8,
    sysctl_tcp_min_snd_mss: i32,
    sysctl_tcp_notsent_lowat: u32,
    sysctl_tcp_limit_output_bytes: i32,
    sysctl_tcp_min_rtt_wlen: i32,
    sysctl_tcp_wmem: [i32; 3],
    sysctl_ip_fwd_use_pmtu: u8,
    __cacheline_group_end__netns_ipv4_read_tx: [u8; 0],
    __cacheline_group_begin__netns_ipv4_read_txrx: [u8; 0],
    sysctl_tcp_shrink_window: u8,
    __cacheline_group_end__netns_ipv4_read_txrx: [u8; 0],
    __cacheline_group_begin__netns_ipv4_read_rx: [u8; 0],
    sysctl_tcp_moderate_rcvbuf: u8,
    sysctl_ip_early_demux: u8,
    sysctl_tcp_early_demux: u8,
    sysctl_tcp_l3mdev_accept: u8,
    sysctl_tcp_reordering: i32,
    sysctl_tcp_rmem: [i32; 3],
    sysctl_tcp_rcvbuf_low_rtt: i32,
    __cacheline_group_end__netns_ipv4_read_rx: [u8; 0],
    __slate_pad_24: [u8; 4],
    __cacheline_group_begin__icmp: [u8; 0],
    icmp_global_credit: __SlateRecord_types_h_190,
    icmp_global_stamp: u32,
    __cacheline_group_end__icmp: [u8; 0],
    __slate_pad_29: [u8; 56],
    __cacheline_group_pad__icmp: __SlateRecord_ipv4_h_96,
    tcp_death_row: inet_timewait_death_row,
    udp_table: *mut udp_table,
    forw_hdr: *mut ctl_table_header,
    frags_hdr: *mut ctl_table_header,
    ipv4_hdr: *mut ctl_table_header,
    route_hdr: *mut ctl_table_header,
    xfrm4_hdr: *mut ctl_table_header,
    devconf_all: *mut ipv4_devconf,
    devconf_dflt: *mut ipv4_devconf,
    ra_chain: *mut ip_ra_chain,
    ra_mutex: mutex,
    rules_ops: *mut fib_rules_ops,
    fib_main: *mut fib_table,
    fib_default: *mut fib_table,
    fib_table_hash_lock: spinlock,
    fib_rules_require_fldissect: u32,
    fib_has_custom_rules: bool,
    fib_has_custom_local_routes: bool,
    fib_offload_disabled: bool,
    fib_table_hash: *mut hlist_head,
    fibnl: *mut sock,
    fib_info_hash: *mut hlist_head,
    fib_info_hash_bits: u32,
    fib_info_cnt: u32,
    mc_autojoin_sk: *mut sock,
    peers: *mut inet_peer_base,
    fqdir: *mut fqdir,
    sysctl_icmp_echo_ignore_all: u8,
    sysctl_icmp_echo_enable_probe: u8,
    sysctl_icmp_echo_ignore_broadcasts: u8,
    sysctl_icmp_ignore_bogus_error_responses: u8,
    sysctl_icmp_errors_use_inbound_ifaddr: u8,
    sysctl_icmp_errors_extension_mask: u8,
    sysctl_icmp_ratelimit: i32,
    sysctl_icmp_ratemask: i32,
    sysctl_icmp_msgs_per_sec: i32,
    sysctl_icmp_msgs_burst: i32,
    ip_rt_min_pmtu: u32,
    ip_rt_mtu_expires: i32,
    ip_rt_min_advmss: i32,
    ip_local_ports: local_ports,
    sysctl_tcp_ecn: u8,
    sysctl_tcp_ecn_option: u8,
    sysctl_tcp_ecn_option_beacon: u8,
    sysctl_tcp_ecn_fallback: u8,
    sysctl_ip_default_ttl: u8,
    sysctl_ip_no_pmtu_disc: u8,
    sysctl_ip_fwd_update_priority: u8,
    sysctl_ip_nonlocal_bind: u8,
    sysctl_ip_autobind_reuse: u8,
    sysctl_ip_dynaddr: u8,
    sysctl_ip_local_port_step_width: u32,
    sysctl_udp_early_demux: u8,
    sysctl_nexthop_compat_mode: u8,
    sysctl_fwmark_reflect: u8,
    sysctl_tcp_fwmark_accept: u8,
    sysctl_tcp_mtu_probing: u8,
    sysctl_tcp_mtu_probe_floor: i32,
    sysctl_tcp_base_mss: i32,
    sysctl_tcp_probe_threshold: i32,
    sysctl_tcp_probe_interval: u32,
    sysctl_tcp_keepalive_time: i32,
    sysctl_tcp_keepalive_intvl: i32,
    sysctl_tcp_keepalive_probes: u8,
    sysctl_tcp_syn_retries: u8,
    sysctl_tcp_synack_retries: u8,
    sysctl_tcp_syncookies: u8,
    sysctl_tcp_migrate_req: u8,
    sysctl_tcp_comp_sack_nr: u8,
    sysctl_tcp_backlog_ack_defer: u8,
    sysctl_tcp_pingpong_thresh: u8,
    sysctl_tcp_retries1: u8,
    sysctl_tcp_retries2: u8,
    sysctl_tcp_orphan_retries: u8,
    sysctl_tcp_tw_reuse: u8,
    sysctl_tcp_tw_reuse_delay: u32,
    sysctl_tcp_fin_timeout: i32,
    sysctl_tcp_sack: u8,
    sysctl_tcp_window_scaling: u8,
    sysctl_tcp_timestamps: u8,
    sysctl_tcp_rto_min_us: i32,
    sysctl_tcp_rto_max_ms: i32,
    sysctl_tcp_recovery: u8,
    sysctl_tcp_thin_linear_timeouts: u8,
    sysctl_tcp_slow_start_after_idle: u8,
    sysctl_tcp_retrans_collapse: u8,
    sysctl_tcp_stdurg: u8,
    sysctl_tcp_rfc1337: u8,
    sysctl_tcp_abort_on_overflow: u8,
    sysctl_tcp_fack: u8,
    sysctl_tcp_max_reordering: i32,
    sysctl_tcp_adv_win_scale: i32,
    sysctl_tcp_dsack: u8,
    sysctl_tcp_app_win: u8,
    sysctl_tcp_frto: u8,
    sysctl_tcp_nometrics_save: u8,
    sysctl_tcp_no_ssthresh_metrics_save: u8,
    sysctl_tcp_workaround_signed_windows: u8,
    sysctl_tcp_challenge_ack_limit: i32,
    sysctl_tcp_min_tso_segs: u8,
    sysctl_tcp_reflect_tos: u8,
    sysctl_tcp_invalid_ratelimit: i32,
    sysctl_tcp_pacing_ss_ratio: i32,
    sysctl_tcp_pacing_ca_ratio: i32,
    sysctl_tcp_child_ehash_entries: u32,
    sysctl_tcp_comp_sack_rtt_percent: i32,
    sysctl_tcp_comp_sack_delay_ns: u64,
    sysctl_tcp_comp_sack_slack_ns: u64,
    sysctl_max_syn_backlog: i32,
    sysctl_tcp_fastopen: i32,
    tcp_congestion_control: *const tcp_congestion_ops,
    tcp_fastopen_ctx: *mut tcp_fastopen_context,
    sysctl_tcp_fastopen_blackhole_timeout: u32,
    tfo_active_disable_times: __SlateRecord_types_h_190,
    tfo_active_disable_stamp: u64,
    tcp_challenge_timestamp: u32,
    tcp_challenge_count: u32,
    sysctl_tcp_plb_enabled: u8,
    sysctl_tcp_plb_idle_rehash_rounds: u8,
    sysctl_tcp_plb_rehash_rounds: u8,
    sysctl_tcp_plb_suspend_rto_sec: u8,
    sysctl_tcp_plb_cong_thresh: i32,
    sysctl_udp_wmem_min: i32,
    sysctl_udp_rmem_min: i32,
    sysctl_fib_notify_on_flag_change: u8,
    sysctl_tcp_syn_linear_timeouts: u8,
    sysctl_igmp_llm_reports: u8,
    sysctl_igmp_max_memberships: i32,
    sysctl_igmp_max_msf: i32,
    sysctl_igmp_qrv: i32,
    ping_group_range: ping_group_range,
    ping_port_rover: u16,
    dev_addr_genid: __SlateRecord_types_h_190,
    sysctl_udp_child_hash_entries: u32,
    sysctl_local_reserved_ports: *mut u64,
    sysctl_ip_prot_sock: i32,
    mrt: *mut mr_table,
    ipmr_notifier_ops: *mut fib_notifier_ops,
    ipmr_seq: __SlateRecord_types_h_190,
    mfc_mutex: mutex,
    sysctl_fib_multipath_hash_seed: sysctl_fib_multipath_hash_seed,
    sysctl_fib_multipath_hash_fields: u32,
    sysctl_fib_multipath_use_neigh: u8,
    sysctl_fib_multipath_hash_policy: u8,
    notifier_ops: *mut fib_notifier_ops,
    fib_seq: u32,
    rt_genid: __SlateRecord_types_h_190,
    ip_id_key: __SlateRecord_siphash_h_21,
    inet_addr_lst: *mut hlist_head,
    addr_chk_work: delayed_work,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dst_ops {
    family: u16,
    gc_thresh: u32,
    gc: Option<unsafe extern "C" fn(*mut dst_ops)>,
    check: Option<unsafe extern "C" fn(*mut dst_entry, u32) -> *mut dst_entry>,
    default_advmss: Option<unsafe extern "C" fn(*const dst_entry) -> u32>,
    mtu: Option<unsafe extern "C" fn(*const dst_entry) -> u32>,
    cow_metrics: Option<unsafe extern "C" fn(*mut dst_entry, u64) -> *mut u32>,
    destroy: Option<unsafe extern "C" fn(*mut dst_entry)>,
    ifdown: Option<unsafe extern "C" fn(*mut dst_entry, *mut net_device)>,
    negative_advice: Option<unsafe extern "C" fn(*mut sock, *mut dst_entry)>,
    link_failure: Option<unsafe extern "C" fn(*mut sk_buff)>,
    update_pmtu: Option<unsafe extern "C" fn(*mut dst_entry, *mut sock, *mut sk_buff, u32, bool)>,
    redirect: Option<unsafe extern "C" fn(*mut dst_entry, *mut sock, *mut sk_buff)>,
    local_out: Option<unsafe extern "C" fn(*mut net, *mut sock, *mut sk_buff) -> i32>,
    neigh_lookup: Option<unsafe extern "C" fn(*const dst_entry, *mut sk_buff, *const ()) -> *mut neighbour>,
    confirm_neigh: Option<unsafe extern "C" fn(*const dst_entry, *const ())>,
    kmem_cachep: *mut kmem_cache,
    pcpuc_entries: percpu_counter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_sysctl_ipv6 {
    hdr: *mut ctl_table_header,
    route_hdr: *mut ctl_table_header,
    icmp_hdr: *mut ctl_table_header,
    frags_hdr: *mut ctl_table_header,
    xfrm6_hdr: *mut ctl_table_header,
    flush_delay: i32,
    ip6_rt_max_size: i32,
    ip6_rt_gc_min_interval: i32,
    ip6_rt_gc_timeout: i32,
    ip6_rt_gc_interval: i32,
    ip6_rt_gc_elasticity: i32,
    ip6_rt_mtu_expires: i32,
    ip6_rt_min_advmss: i32,
    multipath_hash_fields: u32,
    multipath_hash_policy: u8,
    __cacheline_group_begin__sysctl_ipv6_flowlabel: [u8; 0],
    flowlabel_consistency: u8,
    auto_flowlabels: u8,
    flowlabel_state_ranges: u8,
    __cacheline_group_end__sysctl_ipv6_flowlabel: [u8; 0],
    icmpv6_echo_ignore_all: u8,
    icmpv6_echo_ignore_multicast: u8,
    icmpv6_echo_ignore_anycast: u8,
    icmpv6_time: i32,
    icmpv6_ratemask: [u64; 4],
    icmpv6_ratemask_ptr: *mut u64,
    anycast_src_echo_reply: u8,
    bindv6only: u8,
    ip_nonlocal_bind: u8,
    fwmark_reflect: u8,
    idgen_retries: i32,
    idgen_delay: i32,
    flowlabel_reflect: i32,
    max_dst_opts_cnt: i32,
    max_hbh_opts_cnt: i32,
    max_dst_opts_len: i32,
    max_hbh_opts_len: i32,
    seg6_flowlabel: i32,
    ioam6_id: u32,
    ioam6_id_wide: u64,
    skip_notify_on_dev_down: u8,
    fib_notify_on_flag_change: u8,
    icmpv6_error_anycast_as_unicast: u8,
    icmpv6_errors_extension_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_ipv6 {
    ip6_dst_ops: dst_ops,
    sysctl: netns_sysctl_ipv6,
    devconf_all: *mut ipv6_devconf,
    devconf_dflt: *mut ipv6_devconf,
    peers: *mut inet_peer_base,
    fqdir: *mut fqdir,
    fib6_null_entry: *mut fib6_info,
    ip6_null_entry: *mut rt6_info,
    rt6_stats: *mut rt6_statistics,
    ip6_fib_timer: timer_list,
    fib_table_hash: *mut hlist_head,
    fib_table_hash_lock: spinlock,
    fib6_main_tbl: *mut fib6_table,
    fib6_walkers: list_head,
    fib6_walker_lock: rwlock,
    fib6_gc_lock: spinlock,
    ip6_rt_gc_expire: __SlateRecord_types_h_190,
    ip6_rt_last_gc: u64,
    flowlabel_has_excl: u8,
    ndisc_sk: *mut sock,
    tcp_sk: *mut sock,
    igmp_sk: *mut sock,
    mc_autojoin_sk: *mut sock,
    inet6_addr_lst: *mut hlist_head,
    addrconf_hash_lock: spinlock,
    addr_chk_work: delayed_work,
    dev_addr_genid: __SlateRecord_types_h_190,
    fib6_sernum: __SlateRecord_types_h_190,
    seg6_data: *mut seg6_pernet_data,
    notifier_ops: *mut fib_notifier_ops,
    flowlabel_count: i32,
    ip6addrlbl_table: __SlateRecord_ipv6_h_124,
    ioam6_data: *mut ioam6_pernet_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_nexthop {
    rb_root: rb_root,
    devhash: *mut hlist_head,
    seq: u32,
    last_id_allocated: u32,
    notifier_chain: blocking_notifier_head,
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
struct in_addr {
    s_addr: u32,
}


// sizeof(struct sockaddr)
// Address to send to all hosts.
// 224.0.0.106
#[repr(C)]
#[derive(Clone, Copy)]
struct netns_nf {
    proc_netfilter: *mut proc_dir_entry,
    nf_loggers: [*const nf_logger; 11],
    nf_log_dir_header: *mut ctl_table_header,
    hooks_ipv4: [*mut nf_hook_entries; 5],
    hooks_ipv6: [*mut nf_hook_entries; 5],
    defrag_ipv4_users: u32,
    defrag_ipv6_users: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_generic_net {
    timeout: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sockptr_h_20_2 {
    kernel: *mut (),
    user: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_tcp_net {
    timeouts: [u32; 14],
    tcp_loose: u8,
    tcp_be_liberal: u8,
    tcp_max_retrans: u8,
    tcp_ignore_invalid_rst: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_udp_net {
    timeouts: [u32; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_icmp_net {
    timeout: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nf_ip_net {
    generic: nf_generic_net,
    tcp: nf_tcp_net,
    udp: nf_udp_net,
    icmp: nf_icmp_net,
    icmpv6: nf_icmp_net,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_net_h_191_2 {
    buf: *mut u8,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_ct {
    sysctl_log_invalid: u8,
    sysctl_events: u8,
    sysctl_acct: u8,
    sysctl_tstamp: u8,
    sysctl_checksum: u8,
    stat: *mut ip_conntrack_stat,
    nf_conntrack_event_cb: *mut nf_ct_event_notifier,
    nf_ct_proto: nf_ip_net,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1528 {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; 16],
    map_ifindex: u32,
    btf_fd: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    map_extra: u64,
    value_type_btf_obj_fd: i32,
    map_token_fd: i32,
    excl_prog_hash: u64,
    excl_prog_hash_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1581 {
    map_fd: u32,
    key: u64,
    __slate_anon_2: __SlateRecord_bpf_h_1584,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1584 {
    value: u64,
    next_key: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1591 {
    in_batch: u64,
    out_batch: u64,
    keys: u64,
    values: u64,
    count: u32,
    map_fd: u32,
    elem_flags: u64,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1608 {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; 16],
    prog_ifindex: u32,
    expected_attach_type: u32,
    prog_btf_fd: u32,
    func_info_rec_size: u32,
    func_info: u64,
    func_info_cnt: u32,
    line_info_rec_size: u32,
    line_info: u64,
    line_info_cnt: u32,
    attach_btf_id: u32,
    __slate_anon_20: __SlateRecord_bpf_h_1633,
    core_relo_cnt: u32,
    fd_array: u64,
    core_relos: u64,
    core_relo_rec_size: u32,
    log_true_size: u32,
    prog_token_fd: i32,
    fd_array_cnt: u32,
    signature: u64,
    signature_size: u32,
    keyring_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1633 {
    attach_prog_fd: u32,
    attach_btf_obj_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1674 {
    pathname: u64,
    bpf_fd: u32,
    file_flags: u32,
    path_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1687 {
    __slate_anon_0: __SlateRecord_bpf_h_1688,
    attach_bpf_fd: u32,
    attach_type: u32,
    attach_flags: u32,
    replace_bpf_fd: u32,
    __slate_anon_5: __SlateRecord_bpf_h_1696,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1688 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1696 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1703 {
    prog_fd: u32,
    retval: u32,
    data_size_in: u32,
    data_size_out: u32,
    data_in: u64,
    data_out: u64,
    repeat: u32,
    duration: u32,
    ctx_size_in: u32,
    ctx_size_out: u32,
    ctx_in: u64,
    ctx_out: u64,
    flags: u32,
    cpu: u32,
    batch_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1727 {
    __slate_anon_0: __SlateRecord_bpf_h_1728,
    next_id: u32,
    open_flags: u32,
    fd_by_id_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1728 {
    start_id: u32,
    prog_id: u32,
    map_id: u32,
    btf_id: u32,
    link_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1740 {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1746 {
    __slate_anon_0: __SlateRecord_bpf_h_1747,
    attach_type: u32,
    query_flags: u32,
    attach_flags: u32,
    prog_ids: u64,
    __slate_anon_5: __SlateRecord_bpf_h_1755,
    __slate_bits_0: __slate_bits::__SlateBits1004U0,
    prog_attach_flags: u64,
    link_ids: u64,
    link_attach_flags: u64,
    revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1747 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1755 {
    prog_cnt: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1769 {
    name: u64,
    prog_fd: u32,
    __slate_bits_0: __slate_bits::__SlateBits1007U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1776 {
    btf: u64,
    btf_log_buf: u64,
    btf_size: u32,
    btf_log_size: u32,
    btf_log_level: u32,
    btf_log_true_size: u32,
    btf_flags: u32,
    btf_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1794 {
    pid: u32,
    fd: u32,
    flags: u32,
    buf_len: u32,
    buf: u64,
    prog_id: u32,
    fd_type: u32,
    probe_offset: u64,
    probe_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1810 {
    __slate_anon_0: __SlateRecord_bpf_h_1811,
    __slate_anon_1: __SlateRecord_bpf_h_1815,
    attach_type: u32,
    flags: u32,
    __slate_anon_4: __SlateRecord_bpf_h_1821,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1811 {
    prog_fd: u32,
    map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1815 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1821 {
    target_btf_id: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1823,
    perf_event: __SlateRecord_bpf_h_1827,
    kprobe_multi: __SlateRecord_bpf_h_1834,
    tracing: __SlateRecord_bpf_h_1841,
    netfilter: __SlateRecord_bpf_h_1850,
    tcx: __SlateRecord_bpf_h_1856,
    uprobe_multi: __SlateRecord_bpf_h_1863,
    netkit: __SlateRecord_bpf_h_1873,
    cgroup: __SlateRecord_bpf_h_1880,
    tracing_multi: __SlateRecord_bpf_h_1887,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1823 {
    iter_info: u64,
    iter_info_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1827 {
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1834 {
    flags: u32,
    cnt: u32,
    syms: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1841 {
    target_btf_id: u32,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1850 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1856 {
    __slate_anon_0: __SlateRecord_bpf_h_1857,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1857 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1863 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    cnt: u32,
    flags: u32,
    pid: u32,
    path_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1873 {
    __slate_anon_0: __SlateRecord_bpf_h_1874,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1874 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1880 {
    __slate_anon_0: __SlateRecord_bpf_h_1881,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1881 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1887 {
    ids: u64,
    cookies: u64,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1895 {
    link_fd: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1897,
    flags: u32,
    __slate_anon_3: __SlateRecord_bpf_h_1904,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1897 {
    new_prog_fd: u32,
    new_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1904 {
    old_prog_fd: u32,
    old_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1916 {
    link_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1920 {
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1924 {
    link_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1929 {
    prog_fd: u32,
    map_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1935 {
    flags: u32,
    bpffs_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1940 {
    stream_buf: u64,
    stream_buf_len: u32,
    stream_id: u32,
    prog_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1947 {
    map_fd: u32,
    prog_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xfrm_policy_hash {
    table: *mut hlist_head,
    hmask: u32,
    dbits4: u8,
    sbits4: u8,
    dbits6: u8,
    sbits6: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6842 {
    raw_tracepoint: __SlateRecord_bpf_h_6843,
    tracing: __SlateRecord_bpf_h_6849,
    cgroup: __SlateRecord_bpf_h_6856,
    iter: __SlateRecord_bpf_h_6860,
    netns: __SlateRecord_bpf_h_6884,
    xdp: __SlateRecord_bpf_h_6888,
    struct_ops: __SlateRecord_bpf_h_6891,
    netfilter: __SlateRecord_bpf_h_6894,
    kprobe_multi: __SlateRecord_bpf_h_6900,
    uprobe_multi: __SlateRecord_bpf_h_6907,
    tracing_multi: __SlateRecord_bpf_h_6917,
    perf_event: __SlateRecord_bpf_h_6926,
    tcx: __SlateRecord_bpf_h_6959,
    netkit: __SlateRecord_bpf_h_6963,
    sockmap: __SlateRecord_bpf_h_6967,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6843 {
    tp_name: u64,
    tp_name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits1040U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6849 {
    attach_type: u32,
    target_obj_id: u32,
    target_btf_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits1041U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6856 {
    cgroup_id: u64,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6860 {
    target_name: u64,
    target_name_len: u32,
    __slate_anon_2: __SlateRecord_bpf_h_6868,
    __slate_anon_3: __SlateRecord_bpf_h_6873,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6868 {
    map: __SlateRecord_bpf_h_6869,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6869 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6873 {
    cgroup: __SlateRecord_bpf_h_6874,
    task: __SlateRecord_bpf_h_6878,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6874 {
    cgroup_id: u64,
    order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6878 {
    tid: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6884 {
    netns_ino: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6888 {
    ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6891 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6894 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6900 {
    addrs: u64,
    count: u32,
    flags: u32,
    missed: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6907 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    path_size: u32,
    count: u32,
    flags: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6917 {
    attach_type: u32,
    count: u32,
    btf_obj_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits1055U0,
    ids: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6926 {
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits1056U0,
    __slate_anon_2: __SlateRecord_bpf_h_6929,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6929 {
    uprobe: __SlateRecord_bpf_h_6930,
    kprobe: __SlateRecord_bpf_h_6937,
    tracepoint: __SlateRecord_bpf_h_6945,
    event: __SlateRecord_bpf_h_6951,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6930 {
    file_name: u64,
    name_len: u32,
    offset: u32,
    cookie: u64,
    ref_ctr_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6937 {
    func_name: u64,
    name_len: u32,
    offset: u32,
    addr: u64,
    missed: u64,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6945 {
    tp_name: u64,
    name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits1060U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6951 {
    config: u64,
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits1061U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6959 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6963 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6967 {
    map_id: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xfrm_policy_hthresh {
    work: work_struct,
    lock: seqlock,
    lbits4: u8,
    rbits4: u8,
    lbits6: u8,
    rbits6: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_xfrm {
    state_all: list_head,
    state_bydst: *mut hlist_head,
    state_bysrc: *mut hlist_head,
    state_byspi: *mut hlist_head,
    state_byseq: *mut hlist_head,
    state_cache_input: *mut hlist_head,
    state_hmask: u32,
    state_num: u32,
    state_hash_work: work_struct,
    policy_all: list_head,
    policy_byidx: *mut hlist_head,
    policy_idx_hmask: u32,
    idx_generator: u32,
    policy_bydst: [xfrm_policy_hash; 3],
    policy_count: [u32; 6],
    policy_hash_work: work_struct,
    policy_hthresh: xfrm_policy_hthresh,
    inexact_bins: list_head,
    nlsk: *mut sock,
    nlsk_stash: *mut sock,
    sysctl_aevent_etime: u32,
    sysctl_aevent_rseqth: u32,
    sysctl_larval_drop: i32,
    sysctl_acq_expires: u32,
    policy_default: [u8; 3],
    sysctl_hdr: *mut ctl_table_header,
    xfrm4_dst_ops: dst_ops,
    xfrm6_dst_ops: dst_ops,
    xfrm_state_lock: spinlock,
    xfrm_state_hash_generation: seqcount_spinlock,
    xfrm_policy_hash_generation: seqcount_spinlock,
    xfrm_policy_lock: spinlock,
    xfrm_cfg_mutex: mutex,
    nat_keepalive_work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netns_bpf {
    run_array: [*mut bpf_prog_array; 2],
    progs: [*mut bpf_prog; 2],
    links: [list_head; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ref_tracker_dir {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_btf_h_64 {
    size: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_net_trackers_h_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_log {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
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
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ftrace_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_env {
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
struct bpf_arena {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_aux_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_local_storage_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
    passive: refcount_struct,
    rules_mod_lock: spinlock,
    dev_base_seq: u32,
    ifindex: u32,
    nsid_lock: spinlock,
    fnhe_genid: __SlateRecord_types_h_190,
    list: list_head,
    exit_list: list_head,
    defer_free_list: llist_node,
    cleanup_list: llist_node,
    ptype_all: list_head,
    ptype_specific: list_head,
    key_domain: *mut key_tag,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    netns_ids: idr,
    ns: ns_common,
    refcnt_tracker: ref_tracker_dir,
    notrefcnt_tracker: ref_tracker_dir,
    dev_base_head: list_head,
    proc_net: *mut proc_dir_entry,
    proc_net_stat: *mut proc_dir_entry,
    sysctls: ctl_table_set,
    rtnl: *mut sock,
    genl_sock: *mut sock,
    uevent_sock: *mut uevent_sock,
    dev_name_head: *mut hlist_head,
    dev_index_head: *mut hlist_head,
    dev_by_index: xarray,
    netdev_chain: raw_notifier_head,
    hash_mix: u32,
    is_dying: bool,
    loopback_dev: *mut net_device,
    rules_ops: list_head,
    core: netns_core,
    mib: netns_mib,
    packet: netns_packet,
    unx: netns_unix,
    nexthop: netns_nexthop,
    ipv4: netns_ipv4,
    ipv6: netns_ipv6,
    nf: netns_nf,
    ct: netns_ct,
    r#gen: *mut net_generic,
    bpf: netns_bpf,
    xfrm: netns_xfrm,
    net_cookie: u64,
    diag_nlsk: *mut sock,
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
struct __SlateRecord_net_namespace_h_410 {
    net: *mut net,
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
union __SlateRecord_bpf_h_256 {
    kptr: btf_field_kptr,
    graph_root: btf_field_graph_root,
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
union __SlateRecord_bpf_h_329 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_offload_dev {
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
union __SlateRecord_bpf_h_996 {
    __slate_anon_0: __SlateRecord_bpf_h_997,
    arg_type: [u32; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_997 {
    arg1_type: u32,
    arg2_type: u32,
    arg3_type: u32,
    arg4_type: u32,
    arg5_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1006 {
    __slate_anon_0: __SlateRecord_bpf_h_1007,
    arg_btf_id: [*mut u32; 12],
    __slate_anon_2: __SlateRecord_bpf_h_1015,
    arg_size: [u64; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1007 {
    arg1_btf_id: *mut u32,
    arg2_btf_id: *mut u32,
    arg3_btf_id: *mut u32,
    arg4_btf_id: *mut u32,
    arg5_btf_id: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1015 {
    arg1_size: u64,
    arg2_size: u64,
    arg3_size: u64,
    arg4_size: u64,
    arg5_size: u64,
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
union __SlateRecord_bpf_h_1109 {
    ctx_field_size: i32,
    __slate_anon_1: __SlateRecord_bpf_h_1111,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1111 {
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tasklet_struct {
    next: *mut tasklet_struct,
    state: u64,
    count: __SlateRecord_types_h_190,
    use_callback: bool,
    __slate_anon_4: __SlateRecord_interrupt_h_697,
    data: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_reg_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
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
union __SlateRecord_bpf_h_1393 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1423 {
    model: btf_func_model,
    addr: *mut (),
    ftrace_managed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1442 {
    old_image: *mut bpf_tramp_image,
    old_flags: u32,
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
union __SlateRecord_bpf_h_1705 {
    tail_call: __SlateRecord_bpf_h_1706,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1706 {
    map: *mut bpf_map,
    key: u32,
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
struct bpf_kfunc_desc_tab {
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
struct __SlateRecord_bpf_h_1821_2 {
    keyring_serial: i32,
    keyring_type: u8,
    verdict: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_btf_tab {
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
union __SlateRecord_bpf_h_1892 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1926 {
    digest: [u8; 32],
    tag: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_fprog_kern {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1937 {
    __slate_anon_0: __SlateRecord_bpf_h_1938,
    __slate_anon_1: __SlateRecord_bpf_h_1939,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938 {
    __empty_insns: __SlateRecord_bpf_h_1938_2,
    insns: [sock_filter; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939 {
    __empty_insnsi: __SlateRecord_bpf_h_1939_2,
    insnsi: [bpf_insn; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939_2 {
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
    __slate_bits_0: __slate_bits::__SlateBits955U0,
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
union __SlateRecord_bpf_h_1964 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netprio_map {
    rcu: callback_head,
    priomap_len: u32,
    priomap: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_2532 {
    cgroup_storage: [*mut bpf_cgroup_storage; 2],
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_cgroup_storage {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xdp_frame {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sockptr_h_20 {
    __slate_anon_0: __SlateRecord_sockptr_h_20_2,
    __slate_bits_0: __slate_bits::__SlateBits971U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sockopt {
    iter_in: iov_iter,
    iter_out: iov_iter,
    optlen: i32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct socket_wq {
    wait: wait_queue_head,
    fasync_list: *mut fasync_struct,
    flags: u64,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nla_policy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct socket {
    state: u32,
    r#type: i16,
    flags: u64,
    file: *mut file,
    sk: *mut sock,
    ops: *const proto_ops,
    wq: socket_wq,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netlink_h_297 {
    ctx: [u8; 48],
    args: [i64; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_net_h_191 {
    written: u64,
    count: u64,
    arg: __SlateRecord_net_h_191_2,
    error: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proto_ops {
    family: i32,
    owner: *mut module,
    release: Option<unsafe extern "C" fn(*mut socket) -> i32>,
    bind: Option<unsafe extern "C" fn(*mut socket, *mut sockaddr_unsized, i32) -> i32>,
    connect: Option<unsafe extern "C" fn(*mut socket, *mut sockaddr_unsized, i32, i32) -> i32>,
    socketpair: Option<unsafe extern "C" fn(*mut socket, *mut socket) -> i32>,
    accept: Option<unsafe extern "C" fn(*mut socket, *mut socket, *mut proto_accept_arg) -> i32>,
    getname: Option<unsafe extern "C" fn(*mut socket, *mut sockaddr, i32) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut socket, *mut poll_table_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut socket, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut socket, u32, u64) -> i32>,
    gettstamp: Option<unsafe extern "C" fn(*mut socket, *mut (), bool, bool) -> i32>,
    listen: Option<unsafe extern "C" fn(*mut socket, i32) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut socket, i32) -> i32>,
    setsockopt: Option<unsafe extern "C" fn(*mut socket, i32, i32, __SlateRecord_sockptr_h_20, u32) -> i32>,
    getsockopt: Option<unsafe extern "C" fn(*mut socket, i32, i32, *mut u8, *mut i32) -> i32>,
    getsockopt_iter: Option<unsafe extern "C" fn(*mut socket, i32, i32, *mut sockopt) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut socket)>,
    sendmsg: Option<unsafe extern "C" fn(*mut socket, *mut msghdr, u64) -> i32>,
    recvmsg: Option<unsafe extern "C" fn(*mut socket, *mut msghdr, u64, i32) -> i32>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut socket, *mut vm_area_struct) -> i32>,
    splice_read: Option<unsafe extern "C" fn(*mut socket, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut socket)>,
    set_peek_off: Option<unsafe extern "C" fn(*mut sock, i32) -> i32>,
    peek_len: Option<unsafe extern "C" fn(*mut socket) -> i32>,
    read_sock: Option<unsafe extern "C" fn(*mut sock, *mut __SlateRecord_net_h_191, Option<unsafe extern "C" fn(*mut __SlateRecord_net_h_191, *mut sk_buff, u32, u64) -> i32>) -> i32>,
    read_skb: Option<unsafe extern "C" fn(*mut sock, Option<unsafe extern "C" fn(*mut sock, *mut sk_buff) -> i32>) -> i32>,
    sendmsg_locked: Option<unsafe extern "C" fn(*mut sock, *mut msghdr, u64) -> i32>,
    set_rcvlowat: Option<unsafe extern "C" fn(*mut sock, i32) -> i32>,
    set_rcvbuf: Option<unsafe extern "C" fn(*mut sock, i32)>,
}


// double word (64-bit)
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn {
    code: u8,
    __slate_bits_0: __slate_bits::__SlateBits985U0,
    off: i16,
    imm: i32,
}


// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
#[repr(C)]
#[derive(Clone, Copy)]
union bpf_attr {
    __slate_anon_0: __SlateRecord_bpf_h_1528,
    __slate_anon_1: __SlateRecord_bpf_h_1581,
    batch: __SlateRecord_bpf_h_1591,
    __slate_anon_3: __SlateRecord_bpf_h_1608,
    __slate_anon_4: __SlateRecord_bpf_h_1674,
    __slate_anon_5: __SlateRecord_bpf_h_1687,
    test: __SlateRecord_bpf_h_1703,
    __slate_anon_7: __SlateRecord_bpf_h_1727,
    info: __SlateRecord_bpf_h_1740,
    query: __SlateRecord_bpf_h_1746,
    raw_tracepoint: __SlateRecord_bpf_h_1769,
    __slate_anon_11: __SlateRecord_bpf_h_1776,
    task_fd_query: __SlateRecord_bpf_h_1794,
    link_create: __SlateRecord_bpf_h_1810,
    link_update: __SlateRecord_bpf_h_1895,
    link_detach: __SlateRecord_bpf_h_1916,
    enable_stats: __SlateRecord_bpf_h_1920,
    iter_create: __SlateRecord_bpf_h_1924,
    prog_bind_map: __SlateRecord_bpf_h_1929,
    token_create: __SlateRecord_bpf_h_1935,
    prog_stream_read: __SlateRecord_bpf_h_1940,
    prog_assoc_struct_ops: __SlateRecord_bpf_h_1947,
}


// The description below is an attempt at providing documentation to eBPF
// developers about the multiple available eBPF helper functions. It can be
// parsed and used to produce a manual page. The workflow is the following,
// and requires the rst2man utility:
//
//     $ ./scripts/bpf_doc.py  *             --filename include/uapi/linux/bpf.h > /tmp/bpf-helpers.rst
//     $ rst2man /tmp/bpf-helpers.rst > /tmp/bpf-helpers.7
//     $ man /tmp/bpf-helpers.7
//
// Note that in order to produce this external documentation, some RST
// formatting is used in the descriptions to get "bold" and "italics" in
// manual pages. Also note that the few trailing white spaces are
// intentional, removing them would break paragraphs for rst2man.
//
// Start of BPF helper function descriptions:
//
// void *bpf_map_lookup_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Perform a lookup in *map* for an entry associated to *key*.
// 	Return
// 		Map value associated to *key*, or **NULL** if no entry was
// 		found.
//
// long bpf_map_update_elem(struct bpf_map *map, const void *key, const void *value, u64 flags)
// 	Description
// 		Add or update the value of the entry associated to *key* in
// 		*map* with *value*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		Flag value **BPF_NOEXIST** cannot be used for maps of types
// 		**BPF_MAP_TYPE_ARRAY** or **BPF_MAP_TYPE_PERCPU_ARRAY**  (all
// 		elements always exist), the helper would return an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_delete_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Delete entry with *key* from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		For tracing programs, safely attempt to read *size* bytes from
// 		kernel space address *unsafe_ptr* and store the data in *dst*.
//
// 		Generally, use **bpf_probe_read_user**\ () or
// 		**bpf_probe_read_kernel**\ () instead.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_ktime_get_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does not include time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC**)
// 	Return
// 		Current *ktime*.
//
// long bpf_trace_printk(const char *fmt, u32 fmt_size, ...)
// 	Description
// 		This helper is a "printk()-like" facility for debugging. It
// 		prints a message defined by format *fmt* (of size *fmt_size*)
// 		to file *\/sys/kernel/tracing/trace* from TraceFS, if
// 		available. It can take up to three additional **u64**
// 		arguments (as an eBPF helpers, the total number of arguments is
// 		limited to five).
//
// 		Each time the helper is called, it appends a line to the trace.
// 		Lines are discarded while *\/sys/kernel/tracing/trace* is
// 		open, use *\/sys/kernel/tracing/trace_pipe* to avoid this.
// 		The format of the trace is customizable, and the exact output
// 		one will get depends on the options set in
// 		*\/sys/kernel/tracing/trace_options* (see also the
// 		*README* file under the same directory). However, it usually
// 		defaults to something like:
//
// 		::
//
// 			telnet-470   [001] .N.. 419421.045894: 0x00000001: <formatted msg>
//
// 		In the above:
//
// 			* ``telnet`` is the name of the current task.
// 			* ``470`` is the PID of the current task.
// 			* ``001`` is the CPU number on which the task is
// 			  running.
// 			* In ``.N..``, each character refers to a set of
// 			  options (whether irqs are enabled, scheduling
// 			  options, whether hard/softirqs are running, level of
// 			  preempt_disabled respectively). **N** means that
// 			  **TIF_NEED_RESCHED** and **PREEMPT_NEED_RESCHED**
// 			  are set.
// 			* ``419421.045894`` is a timestamp.
// 			* ``0x00000001`` is a fake value used by BPF for the
// 			  instruction pointer register.
// 			* ``<formatted msg>`` is the message formatted with
// 			  *fmt*.
//
// 		The conversion specifiers supported by *fmt* are similar, but
// 		more limited than for printk(). They are **%d**, **%i**,
// 		**%u**, **%x**, **%ld**, **%li**, **%lu**, **%lx**, **%lld**,
// 		**%lli**, **%llu**, **%llx**, **%p**, **%s**. No modifier (size
// 		of field, padding with zeroes, etc.) is available, and the
// 		helper will return **-EINVAL** (but print nothing) if it
// 		encounters an unknown specifier.
//
// 		Also, note that **bpf_trace_printk**\ () is slow, and should
// 		only be used for debugging purposes. For this reason, a notice
// 		block (spanning several lines) is printed to kernel logs and
// 		states that the helper should not be used "for production use"
// 		the first time this helper is used (or more precisely, when
// 		**trace_printk**\ () buffers are allocated). For passing values
// 		to user space, perf events should be preferred.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// u32 bpf_get_prandom_u32(void)
// 	Description
// 		Get a pseudo-random number.
//
// 		From a security point of view, this helper uses its own
// 		pseudo-random internal state, and cannot be used to infer the
// 		seed of other random functions in the kernel. However, it is
// 		essential to note that the generator used by the helper is not
// 		cryptographically secure.
// 	Return
// 		A random 32-bit unsigned value.
//
// u32 bpf_get_smp_processor_id(void)
// 	Description
// 		Get the SMP (symmetric multiprocessing) processor id. Note that
// 		all programs run with migration disabled, which means that the
// 		SMP processor id is stable during all the execution of the
// 		program.
// 	Return
// 		The SMP id of the processor running the program.
// 	Attributes
// 		__bpf_fastcall
//
// long bpf_skb_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len, u64 flags)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. The *flags* are a combination
// 		of the following values:
//
// 		**BPF_F_RECOMPUTE_CSUM**
// 			Automatically update *skb*\ **->csum** after storing the
// 			bytes.
// 		**BPF_F_INVALIDATE_HASH**
// 			Set *skb*\ **->hash**, *skb*\ **->swhash** and *skb* * 			**->l4hash** to 0.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l3_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 size)
// 	Description
// 		Recompute the layer 3 (e.g. IP) checksum for the packet
// 		associated to *skb*. Computation is incremental, so the helper
// 		must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored in *size*.
// 		Alternatively, it is possible to store the difference between
// 		the previous and the new values of the header field in *to*, by
// 		setting *from* and *size* to 0. For both methods, *offset*
// 		indicates the location of the IP checksum within the packet.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l4_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 flags)
// 	Description
// 		Recompute the layer 4 (e.g. TCP, UDP or ICMP) checksum for the
// 		packet associated to *skb*. Computation is incremental, so the
// 		helper must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored on the lowest
// 		four bits of *flags*. Alternatively, it is possible to store
// 		the difference between the previous and the new values of the
// 		header field in *to*, by setting *from* and the four lowest
// 		bits of *flags* to 0. For both methods, *offset* indicates the
// 		location of the IP checksum within the packet. In addition to
// 		the size of the field, *flags* can be added (bitwise OR) actual
// 		flags. With **BPF_F_MARK_MANGLED_0**, a null checksum is left
// 		untouched (unless **BPF_F_MARK_ENFORCE** is added as well), and
// 		for updates resulting in a null checksum the value is set to
// 		**CSUM_MANGLED_0** instead. Flag **BPF_F_PSEUDO_HDR** indicates
// 		that the modified header field is part of the pseudo-header.
// 		Flag **BPF_F_IPV6** should be set for IPv6 packets.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_tail_call(void *ctx, struct bpf_map *prog_array_map, u32 index)
// 	Description
// 		This special helper is used to trigger a "tail call", or in
// 		other words, to jump into another eBPF program. The same stack
// 		frame is used (but values on stack and in registers for the
// 		caller are not accessible to the callee). This mechanism allows
// 		for program chaining, either for raising the maximum number of
// 		available eBPF instructions, or to execute given programs in
// 		conditional blocks. For security reasons, there is an upper
// 		limit to the number of successive tail calls that can be
// 		performed.
//
// 		Upon call of this helper, the program attempts to jump into a
// 		program referenced at index *index* in *prog_array_map*, a
// 		special map of type **BPF_MAP_TYPE_PROG_ARRAY**, and passes
// 		*ctx*, a pointer to the context.
//
// 		If the call succeeds, the kernel immediately runs the first
// 		instruction of the new program. This is not a function call,
// 		and it never returns to the previous program. If the call
// 		fails, then the helper has no effect, and the caller continues
// 		to run its subsequent instructions. A call can fail if the
// 		destination program for the jump does not exist (i.e. *index*
// 		is superior to the number of entries in *prog_array_map*), or
// 		if the maximum number of tail calls has been reached for this
// 		chain of programs. This limit is defined in the kernel by the
// 		macro **MAX_TAIL_CALL_CNT** (not accessible to user space),
// 		which is currently set to 33.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_clone_redirect(struct sk_buff *skb, u32 ifindex, u64 flags)
// 	Description
// 		Clone and redirect the packet associated to *skb* to another
// 		net device of index *ifindex*. Both ingress and egress
// 		interfaces can be used for redirection. The **BPF_F_INGRESS**
// 		value in *flags* is used to make the distinction (ingress path
// 		is selected if the flag is present, egress path otherwise).
// 		This is the only flag supported for now.
//
// 		In comparison with **bpf_redirect**\ () helper,
// 		**bpf_clone_redirect**\ () has the associated cost of
// 		duplicating the packet buffer, but this can be executed out of
// 		the eBPF program. Conversely, **bpf_redirect**\ () is more
// 		efficient, but it is handled through an action code where the
// 		redirection happens only after the eBPF program has returned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure. Positive
// 		error indicates a potential drop or congestion in the target
// 		device. The particular positive error codes are not defined.
//
// u64 bpf_get_current_pid_tgid(void)
// 	Description
// 		Get the current pid and tgid.
// 	Return
// 		A 64-bit integer containing the current tgid and pid, and
// 		created as such:
// 		*current_task*\ **->tgid << 32 \|**
// 		*current_task*\ **->pid**.
//
// u64 bpf_get_current_uid_gid(void)
// 	Description
// 		Get the current uid and gid.
// 	Return
// 		A 64-bit integer containing the current GID and UID, and
// 		created as such: *current_gid* **<< 32 \|** *current_uid*.
//
// long bpf_get_current_comm(void *buf, u32 size_of_buf)
// 	Description
// 		Copy the **comm** attribute of the current task into *buf* of
// 		*size_of_buf*. The **comm** attribute contains the name of
// 		the executable (excluding the path) for the current task. The
// 		*size_of_buf* must be strictly positive. On success, the
// 		helper makes sure that the *buf* is NUL-terminated. On failure,
// 		it is filled with zeroes.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u32 bpf_get_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		Retrieve the classid for the current task, i.e. for the net_cls
// 		cgroup to which *skb* belongs.
//
// 		This helper can be used on TC egress path, but not on ingress.
//
// 		The net_cls cgroup provides an interface to tag network packets
// 		based on a user-provided identifier for all traffic coming from
// 		the tasks belonging to the related cgroup. See also the related
// 		kernel documentation, available from the Linux sources in file
// 		*Documentation/admin-guide/cgroup-v1/net_cls.rst*.
//
// 		The Linux kernel has two versions for cgroups: there are
// 		cgroups v1 and cgroups v2. Both are available to users, who can
// 		use a mixture of them, but note that the net_cls cgroup is for
// 		cgroup v1 only. This makes it incompatible with BPF programs
// 		run on cgroups, which is a cgroup-v2-only feature (a socket can
// 		only hold data for one version of cgroups at a time).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_CGROUP_NET_CLASSID** configuration option set to
// 		"**y**" or to "**m**".
// 	Return
// 		The classid, or 0 for the default unconfigured classid.
//
// long bpf_skb_vlan_push(struct sk_buff *skb, __be16 vlan_proto, u16 vlan_tci)
// 	Description
// 		Push a *vlan_tci* (VLAN tag control information) of protocol
// 		*vlan_proto* to the packet associated to *skb*, then update
// 		the checksum. Note that if *vlan_proto* is different from
// 		**ETH_P_8021Q** and **ETH_P_8021AD**, it is considered to
// 		be **ETH_P_8021Q**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_vlan_pop(struct sk_buff *skb)
// 	Description
// 		Pop a VLAN header from the packet associated to *skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Get tunnel metadata. This helper takes a pointer *key* to an
// 		empty **struct bpf_tunnel_key** of **size**, that will be
// 		filled with tunnel metadata for the packet associated to *skb*.
// 		The *flags* can be set to **BPF_F_TUNINFO_IPV6**, which
// 		indicates that the tunnel is based on IPv6 protocol instead of
// 		IPv4.
//
// 		The **struct bpf_tunnel_key** is an object that generalizes the
// 		principal parameters used by various tunneling protocols into a
// 		single struct. This way, it can be used to easily make a
// 		decision based on the contents of the encapsulation header,
// 		"summarized" in this struct. In particular, it holds the IP
// 		address of the remote end (IPv4 or IPv6, depending on the case)
// 		in *key*\ **->remote_ipv4** or *key*\ **->remote_ipv6**. Also,
// 		this struct exposes the *key*\ **->tunnel_id**, which is
// 		generally mapped to a VNI (Virtual Network Identifier), making
// 		it programmable together with the **bpf_skb_set_tunnel_key** * 		() helper.
//
// 		Let's imagine that the following code is part of a program
// 		attached to the TC ingress interface, on one end of a GRE
// 		tunnel, and is supposed to filter out all messages coming from
// 		remote ends with IPv4 address other than 10.0.0.1:
//
// 		::
//
// 			int ret;
// 			struct bpf_tunnel_key key = {};
//
// 			ret = bpf_skb_get_tunnel_key(skb, &key, sizeof(key), 0);
// 			if (ret < 0)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			if (key.remote_ipv4 != 0x0a000001)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			return TC_ACT_OK;		// accept packet
//
// 		This interface can also be used with all encapsulation devices
// 		that can operate in "collect metadata" mode: instead of having
// 		one network device per specific configuration, the "collect
// 		metadata" mode only requires a single device where the
// 		configuration can be extracted from this helper.
//
// 		This can be used together with various tunnels such as VXLan,
// 		Geneve, GRE or IP in IP (IPIP).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_set_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Populate tunnel metadata for packet associated to *skb.* The
// 		tunnel metadata is set to the contents of *key*, of *size*. The
// 		*flags* can be set to a combination of the following values:
//
// 		**BPF_F_TUNINFO_IPV6**
// 			Indicate that the tunnel is based on IPv6 protocol
// 			instead of IPv4.
// 		**BPF_F_ZERO_CSUM_TX**
// 			For IPv4 packets, add a flag to tunnel metadata
// 			indicating that checksum computation should be skipped
// 			and checksum set to zeroes.
// 		**BPF_F_DONT_FRAGMENT**
// 			Add a flag to tunnel metadata indicating that the
// 			packet should not be fragmented.
// 		**BPF_F_SEQ_NUMBER**
// 			Add a flag to tunnel metadata indicating that a
// 			sequence number should be added to tunnel header before
// 			sending the packet. This flag was added for GRE
// 			encapsulation, but might be used with other protocols
// 			as well in the future.
// 		**BPF_F_NO_TUNNEL_KEY**
// 			Add a flag to tunnel metadata indicating that no tunnel
// 			key should be set in the resulting tunnel header.
//
// 		Here is a typical usage on the transmit path:
//
// 		::
//
// 			struct bpf_tunnel_key key;
// 			     populate key ...
// 			bpf_skb_set_tunnel_key(skb, &key, sizeof(key), 0);
// 			bpf_clone_redirect(skb, vxlan_dev_ifindex, 0);
//
// 		See also the description of the **bpf_skb_get_tunnel_key**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_perf_event_read(struct bpf_map *map, u64 flags)
// 	Description
// 		Read the value of a perf event counter. This helper relies on a
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of
// 		the perf event counter is selected when *map* is updated with
// 		perf event file descriptors. The *map* is an array whose size
// 		is the number of available CPUs, and each cell contains a value
// 		relative to one CPU. The value to retrieve is indicated by
// 		*flags*, that contains the index of the CPU to look up, masked
// 		with **BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		Note that before Linux 4.13, only hardware perf event can be
// 		retrieved.
//
// 		Also, be aware that the newer helper
// 		**bpf_perf_event_read_value**\ () is recommended over
// 		**bpf_perf_event_read**\ () in general. The latter has some ABI
// 		quirks where error and counter value are used as a return code
// 		(which is wrong to do since ranges may overlap). This issue is
// 		fixed with **bpf_perf_event_read_value**\ (), which at the same
// 		time provides more features over the **bpf_perf_event_read** * 		() interface. Please refer to the description of
// 		**bpf_perf_event_read_value**\ () for details.
// 	Return
// 		The value of the perf event counter read from the map, or a
// 		negative error code in case of failure.
//
// long bpf_redirect(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_clone_redirect** * 		(), except that the packet is not cloned, which provides
// 		increased performance.
//
// 		Except for XDP, both ingress and egress interfaces can be used
// 		for redirection. The **BPF_F_INGRESS** value in *flags* is used
// 		to make the distinction (ingress path is selected if the flag
// 		is present, egress path otherwise). Currently, XDP only
// 		supports redirection to the egress interface, and accepts no
// 		flag at all.
//
// 		The same effect can also be attained with the more generic
// 		**bpf_redirect_map**\ (), which uses a BPF map to store the
// 		redirect target instead of providing it directly to the helper.
// 	Return
// 		For XDP, the helper returns **XDP_REDIRECT** on success or
// 		**XDP_ABORTED** on error. For other program types, the values
// 		are **TC_ACT_REDIRECT** on success or **TC_ACT_SHOT** on
// 		error.
//
// u32 bpf_get_route_realm(struct sk_buff *skb)
// 	Description
// 		Retrieve the realm or the route, that is to say the
// 		**tclassid** field of the destination for the *skb*. The
// 		identifier retrieved is a user-provided tag, similar to the
// 		one used with the net_cls cgroup (see description for
// 		**bpf_get_cgroup_classid**\ () helper), but here this tag is
// 		held by a route (a destination entry), not by a task.
//
// 		Retrieving this identifier works with the clsact TC egress hook
// 		(see also **tc-bpf(8)**), or alternatively on conventional
// 		classful egress qdiscs, but not on TC ingress path. In case of
// 		clsact TC egress hook, this has the advantage that, internally,
// 		the destination entry has not been dropped yet in the transmit
// 		path. Therefore, the destination entry does not need to be
// 		artificially held via **netif_keep_dst**\ () for a classful
// 		qdisc until the *skb* is freed.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_IP_ROUTE_CLASSID** configuration option.
// 	Return
// 		The realm of the route for the packet associated to *skb*, or 0
// 		if none was found.
//
// long bpf_perf_event_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		The context of the program *ctx* needs also be passed to the
// 		helper.
//
// 		On user space, a program willing to read the values needs to
// 		call **perf_event_open**\ () on the perf event (either for
// 		one or for all CPUs) and to store the file descriptor into the
// 		*map*. This must be done before the eBPF program can send data
// 		into it. An example is available in file
// 		*samples/bpf/trace_output_user.c* in the Linux kernel source
// 		tree (the eBPF program counterpart is in
// 		*samples/bpf/trace_output.bpf.c*).
//
// 		**bpf_perf_event_output**\ () achieves better performance
// 		than **bpf_trace_printk**\ () for sharing data with user
// 		space, and is much better suitable for streaming data from eBPF
// 		programs.
//
// 		Note that this helper is not restricted to tracing use cases
// 		and can be used with programs attached to TC or XDP as well,
// 		where it allows for passing data to user space listeners. Data
// 		can be:
//
// 		* Only custom structs,
// 		* Only the packet payload, or
// 		* A combination of both.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes(const void *skb, u32 offset, void *to, u32 len)
// 	Description
// 		This helper was provided as an easy way to load data from a
// 		packet. It can be used to load *len* bytes from *offset* from
// 		the packet associated to *skb*, into the buffer pointed by
// 		*to*.
//
// 		Since Linux 4.7, usage of this helper has mostly been replaced
// 		by "direct packet access", enabling packet data to be
// 		manipulated with *skb*\ **->data** and *skb*\ **->data_end**
// 		pointing respectively to the first byte of packet data and to
// 		the byte after the last byte of packet data. However, it
// 		remains useful if one wishes to read large quantities of data
// 		at once from a packet into the eBPF stack.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stackid(void *ctx, struct bpf_map *map, u64 flags)
// 	Description
// 		Walk a user or a kernel stack and return its id. To achieve
// 		this, the helper needs *ctx*, which is a pointer to the context
// 		on which the tracing program is executed, and a pointer to a
// 		*map* of type **BPF_MAP_TYPE_STACK_TRACE**.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		a combination of the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_FAST_STACK_CMP**
// 			Compare stacks by hash only.
// 		**BPF_F_REUSE_STACKID**
// 			If two different stacks hash into the same *stackid*,
// 			discard the old one.
//
// 		The stack id retrieved is a 32 bit long integer handle which
// 		can be further combined with other data (including other stack
// 		ids) and used as a key into maps. This can be useful for
// 		generating a variety of graphs (such as flame graphs or off-cpu
// 		graphs).
//
// 		For walking a stack, this helper is an improvement over
// 		**bpf_probe_read**\ (), which can be used with unrolled loops
// 		but is not efficient and consumes a lot of eBPF instructions.
// 		Instead, **bpf_get_stackid**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The positive or null stack id on success, or a negative error
// 		in case of failure.
//
// s64 bpf_csum_diff(__be32 *from, u32 from_size, __be32 *to, u32 to_size, __wsum seed)
// 	Description
// 		Compute a checksum difference, from the raw buffer pointed by
// 		*from*, of length *from_size* (that must be a multiple of 4),
// 		towards the raw buffer pointed by *to*, of size *to_size*
// 		(same remark). An optional *seed* can be added to the value
// 		(this can be cascaded, the seed may come from a previous call
// 		to the helper).
//
// 		This is flexible enough to be used in several ways:
//
// 		* With *from_size* == 0, *to_size* > 0 and *seed* set to
// 		  checksum, it can be used when pushing new data.
// 		* With *from_size* > 0, *to_size* == 0 and *seed* set to
// 		  checksum, it can be used when removing data from a packet.
// 		* With *from_size* > 0, *to_size* > 0 and *seed* set to 0, it
// 		  can be used to compute a diff. Note that *from_size* and
// 		  *to_size* do not need to be equal.
//
// 		This helper can be used in combination with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace**\ (), to
// 		which one can feed in the difference computed with
// 		**bpf_csum_diff**\ ().
// 	Return
// 		The checksum result, or a negative error code in case of
// 		failure.
//
// long bpf_skb_get_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Retrieve tunnel options metadata for the packet associated to
// 		*skb*, and store the raw tunnel option data to the buffer *opt*
// 		of *size*.
//
// 		This helper can be used with encapsulation devices that can
// 		operate in "collect metadata" mode (please refer to the related
// 		note in the description of **bpf_skb_get_tunnel_key**\ () for
// 		more details). A particular example where this can be used is
// 		in combination with the Geneve encapsulation protocol, where it
// 		allows for pushing (with **bpf_skb_get_tunnel_opt**\ () helper)
// 		and retrieving arbitrary TLVs (Type-Length-Value headers) from
// 		the eBPF program. This allows for full customization of these
// 		headers.
// 	Return
// 		The size of the option data retrieved.
//
// long bpf_skb_set_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Set tunnel options metadata for the packet associated to *skb*
// 		to the option data contained in the raw buffer *opt* of *size*.
//
// 		See also the description of the **bpf_skb_get_tunnel_opt**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_proto(struct sk_buff *skb, __be16 proto, u64 flags)
// 	Description
// 		Change the protocol of the *skb* to *proto*. Currently
// 		supported are transition from IPv4 to IPv6, and from IPv6 to
// 		IPv4. The helper takes care of the groundwork for the
// 		transition, including resizing the socket buffer. The eBPF
// 		program is expected to fill the new headers, if any, via
// 		**skb_store_bytes**\ () and to recompute the checksums with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace** * 		(). The main case for this helper is to perform NAT64
// 		operations out of an eBPF program.
//
// 		Internally, the GSO type is marked as dodgy so that headers are
// 		checked and segments are recalculated by the GSO/GRO engine.
// 		The size for GSO target is adapted as well.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_type(struct sk_buff *skb, u32 type)
// 	Description
// 		Change the packet type for the packet associated to *skb*. This
// 		comes down to setting *skb*\ **->pkt_type** to *type*, except
// 		the eBPF program does not have a write access to *skb* * 		**->pkt_type** beside this helper. Using a helper here allows
// 		for graceful handling of errors.
//
// 		The major use case is to change incoming *skb*s to
// 		**PACKET_HOST** in a programmatic way instead of having to
// 		recirculate via **redirect**\ (..., **BPF_F_INGRESS**), for
// 		example.
//
// 		Note that *type* only allows certain values. At this time, they
// 		are:
//
// 		**PACKET_HOST**
// 			Packet is for us.
// 		**PACKET_BROADCAST**
// 			Send packet to all.
// 		**PACKET_MULTICAST**
// 			Send packet to group.
// 		**PACKET_OTHERHOST**
// 			Send packet to someone else.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_under_cgroup(struct sk_buff *skb, struct bpf_map *map, u32 index)
// 	Description
// 		Check whether *skb* is a descendant of the cgroup2 held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 0, if the *skb* failed the cgroup2 descendant test.
// 		* 1, if the *skb* succeeded the cgroup2 descendant test.
// 		* A negative error code, if an error occurred.
//
// u32 bpf_get_hash_recalc(struct sk_buff *skb)
// 	Description
// 		Retrieve the hash of the packet, *skb*\ **->hash**. If it is
// 		not set, in particular if the hash was cleared due to mangling,
// 		recompute this hash. Later accesses to the hash can be done
// 		directly with *skb*\ **->hash**.
//
// 		Calling **bpf_set_hash_invalid**\ (), changing a packet
// 		prototype with **bpf_skb_change_proto**\ (), or calling
// 		**bpf_skb_store_bytes**\ () with the
// 		**BPF_F_INVALIDATE_HASH** are actions susceptible to clear
// 		the hash and to trigger a new computation for the next call to
// 		**bpf_get_hash_recalc**\ ().
// 	Return
// 		The 32-bit hash.
//
// u64 bpf_get_current_task(void)
// 	Description
// 		Get the current task.
// 	Return
// 		A pointer to the current task struct.
//
// long bpf_probe_write_user(void *dst, const void *src, u32 len)
// 	Description
// 		Attempt in a safe way to write *len* bytes from the buffer
// 		*src* to *dst* in memory. It only works for threads that are in
// 		user context, and *dst* must be a valid user space address.
//
// 		This helper should not be used to implement any kind of
// 		security mechanism because of TOC-TOU attacks, but rather to
// 		debug, divert, and manipulate execution of semi-cooperative
// 		processes.
//
// 		Keep in mind that this feature is meant for experiments, and it
// 		has a risk of crashing the system and running programs.
// 		Therefore, when an eBPF program using this helper is attached,
// 		a warning including PID and process name is printed to kernel
// 		logs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_current_task_under_cgroup(struct bpf_map *map, u32 index)
// 	Description
// 		Check whether the probe is being run is the context of a given
// 		subset of the cgroup2 hierarchy. The cgroup2 to test is held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 1, if current task belongs to the cgroup2.
// 		* 0, if current task does not belong to the cgroup2.
// 		* A negative error code, if an error occurred.
//
// long bpf_skb_change_tail(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Resize (trim or grow) the packet associated to *skb* to the
// 		new *len*. The *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		The basic idea is that the helper performs the needed work to
// 		change the size of the packet, then the eBPF program rewrites
// 		the rest via helpers like **bpf_skb_store_bytes**\ (),
// 		**bpf_l3_csum_replace**\ (), **bpf_l3_csum_replace**\ ()
// 		and others. This helper is a slow path utility intended for
// 		replies with control messages. And because it is targeted for
// 		slow path, the helper itself can afford to be slow: it
// 		implicitly linearizes, unclones and drops offloads from the
// 		*skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_pull_data(struct sk_buff *skb, u32 len)
// 	Description
// 		Pull in non-linear data in case the *skb* is non-linear and not
// 		all of *len* are part of the linear section. Make *len* bytes
// 		from *skb* readable and writable. If a zero value is passed for
// 		*len*, then all bytes in the linear part of *skb* will be made
// 		readable and writable.
//
// 		This helper is only needed for reading and writing with direct
// 		packet access.
//
// 		For direct packet access, testing that offsets to access
// 		are within packet boundaries (test on *skb*\ **->data_end**) is
// 		susceptible to fail if offsets are invalid, or if the requested
// 		data is in non-linear parts of the *skb*. On failure the
// 		program can just bail out, or in the case of a non-linear
// 		buffer, use a helper to make the data available. The
// 		**bpf_skb_load_bytes**\ () helper is a first solution to access
// 		the data. Another one consists in using **bpf_skb_pull_data**
// 		to pull in once the non-linear parts, then retesting and
// 		eventually access the data.
//
// 		At the same time, this also makes sure the *skb* is uncloned,
// 		which is a necessary condition for direct write. As this needs
// 		to be an invariant for the write part only, the verifier
// 		detects writes and adds a prologue that is calling
// 		**bpf_skb_pull_data()** to effectively unclone the *skb* from
// 		the very beginning in case it is indeed cloned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// s64 bpf_csum_update(struct sk_buff *skb, __wsum csum)
// 	Description
// 		Add the checksum *csum* into *skb*\ **->csum** in case the
// 		driver has supplied a checksum for the entire packet into that
// 		field. Return an error otherwise. This helper is intended to be
// 		used in combination with **bpf_csum_diff**\ (), in particular
// 		when the checksum needs to be updated after data has been
// 		written into the packet through direct packet access.
// 	Return
// 		The checksum on success, or a negative error code in case of
// 		failure.
//
// void bpf_set_hash_invalid(struct sk_buff *skb)
// 	Description
// 		Invalidate the current *skb*\ **->hash**. It can be used after
// 		mangling on headers through direct packet access, in order to
// 		indicate that the hash is outdated and to trigger a
// 		recalculation the next time the kernel tries to access this
// 		hash or when the **bpf_get_hash_recalc**\ () helper is called.
// 	Return
// 		void.
//
// long bpf_get_numa_node_id(void)
// 	Description
// 		Return the id of the current NUMA node. The primary use case
// 		for this helper is the selection of sockets for the local NUMA
// 		node, when the program is attached to sockets using the
// 		**SO_ATTACH_REUSEPORT_EBPF** option (see also **socket(7)**),
// 		but the helper is also available to other eBPF program types,
// 		similarly to **bpf_get_smp_processor_id**\ ().
// 	Return
// 		The id of current NUMA node.
//
// long bpf_skb_change_head(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Grows headroom of packet associated to *skb* and adjusts the
// 		offset of the MAC header accordingly, adding *len* bytes of
// 		space. It automatically extends and reallocates memory as
// 		required.
//
// 		This helper can be used on a layer 3 *skb* to push a MAC header
// 		for redirection into a layer 2 device.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_head(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data** by *delta* bytes. Note that
// 		it is possible to use a negative value for *delta*. This helper
// 		can be used to prepare the packet for pushing or popping
// 		headers.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address
// 		*unsafe_ptr* to *dst*. See **bpf_probe_read_kernel_str**\ () for
// 		more details.
//
// 		Generally, use **bpf_probe_read_user_str**\ () or
// 		**bpf_probe_read_kernel_str**\ () instead.
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// u64 bpf_get_socket_cookie(struct sk_buff *skb)
// 	Description
// 		If the **struct sk_buff** pointed by *skb* has a known socket,
// 		retrieve the cookie (generated by the kernel) of this socket.
// 		If no cookie has been set yet, generate a new cookie. Once
// 		generated, the socket cookie remains stable for the life of the
// 		socket. This helper can be useful for monitoring per socket
// 		networking traffic statistics as it provides a global socket
// 		identifier that can be assumed unique.
// 	Return
// 		A 8-byte long unique number on success, or 0 if the socket
// 		field is missing inside *skb*.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_addr *ctx)
// 	Description
// 		Equivalent to bpf_get_socket_cookie() helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_addr** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_ops *ctx)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_ops** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct sock *sk)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*sk*, but gets socket from a BTF **struct sock**. This helper
// 		also works for sleepable programs.
// 	Return
// 		A 8-byte long unique number or 0 if *sk* is NULL.
//
// u32 bpf_get_socket_uid(struct sk_buff *skb)
// 	Description
// 		Get the owner UID of the socked associated to *skb*.
// 	Return
// 		The owner UID of the socket associated to *skb*. If the socket
// 		is **NULL**, or if it is not a full socket (i.e. if it is a
// 		time-wait or a request socket instead), **overflowuid** value
// 		is returned (note that **overflowuid** might also be the actual
// 		UID value for the socket).
//
// long bpf_set_hash(struct sk_buff *skb, u32 hash)
// 	Description
// 		Set the full hash for *skb* (set the field *skb*\ **->hash**)
// 		to value *hash*.
// 	Return
// 		0
//
// long bpf_setsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **setsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **setsockopt(2)** for more information.
// 		The option value of length *optlen* is pointed by *optval*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **setsockopt()**.
// 		It supports the following *level*\ s:
//
// 		* **SOL_SOCKET**, which supports the following *optname*\ s:
// 		  **SO_RCVBUF**, **SO_SNDBUF**, **SO_MAX_PACING_RATE**,
// 		  **SO_PRIORITY**, **SO_RCVLOWAT**, **SO_MARK**,
// 		  **SO_BINDTODEVICE**, **SO_KEEPALIVE**, **SO_REUSEADDR**,
// 		  **SO_REUSEPORT**, **SO_BINDTOIFINDEX**, **SO_TXREHASH**.
// 		* **IPPROTO_TCP**, which supports the following *optname*\ s:
// 		  **TCP_CONGESTION**, **TCP_BPF_IW**,
// 		  **TCP_BPF_SNDCWND_CLAMP**, **TCP_SAVE_SYN**,
// 		  **TCP_KEEPIDLE**, **TCP_KEEPINTVL**, **TCP_KEEPCNT**,
// 		  **TCP_SYNCNT**, **TCP_USER_TIMEOUT**, **TCP_NOTSENT_LOWAT**,
// 		  **TCP_NODELAY**, **TCP_MAXSEG**, **TCP_WINDOW_CLAMP**,
// 		  **TCP_THIN_LINEAR_TIMEOUTS**, **TCP_BPF_DELACK_MAX**,
// 		  **TCP_BPF_RTO_MIN**, **TCP_BPF_SOCK_OPS_CB_FLAGS**.
// 		* **IPPROTO_IP**, which supports *optname* **IP_TOS**.
// 		* **IPPROTO_IPV6**, which supports the following *optname*\ s:
// 		  **IPV6_TCLASS**, **IPV6_AUTOFLOWLABEL**.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_adjust_room(struct sk_buff *skb, s32 len_diff, u32 mode, u64 flags)
// 	Description
// 		Grow or shrink the room for data in the packet associated to
// 		*skb* by *len_diff*, and according to the selected *mode*.
//
// 		By default, the helper will reset any offloaded checksum
// 		indicator of the skb to CHECKSUM_NONE. This can be avoided
// 		by the following flag:
//
// 		* **BPF_F_ADJ_ROOM_NO_CSUM_RESET**: Do not reset offloaded
// 		  checksum data of the skb to CHECKSUM_NONE.
//
// 		There are two supported modes at this time:
//
// 		* **BPF_ADJ_ROOM_MAC**: Adjust room at the mac layer
// 		  (room space is added or removed between the layer 2 and
// 		  layer 3 headers).
//
// 		* **BPF_ADJ_ROOM_NET**: Adjust room at the network layer
// 		  (room space is added or removed between the layer 3 and
// 		  layer 4 headers).
//
// 		The following flags are supported at this time:
//
// 		* **BPF_F_ADJ_ROOM_FIXED_GSO**: Do not adjust gso_size.
// 		  Adjusting mss in this way is not allowed for datagrams.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L3_IPV6**:
// 		  Any new space is reserved to hold a tunnel header.
// 		  Configure skb offsets and other fields accordingly.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L4_GRE**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L4_UDP**:
// 		  Use with ENCAP_L3 flags to further specify the tunnel type.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2**\ (*len*):
// 		  Use with ENCAP_L3/L4 flags to further specify the tunnel
// 		  type; *len* is the length of the inner MAC header.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2_ETH**:
// 		  Use with BPF_F_ADJ_ROOM_ENCAP_L2 flag to further specify the
// 		  L2 type as Ethernet.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_DECAP_L3_IPV6**:
// 		  Indicate the new IP header version after decapsulating the
// 		  outer IP header. Used when the inner and outer IP versions
// 		  are different. These flags only trigger a protocol change
// 		  without clearing any tunnel-specific GSO flags.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_GRE**:
// 		  Clear GRE tunnel GSO flags (SKB_GSO_GRE and SKB_GSO_GRE_CSUM)
// 		  when decapsulating a GRE tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_UDP**:
// 		  Clear UDP tunnel GSO flags (SKB_GSO_UDP_TUNNEL and
// 		  SKB_GSO_UDP_TUNNEL_CSUM) when decapsulating a UDP tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP4**:
// 		  Clear IPIP/SIT tunnel GSO flag (SKB_GSO_IPXIP4) when decapsulating
// 		  a tunnel with an outer IPv4 header (IPv4-in-IPv4 or IPv6-in-IPv4).
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP6**:
// 		  Clear IPv6 encapsulation tunnel GSO flag (SKB_GSO_IPXIP6) when
// 		  decapsulating a tunnel with an outer IPv6 header (IPv6-in-IPv6
// 		  or IPv4-in-IPv6).
//
// 		When using the decapsulation flags above, the skb->encapsulation
// 		flag is automatically cleared if all tunnel-specific GSO flags
// 		(SKB_GSO_UDP_TUNNEL, SKB_GSO_UDP_TUNNEL_CSUM, SKB_GSO_GRE,
// 		SKB_GSO_GRE_CSUM, SKB_GSO_IPXIP4, SKB_GSO_IPXIP6) have been
// 		removed from the packet. This handles cases where all tunnel
// 		layers have been decapsulated.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_redirect_map(struct bpf_map *map, u64 key, u64 flags)
// 	Description
// 		Redirect the packet to the endpoint referenced by *map* at
// 		index *key*. Depending on its type, this *map* can contain
// 		references to net devices (for forwarding packets through other
// 		ports), or to CPUs (for redirecting XDP frames to another CPU;
// 		but this is only implemented for native XDP (with driver
// 		support) as of this writing).
//
// 		The lower two bits of *flags* are used as the return code if
// 		the map lookup fails. This is so that the return value can be
// 		one of the XDP program return codes up to **XDP_TX**, as chosen
// 		by the caller. The higher bits of *flags* can be set to
// 		BPF_F_BROADCAST or BPF_F_EXCLUDE_INGRESS as defined below.
//
// 		With BPF_F_BROADCAST the packet will be broadcasted to all the
// 		interfaces in the map, with BPF_F_EXCLUDE_INGRESS the ingress
// 		interface will be excluded when do broadcasting.
//
// 		See also **bpf_redirect**\ (), which only supports redirecting
// 		to an ifindex, but doesn't require a map to do so.
// 	Return
// 		**XDP_REDIRECT** on success, or the value of the two lower bits
// 		of the *flags* argument on error.
//
// long bpf_sk_redirect_map(struct sk_buff *skb, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		Redirect the packet to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sock_map_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a *map* referencing sockets. The
// 		*skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_meta(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust the address pointed by *xdp_md*\ **->data_meta** by
// 		*delta* (which can be positive or negative). Note that this
// 		operation modifies the address stored in *xdp_md*\ **->data**,
// 		so the latter must be loaded only after the helper has been
// 		called.
//
// 		The use of *xdp_md*\ **->data_meta** is optional and programs
// 		are not required to use it. The rationale is that when the
// 		packet is processed with XDP (e.g. as DoS filter), it is
// 		possible to push further meta data along with it before passing
// 		to the stack, and to give the guarantee that an ingress eBPF
// 		program attached as a TC classifier on the same device can pick
// 		this up for further post-processing. Since TC works with socket
// 		buffers, it remains possible to set from XDP the **mark** or
// 		**priority** pointers, or other pointers for the socket buffer.
// 		Having this scratch space generic and programmable allows for
// 		more flexibility as the user is free to store whatever meta
// 		data they need.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_event_read_value(struct bpf_map *map, u64 flags, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		Read the value of a perf event counter, and store it into *buf*
// 		of size *buf_size*. This helper relies on a *map* of type
// 		**BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of the perf event
// 		counter is selected when *map* is updated with perf event file
// 		descriptors. The *map* is an array whose size is the number of
// 		available CPUs, and each cell contains a value relative to one
// 		CPU. The value to retrieve is indicated by *flags*, that
// 		contains the index of the CPU to look up, masked with
// 		**BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		This helper behaves in a way close to
// 		**bpf_perf_event_read**\ () helper, save that instead of
// 		just returning the value observed, it fills the *buf*
// 		structure. This allows for additional data to be retrieved: in
// 		particular, the enabled and running times (in *buf* * 		**->enabled** and *buf*\ **->running**, respectively) are
// 		copied. In general, **bpf_perf_event_read_value**\ () is
// 		recommended over **bpf_perf_event_read**\ (), which has some
// 		ABI issues and provides fewer functionalities.
//
// 		These values are interesting, because hardware PMU (Performance
// 		Monitoring Unit) counters are limited resources. When there are
// 		more PMU based perf events opened than available counters,
// 		kernel will multiplex these events so each event gets certain
// 		percentage (but not all) of the PMU time. In case that
// 		multiplexing happens, the number of samples or counter value
// 		will not reflect the case compared to when no multiplexing
// 		occurs. This makes comparison between different runs difficult.
// 		Typically, the counter value should be normalized before
// 		comparing to other experiments. The usual normalization is done
// 		as follows.
//
// 		::
//
// 			normalized_counter = counter * t_enabled / t_running
//
// 		Where t_enabled is the time enabled for event and t_running is
// 		the time running for event since last normalization. The
// 		enabled and running times are accumulated since the perf event
// 		open. To achieve scaling factor between two invocations of an
// 		eBPF program, users can use CPU id as the key (which is
// 		typical for perf array usage model) to remember the previous
// 		value and do the calculation inside the eBPF program.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_prog_read_value(struct bpf_perf_event_data *ctx, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		value of the event counter associated to *ctx* and store it in
// 		the structure pointed by *buf* and of size *buf_size*. Enabled
// 		and running times are also stored in the structure (see
// 		description of helper **bpf_perf_event_read_value**\ () for
// 		more details).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_getsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **getsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **getsockopt(2)** for more information.
// 		The retrieved value is stored in the structure pointed by
// 		*opval* and of length *optlen*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **getsockopt()**.
// 		It supports the same set of *optname*\ s that is supported by
// 		the **bpf_setsockopt**\ () helper.  The exceptions are
// 		**TCP_BPF_*** is **bpf_setsockopt**\ () only and
// 		**TCP_SAVED_SYN** is **bpf_getsockopt**\ () only.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_override_return(struct pt_regs *regs, u64 rc)
// 	Description
// 		Used for error injection, this helper uses kprobes to override
// 		the return value of the probed function, and to set it to *rc*.
// 		The first argument is the context *regs* on which the kprobe
// 		works.
//
// 		This helper works by setting the PC (program counter)
// 		to an override function which is run in place of the original
// 		probed function. This means the probed function is not run at
// 		all. The replacement function just returns with the required
// 		value.
//
// 		This helper has security implications, and thus is subject to
// 		restrictions. It is only available if the kernel was compiled
// 		with the **CONFIG_BPF_KPROBE_OVERRIDE** configuration
// 		option, and in this case it only works on functions tagged with
// 		**ALLOW_ERROR_INJECTION** in the kernel code.
// 	Return
// 		0
//
// long bpf_sock_ops_cb_flags_set(struct bpf_sock_ops *bpf_sock, int argval)
// 	Description
// 		Attempt to set the value of the **bpf_sock_ops_cb_flags** field
// 		for the full TCP socket associated to *bpf_sock_ops* to
// 		*argval*.
//
// 		The primary use of this field is to determine if there should
// 		be calls to eBPF programs of type
// 		**BPF_PROG_TYPE_SOCK_OPS** at various points in the TCP
// 		code. A program of the same type can change its value, per
// 		connection and as necessary, when the connection is
// 		established. This field is directly accessible for reading, but
// 		this helper must be used for updates in order to return an
// 		error if an eBPF program tries to set a callback that is not
// 		supported in the current kernel.
//
// 		*argval* is a flag array which can combine these flags:
//
// 		* **BPF_SOCK_OPS_RTO_CB_FLAG** (retransmission time out)
// 		* **BPF_SOCK_OPS_RETRANS_CB_FLAG** (retransmission)
// 		* **BPF_SOCK_OPS_STATE_CB_FLAG** (TCP state change)
// 		* **BPF_SOCK_OPS_RTT_CB_FLAG** (every RTT)
//
// 		Therefore, this function can be used to clear a callback flag by
// 		setting the appropriate bit to zero. e.g. to disable the RTO
// 		callback:
//
// 		**bpf_sock_ops_cb_flags_set(bpf_sock,**
// 			**bpf_sock->bpf_sock_ops_cb_flags & ~BPF_SOCK_OPS_RTO_CB_FLAG)**
//
// 		Here are some examples of where one could call such eBPF
// 		program:
//
// 		* When RTO fires.
// 		* When a packet is retransmitted.
// 		* When the connection terminates.
// 		* When a packet is sent.
// 		* When a packet is received.
// 	Return
// 		Code **-EINVAL** if the socket is not a full TCP socket;
// 		otherwise, a positive number containing the bits that could not
// 		be set is returned (which comes down to 0 if all bits were set
// 		as required).
//
// long bpf_msg_redirect_map(struct sk_msg_buff *msg, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_msg_apply_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, apply the verdict of the eBPF program to
// 		the next *bytes* (number of bytes) of message *msg*.
//
// 		For example, this helper can be used in the following cases:
//
// 		* A single **sendmsg**\ () or **sendfile**\ () system call
// 		  contains multiple logical messages that the eBPF program is
// 		  supposed to read and for which it should apply a verdict.
// 		* An eBPF program only cares to read the first *bytes* of a
// 		  *msg*. If the message has a large payload, then setting up
// 		  and calling the eBPF program repeatedly for all bytes, even
// 		  though the verdict is already known, would create unnecessary
// 		  overhead.
//
// 		When called from within an eBPF program, the helper sets a
// 		counter internal to the BPF infrastructure, that is used to
// 		apply the last verdict to the next *bytes*. If *bytes* is
// 		smaller than the current data being processed from a
// 		**sendmsg**\ () or **sendfile**\ () system call, the first
// 		*bytes* will be sent and the eBPF program will be re-run with
// 		the pointer for start of data pointing to byte number *bytes*
// 		**+ 1**. If *bytes* is larger than the current data being
// 		processed, then the eBPF verdict will be applied to multiple
// 		**sendmsg**\ () or **sendfile**\ () calls until *bytes* are
// 		consumed.
//
// 		Note that if a socket closes with the internal counter holding
// 		a non-zero value, this is not a problem because data is not
// 		being buffered for *bytes* and is sent as it is received.
// 	Return
// 		0
//
// long bpf_msg_cork_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, prevent the execution of the verdict eBPF
// 		program for message *msg* until *bytes* (byte number) have been
// 		accumulated.
//
// 		This can be used when one needs a specific number of bytes
// 		before a verdict can be assigned, even if the data spans
// 		multiple **sendmsg**\ () or **sendfile**\ () calls. The extreme
// 		case would be a user calling **sendmsg**\ () repeatedly with
// 		1-byte long message segments. Obviously, this is bad for
// 		performance, but it is still valid. If the eBPF program needs
// 		*bytes* bytes to validate a header, this helper can be used to
// 		prevent the eBPF program to be called again until *bytes* have
// 		been accumulated.
// 	Return
// 		0
//
// long bpf_msg_pull_data(struct sk_msg_buff *msg, u32 start, u32 end, u64 flags)
// 	Description
// 		For socket policies, pull in non-linear data from user space
// 		for *msg* and set pointers *msg*\ **->data** and *msg* * 		**->data_end** to *start* and *end* bytes offsets into *msg*,
// 		respectively.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it can only parse data that the (**data**, **data_end**)
// 		pointers have already consumed. For **sendmsg**\ () hooks this
// 		is likely the first scatterlist element. But for calls relying
// 		on the **sendpage** handler (e.g. **sendfile**\ ()) this will
// 		be the range (**0**, **0**) because the data is shared with
// 		user space and by default the objective is to avoid allowing
// 		user space to modify data while (or after) eBPF verdict is
// 		being decided. This helper can be used to pull in data and to
// 		set the start and end pointer to given values. Data will be
// 		copied if necessary (i.e. if data was not linear and if start
// 		and end pointers do not point to the same chunk).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_bind(struct bpf_sock_addr *ctx, struct sockaddr *addr, int addr_len)
// 	Description
// 		Bind the socket associated to *ctx* to the address pointed by
// 		*addr*, of length *addr_len*. This allows for making outgoing
// 		connection from the desired IP address, which can be useful for
// 		example when all processes inside a cgroup should use one
// 		single IP address on a host that has multiple IP configured.
//
// 		This helper works for IPv4 and IPv6, TCP and UDP sockets. The
// 		domain (*addr*\ **->sa_family**) must be **AF_INET** (or
// 		**AF_INET6**). It's advised to pass zero port (**sin_port**
// 		or **sin6_port**) which triggers IP_BIND_ADDRESS_NO_PORT-like
// 		behavior and lets the kernel efficiently pick up an unused
// 		port as long as 4-tuple is unique. Passing non-zero port might
// 		lead to degraded performance.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_tail(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data_end** by *delta* bytes. It is
// 		possible to both shrink and grow the packet tail.
// 		Shrink done via *delta* being a negative integer.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_xfrm_state(struct sk_buff *skb, u32 index, struct bpf_xfrm_state *xfrm_state, u32 size, u64 flags)
// 	Description
// 		Retrieve the XFRM state (IP transform framework, see also
// 		**ip-xfrm(8)**) at *index* in XFRM "security path" for *skb*.
//
// 		The retrieved value is stored in the **struct bpf_xfrm_state**
// 		pointed by *xfrm_state* and of length *size*.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_XFRM** configuration option.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stack(void *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		To achieve this, the helper needs *ctx*, which is a pointer
// 		to the context on which the tracing program is executed.
// 		To store the stacktrace, the bpf program provides *buf* with
// 		a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect (build_id, file_offset) instead of ips for user
// 			stack, only valid if **BPF_F_USER_STACK** is also
// 			specified.
//
// 			*file_offset* is an offset relative to the beginning
// 			of the executable or shared object file backing the vma
// 			which the *ip* falls in. It is *not* an offset relative
// 			to that object's base address. Accordingly, it must be
// 			adjusted by adding (sh_addr - sh_offset), where
// 			sh_{addr,offset} correspond to the executable section
// 			containing *file_offset* in the object, for comparisons
// 			to symbols' st_value to be valid.
//
// 		**bpf_get_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes_relative(const void *skb, u32 offset, void *to, u32 len, u32 start_header)
// 	Description
// 		This helper is similar to **bpf_skb_load_bytes**\ () in that
// 		it provides an easy way to load *len* bytes from *offset*
// 		from the packet associated to *skb*, into the buffer pointed
// 		by *to*. The difference to **bpf_skb_load_bytes**\ () is that
// 		a fifth argument *start_header* exists in order to select a
// 		base offset to start from. *start_header* can be one of:
//
// 		**BPF_HDR_START_MAC**
// 			Base offset to load data from is *skb*'s mac header.
// 		**BPF_HDR_START_NET**
// 			Base offset to load data from is *skb*'s network header.
//
// 		In general, "direct packet access" is the preferred method to
// 		access packet data, however, this helper is in particular useful
// 		in socket filters where *skb*\ **->data** does not always point
// 		to the start of the mac header and where "direct packet access"
// 		is not available.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_fib_lookup(void *ctx, struct bpf_fib_lookup *params, int plen, u32 flags)
// 	Description
// 		Do FIB lookup in kernel tables using parameters in *params*.
// 		If lookup is successful and result shows packet is to be
// 		forwarded, the neighbor tables are searched for the nexthop.
// 		If successful (ie., FIB lookup shows forwarding and nexthop
// 		is resolved), the nexthop address is returned in ipv4_dst
// 		or ipv6_dst based on family, smac is set to mac address of
// 		egress device, dmac is set to nexthop mac address, rt_metric
// 		is set to metric from route (IPv4/IPv6 only), and ifindex
// 		is set to the device index of the nexthop from the FIB lookup.
//
// 		*plen* argument is the size of the passed in struct.
// 		*flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_FIB_LOOKUP_DIRECT**
// 			Do a direct table lookup vs full lookup using FIB
// 			rules.
// 		**BPF_FIB_LOOKUP_TBID**
// 			Used with BPF_FIB_LOOKUP_DIRECT.
// 			Use the routing table ID present in *params*->tbid
// 			for the fib lookup.
// 		**BPF_FIB_LOOKUP_OUTPUT**
// 			Perform lookup from an egress perspective (default is
// 			ingress).
// 		**BPF_FIB_LOOKUP_SKIP_NEIGH**
// 			Skip the neighbour table lookup. *params*->dmac
// 			and *params*->smac will not be set as output. A common
// 			use case is to call **bpf_redirect_neigh**\ () after
// 			doing **bpf_fib_lookup**\ ().
// 		**BPF_FIB_LOOKUP_SRC**
// 			Derive and set source IP addr in *params*->ipv{4,6}_src
// 			for the nexthop. If the src addr cannot be derived,
// 			**BPF_FIB_LKUP_RET_NO_SRC_ADDR** is returned. In this
// 			case, *params*->dmac and *params*->smac are not set either.
// 		**BPF_FIB_LOOKUP_MARK**
// 			Use the mark present in *params*->mark for the fib lookup.
// 			This option should not be used with BPF_FIB_LOOKUP_DIRECT,
// 			as it only has meaning for full lookups.
// 		**BPF_FIB_LOOKUP_VLAN**
// 			If the fib lookup resolves to a VLAN device whose
// 			parent is a real (non-VLAN) device, set
// 			*params*->h_vlan_proto and *params*->h_vlan_TCI from
// 			the VLAN device and replace *params*->ifindex with the
// 			parent's ifindex. *params*->h_vlan_TCI carries the VID
// 			only, with PCP and DEI bits zero; a consumer wanting to
// 			set egress priority writes PCP itself. *params*->smac is
// 			the VLAN device's own address, which can differ from the
// 			parent's. Only the immediate parent is resolved; if it
// 			is itself a VLAN device (QinQ) or in another namespace,
// 			the egress cannot be reduced to a physical device plus
// 			one tag and the lookup returns
// 			**BPF_FIB_LKUP_RET_VLAN_FAILURE** with *params*->ifindex
// 			left at the input. To obtain the VLAN device's own
// 			ifindex, repeat the lookup without
// 			**BPF_FIB_LOOKUP_VLAN**, re-initializing *params*
// 			first: output fields overwrite the inputs they share
// 			storage with. The swap and the vlan fields
// 			are written only on success; other output fields keep
// 			the helper's existing behaviour, so a frag-needed result
// 			still reports the route mtu in *params*->mtu_result.
// 			This flag is only valid for XDP programs; tc programs
// 			receive -EINVAL since they can redirect to the VLAN
// 			device directly.
// 		**BPF_FIB_LOOKUP_VLAN_INPUT**
// 			Treat *params*->h_vlan_proto and *params*->h_vlan_TCI
// 			as an input VLAN tag and run the lookup as if ingress
// 			had happened on the VLAN subinterface carrying that tag
// 			on *params*->ifindex. The VID is the low 12 bits of
// 			*params*->h_vlan_TCI; *params*->h_vlan_proto must be
// 			ETH_P_8021Q or ETH_P_8021AD in network byte order, else
// 			**-EINVAL**. If *params*->ifindex is itself a VLAN
// 			device, its inner (QinQ) subinterface is matched; for a
// 			bond or team, pass the master's ifindex. An unmatched
// 			tag, a down device, or one in another namespace returns
// 			**BPF_FIB_LKUP_RET_NOT_FWDED**, mirroring real ingress.
// 			A VID of 0 is looked up literally, so do not set this
// 			flag for priority-tagged frames. Cannot be combined with
// 			**BPF_FIB_LOOKUP_TBID** or **BPF_FIB_LOOKUP_OUTPUT**
// 			(returns **-EINVAL**).
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** tc cls_act programs.
// 	Return
// 		* < 0 if any input argument is invalid
// 		*   0 on success (packet is forwarded, nexthop neighbor exists)
// 		* > 0 one of **BPF_FIB_LKUP_RET_** codes explaining why the
// 		  packet is not forwarded or needs assist from full stack
//
// 		If lookup fails with BPF_FIB_LKUP_RET_FRAG_NEEDED, then the MTU
// 		was exceeded and output params->mtu_result contains the MTU.
//
// long bpf_sock_hash_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a sockhash *map* referencing sockets.
// 		The *skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_redirect_hash(struct sk_msg_buff *msg, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sk_redirect_hash(struct sk_buff *skb, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		skb socket level. If the sk_buff *skb* is allowed to pass (i.e.
// 		if the verdict eBPF program returns **SK_PASS**), redirect it
// 		to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_lwt_push_encap(struct sk_buff *skb, u32 type, void *hdr, u32 len)
// 	Description
// 		Encapsulate the packet associated to *skb* within a Layer 3
// 		protocol header. This header is provided in the buffer at
// 		address *hdr*, with *len* its size in bytes. *type* indicates
// 		the protocol of the header and can be one of:
//
// 		**BPF_LWT_ENCAP_SEG6**
// 			IPv6 encapsulation with Segment Routing Header
// 			(**struct ipv6_sr_hdr**). *hdr* only contains the SRH,
// 			the IPv6 header is computed by the kernel.
// 		**BPF_LWT_ENCAP_SEG6_INLINE**
// 			Only works if *skb* contains an IPv6 packet. Insert a
// 			Segment Routing Header (**struct ipv6_sr_hdr**) inside
// 			the IPv6 header.
// 		**BPF_LWT_ENCAP_IP**
// 			IP encapsulation (GRE/GUE/IPIP/etc). The outer header
// 			must be IPv4 or IPv6, followed by zero or more
// 			additional headers, up to **LWT_BPF_MAX_HEADROOM**
// 			total bytes in all prepended headers. Please note that
// 			if **skb_is_gso**\ (*skb*) is true, no more than two
// 			headers can be prepended, and the inner header, if
// 			present, should be either GRE or UDP/GUE.
//
// 		**BPF_LWT_ENCAP_SEG6**\ \* types can be called by BPF programs
// 		of type **BPF_PROG_TYPE_LWT_IN**; **BPF_LWT_ENCAP_IP** type can
// 		be called by bpf programs of types **BPF_PROG_TYPE_LWT_IN** and
// 		**BPF_PROG_TYPE_LWT_XMIT**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. Only the flags, tag and TLVs
// 		inside the outermost IPv6 Segment Routing Header can be
// 		modified through this helper.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_adjust_srh(struct sk_buff *skb, u32 offset, s32 delta)
// 	Description
// 		Adjust the size allocated to TLVs in the outermost IPv6
// 		Segment Routing Header contained in the packet associated to
// 		*skb*, at position *offset* by *delta* bytes. Only offsets
// 		after the segments are accepted. *delta* can be as well
// 		positive (growing) as negative (shrinking).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_action(struct sk_buff *skb, u32 action, void *param, u32 param_len)
// 	Description
// 		Apply an IPv6 Segment Routing action of type *action* to the
// 		packet associated to *skb*. Each action takes a parameter
// 		contained at address *param*, and of length *param_len* bytes.
// 		*action* can be one of:
//
// 		**SEG6_LOCAL_ACTION_END_X**
// 			End.X action: Endpoint with Layer-3 cross-connect.
// 			Type of *param*: **struct in6_addr**.
// 		**SEG6_LOCAL_ACTION_END_T**
// 			End.T action: Endpoint with specific IPv6 table lookup.
// 			Type of *param*: **int**.
// 		**SEG6_LOCAL_ACTION_END_B6**
// 			End.B6 action: Endpoint bound to an SRv6 policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
// 		**SEG6_LOCAL_ACTION_END_B6_ENCAP**
// 			End.B6.Encap action: Endpoint bound to an SRv6
// 			encapsulation policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_repeat(void *ctx)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded repeat key message. This delays
// 		the generation of a key up event for previously generated
// 		key down event.
//
// 		Some IR protocols like NEC have a special IR message for
// 		repeating last button, for when a button is held down.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_rc_keydown(void *ctx, u32 protocol, u64 scancode, u32 toggle)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded key press with *scancode*,
// 		*toggle* value in the given *protocol*. The scancode will be
// 		translated to a keycode using the rc keymap, and reported as
// 		an input key down event. After a period a key up event is
// 		generated. This period can be extended by calling either
// 		**bpf_rc_keydown**\ () again with the same values, or calling
// 		**bpf_rc_repeat**\ ().
//
// 		Some protocols include a toggle bit, in case the button was
// 		released and pressed again between consecutive scancodes.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		The *protocol* is the decoded protocol number (see
// 		**enum rc_proto** for some predefined values).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// u64 bpf_skb_cgroup_id(struct sk_buff *skb)
// 	Description
// 		Return the cgroup v2 id of the socket associated with the *skb*.
// 		This is roughly similar to the **bpf_get_cgroup_classid**\ ()
// 		helper for cgroup v1 by providing a tag resp. identifier that
// 		can be matched on or used for map lookups e.g. to implement
// 		policy. The cgroup v2 id of a given path in the hierarchy is
// 		exposed in user space through the f_handle API in order to get
// 		to the same 64-bit id.
//
// 		This helper can be used on TC egress path, but not on ingress,
// 		and is available only if the kernel was compiled with the
// 		**CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_get_current_cgroup_id(void)
// 	Description
// 		Get the current cgroup id based on the cgroup within which
// 		the current task is running.
// 	Return
// 		A 64-bit integer containing the current cgroup id based
// 		on the cgroup within which the current task is running.
//
// void *bpf_get_local_storage(void *map, u64 flags)
// 	Description
// 		Get the pointer to the local storage area.
// 		The type and the size of the local storage is defined
// 		by the *map* argument.
// 		The *flags* meaning is specific for each map type,
// 		and has to be 0 for cgroup local storage.
//
// 		Depending on the BPF program type, a local storage area
// 		can be shared between multiple instances of the BPF program,
// 		running simultaneously.
//
// 		A user should care about the synchronization by himself.
// 		For example, by using the **BPF_ATOMIC** instructions to alter
// 		the shared data.
// 	Return
// 		A pointer to the local storage area.
//
// long bpf_sk_select_reuseport(struct sk_reuseport_md *reuse, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Select a **SO_REUSEPORT** socket from a
// 		**BPF_MAP_TYPE_REUSEPORT_SOCKARRAY** *map*.
// 		It checks the selected socket is matching the incoming
// 		request in the socket buffer.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_skb_ancestor_cgroup_id(struct sk_buff *skb, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *skb* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *skb*, then return value will be same as that
// 		of **bpf_skb_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *skb*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_skb_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// struct bpf_sock *bpf_sk_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// struct bpf_sock *bpf_sk_lookup_udp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for UDP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_sk_release(void *sock)
// 	Description
// 		Release the reference held by *sock*. *sock* must be a
// 		non-**NULL** pointer that was returned from
// 		**bpf_sk_lookup_xxx**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_push_elem(struct bpf_map *map, const void *value, u64 flags)
// 	Description
// 		Push an element *value* in *map*. *flags* is one of:
//
// 		**BPF_EXIST**
// 			If the queue/stack is full, the oldest element is
// 			removed to make room for this.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_pop_elem(struct bpf_map *map, void *value)
// 	Description
// 		Pop an element from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_peek_elem(struct bpf_map *map, void *value)
// 	Description
// 		Get an element from *map* without removing it.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_push_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		For socket policies, insert *len* bytes into *msg* at offset
// 		*start*.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it may want to insert metadata or options into the *msg*.
// 		This can later be read and used by any of the lower layer BPF
// 		hooks.
//
// 		This helper may fail if under memory pressure (a malloc
// 		fails) in these cases BPF programs will get an appropriate
// 		error and BPF programs will need to handle them.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_pop_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		Will remove *len* bytes from a *msg* starting at byte *start*.
// 		This may result in **ENOMEM** errors under certain situations if
// 		an allocation and copy are required due to a full ring buffer.
// 		However, the helper will try to avoid doing the allocation
// 		if possible. Other errors can occur if input parameters are
// 		invalid either due to *start* byte not being valid part of *msg*
// 		payload and/or *pop* value being to large.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_pointer_rel(void *ctx, s32 rel_x, s32 rel_y)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded pointer movement.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_spin_lock(struct bpf_spin_lock *lock)
// 	Description
// 		Acquire a spinlock represented by the pointer *lock*, which is
// 		stored as part of a value of a map. Taking the lock allows to
// 		safely update the rest of the fields in that value. The
// 		spinlock can (and must) later be released with a call to
// 		**bpf_spin_unlock**\ (\ *lock*\ ).
//
// 		Spinlocks in BPF programs come with a number of restrictions
// 		and constraints:
//
// 		* **bpf_spin_lock** objects are only allowed inside maps of
// 		  types **BPF_MAP_TYPE_HASH** and **BPF_MAP_TYPE_ARRAY** (this
// 		  list could be extended in the future).
// 		* BTF description of the map is mandatory.
// 		* The BPF program can take ONE lock at a time, since taking two
// 		  or more could cause dead locks.
// 		* Only one **struct bpf_spin_lock** is allowed per map element.
// 		* When the lock is taken, calls (either BPF to BPF or helpers)
// 		  are not allowed.
// 		* The **BPF_LD_ABS** and **BPF_LD_IND** instructions are not
// 		  allowed inside a spinlock-ed region.
// 		* The BPF program MUST call **bpf_spin_unlock**\ () to release
// 		  the lock, on all execution paths, before it returns.
// 		* The BPF program can access **struct bpf_spin_lock** only via
// 		  the **bpf_spin_lock**\ () and **bpf_spin_unlock**\ ()
// 		  helpers. Loading or storing data into the **struct
// 		  bpf_spin_lock** *lock*\ **;** field of a map is not allowed.
// 		* To use the **bpf_spin_lock**\ () helper, the BTF description
// 		  of the map value must be a struct and have **struct
// 		  bpf_spin_lock** *anyname*\ **;** field at the top level.
// 		  Nested lock inside another struct is not allowed.
// 		* The **struct bpf_spin_lock** *lock* field in a map value must
// 		  be aligned on a multiple of 4 bytes in that value.
// 		* Syscall with command **BPF_MAP_LOOKUP_ELEM** does not copy
// 		  the **bpf_spin_lock** field to user space.
// 		* Syscall with command **BPF_MAP_UPDATE_ELEM**, or update from
// 		  a BPF program, do not update the **bpf_spin_lock** field.
// 		* **bpf_spin_lock** cannot be on the stack or inside a
// 		  networking packet (it can only be inside of a map values).
// 		* **bpf_spin_lock** is available to root only.
// 		* Tracing programs and socket filter programs cannot use
// 		  **bpf_spin_lock**\ () due to insufficient preemption checks
// 		  (but this may change in the future).
// 		* **bpf_spin_lock** is not allowed in inner maps of map-in-map.
// 	Return
// 		0
//
// long bpf_spin_unlock(struct bpf_spin_lock *lock)
// 	Description
// 		Release the *lock* previously locked by a call to
// 		**bpf_spin_lock**\ (\ *lock*\ ).
// 	Return
// 		0
//
// struct bpf_sock *bpf_sk_fullsock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_sock** pointer such
// 		that all the fields in this **bpf_sock** can be accessed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_tcp_sock *bpf_tcp_sock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_tcp_sock** pointer from a
// 		**struct bpf_sock** pointer.
// 	Return
// 		A **struct bpf_tcp_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// long bpf_skb_ecn_set_ce(struct sk_buff *skb)
// 	Description
// 		Set ECN (Explicit Congestion Notification) field of IP header
// 		to **CE** (Congestion Encountered) if current value is **ECT**
// 		(ECN Capable Transport). Otherwise, do nothing. Works with IPv6
// 		and IPv4.
// 	Return
// 		1 if the **CE** flag is set (either by the current helper call
// 		or because it was already present), 0 if it is not set.
//
// struct bpf_sock *bpf_get_listener_sock(struct bpf_sock *sk)
// 	Description
// 		Return a **struct bpf_sock** pointer in **TCP_LISTEN** state.
// 		**bpf_sk_release**\ () is unnecessary and not allowed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_sock *bpf_skc_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		This function is identical to **bpf_sk_lookup_tcp**\ (), except
// 		that it also returns timewait or request sockets. Use
// 		**bpf_sk_fullsock**\ () or **bpf_tcp_sock**\ () to access the
// 		full structure.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_tcp_check_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK for
// 		the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK, or a negative
// 		error otherwise.
//
// long bpf_sysctl_get_name(struct bpf_sysctl *ctx, char *buf, size_t buf_len, u64 flags)
// 	Description
// 		Get name of sysctl in /proc/sys/ and copy it into provided by
// 		program buffer *buf* of size *buf_len*.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
//
// 		If *flags* is zero, full name (e.g. "net/ipv4/tcp_mem") is
// 		copied. Use **BPF_F_SYSCTL_BASE_NAME** flag to copy base name
// 		only (e.g. "tcp_mem").
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// long bpf_sysctl_get_current_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get current value of sysctl as it is presented in /proc/sys
// 		(incl. newline, etc), and copy it as a string into provided
// 		by program buffer *buf* of size *buf_len*.
//
// 		The whole value is copied, no matter what file position user
// 		space issued e.g. sys_read at.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if current value was unavailable, e.g. because
// 		sysctl is uninitialized and read returns -EIO for it.
//
// long bpf_sysctl_get_new_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get new value being written by user space to sysctl (before
// 		the actual write happens) and copy it as a string into
// 		provided by program buffer *buf* of size *buf_len*.
//
// 		User space may write new value at file position > 0.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_sysctl_set_new_value(struct bpf_sysctl *ctx, const char *buf, size_t buf_len)
// 	Description
// 		Override new value being written by user space to sysctl with
// 		value provided by program in buffer *buf* of size *buf_len*.
//
// 		*buf* should contain a string in same form as provided by user
// 		space on sysctl write.
//
// 		User space may write new value at file position > 0. To override
// 		the whole sysctl value file position should be set to zero.
// 	Return
// 		0 on success.
//
// 		**-E2BIG** if the *buf_len* is too big.
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_strtol(const char *buf, size_t buf_len, u64 flags, long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to a long integer according to the given base
// 		and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)) followed by a single
// 		optional '**-**' sign.
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtol**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// long bpf_strtoul(const char *buf, size_t buf_len, u64 flags, unsigned long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to an unsigned long integer according to the
// 		given base and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)).
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtoul**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// void *bpf_sk_storage_get(struct bpf_map *map, void *sk, void *value, u64 flags)
// 	Description
// 		Get a bpf-local-storage from a *sk*.
//
// 		Logically, it could be thought of getting the value from
// 		a *map* with *sk* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *sk*) except this
// 		helper enforces the key must be a full socket and the map must
// 		be a **BPF_MAP_TYPE_SK_STORAGE** also.
//
// 		Underneath, the value is stored locally at *sk* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf-local-storages residing at *sk*.
//
// 		*sk* is a kernel **struct sock** pointer for LSM program.
// 		*sk* is a **struct bpf_sock** pointer for other program types.
//
// 		An optional *flags* (**BPF_SK_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf-local-storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_SK_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf-local-storage.  If *value* is
// 		**NULL**, the new bpf-local-storage will be zero initialized.
// 	Return
// 		A bpf-local-storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf-local-storage.
//
// long bpf_sk_storage_delete(struct bpf_map *map, void *sk)
// 	Description
// 		Delete a bpf-local-storage from a *sk*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf-local-storage cannot be found.
// 		**-EINVAL** if sk is not a fullsock (e.g. a request_sock).
//
// long bpf_send_signal(u32 sig)
// 	Description
// 		Send signal *sig* to the process of the current task.
// 		The signal may be delivered to any of this process's threads.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// s64 bpf_tcp_gen_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IP/TCP headers, *iph* and *th*, on the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header with options (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** SYN cookie cannot be issued due to error
//
// 		**-ENOENT** SYN cookie should not be issued (no SYN flood)
//
// 		**-EOPNOTSUPP** kernel configuration does not enable SYN cookies
//
// 		**-EPROTONOSUPPORT** IP packet version is not 4 or 6
//
// long bpf_skb_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct sk_buff.
//
// 		This helper is similar to **bpf_perf_event_output**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from user space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_kernel(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from kernel space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe user address
// 		*unsafe_ptr* to *dst*. The *size* should include the
// 		terminating NUL byte. In case the string length is smaller than
// 		*size*, the target is not padded with further NUL bytes. If the
// 		string length is larger than *size*, just *size*-1 bytes are
// 		copied and the last byte is set to NUL.
//
// 		On success, returns the number of bytes that were written,
// 		including the terminal NUL. This makes this helper useful in
// 		tracing programs for reading strings, and more importantly to
// 		get its length at runtime. See the following snippet:
//
// 		::
//
// 			SEC("kprobe/sys_open")
// 			void bpf_sys_open(struct pt_regs *ctx)
// 			{
// 			        char buf[PATHLEN]; // PATHLEN is defined to 256
// 			        int res = bpf_probe_read_user_str(buf, sizeof(buf),
// 				                                  ctx->di);
//
// 				// Consume buf, for example push it to
// 				// userspace via bpf_perf_event_output(); we
// 				// can use res (the string length) as event
// 				// size, after checking its boundaries.
// 			}
//
// 		In comparison, using **bpf_probe_read_user**\ () helper here
// 		instead to read the string would require to estimate the length
// 		at compile time, and would often result in copying more memory
// 		than necessary.
//
// 		Another useful use case is when parsing individual process
// 		arguments or individual environment variables navigating
// 		*current*\ **->mm->arg_start** and *current* * 		**->mm->env_start**: using this helper and the return value,
// 		one can quickly iterate at the right offset of the memory area.
// 	Return
// 		On success, the strictly positive length of the output string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_probe_read_kernel_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address *unsafe_ptr*
// 		to *dst*. Same semantics as with **bpf_probe_read_user_str**\ () apply.
// 	Return
// 		On success, the strictly positive length of the string, including
// 		the trailing NUL character. On error, a negative value.
//
// long bpf_tcp_send_ack(void *tp, u32 rcv_nxt)
// 	Description
// 		Send out a tcp-ack. *tp* is the in-kernel struct **tcp_sock**.
// 		*rcv_nxt* is the ack_seq to be sent out.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_send_signal_thread(u32 sig)
// 	Description
// 		Send signal *sig* to the thread corresponding to the current task.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// u64 bpf_jiffies64(void)
// 	Description
// 		Obtain the 64bit jiffies
// 	Return
// 		The 64 bit jiffies
//
// long bpf_read_branch_records(struct bpf_perf_event_data *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		branch records (**struct perf_branch_entry**) associated to *ctx*
// 		and store it in the buffer pointed by *buf* up to size
// 		*size* bytes.
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		The *flags* can be set to **BPF_F_GET_BRANCH_RECORDS_SIZE** to
// 		instead return the number of bytes required to store all the
// 		branch entries. If this flag is set, *buf* may be NULL.
//
// 		**-EINVAL** if arguments invalid or **size** not a multiple
// 		of **sizeof**\ (**struct perf_branch_entry**\ ).
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_get_ns_current_pid_tgid(u64 dev, u64 ino, struct bpf_pidns_info *nsdata, u32 size)
// 	Description
// 		Returns 0 on success, values for *pid* and *tgid* as seen from the current
// 		*namespace* will be returned in *nsdata*.
// 	Return
// 		0 on success, or one of the following in case of failure:
//
// 		**-EINVAL** if dev and inum supplied don't match dev_t and inode number
//              with nsfs of current task, or if dev conversion to dev_t lost high bits.
//
// 		**-ENOENT** if pidns does not exists for the current task.
//
// long bpf_xdp_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct xdp_buff.
//
// 		This helper is similar to **bpf_perf_eventoutput**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_get_netns_cookie(void *ctx)
// 	Description
// 		Retrieve the cookie (generated by the kernel) of the network
// 		namespace the input *ctx* is associated with. The network
// 		namespace cookie remains stable for its lifetime and provides
// 		a global identifier that can be assumed unique. If *ctx* is
// 		NULL, then the helper returns the cookie for the initial
// 		network namespace. The cookie itself is very similar to that
// 		of **bpf_get_socket_cookie**\ () helper, but for network
// 		namespaces instead of sockets.
// 	Return
// 		A 8-byte long opaque number.
//
// u64 bpf_get_current_ancestor_cgroup_id(int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of the cgroup associated
// 		with the current task at the *ancestor_level*. The root cgroup
// 		is at *ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with the current task, then return value will be the
// 		same as that of **bpf_get_current_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with the current task.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_get_current_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_sk_assign(struct sk_buff *skb, void *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SCHED_CLS** and
// 		**BPF_PROG_TYPE_SCHED_ACT** programs.
//
// 		Assign the *sk* to the *skb*. When combined with appropriate
// 		routing configuration to receive the packet towards the socket,
// 		will cause *skb* to be delivered to the specified socket.
// 		Subsequent redirection of *skb* via  **bpf_redirect**\ (),
// 		**bpf_clone_redirect**\ () or other methods outside of BPF may
// 		interfere with successful delivery to the socket.
//
// 		This operation is only valid from TC ingress path.
//
// 		The *flags* argument must be zero.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EINVAL** if specified *flags* are not supported.
//
// 		**-ENOENT** if the socket is unavailable for assignment.
//
// 		**-ENETUNREACH** if the socket is unreachable (wrong netns).
//
// 		**-EOPNOTSUPP** if the operation is not supported, for example
// 		a call from outside of TC ingress.
//
// long bpf_sk_assign(struct bpf_sk_lookup *ctx, struct bpf_sock *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SK_LOOKUP** programs.
//
// 		Select the *sk* as a result of a socket lookup.
//
// 		For the operation to succeed passed socket must be compatible
// 		with the packet description provided by the *ctx* object.
//
// 		L4 protocol (**IPPROTO_TCP** or **IPPROTO_UDP**) must
// 		be an exact match. While IP family (**AF_INET** or
// 		**AF_INET6**) must be compatible, that is IPv6 sockets
// 		that are not v6-only can be selected for IPv4 packets.
//
// 		Only TCP listeners and UDP unconnected sockets can be
// 		selected. *sk* can also be NULL to reset any previous
// 		selection.
//
// 		*flags* argument can combination of following values:
//
// 		* **BPF_SK_LOOKUP_F_REPLACE** to override the previous
// 		  socket selection, potentially done by a BPF program
// 		  that ran before us.
//
// 		* **BPF_SK_LOOKUP_F_NO_REUSEPORT** to skip
// 		  load-balancing within reuseport group for the socket
// 		  being selected.
//
// 		On success *ctx->sk* will point to the selected socket.
//
// 	Return
// 		0 on success, or a negative errno in case of failure.
//
// 		* **-EAFNOSUPPORT** if socket family (*sk->family*) is
// 		  not compatible with packet family (*ctx->family*).
//
// 		* **-EEXIST** if socket has been already selected,
// 		  potentially by another program, and
// 		  **BPF_SK_LOOKUP_F_REPLACE** flag was not specified.
//
// 		* **-EINVAL** if unsupported flags were specified.
//
// 		* **-EPROTOTYPE** if socket L4 protocol
// 		  (*sk->protocol*) doesn't match packet protocol
// 		  (*ctx->protocol*).
//
// 		* **-ESOCKTNOSUPPORT** if socket is not in allowed
// 		  state (TCP listening or UDP unconnected).
//
// u64 bpf_ktime_get_boot_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does include the time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_BOOTTIME**)
// 	Return
// 		Current *ktime*.
//
// long bpf_seq_printf(struct seq_file *m, const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		**bpf_seq_printf**\ () uses seq_file **seq_printf**\ () to print
// 		out the format string.
// 		The *m* represents the seq_file. The *fmt* and *fmt_size* are for
// 		the format string itself. The *data* and *data_len* are format string
// 		arguments. The *data* are a **u64** array and corresponding format string
// 		values are stored in the array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data* array.
// 		The *data_len* is the size of *data* in bytes - must be a multiple of 8.
//
// 		Formats **%s**, **%p{i,I}{4,6}** requires to read kernel memory.
// 		Reading kernel memory may fail due to either invalid address or
// 		valid address but requiring a major memory fault. If reading kernel memory
// 		fails, the string for **%s** will be an empty string, and the ip
// 		address for **%p{i,I}{4,6}** will be 0. Not returning error to
// 		bpf program is consistent with what **bpf_trace_printk**\ () does for now.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EBUSY** if per-CPU memory copy buffer is busy, can try again
// 		by returning 1 from bpf program.
//
// 		**-EINVAL** if arguments are invalid, or if *fmt* is invalid/unsupported.
//
// 		**-E2BIG** if *fmt* contains too many format specifiers.
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// long bpf_seq_write(struct seq_file *m, const void *data, u32 len)
// 	Description
// 		**bpf_seq_write**\ () uses seq_file **seq_write**\ () to write the data.
// 		The *m* represents the seq_file. The *data* and *len* represent the
// 		data to write in bytes.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// u64 bpf_sk_cgroup_id(void *sk)
// 	Description
// 		Return the cgroup v2 id of the socket *sk*.
//
// 		*sk* must be a non-**NULL** pointer to a socket, e.g. one
// 		returned from **bpf_sk_lookup_xxx**\ (),
// 		**bpf_sk_fullsock**\ (), etc. The format of returned id is
// 		same as in **bpf_skb_cgroup_id**\ ().
//
// 		This helper is available only if the kernel was compiled with
// 		the **CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_sk_ancestor_cgroup_id(void *sk, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *sk* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *sk*, then return value will be same as that
// 		of **bpf_sk_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *sk*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_sk_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_ringbuf_output(void *ringbuf, void *data, u64 size, u64 flags)
// 	Description
// 		Copy *size* bytes from *data* into a ring buffer *ringbuf*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		An adaptive notification is a notification sent whenever the user-space
// 		process has caught up and consumed all available payloads. In case the user-space
// 		process is still processing a previous payload, then no notification is needed
// 		as it will process the newly added payload automatically.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void *bpf_ringbuf_reserve(void *ringbuf, u64 size, u64 flags)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*.
// 		*flags* must be 0.
// 	Return
// 		Valid pointer with *size* bytes of memory available; NULL,
// 		otherwise.
//
// void bpf_ringbuf_submit(void *data, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard(void *data, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent. Discarded records remain in
// 		the ring buffer until consumed by user space, so a later submit
// 		using adaptive wakeup might not wake up the consumer.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// u64 bpf_ringbuf_query(void *ringbuf, u64 flags)
// 	Description
// 		Query various characteristics of provided ring buffer. What
// 		exactly is queries is determined by *flags*:
//
// 		* **BPF_RB_AVAIL_DATA**: Amount of data not yet consumed.
// 		* **BPF_RB_RING_SIZE**: The size of ring buffer.
// 		* **BPF_RB_CONS_POS**: Consumer position (can wrap around).
// 		* **BPF_RB_PROD_POS**: Producer(s) position (can wrap around).
// 		* **BPF_RB_OVERWRITE_POS**: Overwrite position (can wrap around).
//
// 		Data returned is just a momentary snapshot of actual values
// 		and could be inaccurate, so this facility should be used to
// 		power heuristics and for reporting, not to make 100% correct
// 		calculation.
// 	Return
// 		Requested value, or 0, if *flags* are not recognized.
//
// long bpf_csum_level(struct sk_buff *skb, u64 level)
// 	Description
// 		Change the skbs checksum level by one layer up or down, or
// 		reset it entirely to none in order to have the stack perform
// 		checksum validation. The level is applicable to the following
// 		protocols: TCP, UDP, GRE, SCTP, FCOE. For example, a decap of
// 		| ETH | IP | UDP | GUE | IP | TCP | into | ETH | IP | TCP |
// 		through **bpf_skb_adjust_room**\ () helper with passing in
// 		**BPF_F_ADJ_ROOM_NO_CSUM_RESET** flag would require one	call
// 		to **bpf_csum_level**\ () with **BPF_CSUM_LEVEL_DEC** since
// 		the UDP header is removed. Similarly, an encap of the latter
// 		into the former could be accompanied by a helper call to
// 		**bpf_csum_level**\ () with **BPF_CSUM_LEVEL_INC** if the
// 		skb is still intended to be processed in higher layers of the
// 		stack instead of just egressing at tc.
//
// 		There are three supported level settings at this time:
//
// 		* **BPF_CSUM_LEVEL_INC**: Increases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_DEC**: Decreases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_RESET**: Resets skb->csum_level to 0 and
// 		  sets CHECKSUM_NONE to force checksum validation by the stack.
// 		* **BPF_CSUM_LEVEL_QUERY**: No-op, returns the current
// 		  skb->csum_level.
// 	Return
// 		0 on success, or a negative error in case of failure. In the
// 		case of **BPF_CSUM_LEVEL_QUERY**, the current skb->csum_level
// 		is returned or the error code -EACCES in case the skb is not
// 		subject to CHECKSUM_UNNECESSARY.
//
// struct tcp6_sock *bpf_skc_to_tcp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_sock *bpf_skc_to_tcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_timewait_sock *bpf_skc_to_tcp_timewait_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_timewait_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_request_sock *bpf_skc_to_tcp_request_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_request_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct udp6_sock *bpf_skc_to_udp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *udp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_get_task_stack(struct task_struct *task, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		Note: the user stack will only be populated if the *task* is
// 		the current task; all other tasks will return -EOPNOTSUPP.
// 		To achieve this, the helper needs *task*, which is a valid
// 		pointer to **struct task_struct**. To store the stacktrace, the
// 		bpf program provides *buf* with a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 			The *task* must be the current task.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect buildid+offset instead of ips for user stack,
// 			only valid if **BPF_F_USER_STACK** is also specified.
//
// 		**bpf_get_task_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_load_hdr_opt(struct bpf_sock_ops *skops, void *searchby_res, u32 len, u64 flags)
// 	Description
// 		Load header option.  Support reading a particular TCP header
// 		option for bpf program (**BPF_PROG_TYPE_SOCK_OPS**).
//
// 		If *flags* is 0, it will search the option from the
// 		*skops*\ **->skb_data**.  The comment in **struct bpf_sock_ops**
// 		has details on what skb_data contains under different
// 		*skops*\ **->op**.
//
// 		The first byte of the *searchby_res* specifies the
// 		kind that it wants to search.
//
// 		If the searching kind is an experimental kind
// 		(i.e. 253 or 254 according to RFC6994).  It also
// 		needs to specify the "magic" which is either
// 		2 bytes or 4 bytes.  It then also needs to
// 		specify the size of the magic by using
// 		the 2nd byte which is "kind-length" of a TCP
// 		header option and the "kind-length" also
// 		includes the first 2 bytes "kind" and "kind-length"
// 		itself as a normal TCP header option also does.
//
// 		For example, to search experimental kind 254 with
// 		2 byte magic 0xeB9F, the searchby_res should be
// 		[ 254, 4, 0xeB, 0x9F, 0, 0, .... 0 ].
//
// 		To search for the standard window scale option (3),
// 		the *searchby_res* should be [ 3, 0, 0, .... 0 ].
// 		Note, kind-length must be 0 for regular option.
//
// 		Searching for No-Op (0) and End-of-Option-List (1) are
// 		not supported.
//
// 		*len* must be at least 2 bytes which is the minimal size
// 		of a header option.
//
// 		Supported flags:
//
// 		* **BPF_LOAD_HDR_OPT_TCP_SYN** to search from the
// 		  saved_syn packet or the just-received syn packet.
//
// 	Return
// 		> 0 when found, the header option is copied to *searchby_res*.
// 		The return value is the total length copied. On failure, a
// 		negative error code is returned:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOMSG** if the option is not found.
//
// 		**-ENOENT** if no syn packet is available when
// 		**BPF_LOAD_HDR_OPT_TCP_SYN** is used.
//
// 		**-ENOSPC** if there is not enough space.  Only *len* number of
// 		bytes are copied.
//
// 		**-EFAULT** on failure to parse the header options in the
// 		packet.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_store_hdr_opt(struct bpf_sock_ops *skops, const void *from, u32 len, u64 flags)
// 	Description
// 		Store header option.  The data will be copied
// 		from buffer *from* with length *len* to the TCP header.
//
// 		The buffer *from* should have the whole option that
// 		includes the kind, kind-length, and the actual
// 		option data.  The *len* must be at least kind-length
// 		long.  The kind-length does not have to be 4 byte
// 		aligned.  The kernel will take care of the padding
// 		and setting the 4 bytes aligned value to th->doff.
//
// 		This helper will check for duplicated option
// 		by searching the same option in the outgoing skb.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** If param is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
// 		Nothing has been written
//
// 		**-EEXIST** if the option already exists.
//
// 		**-EFAULT** on failure to parse the existing header options.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_reserve_hdr_opt(struct bpf_sock_ops *skops, u32 len, u64 flags)
// 	Description
// 		Reserve *len* bytes for the bpf header option.  The
// 		space will be used by **bpf_store_hdr_opt**\ () later in
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 		If **bpf_reserve_hdr_opt**\ () is called multiple times,
// 		the total number of bytes will be reserved.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_HDR_OPT_LEN_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// void *bpf_inode_storage_get(struct bpf_map *map, void *inode, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from an *inode*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *inode* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *inode*) except this
// 		helper enforces the key must be an inode and the map must also
// 		be a **BPF_MAP_TYPE_INODE_STORAGE**.
//
// 		Underneath, the value is stored locally at *inode* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *inode*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// int bpf_inode_storage_delete(struct bpf_map *map, void *inode)
// 	Description
// 		Delete a bpf_local_storage from an *inode*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// long bpf_d_path(const struct path *path, char *buf, u32 sz)
// 	Description
// 		Return full path for given **struct path** object, which
// 		needs to be the kernel BTF *path* object. The path is
// 		returned in the provided buffer *buf* of size *sz* and
// 		is zero terminated.
//
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_copy_from_user(void *dst, u32 size, const void *user_ptr)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* and store
// 		the data in *dst*. This is a wrapper of **copy_from_user**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_snprintf_btf(char *str, u32 str_size, struct btf_ptr *ptr, u32 btf_ptr_size, u64 flags)
// 	Description
// 		Use BTF to store a string representation of *ptr*->ptr in *str*,
// 		using *ptr*->type_id.  This value should specify the type
// 		that *ptr*->ptr points to. LLVM __builtin_btf_type_id(type, 1)
// 		can be used to look up vmlinux BTF type ids. Traversing the
// 		data structure using BTF, the type information and values are
// 		stored in the first *str_size* - 1 bytes of *str*.  Safe copy of
// 		the pointer data is carried out to avoid kernel crashes during
// 		operation.  Smaller types can use string space on the stack;
// 		larger programs can use map data to store the string
// 		representation.
//
// 		The string can be subsequently shared with userspace via
// 		bpf_perf_event_output() or ring buffer interfaces.
// 		bpf_trace_printk() is to be avoided as it places too small
// 		a limit on string size to be useful.
//
// 		*flags* is a combination of
//
// 		**BTF_F_COMPACT**
// 			no formatting around type information
// 		**BTF_F_NONAME**
// 			no struct/union member names/types
// 		**BTF_F_PTR_RAW**
// 			show raw (unobfuscated) pointer values;
// 			equivalent to printk specifier %px.
// 		**BTF_F_ZERO**
// 			show zero-valued struct/union members; they
// 			are not displayed by default
//
// 	Return
// 		The number of bytes that were written (or would have been
// 		written if output had to be truncated due to string size),
// 		or a negative error in cases of failure.
//
// long bpf_seq_printf_btf(struct seq_file *m, struct btf_ptr *ptr, u32 ptr_size, u64 flags)
// 	Description
// 		Use BTF to write to seq_write a string representation of
// 		*ptr*->ptr, using *ptr*->type_id as per bpf_snprintf_btf().
// 		*flags* are identical to those used for bpf_snprintf_btf.
// 	Return
// 		0 on success or a negative error in case of failure.
//
// u64 bpf_skb_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		See **bpf_get_cgroup_classid**\ () for the main description.
// 		This helper differs from **bpf_get_cgroup_classid**\ () in that
// 		the cgroup v1 net_cls class is retrieved only from the *skb*'s
// 		associated socket instead of the current process.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_redirect_neigh(u32 ifindex, struct bpf_redir_neigh *params, int plen, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*
// 		and fill in L2 addresses from neighboring subsystem. This helper
// 		is somewhat similar to **bpf_redirect**\ (), except that it
// 		populates L2 addresses as well, meaning, internally, the helper
// 		relies on the neighbor lookup for the L2 address of the nexthop.
//
// 		The helper will perform a FIB lookup based on the skb's
// 		networking header to get the address of the next hop, unless
// 		this is supplied by the caller in the *params* argument. The
// 		*plen* argument indicates the len of *params* and should be set
// 		to 0 if *params* is NULL.
//
// 		The *flags* argument is reserved and must be 0. The helper is
// 		currently only supported for tc BPF program types, and enabled
// 		for IPv4 and IPv6 protocols.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_per_cpu_ptr(const void *percpu_ptr, u32 cpu)
//     Description
//             Take a pointer to a percpu ksym, *percpu_ptr*, and return a
//             pointer to the percpu kernel variable on *cpu*. A ksym is an
//             extern variable decorated with '__ksym'. For ksym, there is a
//             global var (either static or global) defined of the same name
//             in the kernel. The ksym is percpu if the global var is percpu.
//             The returned pointer points to the global percpu var on *cpu*.
//
//             bpf_per_cpu_ptr() has the same semantic as per_cpu_ptr() in the
//             kernel, except that bpf_per_cpu_ptr() may return NULL. This
//             happens if *cpu* is larger than nr_cpu_ids. The caller of
//             bpf_per_cpu_ptr() must check the returned value.
//     Return
//             A pointer pointing to the kernel percpu variable on *cpu*, or
//             NULL, if *cpu* is invalid.
//
// void *bpf_this_cpu_ptr(const void *percpu_ptr)
// 	Description
// 		Take a pointer to a percpu ksym, *percpu_ptr*, and return a
// 		pointer to the percpu kernel variable on this cpu. See the
// 		description of 'ksym' in **bpf_per_cpu_ptr**\ ().
//
// 		bpf_this_cpu_ptr() has the same semantic as this_cpu_ptr() in
// 		the kernel. Different from **bpf_per_cpu_ptr**\ (), it would
// 		never return NULL.
// 	Return
// 		A pointer pointing to the kernel percpu variable on this cpu.
//
// long bpf_redirect_peer(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_redirect**\ (), except
// 		that the redirection happens to the *ifindex*' peer device. If
// 		*flags* is 0, the netns switch takes place from ingress to
// 		ingress without going through the CPU's backlog queue. If the
// 		**BPF_F_EGRESS** flag is provided then redirection happens in
// 		the egress direction of the peer device.
//
// 		*skb*\ **->mark** and *skb*\ **->tstamp** are not cleared during
// 		the netns switch.
//
// 		If the *flags* argument is 0, the helper is currently only
// 		supported for tc BPF program types at the ingress hook and for
// 		veth and netkit target device types. The peer device must reside
// 		in a different network namespace.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *task*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *task* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *task*) except this
// 		helper enforces the key must be a task_struct and the map must also
// 		be a **BPF_MAP_TYPE_TASK_STORAGE**.
//
// 		Underneath, the value is stored locally at *task* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *task*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_task_storage_delete(struct bpf_map *map, struct task_struct *task)
// 	Description
// 		Delete a bpf_local_storage from a *task*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// struct task_struct *bpf_get_current_task_btf(void)
// 	Description
// 		Return a BTF pointer to the "current" task.
// 		This pointer can also be used in helpers that accept an
// 		*ARG_PTR_TO_BTF_ID* of type *task_struct*.
// 	Return
// 		Pointer to the current task.
//
// long bpf_bprm_opts_set(struct linux_binprm *bprm, u64 flags)
// 	Description
// 		Set or clear certain options on *bprm*:
//
// 		**BPF_F_BPRM_SECUREEXEC** Set the secureexec bit
// 		which sets the **AT_SECURE** auxv for glibc. The bit
// 		is cleared if the flag is not specified.
// 	Return
// 		**-EINVAL** if invalid *flags* are passed, zero otherwise.
//
// u64 bpf_ktime_get_coarse_ns(void)
// 	Description
// 		Return a coarse-grained version of the time elapsed since
// 		system boot, in nanoseconds. Does not include time the system
// 		was suspended.
//
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC_COARSE**)
// 	Return
// 		Current *ktime*.
//
// long bpf_ima_inode_hash(struct inode *inode, void *dst, u32 size)
// 	Description
// 		Returns the stored IMA hash of the *inode* (if it's available).
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if IMA is disabled or **-EINVAL** if
// 		invalid arguments are passed.
//
// struct socket *bpf_sock_from_file(struct file *file)
// 	Description
// 		If the given file represents a socket, returns the associated
// 		socket.
// 	Return
// 		A pointer to a struct socket on success or NULL if the file is
// 		not a socket.
//
// long bpf_check_mtu(void *ctx, u32 ifindex, u32 *mtu_len, s32 len_diff, u64 flags)
// 	Description
// 		Check packet size against exceeding MTU of net device (based
// 		on *ifindex*).  This helper will likely be used in combination
// 		with helpers that adjust/change the packet size.
//
// 		The argument *len_diff* can be used for querying with a planned
// 		size change. This allows to check MTU prior to changing packet
// 		ctx. Providing a *len_diff* adjustment that is larger than the
// 		actual packet size (resulting in negative packet size) will in
// 		principle not exceed the MTU, which is why it is not considered
// 		a failure.  Other BPF helpers are needed for performing the
// 		planned size change; therefore the responsibility for catching
// 		a negative packet size belongs in those helpers.
//
// 		Specifying *ifindex* zero means the MTU check is performed
// 		against the current net device.  This is practical if this isn't
// 		used prior to redirect.
//
// 		On input *mtu_len* must be a valid pointer, else verifier will
// 		reject BPF program.  If the value *mtu_len* is initialized to
// 		zero then the ctx packet size is use.  When value *mtu_len* is
// 		provided as input this specify the L3 length that the MTU check
// 		is done against. Remember XDP and TC length operate at L2, but
// 		this value is L3 as this correlate to MTU and IP-header tot_len
// 		values which are L3 (similar behavior as bpf_fib_lookup).
//
// 		The Linux kernel route table can configure MTUs on a more
// 		specific per route level, which is not provided by this helper.
// 		For route level MTU checks use the **bpf_fib_lookup**\ ()
// 		helper.
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** for tc cls_act programs.
//
// 		The *flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_MTU_CHK_SEGS**
// 			This flag will only works for *ctx* **struct sk_buff**.
// 			If packet context contains extra packet segment buffers
// 			(often knows as GSO skb), then MTU check is harder to
// 			check at this point, because in transmit path it is
// 			possible for the skb packet to get re-segmented
// 			(depending on net device features).  This could still be
// 			a MTU violation, so this flag enables performing MTU
// 			check against segments, with a different violation
// 			return code to tell it apart. Check cannot use len_diff.
//
// 		On return *mtu_len* pointer contains the MTU value of the net
// 		device.  Remember the net device configured MTU is the L3 size,
// 		which is returned here and XDP and TC length operate at L2.
// 		Helper take this into account for you, but remember when using
// 		MTU value in your BPF-code.
//
// 	Return
// 		* 0 on success, and populate MTU value in *mtu_len* pointer.
//
// 		* < 0 if any input argument is invalid (*mtu_len* not updated)
//
// 		MTU violations return positive values, but also populate MTU
// 		value in *mtu_len* pointer, as this can be needed for
// 		implementing PMTU handing:
//
// 		* **BPF_MTU_CHK_RET_FRAG_NEEDED**
// 		* **BPF_MTU_CHK_RET_SEGS_TOOBIG**
//
// long bpf_for_each_map_elem(struct bpf_map *map, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For each element in **map**, call **callback_fn** function with
// 		**map**, **callback_ctx** and other map-specific parameters.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0.
//
// 		The following are a list of supported map types and their
// 		respective expected callback signatures:
//
// 		BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_HASH,
// 		BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
// 		BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY
//
// 		long (\*callback_fn)(struct bpf_map \*map, const void \*key, void \*value, void \*ctx);
//
// 		For per_cpu maps, the map_value is the value on the cpu where the
// 		bpf_prog is running.
//
// 		If **callback_fn** return 0, the helper will continue to the next
// 		element. If return value is 1, the helper will skip the rest of
// 		elements and return. Other return values are not used now.
//
// 	Return
// 		The number of traversed map elements for success, **-EINVAL** for
// 		invalid **flags**.
//
// long bpf_snprintf(char *str, u32 str_size, const char *fmt, u64 *data, u32 data_len)
// 	Description
// 		Outputs a string into the **str** buffer of size **str_size**
// 		based on a format string stored in a read-only map pointed by
// 		**fmt**.
//
// 		Each format specifier in **fmt** corresponds to one u64 element
// 		in the **data** array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data*
// 		array. The *data_len* is the size of *data* in bytes - must be
// 		a multiple of 8.
//
// 		Formats **%s** and **%p{i,I}{4,6}** require to read kernel
// 		memory. Reading kernel memory may fail due to either invalid
// 		address or valid address but requiring a major memory fault. If
// 		reading kernel memory fails, the string for **%s** will be an
// 		empty string, and the ip address for **%p{i,I}{4,6}** will be 0.
// 		Not returning error to bpf program is consistent with what
// 		**bpf_trace_printk**\ () does for now.
//
// 	Return
// 		The strictly positive length of the formatted string, including
// 		the trailing zero character. If the return value is greater than
// 		**str_size**, **str** contains a truncated string, guaranteed to
// 		be zero-terminated except when **str_size** is 0.
//
// 		Or **-EBUSY** if the per-CPU memory copy buffer is busy.
//
// long bpf_sys_bpf(u32 cmd, void *attr, u32 attr_size)
// 	Description
// 		Execute bpf syscall with given arguments.
// 	Return
// 		A syscall result.
//
// long bpf_btf_find_by_name_kind(char *name, int name_sz, u32 kind, int flags)
// 	Description
// 		Find BTF type with given name and kind in vmlinux BTF or in module's BTFs.
// 	Return
// 		Returns btf_id and btf_obj_fd in lower and upper 32 bits.
//
// long bpf_sys_close(u32 fd)
// 	Description
// 		Execute close syscall for given FD.
// 	Return
// 		A syscall result.
//
// long bpf_timer_init(struct bpf_timer *timer, struct bpf_map *map, u64 flags)
// 	Description
// 		Initialize the timer.
// 		First 4 bits of *flags* specify clockid.
// 		Only CLOCK_MONOTONIC, CLOCK_REALTIME, CLOCK_BOOTTIME are allowed.
// 		All other bits of *flags* are reserved.
// 		The verifier will reject the program if *timer* is not from
// 		the same *map*.
// 	Return
// 		0 on success.
// 		**-EBUSY** if *timer* is already initialized.
// 		**-EINVAL** if invalid *flags* are passed.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_set_callback(struct bpf_timer *timer, void *callback_fn)
// 	Description
// 		Configure the timer to call *callback_fn* static function.
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_start(struct bpf_timer *timer, u64 nsecs, u64 flags)
// 	Description
// 		Set timer expiration N nanoseconds from the current time. The
// 		configured callback will be invoked in soft irq context on some cpu
// 		and will not repeat unless another bpf_timer_start() is made.
// 		In such case the next invocation can migrate to a different cpu.
// 		Since struct bpf_timer is a field inside map element the map
// 		owns the timer. The bpf_timer_set_callback() will increment refcnt
// 		of BPF program to make sure that callback_fn code stays valid.
// 		When user space reference to a map reaches zero all timers
// 		in a map are cancelled and corresponding program's refcnts are
// 		decremented. This is done to make sure that Ctrl-C of a user
// 		process doesn't leave any timers running. If map is pinned in
// 		bpffs the callback_fn can re-arm itself indefinitely.
// 		bpf_map_update/delete_elem() helpers and user space sys_bpf commands
// 		cancel and free the timer in the given map element.
// 		The map can contain timers that invoke callback_fn-s from different
// 		programs. The same callback_fn can serve different timers from
// 		different maps if key/value layout matches across maps.
// 		Every bpf_timer_set_callback() can have different callback_fn.
//
// 		*flags* can be one of:
//
// 		**BPF_F_TIMER_ABS**
// 			Start the timer in absolute expire value instead of the
// 			default relative one.
// 		**BPF_F_TIMER_CPU_PIN**
// 			Timer will be pinned to the CPU of the caller.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier
// 		or invalid *flags* are passed.
//
// long bpf_timer_cancel(struct bpf_timer *timer)
// 	Description
// 		Cancel the timer and wait for callback_fn to finish if it was running.
// 	Return
// 		0 if the timer was not active.
// 		1 if the timer was active.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EDEADLK** if callback_fn tried to call bpf_timer_cancel() on its
// 		own timer which would have led to a deadlock otherwise.
//
// u64 bpf_get_func_ip(void *ctx)
// 	Description
// 		Get address of the traced function (for tracing and kprobe programs).
//
// 		When called for kprobe program attached as uprobe it returns
// 		probe address for both entry and return uprobe.
//
// 	Return
// 		Address of the traced function for kprobe.
// 		0 for kprobes placed within the function (not at the entry).
// 		Address of the probe for uprobe and return uprobe.
//
// u64 bpf_get_attach_cookie(void *ctx)
// 	Description
// 		Get bpf_cookie value provided (optionally) during the program
// 		attachment. It might be different for each individual
// 		attachment, even if BPF program itself is the same.
// 		Expects BPF program context *ctx* as a first argument.
//
// 		Supported for the following program types:
// 			- kprobe/uprobe;
// 			- tracepoint;
// 			- perf_event.
// 	Return
// 		Value specified by user at BPF link creation/attachment time
// 		or 0, if it was not specified.
//
// long bpf_task_pt_regs(struct task_struct *task)
// 	Description
// 		Get the struct pt_regs associated with **task**.
// 	Return
// 		A pointer to struct pt_regs.
//
// long bpf_get_branch_snapshot(void *entries, u32 size, u64 flags)
// 	Description
// 		Get branch trace from hardware engines like Intel LBR. The
// 		hardware engine is stopped shortly after the helper is
// 		called. Therefore, the user need to filter branch entries
// 		based on the actual use case. To capture branch trace
// 		before the trigger point of the BPF program, the helper
// 		should be called at the beginning of the BPF program.
//
// 		The data is stored as struct perf_branch_entry into output
// 		buffer *entries*. *size* is the size of *entries* in bytes.
// 		*flags* is reserved for now and must be zero.
//
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_trace_vprintk(const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		Behaves like **bpf_trace_printk**\ () helper, but takes an array of u64
// 		to format and can handle more format args as a result.
//
// 		Arguments are to be used as in **bpf_seq_printf**\ () helper.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// struct unix_sock *bpf_skc_to_unix_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *unix_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_kallsyms_lookup_name(const char *name, int name_sz, int flags, u64 *res)
// 	Description
// 		Get the address of a kernel symbol, returned in *res*. *res* is
// 		set to 0 if the symbol is not found.
// 	Return
// 		On success, zero. On error, a negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-EINVAL** if string *name* is not the same size as *name_sz*.
//
// 		**-ENOENT** if symbol is not found.
//
// 		**-EPERM** if caller does not have permission to obtain kernel address.
//
// long bpf_find_vma(struct task_struct *task, u64 addr, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		Find vma of *task* that contains *addr*, call *callback_fn*
// 		function with *task*, *vma*, and *callback_ctx*.
// 		The *callback_fn* should be a static function and
// 		the *callback_ctx* should be a pointer to the stack.
// 		The *flags* is used to control certain aspects of the helper.
// 		Currently, the *flags* must be 0.
//
// 		The expected callback signature is
//
// 		long (\*callback_fn)(struct task_struct \*task, struct vm_area_struct \*vma, void \*callback_ctx);
//
// 	Return
// 		0 on success.
// 		**-ENOENT** if *task->mm* is NULL, or no vma contains *addr*.
// 		**-EBUSY** if failed to try lock mmap_lock.
// 		**-EINVAL** for invalid **flags**.
//
// long bpf_loop(u32 nr_loops, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For **nr_loops**, call **callback_fn** function
// 		with **callback_ctx** as the context parameter.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0. Currently, nr_loops is
// 		limited to 1 << 23 (~8 million) loops.
//
// 		long (\*callback_fn)(u64 index, void \*ctx);
//
// 		where **index** is the current index in the loop. The index
// 		is zero-indexed.
//
// 		If **callback_fn** returns 0, the helper will continue to the next
// 		loop. If return value is 1, the helper will skip the rest of
// 		the loops and return. Other return values are not used now,
// 		and will be rejected by the verifier.
//
// 	Return
// 		The number of loops performed, **-EINVAL** for invalid **flags**,
// 		**-E2BIG** if **nr_loops** exceeds the maximum number of loops.
//
// long bpf_strncmp(const char *s1, u32 s1_sz, const char *s2)
// 	Description
// 		Do strncmp() between **s1** and **s2**. **s1** doesn't need
// 		to be null-terminated and **s1_sz** is the maximum storage
// 		size of **s1**. **s2** must be a read-only string.
// 	Return
// 		An integer less than, equal to, or greater than zero
// 		if the first **s1_sz** bytes of **s1** is found to be
// 		less than, to match, or be greater than **s2**.
//
// long bpf_get_func_arg(void *ctx, u32 n, u64 *value)
// 	Description
// 		Get **n**-th argument register (zero based) of the traced function (for tracing programs)
// 		returned in **value**.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if n >= argument register count of traced function.
//
// long bpf_get_func_ret(void *ctx, u64 *value)
// 	Description
// 		Get return value of the traced function (for tracing programs)
// 		in **value**.
//
// 	Return
// 		0 on success.
// 		**-EOPNOTSUPP** for tracing programs other than BPF_TRACE_FEXIT or BPF_MODIFY_RETURN.
//
// long bpf_get_func_arg_cnt(void *ctx)
// 	Description
// 		Get number of registers of the traced function (for tracing programs) where
// 		function arguments are stored in these registers.
//
// 	Return
// 		The number of argument registers of the traced function.
//
// int bpf_get_retval(void)
// 	Description
// 		Get the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
// 	Return
// 		The BPF program's return value.
//
// int bpf_set_retval(int retval)
// 	Description
// 		Set the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
//
// 		Note that there is the following corner case where the program exports an error
// 		via bpf_set_retval but signals success via 'return 1':
//
// 			bpf_set_retval(-EPERM);
// 			return 1;
//
// 		In this case, the BPF program's return value will use helper's -EPERM. This
// 		still holds true for cgroup/bind{4,6} which supports extra 'return 3' success case.
//
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_xdp_get_buff_len(struct xdp_buff *xdp_md)
// 	Description
// 		Get the total size of a given xdp buff (linear and paged area)
// 	Return
// 		The total size of a given xdp buffer.
//
// long bpf_xdp_load_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		This helper is provided as an easy way to load data from a
// 		xdp buffer. It can be used to load *len* bytes from *offset* from
// 		the frame associated to *xdp_md*, into the buffer pointed by
// 		*buf*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_store_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		Store *len* bytes from buffer *buf* into the frame
// 		associated to *xdp_md*, at *offset*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_copy_from_user_task(void *dst, u32 size, const void *user_ptr, struct task_struct *tsk, u64 flags)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* in *tsk*'s
// 		address space, and stores the data in *dst*. *flags* is not
// 		used yet and is provided for future extensibility. This helper
// 		can only be used by sleepable programs.
// 	Return
// 		0 on success, or a negative error in case of failure. On error
// 		*dst* buffer is zeroed out.
//
// long bpf_skb_set_tstamp(struct sk_buff *skb, u64 tstamp, u32 tstamp_type)
// 	Description
// 		Change the __sk_buff->tstamp_type to *tstamp_type*
// 		and set *tstamp* to the __sk_buff->tstamp together.
//
// 		If there is no need to change the __sk_buff->tstamp_type,
// 		the tstamp value can be directly written to __sk_buff->tstamp
// 		instead.
//
// 		BPF_SKB_TSTAMP_DELIVERY_MONO is the only tstamp that
// 		will be kept during bpf_redirect_*().  A non zero
// 		*tstamp* must be used with the BPF_SKB_TSTAMP_DELIVERY_MONO
// 		*tstamp_type*.
//
// 		A BPF_SKB_TSTAMP_UNSPEC *tstamp_type* can only be used
// 		with a zero *tstamp*.
//
// 		Only IPv4 and IPv6 skb->protocol are supported.
//
// 		This function is most useful when it needs to set a
// 		mono delivery time to __sk_buff->tstamp and then
// 		bpf_redirect_*() to the egress of an iface.  For example,
// 		changing the (rcv) timestamp in __sk_buff->tstamp at
// 		ingress to a mono delivery time and then bpf_redirect_*()
// 		to sch_fq@phy-dev.
// 	Return
// 		0 on success.
// 		**-EINVAL** for invalid input
// 		**-EOPNOTSUPP** for unsupported protocol
//
// long bpf_ima_file_hash(struct file *file, void *dst, u32 size)
// 	Description
// 		Returns a calculated IMA hash of the *file*.
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if the hash calculation failed or **-EINVAL** if
// 		invalid arguments are passed.
//
// void *bpf_kptr_xchg(void *dst, void *ptr)
// 	Description
// 		Exchange kptr at pointer *dst* with *ptr*, and return the old value.
// 		*dst* can be map value or local kptr. *ptr* can be NULL, otherwise
// 		it must be a referenced pointer which will be released when this helper
// 		is called.
// 	Return
// 		The old value of kptr (which can be NULL). The returned pointer
// 		if not NULL, is a reference which must be released using its
// 		corresponding release function, or moved into a BPF map before
// 		program exit.
//
// void *bpf_map_lookup_percpu_elem(struct bpf_map *map, const void *key, u32 cpu)
// 	Description
// 		Perform a lookup in *percpu map* for an entry associated to
// 		*key* on *cpu*.
// 	Return
// 		Map value associated to *key* on *cpu*, or **NULL** if no entry
// 		was found or *cpu* is invalid.
//
// struct mptcp_sock *bpf_skc_to_mptcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *mptcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_dynptr_from_mem(void *data, u64 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Get a dynptr to local memory *data*.
//
// 		*data* must be a ptr to a map value.
// 		The maximum *size* supported is DYNPTR_MAX_SIZE.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if the size exceeds DYNPTR_MAX_SIZE,
// 		-EINVAL if flags is not 0.
//
// long bpf_ringbuf_reserve_dynptr(void *ringbuf, u32 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*
// 		through the dynptr interface. *flags* must be 0.
//
// 		Please note that a corresponding bpf_ringbuf_submit_dynptr or
// 		bpf_ringbuf_discard_dynptr must be called on *ptr*, even if the
// 		reservation fails. This is enforced by the verifier.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void bpf_ringbuf_submit_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*,
// 		through the dynptr interface. This is a no-op if the dynptr is
// 		invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_submit'.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample through the dynptr
// 		interface. This is a no-op if the dynptr is invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_discard'.
// 	Return
// 		Nothing. Always succeeds.
//
// long bpf_dynptr_read(void *dst, u64 len, const struct bpf_dynptr *src, u64 offset, u64 flags)
// 	Description
// 		Read *len* bytes from *src* into *dst*, starting from *offset*
// 		into *src*.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *src*'s data, -EINVAL if *src* is an invalid dynptr or if
// 		*flags* is not 0.
//
// long bpf_dynptr_write(const struct bpf_dynptr *dst, u64 offset, void *src, u64 len, u64 flags)
// 	Description
// 		Write *len* bytes from *src* into *dst*, starting from *offset*
// 		into *dst*.
//
// 		*flags* must be 0 except for skb-type dynptrs.
//
// 		For skb-type dynptrs:
// 		    *  All data slices of the dynptr are automatically
// 		       invalidated after **bpf_dynptr_write**\ (). This is
// 		       because writing may pull the skb and change the
// 		       underlying packet buffer.
//
// 		    *  For *flags*, please see the flags accepted by
// 		       **bpf_skb_store_bytes**\ ().
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *dst*'s data, -EINVAL if *dst* is an invalid dynptr or if *dst*
// 		is a read-only dynptr or if *flags* is not correct. For skb-type dynptrs,
// 		other errors correspond to errors returned by **bpf_skb_store_bytes**\ ().
//
// void *bpf_dynptr_data(const struct bpf_dynptr *ptr, u64 offset, u64 len)
// 	Description
// 		Get a pointer to the underlying dynptr data.
//
// 		*len* must be a statically known value. The returned data slice
// 		is invalidated whenever the dynptr is invalidated.
//
// 		skb and xdp type dynptrs may not use bpf_dynptr_data. They should
// 		instead use bpf_dynptr_slice and bpf_dynptr_slice_rdwr.
// 	Return
// 		Pointer to the underlying dynptr data, NULL if the dynptr is
// 		read-only, if the dynptr is invalid, or if the offset and length
// 		is out of bounds.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv4/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv6/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// long bpf_tcp_raw_check_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// long bpf_tcp_raw_check_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// u64 bpf_ktime_get_tai_ns(void)
// 	Description
// 		A nonsettable system-wide clock derived from wall-clock time but
// 		ignoring leap seconds.  This clock does not experience
// 		discontinuities and backwards jumps caused by NTP inserting leap
// 		seconds as CLOCK_REALTIME does.
//
// 		See: **clock_gettime**\ (**CLOCK_TAI**)
// 	Return
// 		Current *ktime*.
//
// long bpf_user_ringbuf_drain(struct bpf_map *map, void *callback_fn, void *ctx, u64 flags)
// 	Description
// 		Drain samples from the specified user ring buffer, and invoke
// 		the provided callback for each such sample:
//
// 		long (\*callback_fn)(const struct bpf_dynptr \*dynptr, void \*ctx);
//
// 		If **callback_fn** returns 0, the helper will continue to try
// 		and drain the next sample, up to a maximum of
// 		BPF_MAX_USER_RINGBUF_SAMPLES samples. If the return value is 1,
// 		the helper will skip the rest of the samples and return. Other
// 		return values are not used now, and will be rejected by the
// 		verifier.
// 	Return
// 		The number of drained samples if no error was encountered while
// 		draining samples, or 0 if no samples were present in the ring
// 		buffer. If a user-space producer was epoll-waiting on this map,
// 		and at least one sample was drained, they will receive an event
// 		notification notifying them of available space in the ring
// 		buffer. If the BPF_RB_NO_WAKEUP flag is passed to this
// 		function, no wakeup notification will be sent. If the
// 		BPF_RB_FORCE_WAKEUP flag is passed, a wakeup notification will
// 		be sent even if no sample was drained.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EBUSY** if the ring buffer is contended, and another calling
// 		context was concurrently draining the ring buffer.
//
// 		**-EINVAL** if user-space is not properly tracking the ring
// 		buffer due to the producer position not being aligned to 8
// 		bytes, a sample not being aligned to 8 bytes, or the producer
// 		position not matching the advertised length of a sample.
//
// 		**-E2BIG** if user-space has tried to publish a sample which is
// 		larger than the size of the ring buffer, or which cannot fit
// 		within a struct bpf_dynptr.
//
// void *bpf_cgrp_storage_get(struct bpf_map *map, struct cgroup *cgroup, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *cgroup*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *cgroup* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *cgroup*) except this
// 		helper enforces the key must be a cgroup struct and the map must also
// 		be a **BPF_MAP_TYPE_CGRP_STORAGE**.
//
// 		In reality, the local-storage value is embedded directly inside of the
// 		*cgroup* object itself, rather than being located in the
// 		**BPF_MAP_TYPE_CGRP_STORAGE** map. When the local-storage value is
// 		queried for some *map* on a *cgroup* object, the kernel will perform an
// 		O(n) iteration over all of the live local-storage values for that
// 		*cgroup* object until the local-storage value for the *map* is found.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_cgrp_storage_delete(struct bpf_map *map, struct cgroup *cgroup)
// 	Description
// 		Delete a bpf_local_storage from a *cgroup*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
// This helper list is effectively frozen. If you are trying to		 * add a new helper, you should add a kfunc instead which has		 * less stability guarantees. See Documentation/bpf/kfuncs.rst
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_info {
    r#type: u32,
    id: u32,
    prog_id: u32,
    __slate_anon_3: __SlateRecord_bpf_h_6842,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info {
    insn_off: u32,
    type_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netpoll_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_line_info {
    insn_off: u32,
    file_name_off: u32,
    line_off: u32,
    line_col: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_hwtstamp_config {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ip_tunnel_parm_kern {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_config {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_name_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sfp_bus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct udp_tunnel_nic_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct udp_tunnel_nic {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xdp_buff {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xdp_metadata_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_type {
    name_off: u32,
    info: u32,
    __slate_anon_2: __SlateRecord_btf_h_64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_link_topology {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hwtstamp_provider {
}


// BTF_KIND_INT is followed by a u32 and the following
// is the 32 bits arrangement:
// Attributes stored in the BTF_INT_ENCODING
#[repr(C)]
#[derive(Clone, Copy)]
struct btf_member {
    name_off: u32,
    r#type: u32,
    offset: u32,
}


// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_194 {
    rx_packets: u64,
    __rx_packets: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_195 {
    tx_packets: u64,
    __tx_packets: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_196 {
    rx_bytes: u64,
    __rx_bytes: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_197 {
    tx_bytes: u64,
    __tx_bytes: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_198 {
    rx_errors: u64,
    __rx_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_199 {
    tx_errors: u64,
    __tx_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_200 {
    rx_dropped: u64,
    __rx_dropped: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_201 {
    tx_dropped: u64,
    __tx_dropped: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_202 {
    multicast: u64,
    __multicast: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_203 {
    collisions: u64,
    __collisions: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_204 {
    rx_length_errors: u64,
    __rx_length_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_205 {
    rx_over_errors: u64,
    __rx_over_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_206 {
    rx_crc_errors: u64,
    __rx_crc_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_207 {
    rx_frame_errors: u64,
    __rx_frame_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_208 {
    rx_fifo_errors: u64,
    __rx_fifo_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_209 {
    rx_missed_errors: u64,
    __rx_missed_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_210 {
    tx_aborted_errors: u64,
    __tx_aborted_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_211 {
    tx_carrier_errors: u64,
    __tx_carrier_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_212 {
    tx_fifo_errors: u64,
    __tx_fifo_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_213 {
    tx_heartbeat_errors: u64,
    __tx_heartbeat_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_214 {
    tx_window_errors: u64,
    __tx_window_errors: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_215 {
    rx_compressed: u64,
    __rx_compressed: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_216 {
    tx_compressed: u64,
    __tx_compressed: __SlateRecord_types_h_197,
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
struct neigh_parms {
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
    __slate_bits_0: __slate_bits::__SlateBits553U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
}


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct Qdisc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_seq_info {
    seq_ops: *const seq_operations,
    init_seq_private: Option<unsafe extern "C" fn(*mut (), *mut bpf_iter_aux_info) -> i32>,
    fini_seq_private: Option<unsafe extern "C" fn(*mut ())>,
    seq_priv_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_ops {
    map_alloc_check: Option<unsafe extern "C" fn(*mut bpf_attr) -> i32>,
    map_alloc: Option<unsafe extern "C" fn(*mut bpf_attr) -> *mut bpf_map>,
    map_release: Option<unsafe extern "C" fn(*mut bpf_map, *mut file)>,
    map_free: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_get_next_key: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut ()) -> i32>,
    map_release_uref: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_lookup_elem_sys_only: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_lookup_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_and_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i32>,
    map_lookup_and_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_update_batch: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_update_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i64>,
    map_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_push_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u64) -> i64>,
    map_pop_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_peek_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_lookup_percpu_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u32) -> *mut ()>,
    map_get_hash: Option<unsafe extern "C" fn(*mut bpf_map) -> i32>,
    map_fd_get_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, i32) -> *mut ()>,
    map_fd_put_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), bool)>,
    map_gen_lookup: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_insn) -> i32>,
    map_fd_sys_lookup_elem: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    map_seq_show_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut seq_file)>,
    map_check_btf: Option<unsafe extern "C" fn(*mut bpf_map, *const btf, *const btf_type, *const btf_type) -> i32>,
    map_poke_track: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux) -> i32>,
    map_poke_untrack: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux)>,
    map_poke_run: Option<unsafe extern "C" fn(*mut bpf_map, u32, *mut bpf_prog, *mut bpf_prog)>,
    map_direct_value_addr: Option<unsafe extern "C" fn(*const bpf_map, *mut u64, u32) -> i32>,
    map_direct_value_meta: Option<unsafe extern "C" fn(*const bpf_map, u64, *mut u32) -> i32>,
    map_mmap: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_area_struct) -> i32>,
    map_mmap_fault: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_fault) -> u32>,
    map_poll: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *mut poll_table_struct) -> u32>,
    map_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    map_local_storage_charge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32) -> i32>,
    map_local_storage_uncharge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32)>,
    map_owner_storage_ptr: Option<unsafe extern "C" fn(*mut ()) -> *mut *mut bpf_local_storage>,
    map_redirect: Option<unsafe extern "C" fn(*mut bpf_map, u64, u64) -> i64>,
    map_meta_equal: Option<unsafe extern "C" fn(*const bpf_map, *const bpf_map) -> bool>,
    map_set_for_each_callback_args: Option<unsafe extern "C" fn(*mut bpf_verifier_env, *mut bpf_func_state, *mut bpf_func_state) -> i32>,
    map_for_each_callback: Option<unsafe extern "C" fn(*mut bpf_map, Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>, *mut (), u64) -> i64>,
    map_mem_usage: Option<unsafe extern "C" fn(*const bpf_map) -> u64>,
    map_btf_id: *mut i32,
    iter_seq_info: *const bpf_iter_seq_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_kptr {
    btf: *mut btf,
    module: *mut module,
    dtor: Option<unsafe extern "C" fn(*mut ())>,
    btf_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_835 {
    __slate_anon_0: __SlateRecord_netdevice_h_836,
    combined: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_836 {
    count: u16,
    offset: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_graph_root {
    btf: *mut btf,
    value_btf_id: u32,
    node_offset: u32,
    value_rec: *mut btf_record,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field {
    offset: u32,
    size: u32,
    r#type: u32,
    __slate_anon_3: __SlateRecord_bpf_h_256,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_896 {
    encap: __SlateRecord_netdevice_h_897,
    tun: __SlateRecord_netdevice_h_902,
    bridge: __SlateRecord_netdevice_h_915,
    dsa: __SlateRecord_netdevice_h_925,
    mtk_wdma: __SlateRecord_netdevice_h_929,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_897 {
    id: u16,
    proto: u16,
    h_dest: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_902 {
    dst: *mut dst_entry,
    __slate_anon_1: __SlateRecord_netdevice_h_904,
    __slate_anon_2: __SlateRecord_netdevice_h_908,
    inner_proto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_904 {
    src_v4: in_addr,
    src_v6: in6_addr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_908 {
    dst_v4: in_addr,
    dst_v6: in6_addr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_915 {
    vlan_mode: u32,
    vlan_id: u16,
    vlan_proto: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_925 {
    port: i32,
    proto: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_929 {
    wdma_idx: u8,
    queue: u8,
    wcid: u16,
    bss: u8,
    amsdu: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_record {
    cnt: u32,
    field_mask: u32,
    spin_lock_off: i32,
    res_spin_lock_off: i32,
    timer_off: i32,
    wq_off: i32,
    refcount_off: i32,
    task_work_off: i32,
    fields: [btf_field; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_953 {
    id: u16,
    proto: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xdp_dev_bulk_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_xdp_link {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_owner {
    r#type: u32,
    jited: bool,
    xdp_has_frags: bool,
    sleepable: bool,
    storage_cookie: [u64; 2],
    attach_func_proto: *const btf_type,
    expected_attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map {
    sha: [u8; 32],
    ops: *const bpf_map_ops,
    inner_map_meta: *mut bpf_map,
    security: *mut (),
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_extra: u64,
    map_flags: u32,
    id: u32,
    record: *mut btf_record,
    numa_node: i32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    btf: *mut btf,
    name: [u8; 16],
    freeze_mutex: mutex,
    refcnt: __SlateRecord_types_h_197,
    usercnt: __SlateRecord_types_h_197,
    __slate_anon_21: __SlateRecord_bpf_h_329,
    writecnt: __SlateRecord_types_h_197,
    owner_lock: spinlock,
    owner: *mut bpf_map_owner,
    bypass_spec_v1: bool,
    frozen: bool,
    free_after_mult_rcu_gp: bool,
    free_after_rcu_gp: bool,
    sleepable_refcnt: __SlateRecord_types_h_197,
    elem_count: *mut i64,
    cookie: u64,
    excl_prog_sha: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_1024 {
    __slate_anon_0: __SlateRecord_netdevice_h_1026,
    __slate_anon_1: __SlateRecord_netdevice_h_1032,
    xsk: __SlateRecord_netdevice_h_1036,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_1026 {
    flags: u32,
    prog: *mut bpf_prog,
    extack: *mut netlink_ext_ack,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_1032 {
    offmap: *mut bpf_offloaded_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_1036 {
    pool: *mut xsk_buff_pool,
    queue_id: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xsk_buff_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_dev_ops {
    map_get_next_key: Option<unsafe extern "C" fn(*mut bpf_offloaded_map, *mut (), *mut ()) -> i32>,
    map_lookup_elem: Option<unsafe extern "C" fn(*mut bpf_offloaded_map, *mut (), *mut ()) -> i32>,
    map_update_elem: Option<unsafe extern "C" fn(*mut bpf_offloaded_map, *mut (), *mut (), u64) -> i32>,
    map_delete_elem: Option<unsafe extern "C" fn(*mut bpf_offloaded_map, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_offloaded_map {
    map: bpf_map,
    netdev: *mut net_device,
    dev_ops: *const bpf_map_dev_ops,
    dev_priv: *mut (),
    offloads: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_2169 {
    __slate_anon_0: __SlateRecord_netdevice_h_2169_2,
    priv_flags_fast: __SlateRecord_netdevice_h_2169_3,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_2169_2 {
    __slate_bits_0: __slate_bits::__SlateBits1269U0,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_2169_3 {
    __slate_bits_0: __slate_bits::__SlateBits1270U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_mprog_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_netdevice_h_2204 {
    lstats: *mut pcpu_lstats,
    tstats: *mut pcpu_sw_netstats,
    dstats: *mut pcpu_dstats,
}


// bpf_type_flag contains a set of flags that are applicable to the values of
// arg_type, ret_type and reg_type. For example, a pointer value may be null,
// or a memory is read-only. We classify types into two categories: base types
// and extended types. Extended types are base types combined with a type flag.
//
// Currently there are no more than 32 base types in arg_type, ret_type and
// reg_types.
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_proto {
    func: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    gpl_only: bool,
    pkt_access: bool,
    might_sleep: bool,
    allow_fastcall: bool,
    ret_type: u32,
    __slate_anon_6: __SlateRecord_bpf_h_996,
    __slate_anon_7: __SlateRecord_bpf_h_1006,
    ret_btf_id: *mut i32,
    allowed: Option<unsafe extern "C" fn(*const bpf_prog) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn_access_aux {
    reg_type: u32,
    is_ldsx: bool,
    __slate_anon_2: __SlateRecord_bpf_h_1109,
    log: *mut bpf_verifier_log,
    is_retval: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_ops {
    test_run: Option<unsafe extern "C" fn(*mut bpf_prog, *const bpf_attr, *mut bpf_attr) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inet6_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_rx_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_netdevice_h_2264 {
    upper: list_head,
    lower: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xsk_tx_metadata_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ndisc_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct in_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_rmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rtnl_link_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_stat_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_queue_mgmt_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct devlink_port {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dim_irq_moder {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_ops {
    get_func_proto: Option<unsafe extern "C" fn(u32, *const bpf_prog) -> *const bpf_func_proto>,
    is_valid_access: Option<unsafe extern "C" fn(i32, i32, u32, *const bpf_prog, *mut bpf_insn_access_aux) -> bool>,
    gen_prologue: Option<unsafe extern "C" fn(*mut bpf_insn, bool, *const bpf_prog) -> i32>,
    gen_epilogue: Option<unsafe extern "C" fn(*mut bpf_insn, *const bpf_prog, i16) -> i32>,
    gen_ld_abs: Option<unsafe extern "C" fn(*const bpf_insn, *mut bpf_insn) -> i32>,
    convert_ctx_access: Option<unsafe extern "C" fn(u32, *const bpf_insn, *mut bpf_insn, *mut bpf_prog, *mut u32) -> u32>,
    btf_struct_access: Option<unsafe extern "C" fn(*mut bpf_verifier_log, *const bpf_reg_state, i32, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_offload {
    prog: *mut bpf_prog,
    netdev: *mut net_device,
    offdev: *mut bpf_offload_dev,
    dev_priv: *mut (),
    offloads: list_head,
    dev_state: bool,
    opt_failed: bool,
    jited_image: *mut (),
    jited_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_func_model {
    ret_size: u8,
    ret_flags: u8,
    nr_args: u8,
    arg_size: [u8; 12],
    arg_flags: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ksym {
    start: u64,
    end: u64,
    name: [u8; 512],
    lnode: list_head,
    tnode: latch_tree_node,
    prog: bool,
    fp_start: u32,
    fp_end: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_skip {
    prog: *mut bpf_prog,
    nop: *mut (),
    target: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_image {
    image: *mut (),
    size: i32,
    ksym: bpf_ksym,
    pcref: percpu_ref,
    call_orig: bool,
    tr: *mut bpf_trampoline,
    list: list_head,
    nr_skips: i32,
    skips: *mut bpf_tramp_skip,
    __slate_anon_9: __SlateRecord_bpf_h_1393,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_h_928 {
    required_octet: u8,
    __slate_anon_1: __SlateRecord_ieee80211_h_930,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_930 {
    __empty_virtual_map: __SlateRecord_ieee80211_h_930_2,
    virtual_map: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_930_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_trampoline {
    hlist_key: hlist_node,
    hlist_ip: hlist_node,
    fops: *mut ftrace_ops,
    refcnt: refcount_struct,
    flags: u32,
    key: u64,
    ip: u64,
    func: __SlateRecord_bpf_h_1423,
    extension_prog: *mut bpf_prog,
    progs_hlist: [hlist_head; 3],
    progs_cnt: [i32; 3],
    cur_image: *mut bpf_tramp_image,
    images: list_head,
    multi_attach: __SlateRecord_bpf_h_1442,
}


// Fits in 2048B
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info_aux {
    linkage: u16,
    unreliable: bool,
    called: [bool; 2],
    verified: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_jit_poke_descriptor {
    tailcall_target: *mut (),
    tailcall_bypass: *mut (),
    bypass_addr: *mut (),
    aux: *mut (),
    __slate_anon_4: __SlateRecord_bpf_h_1705,
    tailcall_target_stable: bool,
    adj_off: u8,
    reason: u16,
    insn_idx: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_h_999 {
    auth: __SlateRecord_ieee80211_h_1000,
    deauth: __SlateRecord_ieee80211_h_1007,
    assoc_req: __SlateRecord_ieee80211_h_1010,
    assoc_resp: __SlateRecord_ieee80211_h_1016,
    reassoc_resp: __SlateRecord_ieee80211_h_1016,
    s1g_assoc_resp: __SlateRecord_ieee80211_h_1023,
    s1g_reassoc_resp: __SlateRecord_ieee80211_h_1023,
    reassoc_req: __SlateRecord_ieee80211_h_1028,
    disassoc: __SlateRecord_ieee80211_h_1035,
    beacon: __SlateRecord_ieee80211_h_1038,
    probe_req: __SlateRecord_ieee80211_h_1046,
    probe_resp: __SlateRecord_ieee80211_h_1050,
    action: __SlateRecord_ieee80211_h_1058,
    __slate_anon_13: __SlateRecord_ieee80211_h_1206,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1000 {
    auth_alg: u16,
    auth_transaction: u16,
    status_code: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1007 {
    reason_code: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1010 {
    capab_info: u16,
    listen_interval: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1016 {
    capab_info: u16,
    status_code: u16,
    aid: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1023 {
    capab_info: u16,
    status_code: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1028 {
    capab_info: u16,
    listen_interval: u16,
    current_ap: [u8; 6],
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1035 {
    reason_code: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1038 {
    timestamp: u64,
    beacon_int: u16,
    capab_info: u16,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1046 {
    __slate_anon_0: __SlateRecord_ieee80211_h_1048,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1048 {
    __empty_variable: __SlateRecord_ieee80211_h_1048_2,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1048_2 {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1050 {
    timestamp: u64,
    beacon_int: u16,
    capab_info: u16,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1058 {
    category: u8,
    action_code: u8,
    __slate_anon_2: __SlateRecord_ieee80211_h_1061,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_h_1061 {
    wme_action: __SlateRecord_ieee80211_h_1062,
    chan_switch: __SlateRecord_ieee80211_h_1067,
    ext_chan_switch: __SlateRecord_ieee80211_h_1071,
    measurement: __SlateRecord_ieee80211_h_1075,
    addba_req: __SlateRecord_ieee80211_h_1081,
    addba_resp: __SlateRecord_ieee80211_h_1089,
    delba: __SlateRecord_ieee80211_h_1097,
    self_prot: __SlateRecord_ieee80211_h_1101,
    mesh_action: __SlateRecord_ieee80211_h_1105,
    sa_query: __SlateRecord_ieee80211_h_1109,
    ht_smps: __SlateRecord_ieee80211_h_1112,
    ht_notify_cw: __SlateRecord_ieee80211_h_1115,
    tdls_discover_resp: __SlateRecord_ieee80211_h_1118,
    vht_opmode_notif: __SlateRecord_ieee80211_h_1123,
    vht_group_notif: __SlateRecord_ieee80211_h_1126,
    tpc_report: __SlateRecord_ieee80211_h_1130,
    ftm: __SlateRecord_ieee80211_h_1136,
    s1g: __SlateRecord_ieee80211_h_1145,
    wnm_timing_msr: __SlateRecord_ieee80211_h_1149,
    ttlm_req: __SlateRecord_ieee80211_h_1157,
    ttlm_res: __SlateRecord_ieee80211_h_1161,
    ttlm_tear_down: __SlateRecord_ieee80211_h_1166,
    ml_reconf_req: __SlateRecord_ieee80211_h_1170,
    ml_reconf_resp: __SlateRecord_ieee80211_h_1174,
    epcs: __SlateRecord_ieee80211_h_1179,
    eml_omn: __SlateRecord_ieee80211_h_1183,
    uhr_link_reconf_req: __SlateRecord_ieee80211_h_1188,
    uhr_link_reconf_resp: __SlateRecord_ieee80211_h_1193,
    uhr_link_reconf_notif: __SlateRecord_ieee80211_h_1199,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1062 {
    dialog_token: u8,
    status_code: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1067 {
    no_fixed_fields: [u8; 0],
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1071 {
    data: ieee80211_ext_chansw_ie,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1075 {
    dialog_token: u8,
    element_id: u8,
    length: u8,
    msr_elem: ieee80211_msrment_ie,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1081 {
    dialog_token: u8,
    capab: u16,
    timeout: u16,
    start_seq_num: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1089 {
    dialog_token: u8,
    status: u16,
    capab: u16,
    timeout: u16,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1097 {
    params: u16,
    reason_code: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1101 {
    no_fixed_fields: [u8; 0],
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1105 {
    no_fixed_fields: [u8; 0],
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1109 {
    trans_id: [u8; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1112 {
    smps_control: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1115 {
    chanwidth: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1118 {
    dialog_token: u8,
    capability: u16,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1123 {
    operating_mode: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1126 {
    membership: [u8; 8],
    position: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1130 {
    dialog_token: u8,
    tpc_elem_id: u8,
    tpc_elem_length: u8,
    tpc: ieee80211_tpc_report_ie,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1136 {
    dialog_token: u8,
    follow_up: u8,
    tod: [u8; 6],
    toa: [u8; 6],
    tod_error: u16,
    toa_error: u16,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1145 {
    no_fixed_fields: [u8; 0],
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1149 {
    dialog_token: u8,
    follow_up: u8,
    tod: u32,
    toa: u32,
    max_tod_error: u8,
    max_toa_error: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1157 {
    dialog_token: u8,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1161 {
    dialog_token: u8,
    status_code: u16,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1166 {
    no_fixed_fields: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1170 {
    dialog_token: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1174 {
    dialog_token: u8,
    count: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1179 {
    no_fixed_fields: [u8; 0],
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1183 {
    dialog_token: u8,
    control: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1188 {
    dialog_token: u8,
    r#type: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1193 {
    dialog_token: u8,
    r#type: u8,
    count: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1199 {
    dialog_token: u8,
    r#type: u8,
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1206 {
    __empty_body: __SlateRecord_ieee80211_h_1206_2,
    body: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_h_1206_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ctx_arg_aux {
    offset: u32,
    reg_type: u32,
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
    refcounted: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_mod_pair {
    btf: *mut btf,
    module: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_stream {
    capacity: __SlateRecord_types_h_190,
    log: llist_head,
    lock: mutex,
    backlog_head: *mut llist_node,
    backlog_tail: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_aux {
    refcnt: __SlateRecord_types_h_197,
    used_map_cnt: u32,
    used_btf_cnt: u32,
    max_ctx_offset: u32,
    max_pkt_offset: u32,
    max_tp_access: u32,
    stack_depth: u32,
    id: u32,
    func_cnt: u32,
    real_func_cnt: u32,
    func_idx: u32,
    attach_btf_id: u32,
    attach_st_ops_member_off: u32,
    ctx_arg_info_size: u32,
    max_rdonly_access: u32,
    max_rdwr_access: u32,
    subprog_start: u32,
    attach_btf: *mut btf,
    ctx_arg_info: *mut bpf_ctx_arg_aux,
    priv_stack_ptr: *mut (),
    dst_mutex: mutex,
    dst_prog: *mut bpf_prog,
    dst_trampoline: *mut bpf_trampoline,
    saved_dst_prog_type: u32,
    saved_dst_attach_type: u32,
    verifier_zext: bool,
    dev_bound: bool,
    offload_requested: bool,
    attach_btf_trace: bool,
    attach_tracing_prog: bool,
    tramp_linked: bool,
    func_proto_unreliable: bool,
    tail_call_reachable: bool,
    xdp_has_frags: bool,
    exception_cb: bool,
    exception_boundary: bool,
    is_extended: bool,
    jits_use_priv_stack: bool,
    priv_stack_requested: bool,
    changes_pkt_data: bool,
    might_sleep: bool,
    kprobe_write_ctx: bool,
    sig: __SlateRecord_bpf_h_1821_2,
    prog_array_member_cnt: u64,
    ext_mutex: mutex,
    arena: *mut bpf_arena,
    recursion_detected: Option<unsafe extern "C" fn(*mut bpf_prog)>,
    attach_func_proto: *const btf_type,
    attach_func_name: *const u8,
    func: *mut *mut bpf_prog,
    main_prog_aux: *mut bpf_prog_aux,
    jit_data: *mut (),
    poke_tab: *mut bpf_jit_poke_descriptor,
    kfunc_tab: *mut bpf_kfunc_desc_tab,
    kfunc_btf_tab: *mut bpf_kfunc_btf_tab,
    size_poke_tab: u32,
    ksym: bpf_ksym,
    ops: *const bpf_prog_ops,
    st_ops: *const bpf_struct_ops,
    used_maps: *mut *mut bpf_map,
    used_maps_mutex: mutex,
    used_btfs: *mut btf_mod_pair,
    prog: *mut bpf_prog,
    user: *mut user_struct,
    load_time: u64,
    verified_insns: u32,
    cgroup_atype: i32,
    cgroup_storage: [*mut bpf_map; 2],
    name: [u8; 16],
    bpf_exception_cb: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    stack_arg_sp_adjust: u16,
    security: *mut (),
    token: *mut bpf_token,
    offload: *mut bpf_prog_offload,
    btf: *mut btf,
    func_info: *mut bpf_func_info,
    func_info_aux: *mut bpf_func_info_aux,
    linfo: *mut bpf_line_info,
    jited_linfo: *mut *mut (),
    func_info_cnt: u32,
    nr_linfo: u32,
    linfo_idx: u32,
    r#mod: *mut module,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    __slate_anon_85: __SlateRecord_bpf_h_1892,
    stream: [bpf_stream; 2],
    st_ops_assoc_mutex: mutex,
    st_ops_assoc: *mut bpf_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog {
    pages: u16,
    __slate_bits_0: __slate_bits::__SlateBits796U0,
    r#type: u32,
    expected_attach_type: u32,
    len: u32,
    jited_len: u32,
    __slate_anon_22: __SlateRecord_bpf_h_1926,
    stats: *mut bpf_prog_stats,
    active: *mut u8,
    bpf_func: Option<unsafe extern "C" fn(*const (), *const bpf_insn) -> u32>,
    aux: *mut bpf_prog_aux,
    orig_prog: *mut sock_fprog_kern,
    __slate_anon_28: __SlateRecord_bpf_h_1937,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link {
    refcnt: __SlateRecord_types_h_197,
    id: u32,
    r#type: u32,
    ops: *const bpf_link_ops,
    prog: *mut bpf_prog,
    flags: u32,
    attach_type: u32,
    __slate_anon_7: __SlateRecord_bpf_h_1964,
    sleepable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_ops {
    release: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc_deferred: Option<unsafe extern "C" fn(*mut bpf_link)>,
    detach: Option<unsafe extern "C" fn(*mut bpf_link) -> i32>,
    update_prog: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_prog, *mut bpf_prog) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*const bpf_link, *mut seq_file)>,
    fill_link_info: Option<unsafe extern "C" fn(*const bpf_link, *mut bpf_link_info) -> i32>,
    update_map: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_map, *mut bpf_map) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_token {
    work: work_struct,
    refcnt: __SlateRecord_types_h_197,
    userns: *mut user_namespace,
    allowed_cmds: u64,
    allowed_maps: u64,
    allowed_progs: u64,
    allowed_attachs: u64,
    security: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_struct_ops {
    verifier_ops: *const bpf_verifier_ops,
    init: Option<unsafe extern "C" fn(*mut btf) -> i32>,
    check_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *const bpf_prog) -> i32>,
    init_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *mut (), *const ()) -> i32>,
    reg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link) -> i32>,
    unreg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link)>,
    update: Option<unsafe extern "C" fn(*mut (), *mut (), *mut bpf_link) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    cfi_stubs: *mut (),
    owner: *mut module,
    name: *const u8,
    func_models: [btf_func_model; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array_item {
    prog: *mut bpf_prog,
    __slate_anon_1: __SlateRecord_bpf_h_2532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array {
    rcu: callback_head,
    items: [bpf_prog_array_item; 0],
}


// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
#[repr(C)]
#[derive(Clone, Copy)]
struct nlmsghdr {
    nlmsg_len: u32,
    nlmsg_type: u16,
    nlmsg_flags: u16,
    nlmsg_seq: u32,
    nlmsg_pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nlattr {
    nla_len: u16,
    nla_type: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netlink_ext_ack {
    _msg: *const u8,
    bad_attr: *const nlattr,
    policy: *const nla_policy,
    miss_nest: *const nlattr,
    miss_type: u16,
    cookie: [u8; 8],
    cookie_len: u8,
    _msg_buf: [u8; 80],
}


// Always use this macro, this allows later putting the
// message into a separate section or such for things
// like translation or listing all possible messages.
// If string formatting is needed use NL_SET_ERR_MSG_FMT.
#[repr(C)]
#[derive(Clone, Copy)]
struct netlink_callback {
    skb: *mut sk_buff,
    nlh: *const nlmsghdr,
    dump: Option<unsafe extern "C" fn(*mut sk_buff, *mut netlink_callback) -> i32>,
    done: Option<unsafe extern "C" fn(*mut netlink_callback) -> i32>,
    data: *mut (),
    module: *mut module,
    extack: *mut netlink_ext_ack,
    family: u16,
    answer_flags: u16,
    min_dump_alloc: u32,
    prev_seq: u32,
    seq: u32,
    flags: i32,
    strict_check: bool,
    __slate_anon_14: __SlateRecord_netlink_h_297,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ndmsg {
    ndm_family: u8,
    ndm_pad1: u8,
    ndm_pad2: u16,
    ndm_ifindex: i32,
    ndm_state: u16,
    ndm_flags: u8,
    ndm_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rtnl_link_stats64 {
    rx_packets: u64,
    tx_packets: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    rx_errors: u64,
    tx_errors: u64,
    rx_dropped: u64,
    tx_dropped: u64,
    multicast: u64,
    collisions: u64,
    rx_length_errors: u64,
    rx_over_errors: u64,
    rx_crc_errors: u64,
    rx_frame_errors: u64,
    rx_fifo_errors: u64,
    rx_missed_errors: u64,
    tx_aborted_errors: u64,
    tx_carrier_errors: u64,
    tx_fifo_errors: u64,
    tx_heartbeat_errors: u64,
    tx_window_errors: u64,
    rx_compressed: u64,
    tx_compressed: u64,
    rx_nohandler: u64,
    rx_otherhost_dropped: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rtnl_hw_stats64 {
    rx_packets: u64,
    tx_packets: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    rx_errors: u64,
    tx_errors: u64,
    rx_dropped: u64,
    tx_dropped: u64,
    multicast: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ifla_vf_guid {
    vf: u32,
    guid: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ifla_vf_stats {
    rx_packets: u64,
    tx_packets: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    broadcast: u64,
    multicast: u64,
    rx_dropped: u64,
    tx_dropped: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ifla_vf_info {
    vf: u32,
    mac: [u8; 32],
    vlan: u32,
    qos: u32,
    spoofchk: u32,
    linkstate: u32,
    min_tx_rate: u32,
    max_tx_rate: u32,
    rss_query_en: u32,
    trusted: u32,
    vlan_proto: u16,
}


// qdisc flags in net/sch_generic.h
#[repr(C)]
#[derive(Clone, Copy)]
struct net_device_stats {
    __slate_anon_0: __SlateRecord_netdevice_h_194,
    __slate_anon_1: __SlateRecord_netdevice_h_195,
    __slate_anon_2: __SlateRecord_netdevice_h_196,
    __slate_anon_3: __SlateRecord_netdevice_h_197,
    __slate_anon_4: __SlateRecord_netdevice_h_198,
    __slate_anon_5: __SlateRecord_netdevice_h_199,
    __slate_anon_6: __SlateRecord_netdevice_h_200,
    __slate_anon_7: __SlateRecord_netdevice_h_201,
    __slate_anon_8: __SlateRecord_netdevice_h_202,
    __slate_anon_9: __SlateRecord_netdevice_h_203,
    __slate_anon_10: __SlateRecord_netdevice_h_204,
    __slate_anon_11: __SlateRecord_netdevice_h_205,
    __slate_anon_12: __SlateRecord_netdevice_h_206,
    __slate_anon_13: __SlateRecord_netdevice_h_207,
    __slate_anon_14: __SlateRecord_netdevice_h_208,
    __slate_anon_15: __SlateRecord_netdevice_h_209,
    __slate_anon_16: __SlateRecord_netdevice_h_210,
    __slate_anon_17: __SlateRecord_netdevice_h_211,
    __slate_anon_18: __SlateRecord_netdevice_h_212,
    __slate_anon_19: __SlateRecord_netdevice_h_213,
    __slate_anon_20: __SlateRecord_netdevice_h_214,
    __slate_anon_21: __SlateRecord_netdevice_h_215,
    __slate_anon_22: __SlateRecord_netdevice_h_216,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_eht_h_73 {
    __slate_anon_0: __SlateRecord_ieee80211_eht_h_74,
    rx_tx_max_nss: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_eht_h_74 {
    rx_tx_mcs7_max_nss: u8,
    rx_tx_mcs9_max_nss: u8,
    rx_tx_mcs11_max_nss: u8,
    rx_tx_mcs13_max_nss: u8,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct net_device_core_stats {
    rx_dropped: u64,
    tx_dropped: u64,
    rx_nohandler: u64,
    rx_otherhost_dropped: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_eht_h_104 {
    __slate_anon_0: __SlateRecord_ieee80211_eht_h_105,
    rx_tx_max_nss: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_eht_h_105 {
    rx_tx_mcs9_max_nss: u8,
    rx_tx_mcs11_max_nss: u8,
    rx_tx_mcs13_max_nss: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_hw_addr_list {
    list: list_head,
    count: i32,
    tree: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hh_cache {
    hh_len: u32,
    hh_lock: seqlock,
    hh_data: [u64; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct header_ops {
    create: Option<unsafe extern "C" fn(*mut sk_buff, *mut net_device, u16, *const (), *const (), u32) -> i32>,
    parse: Option<unsafe extern "C" fn(*const sk_buff, *const net_device, *mut u8) -> i32>,
    cache: Option<unsafe extern "C" fn(*const neighbour, *mut hh_cache, u16) -> i32>,
    cache_update: Option<unsafe extern "C" fn(*mut hh_cache, *const net_device, *const u8)>,
    validate: Option<unsafe extern "C" fn(*const u8, u32) -> bool>,
    parse_protocol: Option<unsafe extern "C" fn(*const sk_buff) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gro_list {
    list: list_head,
    count: i32,
}


// size of gro hash buckets, must be <= the number of bits in
// gro_node::bitmask
#[repr(C)]
#[derive(Clone, Copy)]
struct gro_node {
    bitmask: u64,
    hash: [gro_list; 8],
    rx_list: list_head,
    rx_count: u32,
    cached_napi_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct napi_config {
    gro_flush_timeout: u64,
    irq_suspend_timeout: u64,
    defer_hard_irqs: u32,
    affinity_mask: cpumask,
    threaded: u8,
    napi_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct napi_struct {
    state: u64,
    poll_list: list_head,
    weight: i32,
    defer_hard_irqs_count: u32,
    poll: Option<unsafe extern "C" fn(*mut napi_struct, i32) -> i32>,
    poll_owner: i32,
    list_owner: i32,
    dev: *mut net_device,
    skb: *mut sk_buff,
    gro: gro_node,
    timer: hrtimer,
    thread: *mut task_struct,
    gro_flush_timeout: u64,
    irq_suspend_timeout: u64,
    defer_hard_irqs: u32,
    napi_id: u32,
    dev_list: list_head,
    napi_hash_node: hlist_node,
    irq: i32,
    notify: irq_affinity_notify,
    napi_rmap_idx: i32,
    index: i32,
    config: *mut napi_config,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_queue {
    dev: *mut net_device,
    dev_tracker: __SlateRecord_net_trackers_h_9,
    qdisc: *mut Qdisc,
    qdisc_sleeping: *mut Qdisc,
    kobj: kobject,
    groups: *mut *const attribute_group,
    tx_maxrate: u64,
    trans_timeout: __SlateRecord_types_h_197,
    sb_dev: *mut net_device,
    dql: dql,
    _xmit_lock: spinlock,
    xmit_lock_owner: i32,
    trans_start: u64,
    state: u64,
    napi: *mut napi_struct,
    numa_node: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xps_map {
    len: u32,
    alloc_len: u32,
    rcu: callback_head,
    queues: [u16; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xps_dev_maps {
    rcu: callback_head,
    nr_ids: u32,
    num_tc: i16,
    attr_map: [*mut xps_map; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_tc_txq {
    __slate_anon_0: __SlateRecord_netdevice_h_835,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_phys_item_id {
    id: [u8; 32],
    id_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net_device_path {
    r#type: u32,
    dev: *const net_device,
    __slate_anon_2: __SlateRecord_netdevice_h_896,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net_device_path_ctx {
    dev: *const net_device,
    daddr: [u8; 6],
    ether_type: u16,
    num_vlans: i32,
    vlan: [__SlateRecord_netdevice_h_953; 2],
    ieee80211: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_xdp_entity {
    prog: *mut bpf_prog,
    link: *mut bpf_xdp_link,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct netdev_bpf {
    command: u32,
    __slate_anon_1: __SlateRecord_netdevice_h_1024,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_ifalias {
    rcuhead: callback_head,
    ifalias: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net_device_ops {
    ndo_init: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    ndo_uninit: Option<unsafe extern "C" fn(*mut net_device)>,
    ndo_open: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    ndo_stop: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    ndo_start_xmit: Option<unsafe extern "C" fn(*mut sk_buff, *mut net_device) -> i32>,
    ndo_features_check: Option<unsafe extern "C" fn(*mut sk_buff, *mut net_device, u64) -> u64>,
    ndo_select_queue: Option<unsafe extern "C" fn(*mut net_device, *mut sk_buff, *mut net_device) -> u16>,
    ndo_change_rx_flags: Option<unsafe extern "C" fn(*mut net_device, i32)>,
    ndo_set_rx_mode: Option<unsafe extern "C" fn(*mut net_device)>,
    ndo_set_rx_mode_async: Option<unsafe extern "C" fn(*mut net_device, *mut netdev_hw_addr_list, *mut netdev_hw_addr_list) -> i32>,
    ndo_work: Option<unsafe extern "C" fn(*mut net_device, u64)>,
    ndo_set_mac_address: Option<unsafe extern "C" fn(*mut net_device, *mut ()) -> i32>,
    ndo_validate_addr: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    ndo_do_ioctl: Option<unsafe extern "C" fn(*mut net_device, *mut ifreq, i32) -> i32>,
    ndo_eth_ioctl: Option<unsafe extern "C" fn(*mut net_device, *mut ifreq, i32) -> i32>,
    ndo_siocbond: Option<unsafe extern "C" fn(*mut net_device, *mut ifreq, i32) -> i32>,
    ndo_siocwandev: Option<unsafe extern "C" fn(*mut net_device, *mut if_settings) -> i32>,
    ndo_siocdevprivate: Option<unsafe extern "C" fn(*mut net_device, *mut ifreq, *mut (), i32) -> i32>,
    ndo_set_config: Option<unsafe extern "C" fn(*mut net_device, *mut ifmap) -> i32>,
    ndo_change_mtu: Option<unsafe extern "C" fn(*mut net_device, i32) -> i32>,
    ndo_neigh_setup: Option<unsafe extern "C" fn(*mut net_device, *mut neigh_parms) -> i32>,
    ndo_tx_timeout: Option<unsafe extern "C" fn(*mut net_device, u32)>,
    ndo_get_stats64: Option<unsafe extern "C" fn(*mut net_device, *mut rtnl_link_stats64)>,
    ndo_has_offload_stats: Option<unsafe extern "C" fn(*const net_device, i32) -> bool>,
    ndo_get_offload_stats: Option<unsafe extern "C" fn(i32, *const net_device, *mut ()) -> i32>,
    ndo_get_stats: Option<unsafe extern "C" fn(*mut net_device) -> *mut net_device_stats>,
    ndo_vlan_rx_add_vid: Option<unsafe extern "C" fn(*mut net_device, u16, u16) -> i32>,
    ndo_vlan_rx_kill_vid: Option<unsafe extern "C" fn(*mut net_device, u16, u16) -> i32>,
    ndo_poll_controller: Option<unsafe extern "C" fn(*mut net_device)>,
    ndo_netpoll_setup: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    ndo_netpoll_cleanup: Option<unsafe extern "C" fn(*mut net_device)>,
    ndo_set_vf_mac: Option<unsafe extern "C" fn(*mut net_device, i32, *mut u8) -> i32>,
    ndo_set_vf_vlan: Option<unsafe extern "C" fn(*mut net_device, i32, u16, u8, u16) -> i32>,
    ndo_set_vf_rate: Option<unsafe extern "C" fn(*mut net_device, i32, i32, i32) -> i32>,
    ndo_set_vf_spoofchk: Option<unsafe extern "C" fn(*mut net_device, i32, bool) -> i32>,
    ndo_set_vf_trust: Option<unsafe extern "C" fn(*mut net_device, i32, bool) -> i32>,
    ndo_get_vf_config: Option<unsafe extern "C" fn(*mut net_device, i32, *mut ifla_vf_info) -> i32>,
    ndo_set_vf_link_state: Option<unsafe extern "C" fn(*mut net_device, i32, i32) -> i32>,
    ndo_get_vf_stats: Option<unsafe extern "C" fn(*mut net_device, i32, *mut ifla_vf_stats) -> i32>,
    ndo_set_vf_port: Option<unsafe extern "C" fn(*mut net_device, i32, *mut *mut nlattr) -> i32>,
    ndo_get_vf_port: Option<unsafe extern "C" fn(*mut net_device, i32, *mut sk_buff) -> i32>,
    ndo_get_vf_guid: Option<unsafe extern "C" fn(*mut net_device, i32, *mut ifla_vf_guid, *mut ifla_vf_guid) -> i32>,
    ndo_set_vf_guid: Option<unsafe extern "C" fn(*mut net_device, i32, u64, i32) -> i32>,
    ndo_set_vf_rss_query_en: Option<unsafe extern "C" fn(*mut net_device, i32, bool) -> i32>,
    ndo_setup_tc: Option<unsafe extern "C" fn(*mut net_device, u32, *mut ()) -> i32>,
    ndo_rx_flow_steer: Option<unsafe extern "C" fn(*mut net_device, *const sk_buff, u16, u32) -> i32>,
    ndo_add_slave: Option<unsafe extern "C" fn(*mut net_device, *mut net_device, *mut netlink_ext_ack) -> i32>,
    ndo_del_slave: Option<unsafe extern "C" fn(*mut net_device, *mut net_device) -> i32>,
    ndo_get_xmit_slave: Option<unsafe extern "C" fn(*mut net_device, *mut sk_buff, bool) -> *mut net_device>,
    ndo_sk_get_lower_dev: Option<unsafe extern "C" fn(*mut net_device, *mut sock) -> *mut net_device>,
    ndo_fix_features: Option<unsafe extern "C" fn(*mut net_device, u64) -> u64>,
    ndo_set_features: Option<unsafe extern "C" fn(*mut net_device, u64) -> i32>,
    ndo_neigh_construct: Option<unsafe extern "C" fn(*mut net_device, *mut neighbour) -> i32>,
    ndo_neigh_destroy: Option<unsafe extern "C" fn(*mut net_device, *mut neighbour)>,
    ndo_fdb_add: Option<unsafe extern "C" fn(*mut ndmsg, *mut *mut nlattr, *mut net_device, *const u8, u16, u16, *mut bool, *mut netlink_ext_ack) -> i32>,
    ndo_fdb_del: Option<unsafe extern "C" fn(*mut ndmsg, *mut *mut nlattr, *mut net_device, *const u8, u16, *mut bool, *mut netlink_ext_ack) -> i32>,
    ndo_fdb_del_bulk: Option<unsafe extern "C" fn(*mut nlmsghdr, *mut net_device, *mut netlink_ext_ack) -> i32>,
    ndo_fdb_dump: Option<unsafe extern "C" fn(*mut sk_buff, *mut netlink_callback, *mut net_device, *mut net_device, *mut i32) -> i32>,
    ndo_fdb_get: Option<unsafe extern "C" fn(*mut sk_buff, *mut *mut nlattr, *mut net_device, *const u8, u16, u32, u32, *mut netlink_ext_ack) -> i32>,
    ndo_mdb_add: Option<unsafe extern "C" fn(*mut net_device, *mut *mut nlattr, u16, *mut netlink_ext_ack) -> i32>,
    ndo_mdb_del: Option<unsafe extern "C" fn(*mut net_device, *mut *mut nlattr, *mut netlink_ext_ack) -> i32>,
    ndo_mdb_del_bulk: Option<unsafe extern "C" fn(*mut net_device, *mut *mut nlattr, *mut netlink_ext_ack) -> i32>,
    ndo_mdb_dump: Option<unsafe extern "C" fn(*mut net_device, *mut sk_buff, *mut netlink_callback) -> i32>,
    ndo_mdb_get: Option<unsafe extern "C" fn(*mut net_device, *mut *mut nlattr, u32, u32, *mut netlink_ext_ack) -> i32>,
    ndo_bridge_setlink: Option<unsafe extern "C" fn(*mut net_device, *mut nlmsghdr, u16, *mut netlink_ext_ack) -> i32>,
    ndo_bridge_getlink: Option<unsafe extern "C" fn(*mut sk_buff, u32, u32, *mut net_device, u32, i32) -> i32>,
    ndo_bridge_dellink: Option<unsafe extern "C" fn(*mut net_device, *mut nlmsghdr, u16) -> i32>,
    ndo_change_carrier: Option<unsafe extern "C" fn(*mut net_device, bool) -> i32>,
    ndo_get_phys_port_id: Option<unsafe extern "C" fn(*mut net_device, *mut netdev_phys_item_id) -> i32>,
    ndo_get_port_parent_id: Option<unsafe extern "C" fn(*mut net_device, *mut netdev_phys_item_id) -> i32>,
    ndo_get_phys_port_name: Option<unsafe extern "C" fn(*mut net_device, *mut u8, u64) -> i32>,
    ndo_dfwd_add_station: Option<unsafe extern "C" fn(*mut net_device, *mut net_device) -> *mut ()>,
    ndo_dfwd_del_station: Option<unsafe extern "C" fn(*mut net_device, *mut ())>,
    ndo_set_tx_maxrate: Option<unsafe extern "C" fn(*mut net_device, i32, u32) -> i32>,
    ndo_get_iflink: Option<unsafe extern "C" fn(*const net_device) -> i32>,
    ndo_fill_metadata_dst: Option<unsafe extern "C" fn(*mut net_device, *mut sk_buff) -> i32>,
    ndo_set_rx_headroom: Option<unsafe extern "C" fn(*mut net_device, i32)>,
    ndo_bpf: Option<unsafe extern "C" fn(*mut net_device, *mut netdev_bpf) -> i32>,
    ndo_xdp_xmit: Option<unsafe extern "C" fn(*mut net_device, i32, *mut *mut xdp_frame, u32) -> i32>,
    ndo_xdp_get_xmit_slave: Option<unsafe extern "C" fn(*mut net_device, *mut xdp_buff) -> *mut net_device>,
    ndo_xsk_wakeup: Option<unsafe extern "C" fn(*mut net_device, u32, u32) -> i32>,
    ndo_tunnel_ctl: Option<unsafe extern "C" fn(*mut net_device, *mut ip_tunnel_parm_kern, i32) -> i32>,
    ndo_get_peer_dev: Option<unsafe extern "C" fn(*mut net_device) -> *mut net_device>,
    ndo_fill_forward_path: Option<unsafe extern "C" fn(*mut net_device_path_ctx, *mut net_device_path) -> i32>,
    ndo_get_tstamp: Option<unsafe extern "C" fn(*mut net_device, *const skb_shared_hwtstamps, bool) -> i64>,
    ndo_hwtstamp_get: Option<unsafe extern "C" fn(*mut net_device, *mut kernel_hwtstamp_config) -> i32>,
    ndo_hwtstamp_set: Option<unsafe extern "C" fn(*mut net_device, *mut kernel_hwtstamp_config, *mut netlink_ext_ack) -> i32>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct net_device {
    __cacheline_group_begin__net_device_read_tx: [u8; 0],
    __slate_anon_1: __SlateRecord_netdevice_h_2169,
    netdev_ops: *const net_device_ops,
    header_ops: *const header_ops,
    _tx: *mut netdev_queue,
    gso_partial_features: u64,
    real_num_tx_queues: u32,
    gso_max_size: u32,
    gso_ipv4_max_size: u32,
    gso_max_segs: u16,
    num_tc: i16,
    mtu: u32,
    needed_headroom: u16,
    tc_to_txq: [netdev_tc_txq; 16],
    xps_maps: [*mut xps_dev_maps; 2],
    nf_hooks_egress: *mut nf_hook_entries,
    tcx_egress: *mut bpf_mprog_entry,
    __cacheline_group_end__net_device_read_tx: [u8; 0],
    __cacheline_group_begin__net_device_read_txrx: [u8; 0],
    __slate_anon_19: __SlateRecord_netdevice_h_2204,
    state: u64,
    flags: u32,
    hard_header_len: u16,
    __slate_bits_0: __slate_bits::__SlateBits230U0,
    features: u64,
    ip6_ptr: *mut inet6_dev,
    __cacheline_group_end__net_device_read_txrx: [u8; 0],
    __cacheline_group_begin__net_device_read_rx: [u8; 0],
    xdp_prog: *mut bpf_prog,
    ptype_specific: list_head,
    ifindex: i32,
    real_num_rx_queues: u32,
    _rx: *mut netdev_rx_queue,
    gro_max_size: u32,
    gro_ipv4_max_size: u32,
    rx_handler: Option<unsafe extern "C" fn(*mut *mut sk_buff) -> u32>,
    rx_handler_data: *mut (),
    nd_net: __SlateRecord_net_namespace_h_410,
    npinfo: *mut netpoll_info,
    tcx_ingress: *mut bpf_mprog_entry,
    __cacheline_group_end__net_device_read_rx: [u8; 0],
    name: [u8; 16],
    name_node: *mut netdev_name_node,
    ifalias: *mut dev_ifalias,
    mem_end: u64,
    mem_start: u64,
    base_addr: u64,
    dev_list: list_head,
    napi_list: list_head,
    unreg_list: list_head,
    close_list: list_head,
    ptype_all: list_head,
    adj_list: __SlateRecord_netdevice_h_2264,
    xdp_features: u32,
    xdp_metadata_ops: *const xdp_metadata_ops,
    xsk_tx_metadata_ops: *const xsk_tx_metadata_ops,
    gflags: u16,
    needed_tailroom: u16,
    hw_features: u64,
    wanted_features: u64,
    vlan_features: u64,
    hw_enc_features: u64,
    mpls_features: u64,
    mangleid_features: u64,
    min_mtu: u32,
    max_mtu: u32,
    r#type: u16,
    min_header_len: u8,
    name_assign_type: u8,
    group: i32,
    stats: net_device_stats,
    core_stats: *mut net_device_core_stats,
    carrier_up_count: __SlateRecord_types_h_190,
    carrier_down_count: __SlateRecord_types_h_190,
    ethtool_ops: *const ethtool_ops,
    ndisc_ops: *const ndisc_ops,
    operstate: u32,
    link_mode: u8,
    if_port: u8,
    dma: u8,
    perm_addr: [u8; 32],
    addr_assign_type: u8,
    addr_len: u8,
    upper_level: u8,
    lower_level: u8,
    threaded: u8,
    neigh_priv_len: u16,
    dev_id: u16,
    dev_port: u16,
    irq: i32,
    priv_len: u32,
    addr_list_lock: spinlock,
    uc: netdev_hw_addr_list,
    mc: netdev_hw_addr_list,
    dev_addrs: netdev_hw_addr_list,
    queues_kset: *mut kset,
    promiscuity: u32,
    allmulti: u32,
    uc_promisc: bool,
    work_node: list_head,
    work_tracker: __SlateRecord_net_trackers_h_9,
    work_pending: u64,
    work_core_pending: u64,
    rx_mode_addr_cache: netdev_hw_addr_list,
    rx_mode_retry_timer: timer_list,
    rx_mode_retry_count: u32,
    ip_ptr: *mut in_device,
    fib_nh_head: hlist_head,
    ieee80211_ptr: *mut wireless_dev,
    dev_addr: *const u8,
    num_rx_queues: u32,
    xdp_zc_max_segs: u32,
    ingress_queue: *mut netdev_queue,
    nf_hooks_ingress: *mut nf_hook_entries,
    broadcast: [u8; 32],
    rx_cpu_rmap: *mut cpu_rmap,
    index_hlist: hlist_node,
    num_tx_queues: u32,
    qdisc: *mut Qdisc,
    tx_queue_len: u32,
    tx_global_lock: spinlock,
    xdp_bulkq: *mut xdp_dev_bulk_queue,
    qdisc_hash: [hlist_head; 16],
    watchdog_timer: timer_list,
    watchdog_timeo: i32,
    watchdog_lock: spinlock,
    watchdog_ref_held: bool,
    proto_down_reason: u32,
    todo_list: list_head,
    pcpu_refcnt: *mut i32,
    refcnt_tracker: ref_tracker_dir,
    link_watch_list: list_head,
    reg_state: u8,
    dismantle: bool,
    moving_ns: bool,
    rtnl_link_initializing: bool,
    needs_free_netdev: bool,
    priv_destructor: Option<unsafe extern "C" fn(*mut net_device)>,
    ml_priv: *mut (),
    ml_priv_type: u32,
    dev: device,
    sysfs_groups: [*const attribute_group; 5],
    sysfs_rx_queue_group: *const attribute_group,
    rtnl_link_ops: *const rtnl_link_ops,
    stat_ops: *const netdev_stat_ops,
    queue_mgmt_ops: *const netdev_queue_mgmt_ops,
    tso_max_size: u32,
    tso_max_segs: u16,
    prio_tc_map: [u8; 16],
    priomap: *mut netprio_map,
    link_topo: *mut phy_link_topology,
    phydev: *mut phy_device,
    sfp_bus: *mut sfp_bus,
    qdisc_tx_busylock: *mut lock_class_key,
    proto_down: bool,
    irq_affinity_auto: bool,
    rx_cpu_rmap_auto: bool,
    __slate_bits_1: __slate_bits::__SlateBits230U1,
    net_notifier_list: list_head,
    udp_tunnel_nic_info: *const udp_tunnel_nic_info,
    udp_tunnel_nic: *mut udp_tunnel_nic,
    cfg: *mut netdev_config,
    cfg_pending: *mut netdev_config,
    ethtool: *mut ethtool_netdev_state,
    xdp_state: [bpf_xdp_entity; 3],
    dev_addr_shadow: [u8; 32],
    linkwatch_dev_tracker: __SlateRecord_net_trackers_h_9,
    watchdog_dev_tracker: __SlateRecord_net_trackers_h_9,
    dev_registered_tracker: __SlateRecord_net_trackers_h_9,
    offload_xstats_l3: *mut rtnl_hw_stats64,
    devlink_port: *mut devlink_port,
    page_pools: hlist_head,
    irq_moder: *mut dim_irq_moder,
    max_pacing_offload_horizon: u64,
    napi_config: *mut napi_config,
    num_napi_configs: u32,
    napi_defer_hard_irqs: u32,
    gro_flush_timeout: u64,
    up: bool,
    request_ops_lock: bool,
    lock: mutex,
    neighbours: [hlist_head; 2],
    hwprov: *mut hwtstamp_provider,
    r#priv: [u8; 0],
}


// Default NAPI poll() weight
// Device drivers are strongly advised to not use bigger value
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct pcpu_sw_netstats {
    rx_packets: __SlateRecord_u64_stats_sync_h_75,
    rx_bytes: __SlateRecord_u64_stats_sync_h_75,
    tx_packets: __SlateRecord_u64_stats_sync_h_75,
    tx_bytes: __SlateRecord_u64_stats_sync_h_75,
    syncp: u64_stats_sync,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct pcpu_dstats {
    rx_packets: __SlateRecord_u64_stats_sync_h_75,
    rx_bytes: __SlateRecord_u64_stats_sync_h_75,
    tx_packets: __SlateRecord_u64_stats_sync_h_75,
    tx_bytes: __SlateRecord_u64_stats_sync_h_75,
    rx_drops: __SlateRecord_u64_stats_sync_h_75,
    tx_drops: __SlateRecord_u64_stats_sync_h_75,
    syncp: u64_stats_sync,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct pcpu_lstats {
    packets: __SlateRecord_u64_stats_sync_h_75,
    bytes: __SlateRecord_u64_stats_sync_h_75,
    syncp: u64_stats_sync,
}


// dev_addrs walker. Should be used only for read access. Call with
// rcu_read_lock held.
// The list of packet types we will receive (as opposed to discard)
// and the routines to invoke.
//
// Why 16. Because with 16 the only overlap we get on a hash of the
// low nibble of the protocol value is RARP/SNAP/X.25.
//
// 	0800	IP
// 	0001	802.3
// 	0002	AX.25
// 	0004	802.2
// 	8035	RARP
// 	0005	SNAP
// 	0805	X.25
// 	0806	ARP
// 	8137	IPX
// 	0009	Localtalk
// 	86DD	IPv6
fn ether_addr_copy(dst: *mut u8, src: *const u8) {
unsafe {
*(dst as *mut u32) = *(src as *const u32);
}
unsafe {
*(dst.wrapping_offset(4 as isize) as *mut u16) = *(src.wrapping_offset(4 as isize) as *const u16);
}
}

#[inline(always)]
fn field_multiplier(field: u64) -> u64 {
if (field | field.wrapping_sub(1)) & (field | field.wrapping_sub(1)).wrapping_add(1) != 0 {
unsafe { __bad_mask() };
}
field & field.wrapping_neg()
}

#[inline(always)]
fn field_mask(field: u64) -> u64 {
field / field_multiplier(field)
}

#[inline(always)]
fn u8_encode_bits(v: u8, field: u8) -> u8 {
let __v0: bool = (unsafe { __slate_intrinsic_70b71f185ebb8c4a(v) }) && ((v as i32) as u64) & !field_mask(field as u64) != 0;
if __v0 {
unsafe { __field_overflow() };
}
(((v as i32) as u64) & field_mask(field as u64)).wrapping_mul(field_multiplier(field as u64)) as u8
}

#[inline(always)]
fn u8_get_bits(v: u8, field: u8) -> u8 {
((((v as i32) & (field as i32)) as u64) / field_multiplier(field as u64)) as u8
}

#[inline(always)]
fn u16_encode_bits(v: u16, field: u16) -> u16 {
let __v0: bool = (unsafe { __slate_intrinsic_b52e897e44daaa8(v) }) && ((v as i32) as u64) & !field_mask(field as u64) != 0;
if __v0 {
unsafe { __field_overflow() };
}
(((v as i32) as u64) & field_mask(field as u64)).wrapping_mul(field_multiplier(field as u64)) as u16
}

// control
// number of ACs
// TID mask
#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_msrment_ie {
    token: u8,
    mode: u8,
    r#type: u8,
    request: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_channel_sw_ie {
    mode: u8,
    new_ch_num: u8,
    count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ext_chansw_ie {
    mode: u8,
    new_operating_class: u8,
    new_ch_num: u8,
    count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sec_chan_offs_ie {
    sec_chan_offs: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_wide_bw_chansw_ie {
    new_channel_width: u8,
    new_center_freq_seg0: u8,
    new_center_freq_seg1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tim_ie {
    dtim_count: u8,
    dtim_period: u8,
    bitmap_ctrl: u8,
    __slate_anon_3: __SlateRecord_ieee80211_h_928,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tpc_report_ie {
    tx_power: u8,
    link_margin: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_addba_ext_ie {
    data: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mgmt {
    frame_control: u16,
    duration: u16,
    da: [u8; 6],
    sa: [u8; 6],
    bssid: [u8; 6],
    seq_ctrl: u16,
    u: __SlateRecord_ieee80211_h_999,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_ch_switch_timing {
    switch_time: u16,
    switch_timeout: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tdls_lnkie {
    ie_type: u8,
    ie_len: u8,
    bssid: [u8; 6],
    init_sta: [u8; 6],
    resp_sta: [u8; 6],
}


// TDLS specific payload type in the LLC/SNAP header
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_timeout_interval_ie {
    r#type: u8,
    value: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_bss_max_idle_period_ie {
    max_idle_period: u16,
    idle_options: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_bssid_index {
    bssid_index: u8,
    dtim_period: u8,
    dtim_count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_1509 {
    rule_cnt: u32,
    rss_context: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_multiple_bssid_configuration {
    bssid_count: u8,
    profile_periodicity: u8,
}


// convert frequencies
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_mcs_info {
    rx_mask: [u8; 10],
    rx_highest: u16,
    tx_params: u8,
    reserved: [u8; 3],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_ht_cap {
    cap_info: u16,
    ampdu_params_info: u8,
    mcs: ieee80211_mcs_info,
    extended_ht_cap_info: u16,
    tx_BF_cap_info: u32,
    antenna_selection_info: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_ht_operation {
    primary_chan: u8,
    ht_param: u8,
    operation_mode: u16,
    stbc_param: u16,
    basic_set: [u8; 16],
}


// block-ack parameters
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_vht_mcs_info {
    rx_mcs_map: u16,
    rx_highest: u16,
    tx_mcs_map: u16,
    tx_highest: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_vht_cap {
    vht_cap_info: u32,
    supp_mcs: ieee80211_vht_mcs_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_138 {
    autoneg: u32,
    link_training: u32,
    link_logical_mismatch: u32,
    bad_signal_integrity: u32,
    cable_issue: u32,
    module: u32,
    __link_ext_substate: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_vht_operation {
    chan_width: u8,
    center_freq_seg0_idx: u8,
    center_freq_seg1_idx: u8,
    basic_mcs_set: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_twt_setup {
    dialog_token: u8,
    element_id: u8,
    length: u8,
    control: u8,
    params: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_he_cap_elem {
    mac_cap_info: [u8; 6],
    phy_cap_info: [u8; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_275 {
    supported: [u64; 2],
    advertising: [u64; 2],
    lp_advertising: [u64; 2],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_he_mcs_nss_supp {
    rx_mcs_80: u16,
    tx_mcs_80: u16,
    rx_mcs_160: u16,
    tx_mcs_160: u16,
    rx_mcs_80p80: u16,
    tx_mcs_80p80: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_he_operation {
    he_oper_params: u32,
    he_mcs_nss_set: u16,
    optional: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_he_spr {
    he_sr_control: u8,
    optional: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_445 {
    __slate_anon_0: __SlateRecord_ethtool_h_445_2,
    stats: __SlateRecord_ethtool_h_445_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_445_2 {
    FramesTransmittedOK: u64,
    SingleCollisionFrames: u64,
    MultipleCollisionFrames: u64,
    FramesReceivedOK: u64,
    FrameCheckSequenceErrors: u64,
    AlignmentErrors: u64,
    OctetsTransmittedOK: u64,
    FramesWithDeferredXmissions: u64,
    LateCollisions: u64,
    FramesAbortedDueToXSColls: u64,
    FramesLostDueToIntMACXmitError: u64,
    CarrierSenseErrors: u64,
    OctetsReceivedOK: u64,
    FramesLostDueToIntMACRcvError: u64,
    MulticastFramesXmittedOK: u64,
    BroadcastFramesXmittedOK: u64,
    FramesWithExcessiveDeferral: u64,
    MulticastFramesReceivedOK: u64,
    BroadcastFramesReceivedOK: u64,
    InRangeLengthErrors: u64,
    OutOfRangeLengthField: u64,
    FrameTooLongErrors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_445_3 {
    FramesTransmittedOK: u64,
    SingleCollisionFrames: u64,
    MultipleCollisionFrames: u64,
    FramesReceivedOK: u64,
    FrameCheckSequenceErrors: u64,
    AlignmentErrors: u64,
    OctetsTransmittedOK: u64,
    FramesWithDeferredXmissions: u64,
    LateCollisions: u64,
    FramesAbortedDueToXSColls: u64,
    FramesLostDueToIntMACXmitError: u64,
    CarrierSenseErrors: u64,
    OctetsReceivedOK: u64,
    FramesLostDueToIntMACRcvError: u64,
    MulticastFramesXmittedOK: u64,
    BroadcastFramesXmittedOK: u64,
    FramesWithExcessiveDeferral: u64,
    MulticastFramesReceivedOK: u64,
    BroadcastFramesReceivedOK: u64,
    InRangeLengthErrors: u64,
    OutOfRangeLengthField: u64,
    FrameTooLongErrors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_he_mu_edca_param_ac_rec {
    aifsn: u8,
    ecw_min_max: u8,
    mu_edca_timer: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_476 {
    __slate_anon_0: __SlateRecord_ethtool_h_476_2,
    stats: __SlateRecord_ethtool_h_476_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_476_2 {
    SymbolErrorDuringCarrier: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_476_3 {
    SymbolErrorDuringCarrier: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mu_edca_param_set {
    mu_qos_info: u8,
    ac_be: ieee80211_he_mu_edca_param_ac_rec,
    ac_bk: ieee80211_he_mu_edca_param_ac_rec,
    ac_vi: ieee80211_he_mu_edca_param_ac_rec,
    ac_vo: ieee80211_he_mu_edca_param_ac_rec,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_509 {
    __slate_anon_0: __SlateRecord_ethtool_h_509_2,
    stats: __SlateRecord_ethtool_h_509_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_509_2 {
    MACControlFramesTransmitted: u64,
    MACControlFramesReceived: u64,
    UnsupportedOpcodesReceived: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_509_3 {
    MACControlFramesTransmitted: u64,
    MACControlFramesReceived: u64,
    UnsupportedOpcodesReceived: u64,
}


// 802.11ax HE PHY capabilities
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_he_6ghz_capa {
    capa: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_535 {
    __slate_anon_0: __SlateRecord_ethtool_h_535_2,
    stats: __SlateRecord_ethtool_h_535_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_535_2 {
    tx_pause_frames: u64,
    rx_pause_frames: u64,
    tx_pause_storm_events: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_535_3 {
    tx_pause_frames: u64,
    rx_pause_frames: u64,
    tx_pause_storm_events: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ttlm_elem {
    control: u8,
    optional: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_mcs_nss_supp_20mhz_only {
    __slate_anon_0: __SlateRecord_ieee80211_eht_h_73,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fec_hist_value {
    sum: u64,
    per_lane: [u64; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_mcs_nss_supp_bw {
    __slate_anon_0: __SlateRecord_ieee80211_eht_h_104,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fec_stat {
    total: u64,
    lanes: [u64; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_cap_elem_fixed {
    mac_cap_info: [u8; 2],
    phy_cap_info: [u8; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_cap_elem {
    fixed: ieee80211_eht_cap_elem_fixed,
    optional: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_632 {
    __slate_anon_0: __SlateRecord_ethtool_h_632_2,
    stats: __SlateRecord_ethtool_h_632_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_632_2 {
    undersize_pkts: u64,
    oversize_pkts: u64,
    fragments: u64,
    jabbers: u64,
    hist: [u64; 11],
    hist_tx: [u64; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_632_3 {
    undersize_pkts: u64,
    oversize_pkts: u64,
    fragments: u64,
    jabbers: u64,
    hist: [u64; 11],
    hist_tx: [u64; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_operation {
    params: u8,
    basic_mcs_nss: ieee80211_eht_mcs_nss_supp_20mhz_only,
    optional: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ethtool_h_662 {
    __slate_anon_0: __SlateRecord_ethtool_h_662_2,
    tx_stats: __SlateRecord_ethtool_h_662_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_662_2 {
    pkts: u64,
    onestep_pkts_unconfirmed: u64,
    lost: u64,
    err: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ethtool_h_662_3 {
    pkts: u64,
    onestep_pkts_unconfirmed: u64,
    lost: u64,
    err: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_operation_info {
    control: u8,
    ccfs0: u8,
    ccfs1: u8,
    optional: [u8; 0],
}


// EHT PHY capabilities as defined in P802.11be_D2.0 section 9.4.2.313.3
#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_bandwidth_indication {
    params: u8,
    info: ieee80211_eht_operation_info,
}


// multi-link device
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_multi_link_elem {
    control: u16,
    variable: [u8; 0],
}


// Described in P802.11be_D3.0
// dot11MSDTimerDuration should default to 5484 (i.e. 171.375)
// dot11MSDOFDMEDthreshold defaults to -72 (i.e. 0)
// dot11MSDTXOPMAX defaults to 1
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_mle_per_sta_profile {
    control: u16,
    sta_info_len: u8,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_uhr_operation {
    params: u16,
    basic_mcs_nss_set: [u8; 4],
    variable: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_uhr_cap_mac {
    mac_cap: [u8; 6],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_uhr_cap_phy {
    cap: u32,
    reserved: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_uhr_cap {
    mac: ieee80211_uhr_cap_mac,
    phy: ieee80211_uhr_cap_phy,
    variable: [u8; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_mesh_chansw_params_ie {
    mesh_ttl: u8,
    mesh_flags: u8,
    mesh_reason: u16,
    mesh_pre_value: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_meshconf_ie {
    meshconf_psel: u8,
    meshconf_pmetric: u8,
    meshconf_congest: u8,
    meshconf_synch: u8,
    meshconf_auth: u8,
    meshconf_form: u8,
    meshconf_cap: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_rann_ie {
    rann_flags: u8,
    rann_hopcount: u8,
    rann_ttl: u8,
    rann_addr: [u8; 6],
    rann_seq: u32,
    rann_interval: u32,
    rann_metric: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_s1g_bcn_compat_ie {
    compat_info: u16,
    beacon_int: u16,
    tsf_completion: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_s1g_oper_ie {
    ch_width: u8,
    oper_class: u8,
    primary_ch: u8,
    oper_ch: u8,
    basic_mcs_nss: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_aid_response_ie {
    aid: u16,
    switch_count: u8,
    response_int: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_s1g_cap {
    capab_info: [u8; 10],
    supp_mcs_nss: [u8; 5],
}


// S1G encoding types
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ieee80211_p2p_noa_desc {
    count: u8,
    duration: u32,
    interval: u32,
    start_time: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_p2p_noa_attr {
    index: u8,
    oppps_ctwindow: u8,
    desc: [ieee80211_p2p_noa_desc; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rfkill {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_drvinfo {
    cmd: u32,
    driver: [u8; 32],
    version: [u8; 32],
    fw_version: [u8; 32],
    bus_info: [u8; 32],
    erom_version: [u8; 32],
    reserved2: [u8; 12],
    n_priv_flags: u32,
    n_stats: u32,
    testinfo_len: u32,
    eedump_len: u32,
    regdump_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_wolinfo {
    cmd: u32,
    supported: u32,
    wolopts: u32,
    sopass: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_tunable {
    cmd: u32,
    id: u32,
    type_id: u32,
    len: u32,
    data: [*mut (); 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_regs {
    cmd: u32,
    version: u32,
    len: u32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_eeprom {
    cmd: u32,
    magic: u32,
    offset: u32,
    len: u32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_modinfo {
    cmd: u32,
    r#type: u32,
    eeprom_len: u32,
    reserved: [u32; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_coalesce {
    cmd: u32,
    rx_coalesce_usecs: u32,
    rx_max_coalesced_frames: u32,
    rx_coalesce_usecs_irq: u32,
    rx_max_coalesced_frames_irq: u32,
    tx_coalesce_usecs: u32,
    tx_max_coalesced_frames: u32,
    tx_coalesce_usecs_irq: u32,
    tx_max_coalesced_frames_irq: u32,
    stats_block_coalesce_usecs: u32,
    use_adaptive_rx_coalesce: u32,
    use_adaptive_tx_coalesce: u32,
    pkt_rate_low: u32,
    rx_coalesce_usecs_low: u32,
    rx_max_coalesced_frames_low: u32,
    tx_coalesce_usecs_low: u32,
    tx_max_coalesced_frames_low: u32,
    pkt_rate_high: u32,
    rx_coalesce_usecs_high: u32,
    rx_max_coalesced_frames_high: u32,
    tx_coalesce_usecs_high: u32,
    tx_max_coalesced_frames_high: u32,
    rate_sample_interval: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_ringparam {
    cmd: u32,
    rx_max_pending: u32,
    rx_mini_max_pending: u32,
    rx_jumbo_max_pending: u32,
    tx_max_pending: u32,
    rx_pending: u32,
    rx_mini_pending: u32,
    rx_jumbo_pending: u32,
    tx_pending: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_channels {
    cmd: u32,
    max_rx: u32,
    max_tx: u32,
    max_other: u32,
    max_combined: u32,
    rx_count: u32,
    tx_count: u32,
    other_count: u32,
    combined_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_pauseparam {
    cmd: u32,
    autoneg: u32,
    rx_pause: u32,
    tx_pause: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_test {
    cmd: u32,
    flags: u32,
    reserved: u32,
    len: u32,
    data: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_stats {
    cmd: u32,
    n_stats: u32,
    data: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_tcpip4_spec {
    ip4src: u32,
    ip4dst: u32,
    psrc: u16,
    pdst: u16,
    tos: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_ah_espip4_spec {
    ip4src: u32,
    ip4dst: u32,
    spi: u32,
    tos: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_usrip4_spec {
    ip4src: u32,
    ip4dst: u32,
    l4_4_bytes: u32,
    tos: u8,
    ip_ver: u8,
    proto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_tcpip6_spec {
    ip6src: [u32; 4],
    ip6dst: [u32; 4],
    psrc: u16,
    pdst: u16,
    tclass: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cfg80211_h_407 {
    only_20mhz: ieee80211_eht_mcs_nss_supp_20mhz_only,
    bw: __SlateRecord_cfg80211_h_409,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_409 {
    _80: ieee80211_eht_mcs_nss_supp_bw,
    _160: ieee80211_eht_mcs_nss_supp_bw,
    _320: ieee80211_eht_mcs_nss_supp_bw,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_ah_espip6_spec {
    ip6src: [u32; 4],
    ip6dst: [u32; 4],
    spi: u32,
    tclass: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_usrip6_spec {
    ip6src: [u32; 4],
    ip6dst: [u32; 4],
    l4_4_bytes: u32,
    tclass: u8,
    l4_proto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union ethtool_flow_union {
    tcp_ip4_spec: ethtool_tcpip4_spec,
    udp_ip4_spec: ethtool_tcpip4_spec,
    sctp_ip4_spec: ethtool_tcpip4_spec,
    ah_ip4_spec: ethtool_ah_espip4_spec,
    esp_ip4_spec: ethtool_ah_espip4_spec,
    usr_ip4_spec: ethtool_usrip4_spec,
    tcp_ip6_spec: ethtool_tcpip6_spec,
    udp_ip6_spec: ethtool_tcpip6_spec,
    sctp_ip6_spec: ethtool_tcpip6_spec,
    ah_ip6_spec: ethtool_ah_espip6_spec,
    esp_ip6_spec: ethtool_ah_espip6_spec,
    usr_ip6_spec: ethtool_usrip6_spec,
    ether_spec: ethhdr,
    hdata: [u8; 52],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_485 {
    data: *const u8,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_flow_ext {
    padding: [u8; 2],
    h_dest: [u8; 6],
    vlan_etype: u16,
    vlan_tci: u16,
    data: [u32; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rx_flow_spec {
    flow_type: u32,
    h_u: ethtool_flow_union,
    h_ext: ethtool_flow_ext,
    m_u: ethtool_flow_union,
    m_ext: ethtool_flow_ext,
    ring_cookie: u64,
    location: u32,
}


// How rings are laid out when accessing virtual functions or
// offloaded queues is device specific. To allow users to do flow
// steering and specify these queues the ring cookie is partitioned
// into a 32bit queue index with an 8 bit virtual function id.
// This also leaves the 3bytes for further specifiers. It is possible
// future devices may support more than 256 virtual functions if
// devices start supporting PCIe w/ARI. However at the moment I
// do not know of any devices that support this so I do not reserve
// space for this at this time. If a future patch consumes the next
// byte it should be aware of this possibility.
#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rxnfc {
    cmd: u32,
    flow_type: u32,
    data: u64,
    fs: ethtool_rx_flow_spec,
    __slate_anon_4: __SlateRecord_ethtool_h_1509,
    rule_locs: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_flash {
    cmd: u32,
    region: u32,
    data: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_dump {
    cmd: u32,
    version: u32,
    flag: u32,
    len: u32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_893 {
    legacy: u32,
    ht_mcs: [u8; 10],
    vht_mcs: [u16; 8],
    he_mcs: [u16; 8],
    eht_mcs: [u16; 16],
    gi: u32,
    he_gi: u32,
    eht_gi: u32,
    he_ltf: u32,
    eht_ltf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fecparam {
    cmd: u32,
    active_fec: u32,
    fec: u32,
    reserved: u32,
}


// Duplex, half or full.
#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_link_settings {
    cmd: u32,
    speed: u32,
    duplex: u8,
    port: u8,
    phy_address: u8,
    autoneg: u8,
    mdio_support: u8,
    eth_tp_mdix: u8,
    eth_tp_mdix_ctrl: u8,
    link_mode_masks_nwords: i8,
    transceiver: u8,
    master_slave_cfg: u8,
    master_slave_state: u8,
    rate_matching: u8,
    reserved: [u32; 7],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_ethtool_ringparam {
    rx_buf_len: u32,
    tcp_data_split: u8,
    tx_push: u8,
    rx_push: u8,
    cqe_size: u32,
    tx_push_buf_len: u32,
    tx_push_buf_max_len: u32,
    hds_thresh: u32,
    hds_thresh_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_link_ext_state_info {
    link_ext_state: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_138,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_link_ext_stats {
    link_down_events: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_1421 {
    data: *const u8,
    len: u64,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ethtool_rxfh_context {
    indir_size: u32,
    key_size: u32,
    indir_user_size: u32,
    priv_size: u16,
    hfunc: u8,
    input_xfrm: u8,
    __slate_bits_0: __slate_bits::__SlateBits1445U0,
    key_off: u32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_1438 {
    data: *const u8,
    len: u64,
}


// declare a link mode bitmap
#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_link_ksettings {
    base: ethtool_link_settings,
    link_modes: __SlateRecord_ethtool_h_275,
    lanes: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_keee {
    supported: [u64; 2],
    advertised: [u64; 2],
    lp_advertised: [u64; 2],
    tx_lpi_timer: u32,
    tx_lpi_enabled: bool,
    eee_active: bool,
    eee_enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_ethtool_coalesce {
    use_cqe_mode_tx: u8,
    use_cqe_mode_rx: u8,
    tx_aggr_max_bytes: u32,
    tx_aggr_max_frames: u32,
    tx_aggr_time_usecs: u32,
    rx_cqe_frames: u32,
    rx_cqe_nsecs: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_eth_mac_stats {
    src: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_445,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_eth_phy_stats {
    src: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_476,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_eth_ctrl_stats {
    src: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_509,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_pause_stats {
    src: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_535,
}


// IEEE 802.3ck/df defines 16 bins for FEC histogram plus one more for
// the end-of-list marker, total 17 items
#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fec_hist_range {
    low: u16,
    high: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fec_hist {
    values: [ethtool_fec_hist_value; 17],
    ranges: *const ethtool_fec_hist_range,
    ranges_buf: [ethtool_fec_hist_range; 17],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_fec_stats {
    corrected_blocks: ethtool_fec_stat,
    uncorrectable_blocks: ethtool_fec_stat,
    corrected_bits: ethtool_fec_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rmon_hist_range {
    low: u16,
    high: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rmon_stats {
    src: u32,
    __slate_anon_1: __SlateRecord_ethtool_h_632,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_ts_stats {
    __slate_anon_0: __SlateRecord_ethtool_h_662,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_module_eeprom {
    offset: u32,
    length: u32,
    page: u8,
    bank: u8,
    i2c_address: u8,
    data: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_module_power_mode_params {
    policy: u32,
    mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_mm_state {
    verify_time: u32,
    max_verify_time: u32,
    verify_status: u32,
    tx_enabled: bool,
    tx_active: bool,
    pmac_enabled: bool,
    verify_enabled: bool,
    tx_min_frag_size: u32,
    rx_min_frag_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_mm_cfg {
    verify_time: u32,
    verify_enabled: bool,
    tx_enabled: bool,
    pmac_enabled: bool,
    tx_min_frag_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_mm_stats {
    MACMergeFrameAssErrorCount: u64,
    MACMergeFrameSmdErrorCount: u64,
    MACMergeFrameAssOkCount: u64,
    MACMergeFragCountRx: u64,
    MACMergeFragCountTx: u64,
    MACMergeHoldCount: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rxfh_param {
    hfunc: u8,
    indir_size: u32,
    indir: *mut u32,
    key_size: u32,
    key: *mut u8,
    rss_context: u32,
    rss_delete: u8,
    input_xfrm: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_rxfh_fields {
    data: u32,
    flow_type: u32,
    rss_context: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_ethtool_ts_info {
    cmd: u32,
    so_timestamping: u32,
    phc_index: i32,
    phc_qualifier: u32,
    phc_source: u32,
    phc_phyindex: i32,
    tx_types: u32,
    rx_filters: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_ops {
    __slate_bits_0: __slate_bits::__SlateBits1177U0,
    rxfh_indir_space: u32,
    rxfh_key_space: u16,
    rxfh_priv_size: u16,
    rxfh_max_num_contexts: u32,
    supported_coalesce_params: u32,
    supported_ring_params: u32,
    supported_hwtstamp_qualifiers: u32,
    op_needs_rtnl: u32,
    get_drvinfo: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_drvinfo)>,
    get_regs_len: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    get_regs: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_regs, *mut ())>,
    get_wol: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_wolinfo)>,
    set_wol: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_wolinfo) -> i32>,
    get_msglevel: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    set_msglevel: Option<unsafe extern "C" fn(*mut net_device, u32)>,
    nway_reset: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    get_link: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    get_link_ext_state: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_link_ext_state_info) -> i32>,
    get_link_ext_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_link_ext_stats)>,
    get_eeprom_len: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    get_eeprom: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eeprom, *mut u8) -> i32>,
    set_eeprom: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eeprom, *mut u8) -> i32>,
    get_coalesce: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_coalesce, *mut kernel_ethtool_coalesce, *mut netlink_ext_ack) -> i32>,
    set_coalesce: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_coalesce, *mut kernel_ethtool_coalesce, *mut netlink_ext_ack) -> i32>,
    get_ringparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_ringparam, *mut kernel_ethtool_ringparam, *mut netlink_ext_ack)>,
    set_ringparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_ringparam, *mut kernel_ethtool_ringparam, *mut netlink_ext_ack) -> i32>,
    get_pause_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_pause_stats)>,
    get_pauseparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_pauseparam)>,
    set_pauseparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_pauseparam) -> i32>,
    self_test: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_test, *mut u64)>,
    get_strings: Option<unsafe extern "C" fn(*mut net_device, u32, *mut u8)>,
    set_phys_id: Option<unsafe extern "C" fn(*mut net_device, u32) -> i32>,
    get_ethtool_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_stats, *mut u64)>,
    begin: Option<unsafe extern "C" fn(*mut net_device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut net_device)>,
    get_priv_flags: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    set_priv_flags: Option<unsafe extern "C" fn(*mut net_device, u32) -> i32>,
    get_sset_count: Option<unsafe extern "C" fn(*mut net_device, i32) -> i32>,
    get_rxnfc: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxnfc, *mut u32) -> i32>,
    set_rxnfc: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxnfc) -> i32>,
    flash_device: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_flash) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut net_device, *mut u32) -> i32>,
    get_rx_ring_count: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    get_rxfh_key_size: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    get_rxfh_indir_size: Option<unsafe extern "C" fn(*mut net_device) -> u32>,
    get_rxfh: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_param) -> i32>,
    set_rxfh: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_param, *mut netlink_ext_ack) -> i32>,
    get_rxfh_fields: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_fields) -> i32>,
    set_rxfh_fields: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_rxfh_fields, *mut netlink_ext_ack) -> i32>,
    create_rxfh_context: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_context, *const ethtool_rxfh_param, *mut netlink_ext_ack) -> i32>,
    modify_rxfh_context: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_context, *const ethtool_rxfh_param, *mut netlink_ext_ack) -> i32>,
    remove_rxfh_context: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rxfh_context, u32, *mut netlink_ext_ack) -> i32>,
    get_channels: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_channels)>,
    set_channels: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_channels) -> i32>,
    get_dump_flag: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_dump) -> i32>,
    get_dump_data: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_dump, *mut ()) -> i32>,
    set_dump: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_dump) -> i32>,
    get_ts_info: Option<unsafe extern "C" fn(*mut net_device, *mut kernel_ethtool_ts_info) -> i32>,
    get_ts_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_ts_stats)>,
    get_module_info: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_modinfo) -> i32>,
    get_module_eeprom: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eeprom, *mut u8) -> i32>,
    get_eee: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_keee) -> i32>,
    set_eee: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_keee) -> i32>,
    get_tunable: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_tunable, *mut ()) -> i32>,
    set_tunable: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_tunable, *const ()) -> i32>,
    get_per_queue_coalesce: Option<unsafe extern "C" fn(*mut net_device, u32, *mut ethtool_coalesce) -> i32>,
    set_per_queue_coalesce: Option<unsafe extern "C" fn(*mut net_device, u32, *mut ethtool_coalesce) -> i32>,
    get_link_ksettings: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_link_ksettings) -> i32>,
    set_link_ksettings: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_link_ksettings) -> i32>,
    get_fec_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_fec_stats, *mut ethtool_fec_hist)>,
    get_fecparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_fecparam) -> i32>,
    set_fecparam: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_fecparam) -> i32>,
    get_ethtool_phy_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_stats, *mut u64)>,
    get_phy_tunable: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_tunable, *mut ()) -> i32>,
    set_phy_tunable: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_tunable, *const ()) -> i32>,
    get_module_eeprom_by_page: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_module_eeprom, *mut netlink_ext_ack) -> i32>,
    set_module_eeprom_by_page: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_module_eeprom, *mut netlink_ext_ack) -> i32>,
    get_eth_phy_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eth_phy_stats)>,
    get_eth_mac_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eth_mac_stats)>,
    get_eth_ctrl_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_eth_ctrl_stats)>,
    get_rmon_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_rmon_stats, *mut *const ethtool_rmon_hist_range)>,
    get_module_power_mode: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_module_power_mode_params, *mut netlink_ext_ack) -> i32>,
    set_module_power_mode: Option<unsafe extern "C" fn(*mut net_device, *const ethtool_module_power_mode_params, *mut netlink_ext_ack) -> i32>,
    get_mm: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_mm_state) -> i32>,
    set_mm: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_mm_cfg, *mut netlink_ext_ack) -> i32>,
    get_mm_stats: Option<unsafe extern "C" fn(*mut net_device, *mut ethtool_mm_stats)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ethtool_netdev_state {
    rss_ctx: xarray,
    rss_lock: mutex,
    rss_indir_user_size: u32,
    __slate_bits_0: __slate_bits::__SlateBits1189U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct nl80211_sta_flag_update {
    mask: u32,
    set: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nl80211_wowlan_tcp_data_seq {
    start: u32,
    offset: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nl80211_wowlan_tcp_data_token {
    offset: u32,
    len: u32,
    token_stream: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nl80211_wowlan_tcp_data_token_feature {
    min_len: u32,
    max_len: u32,
    bufsize: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nl80211_vendor_cmd_info {
    vendor_id: u32,
    subcmd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_hw_trigger_type {
    dummy: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_classdev {
    name: *const u8,
    brightness: u32,
    max_brightness: u32,
    color: u32,
    flags: i32,
    work_flags: u64,
    brightness_set: Option<unsafe extern "C" fn(*mut led_classdev, u32)>,
    brightness_set_blocking: Option<unsafe extern "C" fn(*mut led_classdev, u32) -> i32>,
    brightness_get: Option<unsafe extern "C" fn(*mut led_classdev) -> u32>,
    blink_set: Option<unsafe extern "C" fn(*mut led_classdev, *mut u64, *mut u64) -> i32>,
    pattern_set: Option<unsafe extern "C" fn(*mut led_classdev, *mut led_pattern, u32, i32) -> i32>,
    pattern_clear: Option<unsafe extern "C" fn(*mut led_classdev) -> i32>,
    dev: *mut device,
    groups: *mut *const attribute_group,
    node: list_head,
    default_trigger: *const u8,
    blink_delay_on: u64,
    blink_delay_off: u64,
    blink_timer: timer_list,
    blink_brightness: i32,
    new_blink_brightness: i32,
    flash_resume: Option<unsafe extern "C" fn(*mut led_classdev)>,
    wq: *mut workqueue_struct,
    set_brightness_work: work_struct,
    delayed_set_value: i32,
    delayed_delay_on: u64,
    delayed_delay_off: u64,
    trigger_lock: rw_semaphore,
    trigger: *mut led_trigger,
    trig_list: list_head,
    trigger_data: *mut (),
    activated: bool,
    trigger_type: *mut led_hw_trigger_type,
    hw_control_trigger: *const u8,
    hw_control_is_supported: Option<unsafe extern "C" fn(*mut led_classdev, u64) -> i32>,
    hw_control_set: Option<unsafe extern "C" fn(*mut led_classdev, u64) -> i32>,
    hw_control_get: Option<unsafe extern "C" fn(*mut led_classdev, *mut u64) -> i32>,
    hw_control_get_device: Option<unsafe extern "C" fn(*mut led_classdev) -> *mut device>,
    led_access: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_trigger {
    name: *const u8,
    activate: Option<unsafe extern "C" fn(*mut led_classdev) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut led_classdev)>,
    brightness: u32,
    trigger_type: *mut led_hw_trigger_type,
    leddev_list_lock: spinlock,
    led_cdevs: list_head,
    next_trig: list_head,
    groups: *mut *const attribute_group,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_pattern {
    delta_t: u32,
    brightness: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulatory_request {
    callback_head: callback_head,
    wiphy_idx: i32,
    initiator: u32,
    user_reg_hint_type: u32,
    alpha2: [u8; 3],
    dfs_region: u32,
    intersect: bool,
    processed: bool,
    country_ie_env: u32,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_freq_range {
    start_freq_khz: u32,
    end_freq_khz: u32,
    max_bandwidth_khz: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_power_rule {
    max_antenna_gain: u32,
    max_eirp: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_wmm_ac {
    cw_min: u16,
    cw_max: u16,
    cot: u16,
    aifsn: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_wmm_rule {
    client: [ieee80211_wmm_ac; 4],
    ap: [ieee80211_wmm_ac; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_reg_rule {
    freq_range: ieee80211_freq_range,
    power_rule: ieee80211_power_rule,
    wmm_rule: ieee80211_wmm_rule,
    flags: u32,
    dfs_cac_ms: u32,
    has_wmm: bool,
    psd: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_regdomain {
    callback_head: callback_head,
    n_reg_rules: u32,
    alpha2: [u8; 3],
    dfs_region: u32,
    reg_rules: [ieee80211_reg_rule; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_channel {
    band: u32,
    center_freq: u32,
    freq_offset: u16,
    hw_value: u16,
    flags: u32,
    max_antenna_gain: i32,
    max_power: i32,
    max_reg_power: i32,
    beacon_found: bool,
    orig_flags: u32,
    orig_mag: i32,
    orig_mpwr: i32,
    dfs_state: u32,
    dfs_state_entered: u64,
    dfs_cac_ms: u32,
    cac_start_time: u64,
    psd: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_rate {
    flags: u32,
    bitrate: u16,
    hw_value: u16,
    hw_value_short: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_he_obss_pd {
    enable: bool,
    sr_ctrl: u8,
    non_srg_max_offset: u8,
    min_offset: u8,
    max_offset: u8,
    bss_color_bitmap: [u8; 8],
    partial_bssid_bitmap: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_he_bss_color {
    color: u8,
    enabled: bool,
    partial: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_ht_cap {
    cap: u16,
    ht_supported: bool,
    ampdu_factor: u8,
    ampdu_density: u8,
    mcs: ieee80211_mcs_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_vht_cap {
    vht_supported: bool,
    cap: u32,
    vht_mcs: ieee80211_vht_mcs_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_he_cap {
    has_he: bool,
    he_cap_elem: ieee80211_he_cap_elem,
    he_mcs_nss_supp: ieee80211_he_mcs_nss_supp,
    ppe_thres: [u8; 25],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eht_mcs_nss_supp {
    __slate_anon_0: __SlateRecord_cfg80211_h_407,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_eht_cap {
    has_eht: bool,
    eht_cap_elem: ieee80211_eht_cap_elem_fixed,
    eht_mcs_nss_supp: ieee80211_eht_mcs_nss_supp,
    eht_ppe_thres: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_uhr_cap {
    has_uhr: bool,
    mac: ieee80211_uhr_cap_mac,
    phy: ieee80211_uhr_cap_phy,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6151 {
    preambles: u32,
    bandwidths: u32,
    max_bursts_exponent: i8,
    max_ftms_per_burst: u8,
    __slate_bits_0: __slate_bits::__SlateBits1608U0,
    max_tx_ltf_rep: u8,
    max_rx_ltf_rep: u8,
    max_tx_sts: u8,
    max_rx_sts: u8,
    max_total_ltf_tx: u8,
    max_total_ltf_rx: u8,
    ista: __SlateRecord_cfg80211_h_6170,
    rsta: __SlateRecord_cfg80211_h_6176,
    max_no_of_tx_antennas: u8,
    max_no_of_rx_antennas: u8,
    min_allowed_ranging_interval_edca: u32,
    min_allowed_ranging_interval_ntb: u32,
    r#type: __SlateRecord_cfg80211_h_6186,
    __slate_bits_1: __slate_bits::__SlateBits1608U1,
    pd_preambles: u32,
    pd_bandwidths: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6170 {
    __slate_bits_0: __slate_bits::__SlateBits1609U0,
    max_peers: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6176 {
    __slate_bits_0: __slate_bits::__SlateBits1610U0,
    max_peers: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6186 {
    __slate_bits_0: __slate_bits::__SlateBits1611U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sband_iftype_data {
    types_mask: u16,
    he_cap: ieee80211_sta_he_cap,
    he_6ghz_capa: ieee80211_he_6ghz_capa,
    eht_cap: ieee80211_sta_eht_cap,
    uhr_cap: ieee80211_sta_uhr_cap,
    vendor_elems: __SlateRecord_cfg80211_h_485,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_edmg {
    channels: u8,
    bw_config: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_s1g_cap {
    s1g: bool,
    cap: [u8; 10],
    nss_mcs: [u8; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_supported_band {
    channels: *mut ieee80211_channel,
    bitrates: *mut ieee80211_rate,
    band: u32,
    n_channels: i32,
    n_bitrates: i32,
    ht_cap: ieee80211_sta_ht_cap,
    vht_cap: ieee80211_sta_vht_cap,
    s1g_cap: ieee80211_sta_s1g_cap,
    edmg_cap: ieee80211_edmg,
    n_iftype_data: u16,
    iftype_data: *const ieee80211_sband_iftype_data,
}


// for_each_sband_iftype_data - iterate sband iftype data entries
// @sband: the sband whose iftype_data array to iterate
// @i: iterator counter
// @iftd: iftype data pointer to set
#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_chan_def {
    chan: *mut ieee80211_channel,
    npca_chan: *mut ieee80211_channel,
    width: u32,
    center_freq1: u32,
    center_freq2: u32,
    edmg: ieee80211_edmg,
    freq1_offset: u16,
    punctured: u16,
    npca_punctured: u16,
    s1g_primary_2mhz: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6303 {
    ht: ieee80211_sta_ht_cap,
    vht: ieee80211_sta_vht_cap,
    he: ieee80211_sta_he_cap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_6678 {
    peer: u64,
    vif: u64,
    max_retry: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_conn {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_internal_bss {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_cached_keys {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_cqm_config {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_bitrate_mask {
    control: [__SlateRecord_cfg80211_h_893; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_tid_cfg {
    config_override: bool,
    tids: u8,
    mask: u64,
    noack: u32,
    retry_long: u8,
    retry_short: u8,
    ampdu: u32,
    rtscts: u32,
    amsdu: u32,
    txrate_type: u32,
    txrate_mask: cfg80211_bitrate_mask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_tid_config {
    peer: *const u8,
    n_tid_conf: u32,
    tid_conf: [cfg80211_tid_cfg; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cfg80211_h_7294 {
    client: __SlateRecord_cfg80211_h_7295,
    mesh: __SlateRecord_cfg80211_h_7300,
    ap: __SlateRecord_cfg80211_h_7307,
    ibss: __SlateRecord_cfg80211_h_7312,
    ocb: __SlateRecord_cfg80211_h_7319,
    nan: __SlateRecord_cfg80211_h_7322,
}


#[repr(C, align(2))]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7295 {
    connected_addr: [u8; 6],
    ssid: [u8; 32],
    ssid_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7300 {
    beacon_interval: i32,
    preset_chandef: cfg80211_chan_def,
    chandef: cfg80211_chan_def,
    id: [u8; 32],
    id_len: u8,
    id_up_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7307 {
    preset_chandef: cfg80211_chan_def,
    ssid: [u8; 32],
    ssid_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7312 {
    current_bss: *mut cfg80211_internal_bss,
    chandef: cfg80211_chan_def,
    beacon_interval: i32,
    ssid: [u8; 32],
    ssid_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7319 {
    chandef: cfg80211_chan_def,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7322 {
    cluster_id: [u8; 6],
    n_channels: u8,
    chandefs: *mut cfg80211_chan_def,
    sched_update_pending: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7330 {
    addr: [u8; 6],
    __slate_anon_1: __SlateRecord_cfg80211_h_7332,
    cac_started: bool,
    cac_start_time: u64,
    cac_time_ms: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cfg80211_h_7332 {
    ap: __SlateRecord_cfg80211_h_7333,
    client: __SlateRecord_cfg80211_h_7337,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7333 {
    beacon_interval: u32,
    chandef: cfg80211_chan_def,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cfg80211_h_7337 {
    current_bss: *mut cfg80211_internal_bss,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_set_hw_timestamp {
    macaddr: *const u8,
    enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct survey_info {
    channel: *mut ieee80211_channel,
    time: u64,
    time_busy: u64,
    time_ext_busy: u64,
    time_rx: u64,
    time_tx: u64,
    time_scan: u64,
    time_bss_rx: u64,
    filled: u32,
    noise: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_mbssid_elems {
    cnt: u8,
    elem: [__SlateRecord_cfg80211_h_1421; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_rnr_elems {
    cnt: u8,
    elem: [__SlateRecord_cfg80211_h_1438; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_beacon_data {
    link_id: u32,
    head: *const u8,
    tail: *const u8,
    beacon_ies: *const u8,
    proberesp_ies: *const u8,
    assocresp_ies: *const u8,
    probe_resp: *const u8,
    lci: *const u8,
    civicloc: *const u8,
    mbssid_ies: *mut cfg80211_mbssid_elems,
    rnr_ies: *mut cfg80211_rnr_elems,
    ftm_responder: i8,
    head_len: u64,
    tail_len: u64,
    beacon_ies_len: u64,
    proberesp_ies_len: u64,
    assocresp_ies_len: u64,
    probe_resp_len: u64,
    lci_len: u64,
    civicloc_len: u64,
    he_bss_color: cfg80211_he_bss_color,
    he_bss_color_valid: bool,
    ht_required: bool,
    vht_required: bool,
    ht_oper: *const ieee80211_ht_operation,
    vht_oper: *const ieee80211_vht_operation,
    he_oper: *const ieee80211_he_operation,
    eht_oper: *const ieee80211_eht_operation,
    uhr_oper: *const ieee80211_uhr_operation,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mac_address {
    addr: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_unsol_bcast_probe_resp {
    update: bool,
    interval: u32,
    tmpl_len: u64,
    tmpl: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_csa_settings {
    chandef: cfg80211_chan_def,
    beacon_csa: cfg80211_beacon_data,
    counter_offsets_beacon: *const u16,
    counter_offsets_presp: *const u16,
    n_counter_offsets_beacon: u32,
    n_counter_offsets_presp: u32,
    beacon_after: cfg80211_beacon_data,
    unsol_bcast_probe_resp: cfg80211_unsol_bcast_probe_resp,
    radar_required: bool,
    block_tx: bool,
    count: u8,
    link_id: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rate_info {
    flags: u16,
    legacy: u16,
    mcs: u8,
    nss: u8,
    bw: u8,
    he_gi: u8,
    he_dcm: u8,
    he_ru_alloc: u8,
    n_bonded_ch: u8,
    eht_gi: u8,
    eht_ru_alloc: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sta_bss_parameters {
    flags: u8,
    dtim_period: u8,
    beacon_interval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_txq_stats {
    filled: u32,
    backlog_bytes: u32,
    backlog_packets: u32,
    flows: u32,
    drops: u32,
    ecn_marks: u32,
    overlimit: u32,
    overmemory: u32,
    collisions: u32,
    tx_bytes: u32,
    tx_packets: u32,
    max_flows: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_tid_stats {
    filled: u32,
    rx_msdu: u64,
    tx_msdu: u64,
    tx_msdu_retries: u64,
    tx_msdu_failed: u64,
    txq_stats: cfg80211_txq_stats,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct link_station_info {
    filled: u64,
    connected_time: u32,
    inactive_time: u32,
    assoc_at: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    signal: i8,
    signal_avg: i8,
    chains: u8,
    chain_signal: [i8; 4],
    chain_signal_avg: [i8; 4],
    txrate: rate_info,
    rxrate: rate_info,
    rx_packets: u32,
    tx_packets: u32,
    tx_retries: u32,
    tx_failed: u32,
    rx_dropped_misc: u32,
    bss_param: sta_bss_parameters,
    beacon_loss_count: u32,
    expected_throughput: u32,
    tx_duration: u64,
    rx_duration: u64,
    rx_beacon: u64,
    rx_beacon_signal_avg: u8,
    airtime_weight: u16,
    ack_signal: i8,
    avg_ack_signal: i8,
    pertid: *mut cfg80211_tid_stats,
    rx_mpdu_count: u32,
    fcs_err_count: u32,
    addr: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct station_info {
    filled: u64,
    connected_time: u32,
    inactive_time: u32,
    assoc_at: u64,
    rx_bytes: u64,
    tx_bytes: u64,
    signal: i8,
    signal_avg: i8,
    chains: u8,
    chain_signal: [i8; 4],
    chain_signal_avg: [i8; 4],
    txrate: rate_info,
    rxrate: rate_info,
    rx_packets: u32,
    tx_packets: u32,
    tx_retries: u32,
    tx_failed: u32,
    rx_dropped_misc: u32,
    bss_param: sta_bss_parameters,
    sta_flags: nl80211_sta_flag_update,
    generation: i32,
    beacon_loss_count: u32,
    assoc_req_ies: *const u8,
    assoc_req_ies_len: u64,
    t_offset: i64,
    llid: u16,
    plid: u16,
    plink_state: u8,
    connected_to_gate: u8,
    connected_to_as: u8,
    airtime_link_metric: u32,
    local_pm: u32,
    peer_pm: u32,
    nonpeer_pm: u32,
    expected_throughput: u32,
    airtime_weight: u16,
    ack_signal: i8,
    avg_ack_signal: i8,
    pertid: *mut cfg80211_tid_stats,
    tx_duration: u64,
    rx_duration: u64,
    rx_beacon: u64,
    rx_beacon_signal_avg: u8,
    rx_mpdu_count: u32,
    fcs_err_count: u32,
    mlo_params_valid: bool,
    assoc_link_id: u8,
    mld_addr: [u8; 6],
    assoc_resp_ies: *const u8,
    assoc_resp_ies_len: u64,
    valid_links: u16,
    links: [*mut link_station_info; 15],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mac80211_h_521 {
    rssi: ieee80211_rssi_event,
    mlme: ieee80211_mlme_event,
    ba: ieee80211_ba_event,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sar_sub_specs {
    power: i32,
    freq_range_index: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sar_specs {
    r#type: u32,
    num_sub_specs: u32,
    sub_specs: [cfg80211_sar_sub_specs; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sar_freq_ranges {
    start_freq: u32,
    end_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sar_capa {
    r#type: u32,
    num_freq_ranges: u32,
    freq_ranges: *const cfg80211_sar_freq_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_config {
    dot11MeshRetryTimeout: u16,
    dot11MeshConfirmTimeout: u16,
    dot11MeshHoldingTimeout: u16,
    dot11MeshMaxPeerLinks: u16,
    dot11MeshMaxRetries: u8,
    dot11MeshTTL: u8,
    element_ttl: u8,
    auto_open_plinks: bool,
    dot11MeshNbrOffsetMaxNeighbor: u32,
    dot11MeshHWMPmaxPREQretries: u8,
    path_refresh_time: u32,
    min_discovery_timeout: u16,
    dot11MeshHWMPactivePathTimeout: u32,
    dot11MeshHWMPpreqMinInterval: u16,
    dot11MeshHWMPperrMinInterval: u16,
    dot11MeshHWMPnetDiameterTraversalTime: u16,
    dot11MeshHWMPRootMode: u8,
    dot11MeshConnectedToMeshGate: bool,
    dot11MeshConnectedToAuthServer: bool,
    dot11MeshHWMPRannInterval: u16,
    dot11MeshGateAnnouncementProtocol: bool,
    dot11MeshForwarding: bool,
    rssi_threshold: i32,
    ht_opmode: u16,
    dot11MeshHWMPactivePathToRootTimeout: u32,
    dot11MeshHWMProotInterval: u16,
    dot11MeshHWMPconfirmationInterval: u16,
    power_mode: u32,
    dot11MeshAwakeWindowDuration: u16,
    plink_timeout: u32,
    dot11MeshNolearn: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_ssid {
    ssid: [u8; 32],
    ssid_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_scan_info {
    scan_start_tsf: u64,
    tsf_bssid: [u8; 6],
    aborted: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_856 {
    params: u32,
    nss_set: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_scan_6ghz_params {
    short_ssid: u32,
    channel_idx: u32,
    bssid: [u8; 6],
    unsolicited_probe: bool,
    short_ssid_valid: bool,
    psc_no_listen: bool,
    psd_20: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_scan_request {
    ssids: *mut cfg80211_ssid,
    n_ssids: i32,
    n_channels: u32,
    ie: *const u8,
    ie_len: u64,
    duration: u16,
    duration_mandatory: bool,
    flags: u32,
    rates: [u32; 6],
    wdev: *mut wireless_dev,
    mac_addr: [u8; 6],
    mac_addr_mask: [u8; 6],
    bssid: [u8; 6],
    wiphy: *mut wiphy,
    scan_start: u64,
    no_cck: bool,
    scan_6ghz: bool,
    first_part: bool,
    n_6ghz_params: u32,
    scan_6ghz_params: *mut cfg80211_scan_6ghz_params,
    tsf_report_link_id: i8,
    channels: [*mut ieee80211_channel; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_match_set {
    ssid: cfg80211_ssid,
    bssid: [u8; 6],
    rssi_thold: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sched_scan_plan {
    interval: u32,
    iterations: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_bss_select_adjust {
    band: u32,
    delta: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_sched_scan_request {
    reqid: u64,
    ssids: *mut cfg80211_ssid,
    n_ssids: i32,
    n_channels: u32,
    ie: *const u8,
    ie_len: u64,
    flags: u32,
    match_sets: *mut cfg80211_match_set,
    n_match_sets: i32,
    min_rssi_thold: i32,
    delay: u32,
    scan_plans: *mut cfg80211_sched_scan_plan,
    n_scan_plans: i32,
    mac_addr: [u8; 6],
    mac_addr_mask: [u8; 6],
    relative_rssi_set: bool,
    relative_rssi: i8,
    rssi_adjust: cfg80211_bss_select_adjust,
    wiphy: *mut wiphy,
    dev: *mut net_device,
    scan_start: u64,
    report_results: bool,
    callback_head: callback_head,
    owner_nlportid: u32,
    nl_owner_dead: bool,
    list: list_head,
    channels: [*mut ieee80211_channel; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_bss_ies {
    tsf: u64,
    callback_head: callback_head,
    len: i32,
    from_beacon: bool,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mac80211_h_1780 {
    boottime_ns: u64,
    ack_tx_hwtstamp: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mac80211_h_1790 {
    __slate_anon_0: __SlateRecord_mac80211_h_1791,
    eht: __SlateRecord_mac80211_h_1796,
    uhr: __SlateRecord_mac80211_h_1800,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_1791 {
    __slate_bits_0: __slate_bits::__SlateBits1686U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_1796 {
    __slate_bits_0: __slate_bits::__SlateBits1687U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_1800 {
    __slate_bits_0: __slate_bits::__SlateBits1688U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_bss {
    channel: *mut ieee80211_channel,
    ies: *const cfg80211_bss_ies,
    beacon_ies: *const cfg80211_bss_ies,
    proberesp_ies: *const cfg80211_bss_ies,
    hidden_beacon_bss: *mut cfg80211_bss,
    transmitted_bss: *mut cfg80211_bss,
    nontrans_list: list_head,
    signal: i32,
    ts_boottime: u64,
    beacon_interval: u16,
    capability: u16,
    bssid: [u8; 6],
    chains: u8,
    chain_signal: [i8; 4],
    __slate_bits_0: __slate_bits::__SlateBits1583U0,
    bssid_index: u8,
    max_bssid_indicator: u8,
    use_for: u8,
    cannot_use_reasons: u8,
    __slate_pad_19: [u8; 4],
    r#priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_pkt_pattern {
    mask: *const u8,
    pattern: *const u8,
    pattern_len: i32,
    pkt_offset: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_wowlan_tcp {
    sock: *mut socket,
    src: u32,
    dst: u32,
    src_port: u16,
    dst_port: u16,
    dst_mac: [u8; 6],
    payload_len: i32,
    payload: *const u8,
    payload_seq: nl80211_wowlan_tcp_data_seq,
    data_interval: u32,
    wake_len: u32,
    wake_data: *const u8,
    wake_mask: *const u8,
    tokens_size: u32,
    payload_tok: nl80211_wowlan_tcp_data_token,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_wowlan {
    any: bool,
    disconnect: bool,
    magic_pkt: bool,
    gtk_rekey_failure: bool,
    eap_identity_req: bool,
    four_way_handshake: bool,
    rfkill_release: bool,
    patterns: *mut cfg80211_pkt_pattern,
    tcp: *mut cfg80211_wowlan_tcp,
    n_patterns: i32,
    nd_config: *mut cfg80211_sched_scan_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_gtk_rekey_data {
    kek: *const u8,
    kck: *const u8,
    replay_ctr: *const u8,
    akm: u32,
    kek_len: u8,
    kck_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_dscp_exception {
    dscp: u8,
    up: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_dscp_range {
    low: u8,
    high: u8,
}


// QoS Map Set element length defined in IEEE Std 802.11-2012, 8.4.2.97
#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_qos_map {
    num_des: u8,
    dscp_exception: [cfg80211_dscp_exception; 21],
    up: [cfg80211_dscp_range; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mac80211_h_2472 {
    tkip: __SlateRecord_mac80211_h_2473,
    ccmp: __SlateRecord_mac80211_h_2477,
    aes_cmac: __SlateRecord_mac80211_h_2480,
    aes_gmac: __SlateRecord_mac80211_h_2483,
    gcmp: __SlateRecord_mac80211_h_2486,
    hw: __SlateRecord_mac80211_h_2489,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2473 {
    iv32: u32,
    iv16: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2477 {
    pn: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2480 {
    pn: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2483 {
    pn: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2486 {
    pn: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2489 {
    seq: [u8; 16],
    seq_len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_nan_band_config {
    chan: *mut ieee80211_channel,
    rssi_close: i8,
    rssi_middle: i8,
    awake_dw_interval: u8,
    disable_scan: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_2559 {
    idx: i8,
    count: u8,
    count_cts: u8,
    count_rts: u8,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_nan_conf {
    master_pref: u8,
    bands: u8,
    cluster_id: [u8; 6],
    scan_period: u16,
    scan_dwell_time: u16,
    discovery_beacon_interval: u8,
    enable_dw_notification: bool,
    band_cfgs: [cfg80211_nan_band_config; 6],
    extra_nan_attrs: *const u8,
    extra_nan_attrs_len: u16,
    vendor_elems: *const u8,
    vendor_elems_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_nan_func_filter {
    filter: *const u8,
    len: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_nan_func {
    r#type: u32,
    service_id: [u8; 6],
    publish_type: u8,
    close_range: bool,
    publish_bcast: bool,
    subscribe_active: bool,
    followup_id: u8,
    followup_reqid: u8,
    followup_dest: mac_address,
    ttl: u32,
    serv_spec_info: *const u8,
    serv_spec_info_len: u8,
    srf_include: bool,
    srf_bf: *const u8,
    srf_bf_len: u8,
    srf_bf_idx: u8,
    srf_macs: *mut mac_address,
    srf_num_macs: i32,
    rx_filters: *mut cfg80211_nan_func_filter,
    tx_filters: *mut cfg80211_nan_func_filter,
    num_tx_filters: u8,
    num_rx_filters: u8,
    instance_id: u8,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_ftm_responder_stats {
    filled: u32,
    success_num: u32,
    partial_num: u32,
    failed_num: u32,
    asap_num: u32,
    non_asap_num: u32,
    total_duration_ms: u64,
    unknown_triggers_num: u32,
    reschedule_requests_num: u32,
    out_of_window_triggers_num: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_pmsr_ftm_request_peer {
    preamble: u32,
    burst_period: u16,
    __slate_bits_0: __slate_bits::__SlateBits1596U0,
    num_bursts_exp: u8,
    burst_duration: u8,
    ftms_per_burst: u8,
    ftmr_retries: u8,
    bss_color: u8,
    request_type: u32,
    min_time_between_measurements: u32,
    max_time_between_measurements: u32,
    availability_window: u8,
    nominal_time: u32,
    num_measurements: u32,
    ingress_distance: u64,
    egress_distance: u64,
    __slate_bits_1: __slate_bits::__SlateBits1596U1,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mac80211_h_3305 {
    units_pos: i32,
    accuracy: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_pmsr_request_peer {
    addr: [u8; 6],
    chandef: cfg80211_chan_def,
    __slate_bits_0: __slate_bits::__SlateBits1597U0,
    ftm: cfg80211_pmsr_ftm_request_peer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_pmsr_request {
    cookie: u64,
    drv_data: *mut (),
    n_peers: u32,
    nl_portid: u32,
    timeout: u32,
    mac_addr: [u8; 6],
    mac_addr_mask: [u8; 6],
    list: list_head,
    peers: [cfg80211_pmsr_request_peer; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_iface_limit {
    max: u16,
    types: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_iface_combination {
    limits: *const ieee80211_iface_limit,
    num_different_channels: u32,
    max_interfaces: u16,
    n_limits: u8,
    beacon_int_infra_match: bool,
    radar_detect_widths: u8,
    radar_detect_regions: u8,
    beacon_int_min_gcd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_txrx_stypes {
    tx: u16,
    rx: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_wowlan_tcp_support {
    tok: *const nl80211_wowlan_tcp_data_token_feature,
    data_payload_max: u32,
    data_interval_max: u32,
    wake_payload_max: u32,
    seq: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_wowlan_support {
    flags: u32,
    n_patterns: i32,
    pattern_max_len: i32,
    pattern_min_len: i32,
    max_pkt_offset: i32,
    max_nd_match_sets: i32,
    tcp: *const wiphy_wowlan_tcp_support,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_coalesce_support {
    n_rules: i32,
    max_delay: i32,
    n_patterns: i32,
    pattern_max_len: i32,
    pattern_min_len: i32,
    max_pkt_offset: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_vendor_command {
    info: nl80211_vendor_cmd_info,
    flags: u32,
    doit: Option<unsafe extern "C" fn(*mut wiphy, *mut wireless_dev, *const (), i32) -> i32>,
    dumpit: Option<unsafe extern "C" fn(*mut wiphy, *mut wireless_dev, *mut sk_buff, *const (), i32, *mut u64) -> i32>,
    policy: *const nla_policy,
    maxattr: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_iftype_ext_capab {
    iftype: u32,
    extended_capabilities: *const u8,
    extended_capabilities_mask: *const u8,
    extended_capabilities_len: u8,
    eml_capabilities: u16,
    mld_capa_and_ops: u16,
    ext_mld_capa_and_ops: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfg80211_pmsr_capabilities {
    max_peers: u32,
    __slate_bits_0: __slate_bits::__SlateBits1607U0,
    ftm: __SlateRecord_cfg80211_h_6151,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_iftype_akm_suites {
    iftypes_mask: u16,
    akm_suites: *const u32,
    n_akm_suites: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_radio_cfg {
    rts_threshold: u32,
    radio_debugfsdir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_aes_cbc_macs_h_22 {
    b: [u8; 16],
    w: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_radio_freq_range {
    start_freq: u32,
    end_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_radio {
    freq_range: *const wiphy_radio_freq_range,
    n_freq_range: i32,
    iface_combinations: *const ieee80211_iface_combination,
    n_iface_combinations: i32,
    antenna_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_nan_capa {
    flags: u32,
    op_mode: u8,
    n_antennas: u8,
    max_channel_switch_time: u16,
    dev_capabilities: u8,
    phy: __SlateRecord_cfg80211_h_6303,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct wiphy {
    mtx: mutex,
    perm_addr: [u8; 6],
    addr_mask: [u8; 6],
    addresses: *mut mac_address,
    mgmt_stypes: *const ieee80211_txrx_stypes,
    iface_combinations: *const ieee80211_iface_combination,
    n_iface_combinations: i32,
    software_iftypes: u16,
    n_addresses: u16,
    interface_modes: u16,
    max_acl_mac_addrs: u16,
    flags: u32,
    regulatory_flags: u32,
    features: u32,
    ext_features: [u8; 10],
    ap_sme_capa: u32,
    signal_type: u32,
    bss_priv_size: i32,
    max_scan_ssids: u8,
    max_sched_scan_reqs: u8,
    max_sched_scan_ssids: u8,
    max_match_sets: u8,
    max_scan_ie_len: u16,
    max_sched_scan_ie_len: u16,
    max_sched_scan_plans: u32,
    max_sched_scan_plan_interval: u32,
    max_sched_scan_plan_iterations: u32,
    n_cipher_suites: i32,
    cipher_suites: *const u32,
    n_akm_suites: i32,
    akm_suites: *const u32,
    iftype_akm_suites: *const wiphy_iftype_akm_suites,
    num_iftype_akm_suites: u32,
    retry_short: u8,
    retry_long: u8,
    frag_threshold: u32,
    rts_threshold: u32,
    coverage_class: u8,
    fw_version: [u8; 32],
    hw_version: u32,
    wowlan: *const wiphy_wowlan_support,
    wowlan_config: *mut cfg80211_wowlan,
    max_remain_on_channel_duration: u16,
    max_num_pmkids: u8,
    available_antennas_tx: u32,
    available_antennas_rx: u32,
    probe_resp_offload: u32,
    extended_capabilities: *const u8,
    extended_capabilities_mask: *const u8,
    extended_capabilities_len: u8,
    iftype_ext_capab: *const wiphy_iftype_ext_capab,
    num_iftype_ext_capab: u32,
    privid: *const (),
    bands: [*mut ieee80211_supported_band; 6],
    reg_notifier: Option<unsafe extern "C" fn(*mut wiphy, *mut regulatory_request)>,
    radio_cfg: *mut wiphy_radio_cfg,
    regd: *const ieee80211_regdomain,
    dev: device,
    registered: bool,
    debugfsdir: *mut dentry,
    ht_capa_mod_mask: *const ieee80211_ht_cap,
    vht_capa_mod_mask: *const ieee80211_vht_cap,
    wdev_list: list_head,
    _net: __SlateRecord_net_namespace_h_410,
    coalesce: *const wiphy_coalesce_support,
    vendor_commands: *const wiphy_vendor_command,
    vendor_events: *const nl80211_vendor_cmd_info,
    n_vendor_commands: i32,
    n_vendor_events: i32,
    max_ap_assoc_sta: u16,
    max_num_csa_counters: u8,
    bss_param_support: u32,
    bss_select_support: u32,
    nan_supported_bands: u8,
    nan_capa: wiphy_nan_capa,
    txq_limit: u32,
    txq_memory_limit: u32,
    txq_quantum: u32,
    tx_queue_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits1527U0,
    pmsr_capa: *const cfg80211_pmsr_capabilities,
    tid_config_support: __SlateRecord_cfg80211_h_6678,
    max_data_retry_count: u8,
    sar_capa: *const cfg80211_sar_capa,
    rfkill: *mut rfkill,
    mbssid_max_interfaces: u8,
    ema_max_profile_periodicity: u8,
    max_num_akm_suites: u16,
    hw_timestamp_max_peers: u16,
    n_radio: i32,
    radio: *const wiphy_radio,
    __slate_pad_91: [u8; 16],
    r#priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_work {
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut wiphy, *mut wiphy_work)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_delayed_work {
    work: wiphy_work,
    wiphy: *mut wiphy,
    timer: timer_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wiphy_hrtimer_work {
    work: wiphy_work,
    wiphy: *mut wiphy,
    timer: hrtimer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wireless_dev {
    wiphy: *mut wiphy,
    iftype: u32,
    list: list_head,
    netdev: *mut net_device,
    identifier: u32,
    mgmt_registrations: list_head,
    __slate_bits_0: __slate_bits::__SlateBits1184U0,
    use_4addr: bool,
    is_running: bool,
    registered: bool,
    registering: bool,
    __slate_pad_11: [u8; 1],
    address: [u8; 6],
    conn: *mut cfg80211_conn,
    connect_keys: *mut cfg80211_cached_keys,
    conn_bss_type: u32,
    conn_owner_nlportid: u32,
    disconnect_wk: wiphy_work,
    disconnect_bssid: [u8; 6],
    event_list: list_head,
    event_lock: spinlock,
    __slate_bits_1: __slate_bits::__SlateBits1184U1,
    ps: bool,
    ps_timeout: i32,
    unexpected_nlportid: u32,
    owner_nlportid: u32,
    nl_owner_dead: bool,
    cqm_rssi_work: wiphy_work,
    cqm_config: *mut cfg80211_cqm_config,
    pmsr_list: list_head,
    pmsr_lock: spinlock,
    pmsr_free_wk: wiphy_work,
    unprot_beacon_reported: u64,
    u: __SlateRecord_cfg80211_h_7294,
    links: [__SlateRecord_cfg80211_h_7330; 15],
    valid_links: u16,
    radio_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_70 {
    tkip: __SlateRecord_key_h_71,
    ccmp: __SlateRecord_key_h_84,
    aes_cmac: __SlateRecord_key_h_95,
    aes_gmac: __SlateRecord_key_h_101,
    gcmp: __SlateRecord_key_h_107,
    r#gen: __SlateRecord_key_h_117_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_71 {
    txlock: spinlock,
    tx: tkip_ctx,
    rx: [tkip_ctx_rx; 16],
    mic_failures: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_84 {
    rx_pn: [[u8; 6]; 17],
    tfm: *mut crypto_aead,
    replays: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct crypto_aead {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_95 {
    rx_pn: [u8; 6],
    key: aes_cmac_key,
    replays: u32,
    icverrors: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_101 {
    rx_pn: [u8; 6],
    tfm: *mut crypto_aead,
    replays: u32,
    icverrors: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_107 {
    rx_pn: [[u8; 6]; 17],
    tfm: *mut crypto_aead,
    replays: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_117_2 {
    rx_pn: [[u8; 16]; 17],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codel_params {
    target: u32,
    ce_threshold: u32,
    interval: u32,
    mtu: u32,
    ecn: bool,
    ce_threshold_selector: u8,
    ce_threshold_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codel_vars {
    count: u32,
    lastcount: u32,
    dropping: bool,
    rec_inv_sqrt: u16,
    first_above_time: u32,
    drop_next: u32,
    ldelay: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tx_queue_params {
    txop: u16,
    cw_min: u16,
    cw_max: u16,
    aifs: u8,
    acm: bool,
    uapsd: bool,
    mu_edca: bool,
    mu_edca_param_rec: ieee80211_he_mu_edca_param_ac_rec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_low_level_stats {
    dot11ACKFailureCount: u32,
    dot11RTSFailureCount: u32,
    dot11FCSErrorCount: u32,
    dot11RTSSuccessCount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sta_info_h_259 {
    buf: sk_buff_head,
    time: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_chan_req {
    oper: cfg80211_chan_def,
    ap: cfg80211_chan_def,
    require_npca: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_chanctx_conf {
    def: cfg80211_chan_def,
    min_def: cfg80211_chan_def,
    ap: cfg80211_chan_def,
    radio_idx: i32,
    rx_chains_static: u8,
    rx_chains_dynamic: u8,
    radar_enabled: bool,
    __slate_pad_7: [u8; 1],
    drv_priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_vif_chanctx_switch {
    vif: *mut ieee80211_vif,
    link_conf: *mut ieee80211_bss_conf,
    old_ctx: *mut ieee80211_chanctx_conf,
    new_ctx: *mut ieee80211_chanctx_conf,
}


// The maximum number of IPv4 addresses listed for ARP filtering. If the number
// of addresses for an interface increase beyond this value, hardware ARP
// filtering will be disabled.
#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_rssi_event {
    data: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mlme_event {
    data: u32,
    status: u32,
    reason: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ba_event {
    sta: *mut ieee80211_sta,
    tid: u16,
    ssn: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_event {
    r#type: u32,
    u: __SlateRecord_mac80211_h_521,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mu_group_data {
    membership: [u8; 8],
    position: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sta_info_h_542 {
    signal: ewma_signal,
    chain_signal: [ewma_signal; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sta_info_h_548 {
    filtered: u64,
    retry_failed: u64,
    retry_count: u64,
    lost_packets: u32,
    last_pkt_time: u64,
    msdu_retries: [u64; 17],
    msdu_failed: [u64; 17],
    last_ack: u64,
    last_ack_signal: i8,
    ack_signal_filled: bool,
    avg_ack_signal: ewma_avg_signal,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sta_info_h_562 {
    packets: [u64; 4],
    bytes: [u64; 4],
    last_rate: ieee80211_tx_rate,
    last_rate_info: rate_info,
    msdu: [u64; 17],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ftm_responder_params {
    lci: *const u8,
    civicloc: *const u8,
    lci_len: u64,
    civicloc_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sta_info_h_629 {
    rx_msdu: u64,
    tx_msdu: u64,
    tx_msdu_retries: u64,
    tx_msdu_failed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rate_control_ref {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_fils_discovery {
    min_interval: u32,
    max_interval: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_parsed_tpe_eirp {
    valid: bool,
    power: [i8; 5],
    count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_parsed_tpe_psd {
    valid: bool,
    power: [i8; 16],
    count: u8,
    n: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_parsed_tpe {
    max_local: [ieee80211_parsed_tpe_eirp; 2],
    max_reg_client: [ieee80211_parsed_tpe_eirp; 2],
    psd_local: [ieee80211_parsed_tpe_psd; 2],
    psd_reg_client: [ieee80211_parsed_tpe_psd; 2],
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct ieee80211_bss_npca_params {
    __slate_bits_0: __slate_bits::__SlateBits1667U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_bss_conf {
    vif: *mut ieee80211_vif,
    bss: *mut cfg80211_bss,
    bssid: *const u8,
    link_id: u32,
    addr: [u8; 6],
    htc_trig_based_pkt_ext: u8,
    uora_exists: bool,
    uora_ocw_range: u8,
    frame_time_rts_th: u16,
    he_support: bool,
    twt_requester: bool,
    twt_responder: bool,
    twt_protected: bool,
    twt_broadcast: bool,
    use_cts_prot: bool,
    use_short_preamble: bool,
    use_short_slot: bool,
    enable_beacon: bool,
    dtim_period: u8,
    beacon_int: u16,
    assoc_capability: u16,
    sync_tsf: u64,
    sync_device_ts: u32,
    sync_dtim_count: u8,
    basic_rates: u32,
    beacon_rate: *mut ieee80211_rate,
    mcast_rate: [i32; 6],
    ht_operation_mode: u16,
    cqm_rssi_thold: i32,
    cqm_rssi_hyst: u32,
    cqm_rssi_low: i32,
    cqm_rssi_high: i32,
    chanreq: ieee80211_chan_req,
    mu_group: ieee80211_mu_group_data,
    qos: bool,
    hidden_ssid: bool,
    txpower: i32,
    txpower_type: u32,
    p2p_noa_attr: ieee80211_p2p_noa_attr,
    allow_p2p_go_ps: bool,
    max_idle_period: u16,
    protected_keep_alive: bool,
    ftm_responder: bool,
    ftmr_params: *mut ieee80211_ftm_responder_params,
    nontransmitted: bool,
    tx_bss_conf: *mut ieee80211_bss_conf,
    transmitter_bssid: [u8; 6],
    bssid_index: u8,
    bssid_indicator: u8,
    ema_ap: bool,
    profile_periodicity: u8,
    he_oper: __SlateRecord_mac80211_h_856,
    he_obss_pd: ieee80211_he_obss_pd,
    he_bss_color: cfg80211_he_bss_color,
    fils_discovery: ieee80211_fils_discovery,
    unsol_bcast_probe_resp_interval: u32,
    beacon_tx_rate: cfg80211_bitrate_mask,
    power_type: u32,
    tpe: ieee80211_parsed_tpe,
    pwr_reduction: u8,
    eht_support: bool,
    epcs_support: bool,
    uhr_support: bool,
    csa_active: bool,
    mu_mimo_owner: bool,
    chanctx_conf: *mut ieee80211_chanctx_conf,
    color_change_active: bool,
    color_change_color: u8,
    ht_ldpc: bool,
    vht_ldpc: bool,
    he_ldpc: bool,
    vht_su_beamformer: bool,
    vht_su_beamformee: bool,
    vht_mu_beamformer: bool,
    vht_mu_beamformee: bool,
    he_su_beamformer: bool,
    he_su_beamformee: bool,
    he_mu_beamformer: bool,
    he_full_ul_mumimo: bool,
    eht_su_beamformer: bool,
    eht_su_beamformee: bool,
    eht_mu_beamformer: bool,
    eht_80mhz_full_bw_ul_mumimo: bool,
    eht_disable_mcs15: bool,
    bss_param_ch_cnt: u8,
    bss_param_ch_cnt_link_id: u8,
    enh_bss_param_ch_cnt: u8,
    enh_bss_param_ch_cnt_link_id: u8,
    s1g_long_beacon_period: u8,
    npca: ieee80211_bss_npca_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_nan_channel {
    chanreq: ieee80211_chan_req,
    needed_rx_chains: u8,
    chanctx_conf: *mut ieee80211_chanctx_conf,
    channel_entry: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_nan_peer_map {
    map_id: u8,
    slots: [*mut ieee80211_nan_channel; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_nan_peer_sched {
    seq_id: u8,
    committed_dw: u16,
    max_chan_switch: u16,
    init_ulw: *const u8,
    ulw_size: u16,
    maps: [ieee80211_nan_peer_map; 2],
    n_channels: u8,
    channels: [ieee80211_nan_channel; 0],
}


// there are 40 bytes if you don't need the rateset to be kept
// if you do need the rateset, then you have less space
// maximum number of rate stages
// maximum number of rate table entries
#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tx_rate {
    idx: i8,
    __slate_bits_0: __slate_bits::__SlateBits1672U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_scan_ies {
    ies: [*const u8; 6],
    len: [u64; 6],
    common_ies: *const u8,
    common_ie_len: u64,
}


fn IEEE80211_SKB_RXCB(skb: &mut sk_buff) -> *mut ieee80211_rx_status {
(unsafe { (*(skb as *mut sk_buff)).cb.as_mut_ptr() as *mut u8 }) as *mut ieee80211_rx_status
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_rx_status {
    mactime: u64,
    __slate_anon_1: __SlateRecord_mac80211_h_1780,
    device_timestamp: u32,
    ampdu_reference: u32,
    flag: u32,
    __slate_bits_0: __slate_bits::__SlateBits1683U0,
    enc_flags: u8,
    __slate_bits_1: __slate_bits::__SlateBits1683U1,
    __slate_anon_10: __SlateRecord_mac80211_h_1790,
    rate_idx: u8,
    nss: u8,
    rx_flags: u8,
    band: u8,
    antenna: u8,
    signal: i8,
    chains: u8,
    chain_signal: [i8; 4],
    zero_length_psdu_type: u8,
    __slate_bits_2: __slate_bits::__SlateBits1683U2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_444 {
    bss: *mut cfg80211_bss,
    addr: [u8; 6],
    ap_ht_param: u8,
    ap_vht_cap: ieee80211_vht_cap,
    elems_len: u64,
    elems: *mut u8,
    conn: ieee80211_conn_settings,
    status: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_conf {
    flags: u32,
    power_level: i32,
    dynamic_ps_timeout: i32,
    listen_interval: u16,
    ps_dtim_period: u8,
    long_frame_max_tx_count: u8,
    short_frame_max_tx_count: u8,
    chandef: cfg80211_chan_def,
    radar_enabled: bool,
    smps_mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_channel_switch {
    timestamp: u64,
    device_timestamp: u32,
    block_tx: bool,
    chandef: cfg80211_chan_def,
    count: u8,
    link_id: u8,
    delay: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_eml_params {
    link_id: u8,
    control: u8,
    link_bitmap: u16,
    emlmr_mcs_map_count: u8,
    emlmr_mcs_map_bw: [u8; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_nan_sched_cfg {
    channels: [ieee80211_nan_channel; 3],
    schedule: [*mut ieee80211_nan_channel; 32],
    avail_blob: [u8; 54],
    avail_blob_len: u16,
    deferred: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_631 {
    add_links_data: *mut ieee80211_mgd_assoc_data,
    wk: wiphy_delayed_work,
    removed_links: u16,
    added_links: u16,
    dialog_token: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_640 {
    enabled: bool,
    dialog_token: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_645 {
    status_work: wiphy_hrtimer_work,
    timeout_us: u32,
    links: u16,
    pending: u16,
    pending_init: u16,
    dialog_token: u8,
    acked: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_vif_cfg {
    assoc: bool,
    ibss_joined: bool,
    ibss_creator: bool,
    ps: bool,
    aid: u16,
    eml_cap: u16,
    eml_med_sync_delay: u16,
    mld_capa_op: u16,
    arp_addr_list: [u32; 4],
    arp_addr_cnt: i32,
    ssid: [u8; 32],
    ssid_len: u64,
    s1g: bool,
    idle: bool,
    ap_addr: [u8; 6],
    nan_sched: ieee80211_nan_sched_cfg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_neg_ttlm {
    downlink: [u16; 8],
    uplink: [u16; 8],
    valid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_vif {
    r#type: u32,
    cfg: ieee80211_vif_cfg,
    bss_conf: ieee80211_bss_conf,
    link_conf: [*mut ieee80211_bss_conf; 15],
    valid_links: u16,
    active_links: u16,
    dormant_links: u16,
    suspended_links: u16,
    neg_ttlm: ieee80211_neg_ttlm,
    addr: [u8; 6],
    addr_valid: bool,
    p2p: bool,
    cab_queue: u8,
    hw_queue: [u8; 4],
    txq: *mut ieee80211_txq,
    txq_mgmt: *mut ieee80211_txq,
    netdev_features: u64,
    driver_flags: u32,
    offload_flags: u32,
    probe_req_reg: bool,
    rx_mcast_action_reg: bool,
    __slate_pad_21: [u8; 6],
    drv_priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_key_conf {
    tx_pn: __SlateRecord_types_h_197,
    cipher: u32,
    icv_len: u8,
    iv_len: u8,
    hw_key_idx: u8,
    keyidx: i8,
    flags: u16,
    link_id: i8,
    keylen: u8,
    key: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_key_seq {
    __slate_anon_0: __SlateRecord_mac80211_h_2472,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_rates {
    callback_head: callback_head,
    rate: [__SlateRecord_mac80211_h_2559; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_txpwr {
    power: i16,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_rmc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_aggregates {
    max_amsdu_len: u16,
    max_rc_amsdu_len: u16,
    max_tid_amsdu_len: [u16; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_link_sta {
    sta: *mut ieee80211_sta,
    addr: [u8; 6],
    link_id: u8,
    smps_mode: u32,
    supp_rates: [u32; 6],
    ht_cap: ieee80211_sta_ht_cap,
    vht_cap: ieee80211_sta_vht_cap,
    he_cap: ieee80211_sta_he_cap,
    he_6ghz_capa: ieee80211_he_6ghz_capa,
    eht_cap: ieee80211_sta_eht_cap,
    uhr_cap: ieee80211_sta_uhr_cap,
    s1g_cap: ieee80211_sta_s1g_cap,
    agg: ieee80211_sta_aggregates,
    rx_nss: u8,
    bandwidth: u32,
    txpwr: ieee80211_sta_txpwr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta {
    addr: [u8; 6],
    aid: u16,
    max_rx_aggregation_subframes: u16,
    wme: bool,
    uapsd_queues: u8,
    max_sp: u8,
    rates: *mut ieee80211_sta_rates,
    tdls: bool,
    tdls_initiator: bool,
    mfp: bool,
    mlo: bool,
    spp_amsdu: bool,
    max_amsdu_subframes: u8,
    eml_cap: u16,
    cur: *mut ieee80211_sta_aggregates,
    support_p2p_ps: bool,
    txq: [*mut ieee80211_txq; 17],
    valid_links: u16,
    ext_mld_capa_ops: u16,
    epp_peer: bool,
    deflink: ieee80211_link_sta,
    link: [*mut ieee80211_link_sta; 15],
    nmi: *mut ieee80211_sta,
    nan_sched: *mut ieee80211_nan_peer_sched,
    drv_priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tx_control {
    sta: *mut ieee80211_sta,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_1011 {
    func_lock: spinlock,
    function_inst_ids: idr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_txq {
    vif: *mut ieee80211_vif,
    sta: *mut ieee80211_sta,
    tid: u8,
    ac: u8,
    __slate_pad_4: [u8; 6],
    drv_priv: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_hw {
    conf: ieee80211_conf,
    wiphy: *mut wiphy,
    rate_control_algorithm: *const u8,
    r#priv: *mut (),
    flags: [u64; 1],
    extra_tx_headroom: u32,
    extra_beacon_tailroom: u32,
    vif_data_size: i32,
    sta_data_size: i32,
    chanctx_data_size: i32,
    txq_data_size: i32,
    queues: u16,
    max_listen_interval: u16,
    max_signal: i8,
    max_rates: u8,
    max_report_rates: u8,
    max_rate_tries: u8,
    max_rx_aggregation_subframes: u16,
    max_tx_aggregation_subframes: u16,
    max_tx_fragments: u8,
    offchannel_tx_hw_queue: u8,
    radiotap_mcs_details: u8,
    radiotap_vht_details: u16,
    radiotap_timestamp: __SlateRecord_mac80211_h_3305,
    netdev_features: u64,
    uapsd_queues: u8,
    uapsd_max_sp_len: u8,
    max_nan_de_entries: u8,
    tx_sk_pacing_shift: u8,
    weight_multiplier: u8,
    max_mtu: u32,
    tx_power_levels: *const i8,
    max_txpwr_levels_idx: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_1048 {
    switch_work: wiphy_hrtimer_work,
    ap_chandef: cfg80211_chan_def,
    tpe: ieee80211_parsed_tpe,
    time: i64,
    waiting_bcn: bool,
    ignored_same_chan: bool,
    blocked_tx: bool,
}


fn _ieee80211_hw_check(hw: *mut ieee80211_hw, flg: u32) -> bool {
test_bit__2!(flg, unsafe { (*hw).flags.as_mut_ptr() as *mut u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_1114 {
    finalize_work: wiphy_work,
    chanreq: ieee80211_chan_req,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_i_h_1138 {
    mgd: ieee80211_link_data_managed,
    ap: ieee80211_link_data_ap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ieee80211_i_h_1147 {
    he_and_lower: u32,
    eht: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ieee80211_i_h_1224 {
    ap: ieee80211_if_ap,
    vlan: ieee80211_if_vlan,
    mgd: ieee80211_if_managed,
    ibss: ieee80211_if_ibss,
    mesh: ieee80211_if_mesh,
    ocb: ieee80211_if_ocb,
    mntr: ieee80211_if_mntr,
    nan: ieee80211_if_nan,
    nan_data: ieee80211_if_nan_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_scan_request {
    ies: ieee80211_scan_ies,
    req: cfg80211_scan_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tdls_ch_sw_params {
    sta: *mut ieee80211_sta,
    chandef: *mut cfg80211_chan_def,
    action_code: u8,
    status: u32,
    timestamp: u32,
    switch_time: u16,
    switch_timeout: u16,
    tmpl_skb: *mut sk_buff,
    ch_sw_tm_ie: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ampdu_params {
    action: u32,
    sta: *mut ieee80211_sta,
    tid: u16,
    ssn: u16,
    buf_size: u16,
    amsdu: bool,
    timeout: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_agg_rx_c_192 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_agg_rx_c_221 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_agg_rx_c_413_4 {
}


mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits553U0 {
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
    pub struct __SlateBits585U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits287U0 {
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
    pub struct __SlateBits76U0 {
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
    pub struct __SlateBits76U1 {
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
    #[bitfields::bitfield([u8; 5], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1269U0 {
        #[bits(32)]
        pub priv_flags: u64,
        #[bits(1)]
        pub lltx: u64,
        #[bits(2)]
        pub netmem_tx: u64,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield([u8; 5], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1270U0 {
        #[bits(32)]
        pub priv_flags: u64,
        #[bits(1)]
        pub lltx: u64,
        #[bits(2)]
        pub netmem_tx: u64,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits681U0 {
        #[bits(1)]
        pub cloned: u8,
        #[bits(1)]
        pub nohdr: u8,
        #[bits(2)]
        pub fclone: u8,
        #[bits(1)]
        pub peeked: u8,
        #[bits(1)]
        pub head_frag: u8,
        #[bits(1)]
        pub pfmemalloc: u8,
        #[bits(1)]
        pub pp_recycle: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U0 {
        #[bits(3)]
        pub pkt_type: u8,
        #[bits(1)]
        pub ignore_df: u8,
        #[bits(1)]
        pub dst_pending_confirm: u8,
        #[bits(2)]
        pub ip_summed: u8,
        #[bits(1)]
        pub ooo_okay: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U1 {
        #[bits(2)]
        pub tstamp_type: u8,
        #[bits(1)]
        pub tc_at_ingress: u8,
        #[bits(1)]
        pub tc_skip_classify: u8,
        #[bits(1)]
        pub remcsum_offload: u8,
        #[bits(1)]
        pub csum_complete_sw: u8,
        #[bits(2)]
        pub csum_level: u8,
        #[bits(1)]
        pub inner_protocol_type: u8,
        #[bits(1)]
        pub l4_hash: u8,
        #[bits(1)]
        pub sw_hash: u8,
        #[bits(1)]
        pub wifi_acked_valid: u8,
        #[bits(1)]
        pub wifi_acked: u8,
        #[bits(1)]
        pub no_fcs: u8,
        #[bits(1)]
        pub encapsulation: u8,
        #[bits(1)]
        pub encap_hdr_csum: u8,
        #[bits(1)]
        pub csum_valid: u8,
        #[bits(2)]
        pub ndisc_nodetype: u8,
        #[bits(1)]
        pub redirected: u8,
        #[bits(1)]
        pub nf_skip_egress: u8,
        #[bits(1)]
        pub slow_gro: u8,
        #[bits(1)]
        pub unreadable: u8,
        #[bits(1, access = na)]
        pub __slate_pad_20: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U2 {
        #[bits(2)]
        pub tc_depth: u8,
        #[bits(6, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits825U0 {
        #[bits(3)]
        pub pkt_type: u8,
        #[bits(1)]
        pub ignore_df: u8,
        #[bits(1)]
        pub dst_pending_confirm: u8,
        #[bits(2)]
        pub ip_summed: u8,
        #[bits(1)]
        pub ooo_okay: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits825U1 {
        #[bits(2)]
        pub tstamp_type: u8,
        #[bits(1)]
        pub tc_at_ingress: u8,
        #[bits(1)]
        pub tc_skip_classify: u8,
        #[bits(1)]
        pub remcsum_offload: u8,
        #[bits(1)]
        pub csum_complete_sw: u8,
        #[bits(2)]
        pub csum_level: u8,
        #[bits(1)]
        pub inner_protocol_type: u8,
        #[bits(1)]
        pub l4_hash: u8,
        #[bits(1)]
        pub sw_hash: u8,
        #[bits(1)]
        pub wifi_acked_valid: u8,
        #[bits(1)]
        pub wifi_acked: u8,
        #[bits(1)]
        pub no_fcs: u8,
        #[bits(1)]
        pub encapsulation: u8,
        #[bits(1)]
        pub encap_hdr_csum: u8,
        #[bits(1)]
        pub csum_valid: u8,
        #[bits(2)]
        pub ndisc_nodetype: u8,
        #[bits(1)]
        pub redirected: u8,
        #[bits(1)]
        pub nf_skip_egress: u8,
        #[bits(1)]
        pub slow_gro: u8,
        #[bits(1)]
        pub unreadable: u8,
        #[bits(1, access = na)]
        pub __slate_pad_20: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits825U2 {
        #[bits(2)]
        pub tc_depth: u8,
        #[bits(6, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits796U0 {
        #[bits(1)]
        pub jited: u32,
        #[bits(1)]
        pub jit_requested: u32,
        #[bits(1)]
        pub jit_required: u32,
        #[bits(1)]
        pub gpl_compatible: u32,
        #[bits(1)]
        pub cb_access: u32,
        #[bits(1)]
        pub dst_needed: u32,
        #[bits(1)]
        pub blinding_requested: u32,
        #[bits(1)]
        pub blinded: u32,
        #[bits(1)]
        pub is_func: u32,
        #[bits(1)]
        pub kprobe_override: u32,
        #[bits(1)]
        pub has_callchain_buf: u32,
        #[bits(1)]
        pub enforce_expected_attach_type: u32,
        #[bits(1)]
        pub call_get_stack: u32,
        #[bits(1)]
        pub call_get_func_ip: u32,
        #[bits(1)]
        pub call_session_cookie: u32,
        #[bits(1)]
        pub tstamp_type_access: u32,
        #[bits(1)]
        pub sleepable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_17: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits985U0 {
        #[bits(4)]
        pub dst_reg: u8,
        #[bits(4)]
        pub src_reg: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1004U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1007U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1040U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1041U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1055U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1056U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1060U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1061U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits230U0 {
        #[bits(8)]
        pub pcpu_stat_type: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1177U0 {
        #[bits(8)]
        pub supported_input_xfrm: u32,
        #[bits(1)]
        pub cap_link_lanes_supported: u32,
        #[bits(1)]
        pub rxfh_per_ctx_fields: u32,
        #[bits(1)]
        pub rxfh_per_ctx_key: u32,
        #[bits(1)]
        pub cap_rss_rxnfc_adds: u32,
        #[bits(4, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1445U0 {
        #[bits(1)]
        pub indir_configured: u8,
        #[bits(1)]
        pub key_configured: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits971U0 {
        #[bits(1)]
        pub is_kernel: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits686U0 {
        #[bits(1)]
        pub msg_control_is_user: bool,
        #[bits(1)]
        pub msg_get_inq: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits702U0 {
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
    pub struct __SlateBits699U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits702U1 {
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
    pub struct __SlateBits702U2 {
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
    pub struct __SlateBits702U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1527U0 {
        #[bits(1)]
        pub support_mbssid: u8,
        #[bits(1)]
        pub support_only_he_mbssid: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1607U0 {
        #[bits(1)]
        pub report_ap_tsf: u8,
        #[bits(1)]
        pub randomize_mac_addr: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1608U0 {
        #[bits(1)]
        pub supported: u8,
        #[bits(1)]
        pub asap: u8,
        #[bits(1)]
        pub non_asap: u8,
        #[bits(1)]
        pub request_lci: u8,
        #[bits(1)]
        pub request_civicloc: u8,
        #[bits(1)]
        pub trigger_based: u8,
        #[bits(1)]
        pub non_trigger_based: u8,
        #[bits(1)]
        pub support_6ghz: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1609U0 {
        #[bits(1)]
        pub support_ntb: u8,
        #[bits(1)]
        pub support_tb: u8,
        #[bits(1)]
        pub support_edca: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1610U0 {
        #[bits(1)]
        pub support_ntb: u8,
        #[bits(1)]
        pub support_tb: u8,
        #[bits(1)]
        pub support_edca: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1611U0 {
        #[bits(1)]
        pub infra_support: u8,
        #[bits(1)]
        pub pd_support: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1608U1 {
        #[bits(1)]
        pub concurrent_ista_rsta_support: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1184U0 {
        #[bits(1)]
        pub mgmt_registrations_need_update: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1184U1 {
        #[bits(1)]
        pub connected: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits230U1 {
        #[bits(1)]
        pub see_all_hwtstamp_requests: u64,
        #[bits(1)]
        pub change_proto_down: u64,
        #[bits(1)]
        pub netns_immutable: u64,
        #[bits(1)]
        pub fcoe_mtu: u64,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1189U0 {
        #[bits(1)]
        pub phys_id_busy: u32,
        #[bits(1)]
        pub wol_enabled: u32,
        #[bits(1)]
        pub module_fw_flash_in_progress: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits659U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits955U0 {
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
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits76U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits130U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits597U0 {
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
    pub struct __SlateBits129U0 {
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
    pub struct __SlateBits418U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1583U0 {
        #[bits(1)]
        pub proberesp_ecsa_stuck: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1667U0 {
        #[bits(4)]
        pub min_dur_thresh: u32,
        #[bits(6)]
        pub switch_delay: u32,
        #[bits(6)]
        pub switch_back_delay: u32,
        #[bits(2)]
        pub init_qsrc: u32,
        #[bits(1)]
        pub moplen: u32,
        #[bits(1)]
        pub enabled: u32,
        #[bits(4, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1726U0 {
        #[bits(1)]
        pub success: u8,
        #[bits(1)]
        pub was_assoc: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1597U0 {
        #[bits(1)]
        pub report_ap_tsf: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1596U0 {
        #[bits(1)]
        pub requested: u8,
        #[bits(1)]
        pub asap: u8,
        #[bits(1)]
        pub request_lci: u8,
        #[bits(1)]
        pub request_civicloc: u8,
        #[bits(1)]
        pub trigger_based: u8,
        #[bits(1)]
        pub non_trigger_based: u8,
        #[bits(1)]
        pub lmr_feedback: u8,
        #[bits(1)]
        pub rsta: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1596U1 {
        #[bits(1)]
        pub pd_suppress_range_results: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1765U0 {
        #[bits(1)]
        pub check_sequential_pn: u8,
        #[bits(1)]
        pub is_protected: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1794U0 {
        #[bits(1)]
        pub connection_loss: u8,
        #[bits(1)]
        pub driver_disconnect: u8,
        #[bits(1)]
        pub reconnect: u8,
        #[bits(1)]
        pub associated: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1683U0 {
        #[bits(13)]
        pub freq: u16,
        #[bits(1)]
        pub freq_offset: u16,
        #[bits(2, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1683U1 {
        #[bits(3)]
        pub encoding: u8,
        #[bits(4)]
        pub bw: u8,
        #[bits(1, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1686U0 {
        #[bits(3)]
        pub he_ru: u8,
        #[bits(2)]
        pub he_gi: u8,
        #[bits(1)]
        pub he_dcm: u8,
        #[bits(2, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1687U0 {
        #[bits(4)]
        pub ru: u8,
        #[bits(2)]
        pub gi: u8,
        #[bits(2, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1688U0 {
        #[bits(4)]
        pub ru: u8,
        #[bits(2)]
        pub gi: u8,
        #[bits(1)]
        pub elr: u8,
        #[bits(1)]
        pub im: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1683U2 {
        #[bits(1)]
        pub link_valid: u8,
        #[bits(4)]
        pub link_id: u8,
        #[bits(3, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1762U0 {
        #[bits(1)]
        pub key: u8,
        #[bits(1)]
        pub internal_forward: u8,
        #[bits(1)]
        pub uses_rss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1758U0 {
        #[bits(1)]
        pub auto_seq: u8,
        #[bits(1)]
        pub removed: u8,
        #[bits(1)]
        pub started: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1672U0 {
        #[bits(5)]
        pub count: u16,
        #[bits(11)]
        pub flags: u16,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_prep_tx_info {
    duration: u16,
    subtype: u16,
    __slate_bits_0: __slate_bits::__SlateBits1726U0,
    link_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_ops {
    tx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_tx_control, *mut sk_buff)>,
    start: Option<unsafe extern "C" fn(*mut ieee80211_hw) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut ieee80211_hw, bool)>,
    suspend: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut cfg80211_wowlan) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut ieee80211_hw) -> i32>,
    set_wakeup: Option<unsafe extern "C" fn(*mut ieee80211_hw, bool)>,
    add_interface: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> i32>,
    change_interface: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, bool) -> i32>,
    remove_interface: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    config: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, u32) -> i32>,
    bss_info_changed: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf, u64)>,
    vif_cfg_changed: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u64)>,
    link_info_changed: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf, u64)>,
    start_ap: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf) -> i32>,
    stop_ap: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf)>,
    prepare_multicast: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut netdev_hw_addr_list) -> u64>,
    configure_filter: Option<unsafe extern "C" fn(*mut ieee80211_hw, u32, *mut u32, u64)>,
    config_iface_filter: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, u32)>,
    set_tim: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta, bool) -> i32>,
    set_key: Option<unsafe extern "C" fn(*mut ieee80211_hw, u32, *mut ieee80211_vif, *mut ieee80211_sta, *mut ieee80211_key_conf) -> i32>,
    update_tkip_key: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_key_conf, *mut ieee80211_sta, u32, *mut u16)>,
    set_rekey_data: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_gtk_rekey_data)>,
    set_default_unicast_key: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, i32)>,
    hw_scan: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_scan_request) -> i32>,
    cancel_hw_scan: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    sched_scan_start: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_sched_scan_request, *mut ieee80211_scan_ies) -> i32>,
    sched_scan_stop: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> i32>,
    sw_scan_start: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *const u8)>,
    sw_scan_complete: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    get_stats: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_low_level_stats) -> i32>,
    get_key_seq: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_key_conf, *mut ieee80211_key_seq)>,
    set_frag_threshold: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, u32) -> i32>,
    set_rts_threshold: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, u32) -> i32>,
    sta_add: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta) -> i32>,
    sta_remove: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta) -> i32>,
    sta_notify: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, *mut ieee80211_sta)>,
    sta_set_txpwr: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta) -> i32>,
    sta_state: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, u32, u32) -> i32>,
    sta_pre_rcu_remove: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta)>,
    link_sta_rc_update: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_link_sta, u32)>,
    sta_rate_tbl_update: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta)>,
    sta_statistics: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, *mut station_info)>,
    conf_tx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, u16, *const ieee80211_tx_queue_params) -> i32>,
    get_tsf: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> u64>,
    set_tsf: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u64)>,
    offset_tsf: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, i64)>,
    reset_tsf: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    tx_last_beacon: Option<unsafe extern "C" fn(*mut ieee80211_hw) -> i32>,
    link_sta_statistics: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_link_sta, *mut link_station_info)>,
    ampdu_action: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_ampdu_params) -> i32>,
    get_survey: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, *mut survey_info) -> i32>,
    rfkill_poll: Option<unsafe extern "C" fn(*mut ieee80211_hw)>,
    set_coverage_class: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, i16)>,
    flush: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, bool)>,
    flush_sta: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta)>,
    channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_channel_switch)>,
    set_antenna: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, u32, u32) -> i32>,
    get_antenna: Option<unsafe extern "C" fn(*mut ieee80211_hw, i32, *mut u32, *mut u32) -> i32>,
    remain_on_channel: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_channel, i32, u32) -> i32>,
    cancel_remain_on_channel: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> i32>,
    set_ringparam: Option<unsafe extern "C" fn(*mut ieee80211_hw, u32, u32) -> i32>,
    get_ringparam: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut u32, *mut u32, *mut u32, *mut u32)>,
    tx_frames_pending: Option<unsafe extern "C" fn(*mut ieee80211_hw) -> bool>,
    set_bitrate_mask: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *const cfg80211_bitrate_mask) -> i32>,
    event_callback: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *const ieee80211_event)>,
    allow_buffered_frames: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta, u16, i32, u32, bool)>,
    release_buffered_frames: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta, u16, i32, u32, bool)>,
    get_et_sset_count: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, i32) -> i32>,
    get_et_stats: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ethtool_stats, *mut u64)>,
    get_et_strings: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, *mut u8)>,
    mgd_prepare_tx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_prep_tx_info)>,
    mgd_complete_tx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_prep_tx_info)>,
    mgd_protect_tdls_discover: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32)>,
    add_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_chanctx_conf) -> i32>,
    remove_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_chanctx_conf)>,
    change_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_chanctx_conf, u32)>,
    assign_vif_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf, *mut ieee80211_chanctx_conf) -> i32>,
    unassign_vif_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf, *mut ieee80211_chanctx_conf)>,
    switch_vif_chanctx: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif_chanctx_switch, i32, u32) -> i32>,
    reconfig_complete: Option<unsafe extern "C" fn(*mut ieee80211_hw, u32)>,
    ipv6_addr_change: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut inet6_dev)>,
    channel_switch_beacon: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_chan_def)>,
    pre_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_channel_switch) -> i32>,
    post_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf) -> i32>,
    abort_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_bss_conf)>,
    channel_switch_rx_beacon: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_channel_switch)>,
    join_ibss: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> i32>,
    leave_ibss: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    get_expected_throughput: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta) -> u32>,
    get_txpower: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u32, *mut i32) -> i32>,
    tdls_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, u8, *mut cfg80211_chan_def, *mut sk_buff, u32) -> i32>,
    tdls_cancel_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta)>,
    tdls_recv_channel_switch: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_tdls_ch_sw_params)>,
    wake_tx_queue: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_txq)>,
    sync_rx_queues: Option<unsafe extern "C" fn(*mut ieee80211_hw)>,
    start_nan: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_nan_conf) -> i32>,
    stop_nan: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif) -> i32>,
    nan_change_conf: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_nan_conf, u32) -> i32>,
    add_nan_func: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *const cfg80211_nan_func) -> i32>,
    del_nan_func: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u8)>,
    nan_peer_sched_changed: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta) -> i32>,
    can_aggregate_in_amsdu: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut sk_buff, *mut sk_buff) -> bool>,
    get_ftm_responder_stats: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_ftm_responder_stats) -> i32>,
    start_pmsr: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_pmsr_request) -> i32>,
    abort_pmsr: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_pmsr_request)>,
    set_tid_config: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, *mut cfg80211_tid_config) -> i32>,
    reset_tid_config: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, u8) -> i32>,
    update_vif_offload: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif)>,
    sta_set_4addr: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, bool)>,
    set_sar_specs: Option<unsafe extern "C" fn(*mut ieee80211_hw, *const cfg80211_sar_specs) -> i32>,
    sta_set_decap_offload: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, bool)>,
    add_twt_setup: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta, *mut ieee80211_twt_setup)>,
    twt_teardown_request: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_sta, u8)>,
    set_radar_background: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut cfg80211_chan_def) -> i32>,
    net_fill_forward_path: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, *mut net_device_path_ctx, *mut net_device_path) -> i32>,
    can_activate_links: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u16) -> bool>,
    change_vif_links: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, u16, u16, *mut *mut ieee80211_bss_conf) -> i32>,
    change_sta_links: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, u16, u16) -> i32>,
    set_hw_timestamp: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut cfg80211_set_hw_timestamp) -> i32>,
    net_setup_tc: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut net_device, u32, *mut ()) -> i32>,
    can_neg_ttlm: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_neg_ttlm) -> u32>,
    prep_add_interface: Option<unsafe extern "C" fn(*mut ieee80211_hw, u32)>,
    set_eml_op_mode: Option<unsafe extern "C" fn(*mut ieee80211_hw, *mut ieee80211_vif, *mut ieee80211_sta, *mut ieee80211_eml_params) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_tpt_blink {
    throughput: i32,
    blink_time: i32,
}


// for_each_interface - iterate interfaces under wiphy mutex
// @vif: the iterator variable
// @hw: the HW to iterate for
// @flags: the iteration flags, see &enum ieee80211_interface_iteration_flags
// for_each_station - iterate stations under wiphy mutex
// @sta: the iterator variable
// @hw: the HW to iterate for
core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ieee80211_stop_rx_ba_session: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ieee80211_manage_rx_ba_offl: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ieee80211_rx_ba_timer_expired: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous", sym ieee80211_stop_rx_ba_session, sym ieee80211_manage_rx_ba_offl, sym ieee80211_rx_ba_timer_expired, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_lock_head {
}


// Maximum chain length before rehash
//
// The maximum (not average) chain length grows with the size of the hash
// table, at a rate of (log N)/(log log N).
//
// The value of 16 is selected so that even if the hash table grew to
// 2^32 you would not expect the maximum chain length to exceed it
// unless we are under attack (or extremely unlucky).
//
// As this limit is only to detect attacks, we don't need to set it to a
// lower value as you'd need the chain length to vastly exceed 16 to have
// any real effect on the system.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct bucket_table {
    size: u32,
    nest: u32,
    hash_rnd: u32,
    walkers: list_head,
    rcu: callback_head,
    future_tbl: *mut bucket_table,
    dep_map: lockdep_map,
    __slate_pad_7: [u8; 8],
    buckets: [*mut rhash_lock_head; 0],
}


// NULLS_MARKER() expects a hash value with the low
// bits mostly likely to be significant, and it discards
// the msb.
// We give it an address, in which the bottom bit is
// always 0, and the msb might be significant.
// So we shift the address down one bit to align with
// expectations and avoid losing a significant bit.
//
// We never store the NULLS_MARKER in the hash table
// itself as we need the lsb for locking.
// Instead we store a NULL
// rht_for_each_from - iterate over hash chain from given head
// @pos:	the &struct rhash_head to use as a loop cursor.
// @head:	the &struct rhash_head to start from
// @tbl:	the &struct bucket_table
// @hash:	the hash value / bucket index
// rht_for_each_rcu_from - iterate over rcu hash chain from given head
// @pos:	the &struct rhash_head to use as a loop cursor.
// @head:	the &struct rhash_head to start from
// @tbl:	the &struct bucket_table
// @hash:	the hash value / bucket index
//
// This hash chain list-traversal primitive may safely run concurrently with
// the _rcu mutation primitives such as rhashtable_insert() as long as the
// traversal is guarded by rcu_read_lock().
#[repr(C)]
#[derive(Clone, Copy)]
struct fq_flow {
    tin: *mut fq_tin,
    flowchain: list_head,
    queue: sk_buff_head,
    backlog: u32,
    deficit: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fq_tin {
    new_flows: list_head,
    old_flows: list_head,
    tin_list: list_head,
    default_flow: fq_flow,
    backlog_bytes: u32,
    backlog_packets: u32,
    overlimit: u32,
    collisions: u32,
    flows: u32,
    tx_bytes: u32,
    tx_packets: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fq {
    flows: *mut fq_flow,
    flows_bitmap: *mut u64,
    tin_backlog: list_head,
    lock: spinlock,
    flows_cnt: u32,
    limit: u32,
    memory_limit: u32,
    memory_usage: u32,
    quantum: u32,
    backlog: u32,
    overlimit: u32,
    overmemory: u32,
    collisions: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union aes_enckey_arch {
    rndkeys: [u32; 60],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct aes_enckey {
    len: u32,
    nrounds: u32,
    padding: [u32; 2],
    k: aes_enckey_arch,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct aes_cmac_key {
    aes: aes_enckey,
    k_final: [__SlateRecord_aes_cbc_macs_h_22; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arc4_ctx {
    S: [u32; 256],
    x: u32,
    y: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tkip_ctx {
    p1k: [u16; 5],
    p1k_iv32: u32,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tkip_ctx_rx {
    ctx: tkip_ctx,
    iv32: u32,
    iv16: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_key {
    local: *mut ieee80211_local,
    sdata: *mut ieee80211_sub_if_data,
    sta: *mut sta_info,
    list: list_head,
    flags: u32,
    u: __SlateRecord_key_h_70,
    color: u32,
    conf: ieee80211_key_conf,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ewma_avg_signal {
    internal: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct airtime_info {
    rx_airtime: u64,
    tx_airtime: u64,
    last_active: u64,
    deficit: i32,
    aql_tx_pending: __SlateRecord_types_h_190,
    aql_limit_low: u32,
    aql_limit_high: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tid_ampdu_tx {
    callback_head: callback_head,
    session_timer: timer_list,
    addba_resp_timer: timer_list,
    pending: sk_buff_head,
    sta: *mut sta_info,
    state: u64,
    last_tx: u64,
    timeout: u16,
    dialog_token: u8,
    stop_initiator: u8,
    tx_stop: bool,
    buf_size: u16,
    ssn: u16,
    failed_bar_ssn: u16,
    bar_pending: bool,
    amsdu: bool,
    ndp: bool,
    tid: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tid_ampdu_rx {
    callback_head: callback_head,
    reorder_lock: spinlock,
    reorder_buf_filtered: u64,
    sta: *mut sta_info,
    session_timer: timer_list,
    reorder_timer: timer_list,
    last_rx: u64,
    head_seq_num: u16,
    stored_mpdu_num: u16,
    ssn: u16,
    buf_size: u16,
    timeout: u16,
    tid: u8,
    __slate_bits_0: __slate_bits::__SlateBits1758U0,
    reorder: [__SlateRecord_sta_info_h_259; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sta_ampdu_mlme {
    tid_rx: [*mut tid_ampdu_rx; 16],
    tid_rx_token: [u8; 16],
    tid_rx_timer_expired: [u64; 1],
    tid_rx_stop_requested: [u64; 1],
    tid_rx_manage_offl: [u64; 1],
    agg_session_valid: [u64; 1],
    unexpected_agg: [u64; 1],
    work: wiphy_work,
    tid_tx: [*mut tid_ampdu_tx; 16],
    tid_start_tx: [*mut tid_ampdu_tx; 16],
    last_addba_req_time: [u64; 16],
    addba_req_num: [u8; 16],
    dialog_token_allocator: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_fast_tx {
    key: *mut ieee80211_key,
    hdr_len: u8,
    sa_offs: u8,
    da_offs: u8,
    pn_offs: u8,
    band: u8,
    __slate_pad_6: [u8; 1],
    hdr: [u8; 56],
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_fast_rx {
    dev: *mut net_device,
    vif_type: u32,
    vif_addr: [u8; 6],
    rfc1042_hdr: [u8; 6],
    control_port_protocol: u16,
    expected_ds_bits: u16,
    icv_len: u8,
    __slate_bits_0: __slate_bits::__SlateBits1762U0,
    da_offs: u8,
    sa_offs: u8,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ewma_signal {
    internal: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_rx_stats {
    packets: u64,
    last_rx: u64,
    num_duplicates: u64,
    fragments: u64,
    dropped: u64,
    last_signal: i32,
    chains: u8,
    chain_signal_last: [i8; 4],
    last_rate: u32,
    syncp: u64_stats_sync,
    bytes: __SlateRecord_u64_stats_sync_h_75,
    msdu: [__SlateRecord_u64_stats_sync_h_75; 17],
}


// IEEE 802.11-2016 (10.6 "Defragmentation") recommends support for "concurrent
// reception of at least one MSDU per access category per associated STA"
// on APs, or "at least one MSDU per access category" on other interface types.
//
// This limit can be increased by changing this define, at the cost of slower
// frame reassembly and increased memory use while fragments are pending.
#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_fragment_entry {
    skb_list: sk_buff_head,
    first_frag_time: u64,
    seq: u16,
    extra_len: u16,
    last_frag: u16,
    rx_queue: u8,
    __slate_bits_0: __slate_bits::__SlateBits1765U0,
    last_pn: [u8; 6],
    key_color: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_fragment_cache {
    entries: [ieee80211_fragment_entry; 4],
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct link_sta_info {
    addr: [u8; 6],
    link_id: u8,
    op_mode_nss: u8,
    capa_nss: u8,
    link_hash_node: rhlist_head,
    sta: *mut sta_info,
    gtk: [*mut ieee80211_key; 8],
    pcpu_rx_stats: *mut ieee80211_sta_rx_stats,
    rx_stats: ieee80211_sta_rx_stats,
    rx_stats_avg: __SlateRecord_sta_info_h_542,
    status_stats: __SlateRecord_sta_info_h_548,
    tx_stats: __SlateRecord_sta_info_h_562,
    op_mode_bw: u32,
    rx_omi_bw_rx: u32,
    rx_omi_bw_tx: u32,
    rx_omi_bw_staging: u32,
    uhr_usable_tx_width: u32,
    uhr_dbe_enabled: bool,
    r#pub: *mut ieee80211_link_sta,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_removed_link_stats {
    rx_packets: u32,
    tx_packets: u32,
    rx_bytes: u64,
    tx_bytes: u64,
    tx_retries: u32,
    tx_failed: u32,
    rx_dropped_misc: u32,
    beacon_loss_count: u32,
    expected_throughput: u32,
    pertid_stats: __SlateRecord_sta_info_h_629,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sta_info {
    list: list_head,
    free_list: list_head,
    callback_head: callback_head,
    hash_node: rhlist_head,
    addr: [u8; 6],
    local: *mut ieee80211_local,
    sdata: *mut ieee80211_sub_if_data,
    ptk: [*mut ieee80211_key; 4],
    ptk_idx: u8,
    rate_ctrl: *mut rate_control_ref,
    rate_ctrl_priv: *mut (),
    rate_ctrl_lock: spinlock,
    lock: spinlock,
    fast_tx: *mut ieee80211_fast_tx,
    fast_rx: *mut ieee80211_fast_rx,
    drv_deliver_wk: work_struct,
    listen_interval: u16,
    dead: bool,
    removed: bool,
    uploaded: bool,
    sta_state: u32,
    _flags: u64,
    ps_lock: spinlock,
    ps_tx_buf: [sk_buff_head; 4],
    tx_filtered: [sk_buff_head; 4],
    driver_buffered_tids: u64,
    txq_buffered_tids: u64,
    assoc_at: u64,
    last_connected: i64,
    last_seq_ctrl: [u16; 17],
    tid_seq: [u16; 16],
    airtime: [airtime_info; 4],
    airtime_weight: u16,
    ampdu_mlme: sta_ampdu_mlme,
    reserved_tid: u8,
    amsdu_mesh_control: i8,
    tdls_chandef: cfg80211_chan_def,
    frags: ieee80211_fragment_cache,
    cur: ieee80211_sta_aggregates,
    deflink: link_sta_info,
    link: [*mut link_sta_info; 15],
    rem_link_stats: ieee80211_sta_removed_link_stats,
    sta: ieee80211_sta,
}


fn test_sta_flag(sta: *mut sta_info, flag: u32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*sta)._flags) })
}

// common
// STA_STATS_RATE_TYPE_LEGACY
// STA_STATS_RATE_TYPE_HT
// STA_STATS_RATE_TYPE_VHT
// HT, VHT & S1G
// STA_STATS_RATE_TYPE_HE
// STA_STATS_RATE_TYPE_EHT
// STA_STATS_RATE_TYPE_UHR
// STA_STATS_RATE_TYPE_S1G
#[repr(C)]
#[derive(Clone, Copy)]
struct beacon_data {
    head: *mut u8,
    tail: *mut u8,
    head_len: i32,
    tail_len: i32,
    meshconf: *mut ieee80211_meshconf_ie,
    cntdwn_counter_offsets: [u16; 2],
    cntdwn_current_counter: u8,
    mbssid_ies: *mut cfg80211_mbssid_elems,
    rnr_ies: *mut cfg80211_rnr_elems,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct probe_resp {
    callback_head: callback_head,
    len: i32,
    cntdwn_counter_offsets: [u16; 2],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fils_discovery_data {
    callback_head: callback_head,
    len: i32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unsol_bcast_probe_resp_data {
    callback_head: callback_head,
    len: i32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct s1g_short_beacon_data {
    callback_head: callback_head,
    short_head: *mut u8,
    short_tail: *mut u8,
    short_head_len: i32,
    short_tail_len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ps_data {
    tim: [u8; 256],
    bc_buf: sk_buff_head,
    num_sta_ps: __SlateRecord_types_h_190,
    dtim_count: i32,
    dtim_bc_mc: bool,
    sb_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_ap {
    vlans: list_head,
    ps: ps_data,
    num_mcast_sta: __SlateRecord_types_h_190,
    multicast_to_unicast: bool,
    active: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_vlan {
    list: list_head,
    sta: *mut sta_info,
    num_mcast_sta: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_stats {
    fwded_mcast: u32,
    fwded_unicast: u32,
    fwded_frames: u32,
    dropped_frames_ttl: u32,
    dropped_frames_no_route: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_preq_queue {
    list: list_head,
    dst: [u8; 6],
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_conn_settings {
    mode: u32,
    bw_limit: u32,
    dbe_enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mgd_auth_data {
    bss: *mut cfg80211_bss,
    timeout: u64,
    tries: i32,
    algorithm: u16,
    expected_transaction: u16,
    userspace_selectors: [u64; 2],
    key: [u8; 13],
    key_len: u8,
    key_idx: u8,
    done: bool,
    waiting: bool,
    peer_confirmed: bool,
    timeout_started: bool,
    link_id: i32,
    ap_addr: [u8; 6],
    trans: u16,
    status: u16,
    data_len: u64,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mgd_assoc_data {
    link: [__SlateRecord_ieee80211_i_h_444; 15],
    ap_addr: [u8; 6],
    supp_rates: *const u8,
    supp_rates_len: u8,
    timeout: u64,
    tries: i32,
    prev_ap_addr: [u8; 6],
    ssid: [u8; 32],
    ssid_len: u8,
    wmm: bool,
    uapsd: bool,
    need_beacon: bool,
    synced: bool,
    timeout_started: bool,
    comeback: bool,
    s1g: bool,
    spp_amsdu: bool,
    assoc_link_id: i8,
    ext_mld_capa_ops: u16,
    fils_nonces: [u8; 32],
    fils_kek: [u8; 64],
    fils_kek_len: u64,
    ie_len: u64,
    ie_pos: *mut u8,
    ie: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sta_tx_tspec {
    time_slice_start: u64,
    admitted_time: u32,
    tsid: u8,
    up: i8,
    consumed_tx_time: u32,
    action: u32,
    downgraded: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_adv_ttlm_info {
    switch_time: u16,
    duration: u32,
    map: u16,
    active: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ewma_beacon_signal {
    internal: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_managed {
    timer: timer_list,
    conn_mon_timer: timer_list,
    bcn_mon_timer: timer_list,
    monitor_work: wiphy_work,
    beacon_connection_loss_work: wiphy_work,
    csa_connection_drop_work: wiphy_work,
    beacon_timeout: u64,
    probe_timeout: u64,
    probe_send_count: i32,
    nullfunc_failed: bool,
    __slate_bits_0: __slate_bits::__SlateBits1794U0,
    auth_data: *mut ieee80211_mgd_auth_data,
    assoc_data: *mut ieee80211_mgd_assoc_data,
    userspace_selectors: [u64; 2],
    powersave: bool,
    broken_ap: bool,
    flags: u32,
    mcast_seq_last: u16,
    status_acked: bool,
    status_received: bool,
    status_fc: u16,
    mfp: u32,
    uapsd_queues: u32,
    uapsd_max_sp_len: u32,
    use_4addr: u8,
    rssi_min_thold: i32,
    rssi_max_thold: i32,
    ht_capa: ieee80211_ht_cap,
    ht_capa_mask: ieee80211_ht_cap,
    vht_capa: ieee80211_vht_cap,
    vht_capa_mask: ieee80211_vht_cap,
    s1g_capa: ieee80211_s1g_cap,
    s1g_capa_mask: ieee80211_s1g_cap,
    tdls_peer: [u8; 6],
    tdls_peer_del_work: wiphy_delayed_work,
    orig_teardown_skb: *mut sk_buff,
    teardown_skb: *mut sk_buff,
    teardown_lock: spinlock,
    tdls_wider_bw_prohibited: bool,
    tx_tspec: [ieee80211_sta_tx_tspec; 4],
    tx_tspec_wk: wiphy_delayed_work,
    assoc_req_ies: *mut u8,
    assoc_req_ies_len: u64,
    ml_reconf_work: wiphy_hrtimer_work,
    removed_links: u16,
    ttlm_work: wiphy_hrtimer_work,
    ttlm_info: ieee80211_adv_ttlm_info,
    teardown_ttlm_work: wiphy_work,
    dialog_token_alloc: u8,
    neg_ttlm_timeout_work: wiphy_delayed_work,
    reconf: __SlateRecord_ieee80211_i_h_631,
    epcs: __SlateRecord_ieee80211_i_h_640,
    uhr_omp: __SlateRecord_ieee80211_i_h_645,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_ibss {
    timer: timer_list,
    csa_connection_drop_work: wiphy_work,
    last_scan_completed: u64,
    basic_rates: u32,
    fixed_bssid: bool,
    fixed_channel: bool,
    privacy: bool,
    control_port: bool,
    userspace_handles_dfs: bool,
    __slate_pad_9: [u8; 1],
    bssid: [u8; 6],
    ssid: [u8; 32],
    ssid_len: u8,
    ie_len: u8,
    ie: *mut u8,
    chandef: cfg80211_chan_def,
    ibss_join_req: u64,
    presp: *mut beacon_data,
    ht_capa: ieee80211_ht_cap,
    ht_capa_mask: ieee80211_ht_cap,
    incomplete_lock: spinlock,
    incomplete_stations: list_head,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_ocb {
    housekeeping_timer: timer_list,
    wrkq_flags: u64,
    incomplete_lock: spinlock,
    incomplete_stations: list_head,
    joined: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_mesh_sync_ops {
    rx_bcn_presp: Option<unsafe extern "C" fn(*mut ieee80211_sub_if_data, u16, *mut ieee80211_mgmt, u32, *const ieee80211_meshconf_ie, *mut ieee80211_rx_status)>,
    adjust_tsf: Option<unsafe extern "C" fn(*mut ieee80211_sub_if_data, *mut beacon_data)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_csa_settings {
    callback_head: callback_head,
    settings: cfg80211_csa_settings,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_table {
    known_gates: hlist_head,
    gates_lock: spinlock,
    rhead: rhashtable,
    walk_head: hlist_head,
    walk_lock: spinlock,
    entries: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mesh_tx_cache {
    rht: rhashtable,
    walk_head: hlist_head,
    walk_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_mesh {
    housekeeping_timer: timer_list,
    mesh_path_timer: timer_list,
    mesh_path_root_timer: timer_list,
    wrkq_flags: u64,
    mbss_changed: [u64; 1],
    userspace_handles_dfs: bool,
    mesh_id: [u8; 32],
    mesh_id_len: u64,
    mesh_pp_id: u8,
    mesh_pm_id: u8,
    mesh_cc_id: u8,
    mesh_sp_id: u8,
    mesh_auth_id: u8,
    sn: u32,
    preq_id: u32,
    mpaths: __SlateRecord_types_h_190,
    last_sn_update: u64,
    next_perr: u64,
    last_preq: u64,
    rmc: *mut mesh_rmc,
    mesh_preq_queue_lock: spinlock,
    preq_queue: mesh_preq_queue,
    preq_queue_len: i32,
    mshstats: mesh_stats,
    mshcfg: mesh_config,
    estab_plinks: __SlateRecord_types_h_190,
    mesh_seqnum: __SlateRecord_types_h_190,
    accepting_plinks: bool,
    num_gates: i32,
    beacon: *mut beacon_data,
    ie: *const u8,
    ie_len: u8,
    security: u32,
    user_mpm: bool,
    sync_ops: *const ieee80211_mesh_sync_ops,
    sync_offset_clockdrift_max: i64,
    sync_offset_lock: spinlock,
    nonpeer_pm: u32,
    ps_peers_light_sleep: i32,
    ps_peers_deep_sleep: i32,
    ps: ps_data,
    csa: *mut mesh_csa_settings,
    csa_role: u32,
    chsw_ttl: u8,
    pre_value: u16,
    meshconf_offset: i32,
    mesh_paths: mesh_table,
    mpp_paths: mesh_table,
    mesh_paths_generation: i32,
    mpp_paths_generation: i32,
    tx_cache: mesh_tx_cache,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_chanctx {
    list: list_head,
    callback_head: callback_head,
    replace_state: u32,
    replace_ctx: *mut ieee80211_chanctx,
    mode: u32,
    driver_present: bool,
    req: ieee80211_chan_req,
    radar_detected: bool,
    will_be_used: bool,
    conf: ieee80211_chanctx_conf,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mac80211_qos_map {
    qos_map: cfg80211_qos_map,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_mntr {
    flags: u32,
    mu_follow_addr: [u8; 6],
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_nan {
    conf: cfg80211_nan_conf,
    started: bool,
    de: __SlateRecord_ieee80211_i_h_1011,
    removed_channels: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_if_nan_data {
    nmi: *mut ieee80211_sub_if_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_link_data_managed {
    bssid: [u8; 6],
    dtim_period: u8,
    req_smps: u32,
    driver_smps_mode: u32,
    conn: ieee80211_conn_settings,
    p2p_noa_index: i16,
    tdls_chan_switch_prohibited: bool,
    have_beacon: bool,
    tracking_signal_avg: bool,
    disable_wmm_tracking: bool,
    operating_11g_mode: bool,
    csa: __SlateRecord_ieee80211_i_h_1048,
    request_smps_work: wiphy_work,
    recalc_smps: wiphy_work,
    beacon_crc_valid: bool,
    beacon_crc: u32,
    ave_beacon_signal: ewma_beacon_signal,
    last_ave_beacon_signal: i32,
    count_beacon_signal: u32,
    beacon_loss_count: u32,
    last_cqm_event_signal: i32,
    wmm_last_param_set: i32,
    mu_edca_last_param_set: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_link_data_ap {
    beacon: *mut beacon_data,
    probe_resp: *mut probe_resp,
    fils_discovery: *mut fils_discovery_data,
    unsol_bcast_probe_resp: *mut unsol_bcast_probe_resp_data,
    s1g_short_beacon: *mut s1g_short_beacon_data,
    next_beacon: *mut cfg80211_beacon_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_link_data {
    sdata: *mut ieee80211_sub_if_data,
    link_id: u32,
    gtk: [*mut ieee80211_key; 8],
    default_multicast_key: *mut ieee80211_key,
    default_mgmt_key: *mut ieee80211_key,
    default_beacon_key: *mut ieee80211_key,
    operating_11g_mode: bool,
    csa: __SlateRecord_ieee80211_i_h_1114,
    color_change_finalize_work: wiphy_work,
    color_collision_detect_work: wiphy_delayed_work,
    color_bitmap: u64,
    reserved_chanctx: *mut ieee80211_chanctx,
    reserved: ieee80211_chan_req,
    reserved_radar_required: bool,
    reserved_ready: bool,
    needed_rx_chains: u8,
    smps_mode: u32,
    user_power_level: i32,
    ap_power_level: i32,
    radar_required: bool,
    dfs_cac_timer_work: wiphy_hrtimer_work,
    u: __SlateRecord_ieee80211_i_h_1138,
    tx_conf: [ieee80211_tx_queue_params; 4],
    conf: *mut ieee80211_bss_conf,
    bss_bw: __SlateRecord_ieee80211_i_h_1147,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_sub_if_data {
    list: list_head,
    wdev: wireless_dev,
    key_list: list_head,
    crypto_tx_tailroom_needed_cnt: i32,
    crypto_tx_tailroom_pending_dec: i32,
    dec_tailroom_needed_wk: wiphy_delayed_work,
    dev: *mut net_device,
    local: *mut ieee80211_local,
    flags: u32,
    state: u64,
    name: [u8; 16],
    frags: ieee80211_fragment_cache,
    noack_map: u16,
    wmm_acm: u8,
    keys: [*mut ieee80211_key; 4],
    default_unicast_key: *mut ieee80211_key,
    sequence_number: u16,
    mld_mcast_seq: u16,
    control_port_protocol: u16,
    control_port_no_encrypt: bool,
    control_port_no_preauth: bool,
    control_port_over_nl80211: bool,
    num_tx_queued: __SlateRecord_types_h_190,
    qos_map: *mut mac80211_qos_map,
    work: wiphy_work,
    skb_queue: sk_buff_head,
    status_queue: sk_buff_head,
    bss: *mut ieee80211_if_ap,
    rc_rateidx_mask: [u32; 6],
    rc_has_mcs_mask: [bool; 6],
    rc_rateidx_mcs_mask: [[u8; 10]; 6],
    rc_has_vht_mcs_mask: [bool; 6],
    rc_rateidx_vht_mcs_mask: [[u16; 8]; 6],
    beacon_rateidx_mask: [u32; 6],
    beacon_rate_set: bool,
    u: __SlateRecord_ieee80211_i_h_1224,
    deflink: ieee80211_link_data,
    link: [*mut ieee80211_link_data; 15],
    activate_links_work: wiphy_work,
    desired_active_links: u16,
    restart_active_links: u16,
    tx_handlers_drop: u32,
    vif: ieee80211_vif,
}


fn vif_to_sdata(p: *mut ieee80211_vif) -> *mut ieee80211_sub_if_data {
container_of__3!(p, ieee80211_sub_if_data)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tpt_led_trigger {
    name: [u8; 32],
    blink_table: *const ieee80211_tpt_blink,
    blink_table_len: u32,
    timer: timer_list,
    local: *mut ieee80211_local,
    prev_traffic: u64,
    tx_bytes: u64,
    rx_bytes: u64,
    active: u32,
    want: u32,
    running: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_local {
    hw: ieee80211_hw,
    fq: fq,
    cvars: *mut codel_vars,
    cparams: codel_params,
    active_txq_lock: [spinlock; 4],
    active_txqs: [list_head; 4],
    schedule_round: [u16; 4],
    handle_wake_tx_queue_lock: spinlock,
    airtime_flags: u16,
    aql_txq_limit_mc: u32,
    aql_txq_limit_low: [u32; 4],
    aql_txq_limit_high: [u32; 4],
    aql_threshold: u32,
    aql_total_pending_airtime: __SlateRecord_types_h_190,
    aql_mc_pending_airtime: __SlateRecord_types_h_190,
    aql_ac_pending_airtime: [__SlateRecord_types_h_190; 4],
    ops: *const ieee80211_ops,
    workqueue: *mut workqueue_struct,
    queue_stop_reasons: [u64; 16],
    q_stop_reasons: [[i32; 11]; 16],
    queue_stop_reason_lock: spinlock,
    open_count: i32,
    monitors: i32,
    virt_monitors: i32,
    tx_mntrs: i32,
    fif_fcsfail: i32,
    fif_plcpfail: i32,
    fif_control: i32,
    fif_other_bss: i32,
    fif_pspoll: i32,
    fif_probe_req: i32,
    probe_req_reg: bool,
    rx_mcast_action_reg: bool,
    filter_flags: u32,
    dflt_chandef: cfg80211_chan_def,
    emulate_chanctx: bool,
    filter_lock: spinlock,
    reconfig_filter: wiphy_work,
    mc_list: netdev_hw_addr_list,
    tim_in_locked_section: bool,
    suspended: bool,
    suspending: bool,
    resuming: bool,
    quiescing: bool,
    started: bool,
    in_reconfig: bool,
    reconfig_failure: bool,
    wowlan: bool,
    radar_detected_work: wiphy_work,
    rx_chains: u8,
    sband_allocated: u8,
    tx_headroom: i32,
    tasklet: tasklet_struct,
    skb_queue: sk_buff_head,
    skb_queue_unreliable: sk_buff_head,
    rx_path_lock: spinlock,
    tim_lock: spinlock,
    num_sta: u64,
    sta_list: list_head,
    sta_hash: rhltable,
    link_sta_hash: rhltable,
    sta_cleanup: timer_list,
    sta_generation: i32,
    pending: [sk_buff_head; 16],
    tx_pending_tasklet: tasklet_struct,
    wake_txqs_tasklet: tasklet_struct,
    agg_queue_stop: [__SlateRecord_types_h_190; 16],
    iff_allmultis: __SlateRecord_types_h_190,
    rate_ctrl: *mut rate_control_ref,
    wep_tx_ctx: arc4_ctx,
    wep_rx_ctx: arc4_ctx,
    wep_iv: u32,
    interfaces: list_head,
    mon_list: list_head,
    iflist_mtx: mutex,
    scanning: u64,
    scan_ssid: cfg80211_ssid,
    int_scan_req: *mut cfg80211_scan_request,
    scan_req: *mut cfg80211_scan_request,
    hw_scan_req: *mut ieee80211_scan_request,
    scan_chandef: cfg80211_chan_def,
    hw_scan_band: u32,
    scan_channel_idx: i32,
    scan_ies_len: i32,
    hw_scan_ies_bufsize: i32,
    scan_info: cfg80211_scan_info,
    sched_scan_stopped_work: wiphy_work,
    sched_scan_sdata: *mut ieee80211_sub_if_data,
    sched_scan_req: *mut cfg80211_sched_scan_request,
    scan_addr: [u8; 6],
    leave_oper_channel_time: u64,
    next_scan_state: u32,
    scan_work: wiphy_delayed_work,
    scan_sdata: *mut ieee80211_sub_if_data,
    tmp_channel: *mut ieee80211_channel,
    chanctx_list: list_head,
    tx_led: led_trigger,
    rx_led: led_trigger,
    assoc_led: led_trigger,
    radio_led: led_trigger,
    tpt_led: led_trigger,
    tx_led_active: __SlateRecord_types_h_190,
    rx_led_active: __SlateRecord_types_h_190,
    assoc_led_active: __SlateRecord_types_h_190,
    radio_led_active: __SlateRecord_types_h_190,
    tpt_led_active: __SlateRecord_types_h_190,
    tpt_led_trigger: *mut tpt_led_trigger,
    total_ps_buffered: i32,
    pspolling: bool,
    ps_sdata: *mut ieee80211_sub_if_data,
    dynamic_ps_enable_work: wiphy_work,
    dynamic_ps_disable_work: wiphy_work,
    dynamic_ps_timer: timer_list,
    ifa_notifier: notifier_block,
    ifa6_notifier: notifier_block,
    dynamic_ps_forced_timeout: i32,
    user_power_level: i32,
    restart_work: work_struct,
    roc_work: wiphy_delayed_work,
    roc_list: list_head,
    hw_roc_start: wiphy_work,
    hw_roc_done: wiphy_work,
    hw_roc_start_time: u64,
    ack_status_frames: idr,
    ack_status_lock: spinlock,
    monitor_sdata: *mut ieee80211_sub_if_data,
    monitor_chanreq: ieee80211_chan_req,
    ext_capa: [u8; 8],
    wbrf_supported: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ieee802_11_elems {
    ie_start: *const u8,
    total_len: u64,
    crc: u32,
    frame_type: u8,
    from_ap: bool,
    lnk_id: *const ieee80211_tdls_lnkie,
    ch_sw_timing: *const ieee80211_ch_switch_timing,
    ext_capab: *const u8,
    ssid: *const u8,
    supp_rates: *const u8,
    ds_params: *const u8,
    tim: *const ieee80211_tim_ie,
    rsn: *const u8,
    rsnx: *const u8,
    erp_info: *const u8,
    ext_supp_rates: *const u8,
    wmm_info: *const u8,
    wmm_param: *const u8,
    ht_cap_elem: *const ieee80211_ht_cap,
    ht_operation: *const ieee80211_ht_operation,
    vht_cap_elem: *const ieee80211_vht_cap,
    vht_operation: *const ieee80211_vht_operation,
    mesh_config: *const ieee80211_meshconf_ie,
    he_cap: *const u8,
    he_operation: *const ieee80211_he_operation,
    he_spr: *const ieee80211_he_spr,
    mu_edca_param_set: *const ieee80211_mu_edca_param_set,
    he_6ghz_capa: *const ieee80211_he_6ghz_capa,
    uora_element: *const u8,
    mesh_id: *const u8,
    peering: *const u8,
    awake_window: *const u16,
    preq: *const u8,
    prep: *const u8,
    perr: *const u8,
    rann: *const ieee80211_rann_ie,
    ch_switch_ie: *const ieee80211_channel_sw_ie,
    ext_chansw_ie: *const ieee80211_ext_chansw_ie,
    wide_bw_chansw_ie: *const ieee80211_wide_bw_chansw_ie,
    max_channel_switch_time: *const u8,
    country_elem: *const u8,
    pwr_constr_elem: *const u8,
    cisco_dtpc_elem: *const u8,
    timeout_int: *const ieee80211_timeout_interval_ie,
    opmode_notif: *const u8,
    sec_chan_offs: *const ieee80211_sec_chan_offs_ie,
    mesh_chansw_params_ie: *mut ieee80211_mesh_chansw_params_ie,
    max_idle_period_ie: *const ieee80211_bss_max_idle_period_ie,
    mbssid_config_ie: *const ieee80211_multiple_bssid_configuration,
    bssid_index: *const ieee80211_bssid_index,
    max_bssid_indicator: u8,
    dtim_count: u8,
    dtim_period: u8,
    addba_ext_ie: *const ieee80211_addba_ext_ie,
    s1g_capab: *const ieee80211_s1g_cap,
    s1g_oper: *const ieee80211_s1g_oper_ie,
    s1g_bcn_compat: *const ieee80211_s1g_bcn_compat_ie,
    aid_resp: *const ieee80211_aid_response_ie,
    eht_cap: *const ieee80211_eht_cap_elem,
    eht_operation: *const ieee80211_eht_operation,
    ml_basic: *const ieee80211_multi_link_elem,
    ml_reconf: *const ieee80211_multi_link_elem,
    ml_epcs: *const ieee80211_multi_link_elem,
    bandwidth_indication: *const ieee80211_bandwidth_indication,
    ttlm: [*const ieee80211_ttlm_elem; 2],
    uhr_cap: *const ieee80211_uhr_cap,
    uhr_operation: *const ieee80211_uhr_operation,
    tpe: ieee80211_parsed_tpe,
    csa_tpe: ieee80211_parsed_tpe,
    ext_capab_len: u8,
    ssid_len: u8,
    supp_rates_len: u8,
    tim_len: u8,
    rsn_len: u8,
    rsnx_len: u8,
    ext_supp_rates_len: u8,
    wmm_info_len: u8,
    wmm_param_len: u8,
    he_cap_len: u8,
    mesh_id_len: u8,
    peering_len: u8,
    preq_len: u8,
    prep_len: u8,
    perr_len: u8,
    country_elem_len: u8,
    bssid_index_len: u8,
    eht_cap_len: u8,
    uhr_cap_len: u8,
    uhr_operation_len: u8,
    ml_basic_len: u64,
    ml_reconf_len: u64,
    ml_epcs_len: u64,
    ttlm_num: u8,
    prof: *mut ieee80211_mle_per_sta_profile,
    sta_prof_len: u64,
    mbssid_nontx_profile_missing: bool,
    parse_error: u8,
}


#[unsafe(no_mangle)]
extern "C" fn __ieee80211_stop_rx_ba_session(sta: *mut sta_info, tid: u16, initiator: u16, reason: u16, tx: bool) {
let local: *mut ieee80211_local = unsafe { (*sta).local };
let mut tid_rx: *mut tid_ampdu_rx = unsafe { core::mem::zeroed() };
let mut params: ieee80211_ampdu_params = {
let mut __t0: ieee80211_ampdu_params = unsafe { core::mem::zeroed() };
__t0.action = 1u32;
__t0.sta = unsafe { core::ptr::addr_of_mut!((*sta).sta) };
__t0.tid = tid;
__t0.ssn = 0u16;
__t0.amsdu = false;
__t0.timeout = 0u16;
__t0
};
lockdep_assert_wiphy!(unsafe { (*(*sta).local).hw.wiphy });
tid_rx = rcu_dereference_protected__16!(unsafe { *((*sta).ampdu_mlme.tid_rx.as_mut_ptr() as *mut *mut tid_ampdu_rx).wrapping_offset((tid as i32) as isize) }, unsafe { lockdep_is_held(core::ptr::addr_of_mut!((*(*(*sta).local).hw.wiphy).mtx) as *const ()) });
if test_bit__4!(tid, unsafe { (*sta).ampdu_mlme.agg_session_valid.as_mut_ptr() as *mut u64 }) == 0 {
return;
}
loop {
loop {
if !(__native_word__36!() || ((const { core::mem::size_of::<*mut tid_ampdu_rx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2222() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__39!(*unsafe { (*sta).ampdu_mlme.tid_rx.as_mut_ptr() as *mut *mut tid_ampdu_rx }.wrapping_offset((tid as i32) as isize), RCU_INITIALIZER!(core::ptr::null_mut::<()>()));
if true {
break;
}
}
__clear_bit!(tid, unsafe { (*sta).ampdu_mlme.agg_session_valid.as_mut_ptr() as *mut u64 });
loop {
if false {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x017%s: Rx BA session stop requested for %pM tid %u %s reason: %d\n".as_ptr() as *const u8, (*(*sta).sdata).name.as_mut_ptr() as *mut u8, (*sta).sta.addr.as_mut_ptr() as *mut u8, tid as i32, if (initiator as i32) == 0 { c"recipient".as_ptr() as *mut u8 } else { c"initiator".as_ptr() as *mut u8 }, reason as i32) };
}
if true {
break;
}
}
if (unsafe { drv_ampdu_action(local, (*sta).sdata, core::ptr::addr_of_mut!(params)) }) != 0 {
loop {
let mut __v11: i32;
while false {
}
__v11 = unsafe { _printk(c"\x016%s: HW problem - can not stop rx aggregation for %pM tid %d\n".as_ptr() as *const u8, (*(*sta).sdata).name.as_mut_ptr() as *mut u8, (*sta).sta.addr.as_mut_ptr() as *mut u8, tid as i32) };
if true {
break;
}
}
}
// check if this is a self generated aggregation halt
if (initiator as i32) == 0 && tx {
unsafe { ieee80211_send_delba((*sta).sdata, ((*sta).sta.addr.as_mut_ptr() as *mut u8) as *const u8, tid, 0u16, reason, ieee80211_s1g_use_ndp_ba((*sta).sdata as *const ieee80211_sub_if_data, sta as *const sta_info)) };
}
// return here in case tid_rx is not assigned - which will happen if
// IEEE80211_HW_SUPPORTS_REORDERING_BUFFER is set.
if tid_rx.is_null() {
return;
}
unsafe { timer_shutdown_sync(core::ptr::addr_of_mut!((*tid_rx).session_timer)) };
// make sure ieee80211_sta_reorder_release() doesn't re-arm the timer
spin_lock_bh(unsafe { core::ptr::addr_of_mut!((*tid_rx).reorder_lock) });
unsafe {
(*tid_rx).__slate_bits_0.__set_removed(1u8);
}
spin_unlock_bh(unsafe { core::ptr::addr_of_mut!((*tid_rx).reorder_lock) });
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*tid_rx).reorder_timer)) };
unsafe { call_rcu(core::ptr::addr_of_mut!((*tid_rx).callback_head), Some(ieee80211_free_tid_rx)) };
}

#[unsafe(no_mangle)]
extern "C" fn __ieee80211_start_rx_ba_session(sta: *mut sta_info, dialog_token: u8, timeout: u16, start_seq_num: u16, ba_policy: u16, tid: u16, mut buf_size: u16, tx: bool, auto_seq: bool, req_ndp: bool, addba_ext_data: u8) {
let mut __slate_compound_42: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_43: u64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_44: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_46: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_ptr: *mut tid_ampdu_rx = unsafe { core::mem::zeroed() };
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut __v17: bool = unsafe { core::mem::zeroed() };
let mut __v18: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_2224: *mut tid_ampdu_rx = unsafe { core::mem::zeroed() };
let mut __v23: bool = unsafe { core::mem::zeroed() };
let mut tid_rx: core::mem::MaybeUninit<*mut tid_ampdu_rx> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut tid_ampdu_rx = core::ptr::addr_of_mut!(tid_rx) as *mut *mut tid_ampdu_rx;
let mut __v27: bool = unsafe { core::mem::zeroed() };
let mut __v31: bool = unsafe { core::mem::zeroed() };
let mut max_buf_size: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u16 = core::ptr::addr_of_mut!(max_buf_size) as *mut u16;
let mut status: u16 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut params: core::mem::MaybeUninit<ieee80211_ampdu_params> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut ieee80211_ampdu_params = core::ptr::addr_of_mut!(params) as *mut ieee80211_ampdu_params;
let mut tid_agg_rx: core::mem::MaybeUninit<*mut tid_ampdu_rx> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut tid_ampdu_rx = core::ptr::addr_of_mut!(tid_agg_rx) as *mut *mut tid_ampdu_rx;
let mut local: *mut ieee80211_local = unsafe { core::mem::zeroed() };
unsafe {
'__join_4: {
local = (*(*sta).sdata).local;
let __slate_auto_bytes_2 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_37, {
let mut __t2: ieee80211_ampdu_params = core::mem::zeroed();
__t2.action = 0u32;
__t2.sta = core::ptr::addr_of_mut!((*sta).sta);
__t2.tid = tid;
__t2.ssn = start_seq_num;
__t2.amsdu = false;
__t2.timeout = timeout;
__t2
});
let __slate_auto_bytes_3 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ret = 95i32.wrapping_neg();
status = 37;
let __slate_auto_bytes_4 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 2usize);
lockdep_assert_wiphy!(unsafe { (*(*sta).local).hw.wiphy });
if (tid as i32) >= 8 {
while false {
}
} else {
if tx {
__v31 = ieee80211_s1g_use_ndp_ba((*sta).sdata as *const ieee80211_sub_if_data, sta as *const sta_info);
} else {
__v31 = false;
}
if __v31 && !req_ndp {
// According to IEEE 802.11-2024: Inform S1G originator
// ADDBA rejected as NDP BlockAck is preferred
status = 109;
while false {
}
} else {
if (*sta).sta.valid_links == 0 && !(*sta).sta.deflink.ht_cap.ht_supported && !(*sta).sta.deflink.he_cap.has_he && !(*sta).sta.deflink.s1g_cap.s1g {
while false {
}
// send a response anyway, it's an error case if we get here
} else {
if test_sta_flag(sta, 8u32) != 0 {
while false {
}
} else {
if (*sta).sta.valid_links != 0 || (*sta).sta.deflink.eht_cap.has_eht {
*__slate_slot_33 = 1024;
} else {
if (*sta).sta.deflink.he_cap.has_he {
*__slate_slot_33 = 256;
} else {
*__slate_slot_33 = 64;
}
}
// sanity check for incoming parameters:
// check if configuration can support the BA policy
// and if buffer size does not exceeds max value
//
// XXX: check own ht delayed BA capability??
if (ba_policy as i32) != 1 && ((*sta).sta.valid_links != 0 || ((*sta).sta.deflink.ht_cap.cap as i32) & 1024 == 0 || ((*((*sta).sta.deflink.s1g_cap.cap.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) as i32) as u64) & BIT!(1i32) == 0) || (buf_size as i32) > (*__slate_slot_33 as i32) {
status = 38;
loop {
if false {
__v27 = net_ratelimit() != 0;
} else {
__v27 = false;
}
if true {
break;
}
}
} else {
// determine default buffer size
if (buf_size as i32) == 0 {
buf_size = *__slate_slot_33;
}
// make sure the size doesn't exceed the maximum supported by the hw
if (buf_size as i32) > ((*sta).sta.max_rx_aggregation_subframes as i32) {
buf_size = (*sta).sta.max_rx_aggregation_subframes;
}
(*__slate_slot_37).buf_size = buf_size;
while false {
}
if test_bit__4!(tid, (*sta).ampdu_mlme.agg_session_valid.as_mut_ptr() as *mut u64) != 0 {
if (*((*sta).ampdu_mlme.tid_rx_token.as_mut_ptr() as *mut u8).wrapping_offset((tid as i32) as isize) as i32) == (dialog_token as i32) {
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
loop {
if false {
__v23 = net_ratelimit() != 0;
} else {
__v23 = false;
}
if true {
break;
}
}
// We have no API to update the timeout value in the
// driver so reject the timeout update if the timeout
// changed. If it did not change, i.e., no real update,
// just reply with success.
rcu_read_lock();
while false {
}
__UNIQUE_ID_rcu_2224 = core::ptr::read_volatile(((*sta).ampdu_mlme.tid_rx.as_mut_ptr() as *mut *mut tid_ampdu_rx).wrapping_offset((tid as i32) as isize) as *const *mut tid_ampdu_rx);
loop {
if false {
if false {
__v17 = true;
} else {
__v17 = rcu_read_lock_held() != 0;
}
__v18 = !__v17;
} else {
__v18 = false;
}
if !__v18 {
break;
}
}
*__slate_slot_24 = __UNIQUE_ID_rcu_2224;
if !(*__slate_slot_24).is_null() && ((*(*__slate_slot_24)).timeout as i32) == (timeout as i32) {
status = 0;
} else {
status = 37;
}
rcu_read_unlock();
break '__join_4;
} else {
loop {
if false {
__v16 = net_ratelimit() != 0;
} else {
__v16 = false;
}
if true {
break;
}
}
// delete existing Rx BA session on the same tid
__ieee80211_stop_rx_ba_session(sta, tid, 0u16, 32u16, false);
}
}
if _ieee80211_hw_check(core::ptr::addr_of_mut!((*local).hw), 34u32) {
ret = drv_ampdu_action(local, (*sta).sdata, core::ptr::addr_of_mut!(*__slate_slot_37));
while false {
}
if ret == 0 {
status = 0;
}
} else {
__count = buf_size as u64;
__obj_size = struct_size_t__4!(__count);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_44 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_44)
});
alloc_tag_restore!();
} else {
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_46 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_46)
});
}
__obj_ptr = *__slate_slot_7 as *mut tid_ampdu_rx;
if !__obj_ptr.is_null() {
let __v4: u64 = __count;
*core::ptr::addr_of_mut!(*{
__slate_compound_43 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_43)
}) = __v4;
}
// prepare A-MPDU MLME for Rx aggregation
*__slate_slot_38 = __obj_ptr;
if !(*__slate_slot_38).is_null() {
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_38)).reorder_lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_38)).reorder_lock) = *{
__slate_compound_42 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_42)
};
if true {
break;
}
}
// rx timer
timer_setup!(core::ptr::addr_of_mut!((*(*__slate_slot_38)).session_timer), Some(sta_rx_agg_session_timer_expired), 524288i32);
// rx reorder timer
timer_setup!(core::ptr::addr_of_mut!((*(*__slate_slot_38)).reorder_timer), Some(sta_rx_agg_reorder_timer_expired), 0i32);
*__slate_slot_36 = 0;
while *__slate_slot_36 < (buf_size as i32) {
__skb_queue_head_init(core::ptr::addr_of_mut!((*(core::ptr::addr_of_mut!((*(*__slate_slot_38)).reorder) as *mut __SlateRecord_sta_info_h_259).wrapping_offset(*__slate_slot_36 as isize)).buf));
let __v3: i32 = *__slate_slot_36;
*__slate_slot_36 = __v3.wrapping_add(1);
}
ret = drv_ampdu_action(local, (*sta).sdata, core::ptr::addr_of_mut!(*__slate_slot_37));
while false {
}
if ret != 0 {
kfree(*__slate_slot_38 as *const ());
} else {
// update data
(*(*__slate_slot_38)).ssn = start_seq_num;
(*(*__slate_slot_38)).head_seq_num = start_seq_num;
(*(*__slate_slot_38)).buf_size = buf_size;
(*(*__slate_slot_38)).timeout = timeout;
(*(*__slate_slot_38)).stored_mpdu_num = 0;
(*(*__slate_slot_38)).__slate_bits_0.__set_auto_seq(auto_seq as u8);
(*(*__slate_slot_38)).__slate_bits_0.__set_started(0u8);
(*(*__slate_slot_38)).reorder_buf_filtered = 0;
(*(*__slate_slot_38)).tid = tid as u8;
(*(*__slate_slot_38)).sta = sta;
status = 0;
// activate it for RX
_r_a_p__v = *__slate_slot_38 as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__41!(*((*sta).ampdu_mlme.tid_rx.as_mut_ptr() as *mut *mut tid_ampdu_rx).wrapping_offset((tid as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut tid_ampdu_rx));
if true {
break;
}
}
}
if true {
break;
}
}
}
if true {
break;
}
}
if timeout != 0 {
mod_timer(core::ptr::addr_of_mut!((*(*__slate_slot_38)).session_timer), TU_TO_EXP_TIME!(timeout));
(*(*__slate_slot_38)).last_rx = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
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
if (status as i32) == 0 {
__set_bit__4!(tid, (*sta).ampdu_mlme.agg_session_valid.as_mut_ptr() as *mut u64);
__clear_bit!(tid, (*sta).ampdu_mlme.unexpected_agg.as_mut_ptr() as *mut u64);
*((*sta).ampdu_mlme.tid_rx_token.as_mut_ptr() as *mut u8).wrapping_offset((tid as i32) as isize) = dialog_token;
}
if tx {
ieee80211_send_addba_resp(sta, (*sta).sta.addr.as_mut_ptr() as *mut u8, tid, dialog_token, status, 1u16, buf_size, timeout, addba_ext_data);
}
}
}

fn ieee80211_mgmt_ba(skb: *mut sk_buff, da: *const u8, sdata: *mut ieee80211_sub_if_data) -> *mut ieee80211_mgmt {
let mgmt: *mut ieee80211_mgmt = skb_put_zero(skb, 24u32) as *mut ieee80211_mgmt;
ether_addr_copy(unsafe { (*mgmt).da.as_mut_ptr() as *mut u8 }, da);
ether_addr_copy(unsafe { (*mgmt).sa.as_mut_ptr() as *mut u8 }, (unsafe { (*sdata).vif.addr.as_mut_ptr() as *mut u8 }) as *const u8);
if ((unsafe { (*sdata).vif.r#type }) as u32) == 3 || ((unsafe { (*sdata).vif.r#type }) as u32) == 4 || ((unsafe { (*sdata).vif.r#type }) as u32) == 7 {
ether_addr_copy(unsafe { (*mgmt).bssid.as_mut_ptr() as *mut u8 }, (unsafe { (*sdata).vif.addr.as_mut_ptr() as *mut u8 }) as *const u8);
} else {
if ((unsafe { (*sdata).vif.r#type }) as u32) == 2 {
ether_addr_copy(unsafe { (*mgmt).bssid.as_mut_ptr() as *mut u8 }, (unsafe { (*sdata).vif.cfg.ap_addr.as_mut_ptr() as *mut u8 }) as *const u8);
} else {
if ((unsafe { (*sdata).vif.r#type }) as u32) == 1 {
ether_addr_copy(unsafe { (*mgmt).bssid.as_mut_ptr() as *mut u8 }, (unsafe { (*sdata).u.ibss.bssid.as_mut_ptr() as *mut u8 }) as *const u8);
}
}
}
unsafe {
(*mgmt).frame_control = 208u16;
}
mgmt
}

fn ieee80211_tx_skb(sdata: *mut ieee80211_sub_if_data, skb: *mut sk_buff) {
// Send all internal mgmt frames on VO. Accordingly set TID to 7.
unsafe { ieee80211_tx_skb_tid(sdata, skb, 7i32, 1i32.wrapping_neg()) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ieee80211_elems_parse_params {
    mode: u32,
    start: *const u8,
    len: u64,
    r#type: u8,
    filter: u64,
    crc: u32,
    bss: *mut cfg80211_bss,
    link_id: i32,
    from_ap: bool,
}


fn ieee802_11_parse_elems(start: *const u8, len: u64, r#type: u8, bss: *mut cfg80211_bss) -> *mut ieee802_11_elems {
let mut params: ieee80211_elems_parse_params = {
let mut __t0: ieee80211_elems_parse_params = unsafe { core::mem::zeroed() };
__t0.mode = 6u32;
__t0.start = start;
__t0.len = len;
__t0.r#type = r#type;
__t0.bss = bss;
__t0.link_id = 1i32.wrapping_neg();
__t0
};
unsafe { ieee802_11_parse_elems_full(core::ptr::addr_of_mut!(params)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// DOC: RX A-MPDU aggregation
//
// Aggregation on the RX side requires only implementing the
// @ampdu_action callback that is invoked to start/stop any
// block-ack sessions for RX aggregation.
//
// When RX aggregation is started by the peer, the driver is
// notified via @ampdu_action function, with the
// %IEEE80211_AMPDU_RX_START action, and may reject the request
// in which case a negative response is sent to the peer, if it
// accepts it a positive response is sent.
//
// While the session is active, the device/driver are required
// to de-aggregate frames and pass them up one by one to mac80211,
// which will handle the reorder buffer.
//
// When the aggregation session is stopped again by the peer or
// ourselves, the driver's @ampdu_action function will be called
// with the action %IEEE80211_AMPDU_RX_STOP. In this case, the
// call must not fail.
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
#[unsafe(link_section = ".text.slate_distinct.agg_rx.ieee80211_free_tid_rx")]
extern "C" fn ieee80211_free_tid_rx(h: *mut callback_head) {
let tid_rx: *mut tid_ampdu_rx = container_of__9!(h, tid_ampdu_rx);
let mut i: i32 = 0;
while i < ((unsafe { (*tid_rx).buf_size }) as i32) {
__skb_queue_purge(unsafe { core::ptr::addr_of_mut!((*(core::ptr::addr_of_mut!((*tid_rx).reorder) as *mut __SlateRecord_sta_info_h_259).wrapping_offset(i as isize)).buf) });
i = i.wrapping_add(1);
}
unsafe { kfree(tid_rx as *const ()) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.agg_rx.ieee80211_stop_rx_ba_session")]
extern "C" fn ieee80211_stop_rx_ba_session(vif: *mut ieee80211_vif, ba_rx_bitmap: u16, addr: *const u8) {
let sdata: *mut ieee80211_sub_if_data = vif_to_sdata(vif);
let mut sta: *mut sta_info = unsafe { core::mem::zeroed() };
rcu_read_lock();
sta = unsafe { sta_info_get_bss(sdata, addr) };
if sta.is_null() {
rcu_read_unlock();
return;
}
for i in 0i32..16i32 {
if ((ba_rx_bitmap as i32) as u64) & BIT!(i) != 0 {
set_bit(i as i64, (unsafe { (*sta).ampdu_mlme.tid_rx_stop_requested.as_mut_ptr() as *mut u64 }) as *mut u64);
}
}
unsafe { wiphy_work_queue((*(*sta).local).hw.wiphy, core::ptr::addr_of_mut!((*sta).ampdu_mlme.work)) };
rcu_read_unlock();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ieee80211_stop_rx_ba_session_2223: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ieee80211_vif, u16, *const u8)>, *mut ()>(Some(ieee80211_stop_rx_ba_session)) };

/// After accepting the AddBA Request we activated a timer,
/// resetting it after each frame that arrives from the originator.
#[unsafe(link_section = ".text.slate_distinct.agg_rx.sta_rx_agg_session_timer_expired")]
extern "C" fn sta_rx_agg_session_timer_expired(t: *mut timer_list) {
let tid_rx: *mut tid_ampdu_rx = timer_container_of__4!(t);
let sta: *mut sta_info = unsafe { (*tid_rx).sta };
let tid: u8 = unsafe { (*tid_rx).tid };
let timeout: u64 = unsafe { (*tid_rx).last_rx }.wrapping_add(TU_TO_JIFFIES!(unsafe { (*tid_rx).timeout }));
if time_is_after_jiffies!(timeout) {
unsafe { mod_timer(core::ptr::addr_of_mut!((*tid_rx).session_timer), timeout) };
return;
}
loop {
if false {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x017%s: RX session timer expired on %pM tid %d\n".as_ptr() as *const u8, (*(*sta).sdata).name.as_mut_ptr() as *mut u8, (*sta).sta.addr.as_mut_ptr() as *mut u8, tid as i32) };
}
if true {
break;
}
}
set_bit(tid as i64, (unsafe { (*sta).ampdu_mlme.tid_rx_timer_expired.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe { wiphy_work_queue((*(*sta).local).hw.wiphy, core::ptr::addr_of_mut!((*sta).ampdu_mlme.work)) };
}

#[unsafe(link_section = ".text.slate_distinct.agg_rx.sta_rx_agg_reorder_timer_expired")]
extern "C" fn sta_rx_agg_reorder_timer_expired(t: *mut timer_list) {
let tid_rx: *mut tid_ampdu_rx = timer_container_of__2!(t);
rcu_read_lock();
unsafe { ieee80211_release_reorder_timeout((*tid_rx).sta, (*tid_rx).tid as i32) };
rcu_read_unlock();
}

#[unsafe(no_mangle)]
extern "C" fn ieee80211_add_addbaext(skb: *mut sk_buff, req_addba_ext_data: u8, buf_size: u16) {
let mut pos: *mut u8 = unsafe { core::mem::zeroed() };
pos = skb_put_zero(skb, 2u64.wrapping_add(core::mem::size_of::<ieee80211_addba_ext_ie>() as u64) as u32) as *mut u8;
let __v0: *mut u8 = pos;
pos = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = 159;
}
let __v2: *mut u8 = pos;
pos = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = core::mem::size_of::<ieee80211_addba_ext_ie>() as u8;
}
let addba_ext: *mut ieee80211_addba_ext_ie = pos as *mut ieee80211_addba_ext_ie;
unsafe {
(*addba_ext).data = BIT!(0i32) as u8;
}
if req_addba_ext_data != 0 {
unsafe {
(*addba_ext).data = (((*addba_ext).data as i32) & (req_addba_ext_data as i32)) as u8;
}
}
let __v7: u8 = u8_encode_bits(((buf_size as i32) >> 10i32) as u8, GENMASK__4!(7i32, 5i32) as u8);
unsafe {
(*addba_ext).data = (((*addba_ext).data as i32) | (__v7 as i32)) as u8;
}
}

#[unsafe(no_mangle)]
extern "C" fn ieee80211_retrieve_addba_ext_data(sta: *mut sta_info, elem_data: *const (), elem_len: i64, buf_size: *mut u16) -> u8 {
let mut data: u8 = unsafe { core::mem::zeroed() };
let mut buf_size_1k: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u8 = core::ptr::addr_of_mut!(buf_size_1k) as *mut u8;
let mut elems: core::mem::MaybeUninit<*mut ieee802_11_elems> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ieee802_11_elems = core::ptr::addr_of_mut!(elems) as *mut *mut ieee802_11_elems;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
data = 0;
if !(*sta).sta.deflink.he_cap.has_he {
return 0;
} else {
if elem_len <= 0 {
return 0;
} else {
*__slate_slot_5 = ieee802_11_parse_elems(elem_data as *const u8, elem_len as u64, 208u8, core::ptr::null_mut::<cfg80211_bss>());
if !((*__slate_slot_5).is_null() || (*(*__slate_slot_5)).parse_error != 0 || unsafe { (*(*__slate_slot_5)).addba_ext_ie }.is_null()) {
data = (*(*(*__slate_slot_5)).addba_ext_ie).data;
if !buf_size.is_null() && ((*sta).sta.valid_links != 0 || (*sta).sta.deflink.eht_cap.has_eht) {
*__slate_slot_4 = u8_get_bits((*(*(*__slate_slot_5)).addba_ext_ie).data, GENMASK__7!(7i32, 5i32) as u8);
*buf_size = ((*buf_size as i32) | ((*__slate_slot_4 as u16) as i32) << 10i32) as u16;
}
}
kfree(*__slate_slot_5 as *const ());
return data;
}
}
}
}

fn ieee80211_send_addba_resp(sta: *mut sta_info, da: *mut u8, tid: u16, dialog_token: u8, status: u16, policy: u16, buf_size: u16, timeout: u16, req_addba_ext_data: u8) {
let sdata: *mut ieee80211_sub_if_data = unsafe { (*sta).sdata };
let local: *mut ieee80211_local = unsafe { (*sdata).local };
let amsdu: bool = _ieee80211_hw_check(unsafe { core::ptr::addr_of_mut!((*local).hw) }, 31u32);
let use_ndp: bool = unsafe { ieee80211_s1g_use_ndp_ba(sdata as *const ieee80211_sub_if_data, sta as *const sta_info) };
let mut capab: u16 = 0;
let skb: *mut sk_buff = dev_alloc_skb(52u64.wrapping_add(core::mem::size_of::<ieee80211_addba_ext_ie>() as u64).wrapping_add((unsafe { (*local).hw.extra_tx_headroom }) as u64) as u32);
if skb.is_null() {
return;
}
skb_reserve(skb, (unsafe { (*local).hw.extra_tx_headroom }) as i32);
let mgmt: *mut ieee80211_mgmt = ieee80211_mgmt_ba(skb, da as *const u8, sdata);
unsafe { skb_put(skb, 9u32) };
unsafe {
(*mgmt).u.action.category = 3;
}
unsafe {
(*mgmt).u.action.action_code = (if use_ndp { 129i32 } else { 1i32 }) as u8;
}
unsafe {
(*mgmt).u.action.__slate_anon_2.addba_resp.dialog_token = dialog_token;
}
capab = u16_encode_bits(amsdu as u16, 1u16);
let __v0: u16 = u16_encode_bits(policy, 2u16);
capab = ((capab as i32) | (__v0 as i32)) as u16;
let __v3: u16 = u16_encode_bits(tid, 60u16);
capab = ((capab as i32) | (__v3 as i32)) as u16;
let __v6: u16 = u16_encode_bits(buf_size, 65472u16);
capab = ((capab as i32) | (__v6 as i32)) as u16;
unsafe {
(*mgmt).u.action.__slate_anon_2.addba_resp.capab = capab;
}
unsafe {
(*mgmt).u.action.__slate_anon_2.addba_resp.timeout = timeout;
}
unsafe {
(*mgmt).u.action.__slate_anon_2.addba_resp.status = status;
}
if (unsafe { (*sta).sta.valid_links }) != 0 || unsafe { (*sta).sta.deflink.he_cap.has_he } {
ieee80211_add_addbaext(skb, req_addba_ext_data, buf_size);
}
ieee80211_tx_skb(sdata, skb);
}

#[unsafe(no_mangle)]
extern "C" fn ieee80211_process_addba_request(local: *mut ieee80211_local, sta: *mut sta_info, mgmt: *mut ieee80211_mgmt, len: u64) {
let req_ndp: bool = ((unsafe { (*mgmt).u.action.action_code }) as i32) == 128;
let mut capab: u16 = 0;
let mut tid: u16 = 0;
let mut timeout: u16 = 0;
let mut ba_policy: u16 = 0;
let mut buf_size: u16 = 0;
let mut start_seq_num: u16 = 0;
let mut dialog_token: u8 = 0;
let mut addba_ext_data: u8 = 0;
// extract session parameters from addba request frame
dialog_token = unsafe { (*mgmt).u.action.__slate_anon_2.addba_req.dialog_token };
timeout = unsafe { (*mgmt).u.action.__slate_anon_2.addba_req.timeout };
start_seq_num = (((unsafe { (*mgmt).u.action.__slate_anon_2.addba_req.start_seq_num }) as i32) >> 4i32) as u16;
capab = unsafe { (*mgmt).u.action.__slate_anon_2.addba_req.capab };
ba_policy = (((capab as i32) & 2) >> 1i32) as u16;
tid = (((capab as i32) & 60) >> 2i32) as u16;
buf_size = (((capab as i32) & 65472) >> 6i32) as u16;
addba_ext_data = ieee80211_retrieve_addba_ext_data(sta, (unsafe { core::ptr::addr_of_mut!((*mgmt).u.action.__slate_anon_2.addba_req.variable) as *mut u8 }) as *const (), len.wrapping_sub(offsetof__36!()) as i64, core::ptr::addr_of_mut!(buf_size));
__ieee80211_start_rx_ba_session(sta, dialog_token, timeout, start_seq_num, ba_policy, tid, buf_size, true, false, req_ndp, addba_ext_data);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.agg_rx.ieee80211_manage_rx_ba_offl")]
extern "C" fn ieee80211_manage_rx_ba_offl(vif: *mut ieee80211_vif, addr: *const u8, tid: u32) {
let mut sta: core::mem::MaybeUninit<*mut sta_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut sta_info = core::ptr::addr_of_mut!(sta) as *mut *mut sta_info;
unsafe {
let sdata: *mut ieee80211_sub_if_data = vif_to_sdata(vif);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
*__slate_slot_0 = sta_info_get_bss(sdata, addr);
if !(*__slate_slot_0).is_null() {
set_bit(tid as i64, ((*(*__slate_slot_0)).ampdu_mlme.tid_rx_manage_offl.as_mut_ptr() as *mut u64) as *mut u64);
wiphy_work_queue((*(*(*__slate_slot_0)).local).hw.wiphy, core::ptr::addr_of_mut!((*(*__slate_slot_0)).ampdu_mlme.work));
}
rcu_read_unlock();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ieee80211_manage_rx_ba_offl_2229: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ieee80211_vif, *const u8, u32)>, *mut ()>(Some(ieee80211_manage_rx_ba_offl)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.agg_rx.ieee80211_rx_ba_timer_expired")]
extern "C" fn ieee80211_rx_ba_timer_expired(vif: *mut ieee80211_vif, addr: *const u8, tid: u32) {
let mut sta: core::mem::MaybeUninit<*mut sta_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut sta_info = core::ptr::addr_of_mut!(sta) as *mut *mut sta_info;
unsafe {
let sdata: *mut ieee80211_sub_if_data = vif_to_sdata(vif);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
*__slate_slot_0 = sta_info_get_bss(sdata, addr);
if !(*__slate_slot_0).is_null() {
set_bit(tid as i64, ((*(*__slate_slot_0)).ampdu_mlme.tid_rx_timer_expired.as_mut_ptr() as *mut u64) as *mut u64);
wiphy_work_queue((*(*(*__slate_slot_0)).local).hw.wiphy, core::ptr::addr_of_mut!((*(*__slate_slot_0)).ampdu_mlme.work));
}
rcu_read_unlock();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ieee80211_rx_ba_timer_expired_2230: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ieee80211_vif, *const u8, u32)>, *mut ()>(Some(ieee80211_rx_ba_timer_expired)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_skb_dst_3: [u8; 1] = *b"\0";

static __slate_asm_string_skb_dst_4: [u8; 52] = *b"/home/takashi/c-corpus/linux/include/linux/skbuff.h\0";
