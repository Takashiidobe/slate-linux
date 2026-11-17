
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
//!  linux/fs/ext4/namei.c
//!
//! Copyright (C) 1992, 1993, 1994, 1995
//! Remy Card (card@masi.ibp.fr)
//! Laboratoire MASI - Institut Blaise Pascal
//! Universite Pierre et Marie Curie (Paris VI)
//!
//!  from
//!
//!  linux/fs/minix/namei.c
//!
//!  Copyright (C) 1991, 1992  Linus Torvalds
//!
//!  Big-endian to little-endian byte-swapping/bitmaps by
//!        David S. Miller (davem@caip.rutgers.edu), 1995
//!  Directory entry file type support and forward compatibility hooks
//! 	for B-tree directories by Theodore Ts'o (tytso@mit.edu), 1998
//!  Hash Tree Directory indexing (c)
//! 	Daniel Phillips, 2001
//!  Hash Tree Directory indexing porting
//! 	Christopher Li, 2002
//!  Hash Tree Directory indexing cleanup
//! 	Theodore Ts'o, 2002
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Internal kernel timer frequency
//! Sign Flag
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! This counts to 15. Any more, it will return 16th argument.
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! writeable
//! Operation not permitted
//! File name too long
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
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
//! @a is a power of 2 value
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
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Used in tsk->__state:
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! these are defined by POSIX and also present in glibc's dirent.h
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! no-op
//! Don't overwrite target
//! File supports atomic writes
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
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
//! We borrow the lowest bit in the i_version to use as a flag to tell whether
//! it has been queried since we last incremented it. If it has, then we must
//! increment it on the next change. After that, we can clear the flag and
//! avoid incrementing it again until it has again been queried.
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! Miscellaneous stuff.
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! Set this bit for virtual address instead of SG list.
//! Largest digest size among all hash algorithms supported by fs-verity.
//! Currently assumed to be <= size of fsverity_descriptor::root_hash.
//! Besides EXTENT_STATUS_REFERENCED, all these extent type masks
//! are exclusive, only one type can be set at a time.
//! Fast commit tags
//! Number of credits needed if we need to insert an entry into a
//! directory.  For each new index block, we need 4 blocks (old index
//! block, new index block, bitmap block, bg summary).  For normal
//! htree directories there are 2 levels; if the largedir feature
//! enabled it's 3 levels.
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! a_type field in acl_user_posix_entry_t
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
// __must_not_hold() - function attribute, caller must not hold context lock
//
// Function attribute declaring that the caller must not hold the given context
// lock instance(s).
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
// Overflow Flag
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// File exists
// Cross-device link
// Invalid argument
// Text file busy
// No space left on device
// Read-only file system
// Too many links
// Directory not empty
// Bad CRC detected
// Value too large for defined data type
// Operation not supported on transport endpoint
// Operation now in progress
// Stale file handle
// Filesystem is corrupted
// Required key not available
// Memory page has hardware error
// restart by calling sys_restart_syscall
// Operation is not supported
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
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
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// Exchange source and dest
// Whiteout source
// per-IO O_SYNC
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
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Implementation specific max size
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// struct hash_alg_common - define properties of message digest
// @digestsize: Size of the result of the transformation. A buffer of this size
// 	        must be available to the @final and @finup calls, so they can
// 	        store the resulting hash into it. For various predefined sizes,
// 	        search include/crypto/ using
// 	        git grep _DIGEST_SIZE include/crypto.
// @statesize: Size of the block for partial state of the transformation. A
// 	       buffer of this size must be passed to the @export function as it
// 	       will save the partial state of the transformation into it. On the
// 	       other side, the @import function will load the state from a
// 	       buffer of this size as well.
// @base: Start of data structure of cipher algorithm. The common data
// 	  structure of crypto_alg contains information common to all ciphers.
// 	  The hash_alg_common data structure now adds the hash-specific
// 	  information.
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
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
macro_rules! ARRAY_SIZE__28 {
() => {
((core::mem::size_of::<[*mut buffer_head; 8]>() as u64) / (core::mem::size_of::<*mut buffer_head>() as u64)).wrapping_add(__must_be_array__41!() as u64)
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

macro_rules! BUFFER_TRACE {
() => {
while false {
}
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
};
}

macro_rules! EXT4_BLOCK_SIZE_BITS {
($s:expr) => {
unsafe { (*$s).s_blocksize_bits }
};
}

macro_rules! EXT4_BLOCK_SIZE {
($s:expr) => {
unsafe { (*$s).s_blocksize }
};
}

macro_rules! EXT4_DATA_TRANS_BLOCKS {
($sb:expr) => {
EXT4_SINGLEDATA_TRANS_BLOCKS!($sb).wrapping_add(6).wrapping_sub(2).wrapping_add(EXT4_MAXQUOTAS_TRANS_BLOCKS!($sb) as u32)
};
}

/// Access the hashes at the end of ext4_dir_entry_2
macro_rules! EXT4_DIRENT_HASHES {
($entry:expr) => {
((unsafe { ((($entry) as *mut ()) as *const u8).wrapping_offset((8i32.wrapping_add((*$entry).name_len as i32).wrapping_add(3i32) & -4i32) as isize) }) as *mut ()) as *mut ext4_dir_entry_hash
};
}

macro_rules! EXT4_DIRENT_HASH {
($entry:expr) => {
unsafe { (*EXT4_DIRENT_HASHES!($entry)).hash }
};
}

macro_rules! EXT4_DIRENT_MINOR_HASH {
($entry:expr) => {
unsafe { (*EXT4_DIRENT_HASHES!($entry)).minor_hash }
};
}

macro_rules! EXT4_DIRENT_TAIL {
($block:expr, $blocksize:expr) => {
((unsafe { ((($block) as *mut ()) as *const u8).wrapping_offset((($blocksize) as u64).wrapping_sub(core::mem::size_of::<ext4_dir_entry_tail>() as u64) as isize) }) as *mut ()) as *mut ext4_dir_entry_tail
};
}

macro_rules! EXT4_DIRENT_TAIL__2 {
($block:expr, $blocksize:expr) => {
((unsafe { ((($block) as *mut ()) as *const u8).wrapping_offset($blocksize.wrapping_sub(core::mem::size_of::<ext4_dir_entry_tail>() as u64) as isize) }) as *mut ()) as *mut ext4_dir_entry_tail
};
}

macro_rules! EXT4_DIR_LINK_EMPTY {
($dir:expr) => {
(unsafe { (*$dir).__slate_anon_12.i_nlink }) == 2 || (unsafe { (*$dir).__slate_anon_12.i_nlink }) == 1
};
}

macro_rules! EXT4_DIR_LINK_MAX {
($dir:expr) => {
{
let __v: bool = if (unsafe { (*$dir).__slate_anon_12.i_nlink }) >= 65000 {
let __v2: bool = ext4_has_feature_dir_nlink(unsafe { (*$dir).i_sb }) && is_dx!($dir);
!__v2
} else {
false
};
__v as i64
}
};
}

macro_rules! EXT4_FIRST_INO {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_first_ino }
};
}

macro_rules! EXT4_I {
($inode:expr) => {
{
let mut __v: *mut ext4_inode_info;
__v = ((unsafe { ((($inode) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(ext4_inode_info) as isize)) }) as *mut ()) as *mut ext4_inode_info;
__v
}
};
}

macro_rules! EXT4_LBLK_TO_B {
($inode:expr, $lblk:expr) => {
(($lblk) as i64) << ((unsafe { (*$inode).i_blkbits }) as i32)
};
}

macro_rules! EXT4_MAXQUOTAS_TRANS_BLOCKS {
($sb:expr) => {
3i32.wrapping_mul(EXT4_QUOTA_TRANS_BLOCKS!($sb))
};
}

/// Amount of blocks needed for quota update - we know that the structure was
/// allocated so we need to update only data block
macro_rules! EXT4_QUOTA_TRANS_BLOCKS {
($sb:expr) => {
if ext4_quota_capable($sb) { 1i32 } else { 0i32 }
};
}

macro_rules! EXT4_SINGLEDATA_TRANS_BLOCKS {
($sb:expr) => {
if ext4_has_feature_extents($sb) { 20u32 } else { 8u32 }
};
}

macro_rules! IS_CASEFOLDED {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 32768u32
};
}

macro_rules! IS_DIRSYNC {
($inode:expr) => {
__IS_FLG!($inode, BIT!(4i32) | BIT!(7i32)) != 0 || (unsafe { (*$inode).i_flags }) & 65u32 != 0
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

macro_rules! IS_ENCRYPTED {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 16384u32
};
}

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! S_ISDIR {
($m:expr) => {
(($m) as i32) & 61440 == 16384
};
}

macro_rules! S_ISLNK {
($m:expr) => {
(($m) as i32) & 61440 == 40960
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VFS_WARN_ON_INODE {
() => {
BUILD_BUG_ON_INVALID!();
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

/// Note that nosuid etc flags are inode-specific: setting some file-system
/// flags just means all the inodes inherit those flags by default. It might be
/// possible to override it selectively if you really wanted to with some
/// ioctl() that is not currently implemented.
///
/// Exception: SB_RDONLY is always applied to the entire file system.
///
/// Unfortunately, it is possible to change a filesystems flags with it mounted
/// with files in use.  This means that all of the inodes will not have their
/// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
/// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
macro_rules! __IS_FLG {
($inode:expr, $flg:expr) => {
(unsafe { (*(*$inode).i_sb).s_flags }) & $flg
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, ($val) as u32) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i64, $val) };
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

macro_rules! __must_be_array__41 {
() => {
core::mem::size_of::<__SlateRecord_namei_c_1602>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
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
macro_rules! __verify_pcpu_ptr__3 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<__SlateRecord_types_h_197>() as *const ();
if true {
break;
}
}
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
let __vpp_verify: *const () = core::ptr::null_mut::<i64>() as *const ();
if true {
break;
}
}
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
macro_rules! __verify_pcpu_ptr__2 {
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

macro_rules! cond_resched__3 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/fs/ext4/namei.c".as_ptr() as *const u8, 1313i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__10 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/fs/ext4/namei.c".as_ptr() as *const u8, 1594i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__12 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/fs/ext4/namei.c".as_ptr() as *const u8, 1227i32, 0u32);
__v = _cond_resched();
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! ext4_check_dir_entry {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_empty_dir".as_ptr() as *const u8, 3103u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, $offset) })
};
}

macro_rules! ext4_check_dir_entry__2 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_empty_dir".as_ptr() as *const u8, 3093u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__3 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_empty_dir".as_ptr() as *const u8, 3127u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, $offset) })
};
}

macro_rules! ext4_check_dir_entry__5 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_get_first_dir_block".as_ptr() as *const u8, 3559u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, $offset) })
};
}

macro_rules! ext4_check_dir_entry__7 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_find_dest_de".as_ptr() as *const u8, 2052u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, ($buf) as *mut u8, $size, $offset) })
};
}

macro_rules! ext4_check_dir_entry__9 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"htree_dirblock_to_tree".as_ptr() as *const u8, 1080u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__11 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_get_first_dir_block".as_ptr() as *const u8, 3547u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__12 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"make_indexed_dir".as_ptr() as *const u8, 2269u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__14 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_generic_delete_entry".as_ptr() as *const u8, 2675u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, ($buf) as *mut u8, $size, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__15 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"dx_make_map".as_ptr() as *const u8, 1296u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, ($size) as i32, ($offset) as u32) })
};
}

macro_rules! ext4_check_dir_entry__16 {
($dir:expr, $de:expr, $bh:expr, $buf:expr, $size:expr, $offset:expr) => {
unlikely__2!(unsafe { __ext4_check_dir_entry(c"ext4_search_dir".as_ptr() as *const u8, 1476u32, $dir, core::ptr::null_mut::<file>(), $de, $bh, $buf, $size, $offset) })
};
}

macro_rules! ext4_handle_dirty_metadata__11 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_init_symlink_block".as_ptr() as *const u8, 3357u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__28 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_handle_dirty_dx_node".as_ptr() as *const u8, 519u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__59 {
($handle:expr, $inode:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_handle_dirty_dirblock".as_ptr() as *const u8, 409u32, $handle, $inode, $bh) }
};
}

macro_rules! ext4_iget__4 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_get_parent".as_ptr() as *const u8, 1838u32) }
};
}

macro_rules! ext4_iget__19 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, ($ino) as u64, ($flags) as u32, c"ext4_lookup".as_ptr() as *const u8, 1787u32) }
};
}

macro_rules! ext4_journal_get_write_access__2 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_dx_add_entry".as_ptr() as *const u8, 2573u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__3 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_setent".as_ptr() as *const u8, 3650u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__9 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_append".as_ptr() as *const u8, 93u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__23 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"do_split".as_ptr() as *const u8, 1934u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__27 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_init_symlink_block".as_ptr() as *const u8, 3349u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__34 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"make_indexed_dir".as_ptr() as *const u8, 2235u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__36 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_delete_entry".as_ptr() as *const u8, 2729u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__37 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"do_split".as_ptr() as *const u8, 1940u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__42 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_rename_dir_prepare".as_ptr() as *const u8, 3611u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__43 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_dx_add_entry".as_ptr() as *const u8, 2559u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__49 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"add_dirent_to_buf".as_ptr() as *const u8, 2130u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__59 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_dx_add_entry".as_ptr() as *const u8, 2501u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_start__6 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 3465u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__12 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 3883u32, $type, $nblocks, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__43 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 3177u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__47 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 4113u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_start__48 {
($inode:expr, $type:expr, $nblocks:expr) => {
__ext4_journal_start($inode, 3262u32, $type, ($nblocks) as i32, 0i32, ext4_trans_default_revoke_credits(unsafe { (*$inode).i_sb }))
};
}

macro_rules! ext4_journal_stop {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"__ext4_unlink".as_ptr() as *const u8, 3295u32, $handle) };
};
}

macro_rules! ext4_journal_stop__7 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_mkdir".as_ptr() as *const u8, 3038u32, $handle) };
};
}

macro_rules! ext4_journal_stop__13 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_rename".as_ptr() as *const u8, 4040u32, $handle) };
};
}

macro_rules! ext4_journal_stop__15 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_tmpfile".as_ptr() as *const u8, 2920u32, $handle) };
};
}

macro_rules! ext4_journal_stop__21 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_create".as_ptr() as *const u8, 2842u32, $handle) };
};
}

macro_rules! ext4_journal_stop__26 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_symlink".as_ptr() as *const u8, 3447u32, $handle) };
};
}

macro_rules! ext4_journal_stop__30 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_mkdir".as_ptr() as *const u8, 3055u32, $handle) };
};
}

macro_rules! ext4_journal_stop__40 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_cross_rename".as_ptr() as *const u8, 4190u32, $handle) };
};
}

macro_rules! ext4_journal_stop__49 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_rename".as_ptr() as *const u8, 4043u32, $handle) };
};
}

macro_rules! ext4_journal_stop__52 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_whiteout_for_rename".as_ptr() as *const u8, 3779u32, $handle) };
};
}

macro_rules! ext4_journal_stop__60 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_mknod".as_ptr() as *const u8, 2876u32, $handle) };
};
}

macro_rules! ext4_journal_stop__64 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_tmpfile".as_ptr() as *const u8, 2915u32, $handle) };
};
}

macro_rules! ext4_journal_stop__69 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_rmdir".as_ptr() as *const u8, 3224u32, $handle) };
};
}

macro_rules! ext4_journal_stop__70 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_symlink".as_ptr() as *const u8, 3401u32, $handle) };
};
}

macro_rules! ext4_journal_stop__77 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"__ext4_link".as_ptr() as *const u8, 3488u32, $handle) };
};
}

macro_rules! ext4_journal_stop__84 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_symlink".as_ptr() as *const u8, 3437u32, $handle) };
};
}

macro_rules! ext4_mark_inode_dirty__9 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"make_indexed_dir".as_ptr() as *const u8, 2342u32) };
};
}

macro_rules! ext4_mark_inode_dirty__12 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_append".as_ptr() as *const u8, 88u32) }
};
}

macro_rules! ext4_mark_inode_dirty__13 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_mkdir".as_ptr() as *const u8, 3045u32) }
};
}

macro_rules! ext4_mark_inode_dirty__14 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_add_entry".as_ptr() as *const u8, 2408u32) }
};
}

macro_rules! ext4_mark_inode_dirty__17 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_mkdir".as_ptr() as *const u8, 3027u32) }
};
}

macro_rules! ext4_mark_inode_dirty__19 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 3933u32) }
};
}

macro_rules! ext4_mark_inode_dirty__23 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 3988u32) }
};
}

macro_rules! ext4_mark_inode_dirty__25 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rmdir".as_ptr() as *const u8, 3204u32) }
};
}

macro_rules! ext4_mark_inode_dirty__26 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_symlink".as_ptr() as *const u8, 3443u32) };
};
}

macro_rules! ext4_mark_inode_dirty__28 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_update_dir_count".as_ptr() as *const u8, 3752u32) };
};
}

macro_rules! ext4_mark_inode_dirty__29 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 4022u32) }
};
}

macro_rules! ext4_mark_inode_dirty__30 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 3957u32) }
};
}

macro_rules! ext4_mark_inode_dirty__33 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_cross_rename".as_ptr() as *const u8, 4165u32) }
};
}

macro_rules! ext4_mark_inode_dirty__34 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_mkdir".as_ptr() as *const u8, 3035u32) }
};
}

macro_rules! ext4_mark_inode_dirty__38 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_setent".as_ptr() as *const u8, 3658u32) }
};
}

macro_rules! ext4_mark_inode_dirty__48 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_unlink".as_ptr() as *const u8, 3277u32) }
};
}

macro_rules! ext4_mark_inode_dirty__49 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_add_nondir".as_ptr() as *const u8, 2801u32) };
};
}

macro_rules! ext4_mark_inode_dirty__50 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"add_dirent_to_buf".as_ptr() as *const u8, 2153u32) }
};
}

macro_rules! ext4_mark_inode_dirty__60 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 3993u32) }
};
}

macro_rules! ext4_mark_inode_dirty__67 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_add_nondir".as_ptr() as *const u8, 2793u32) }
};
}

macro_rules! ext4_mark_inode_dirty__68 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename".as_ptr() as *const u8, 4036u32) };
};
}

macro_rules! ext4_mark_inode_dirty__71 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rename_dir_finish".as_ptr() as *const u8, 3634u32) }
};
}

macro_rules! ext4_mark_inode_dirty__73 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_cross_rename".as_ptr() as *const u8, 4162u32) }
};
}

macro_rules! ext4_mark_inode_dirty__75 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"ext4_rmdir".as_ptr() as *const u8, 3210u32) }
};
}

macro_rules! ext4_mark_inode_dirty__77 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_unlink".as_ptr() as *const u8, 3291u32) }
};
}

macro_rules! ext4_mark_inode_dirty__78 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_link".as_ptr() as *const u8, 3477u32) }
};
}

macro_rules! ext4_new_inode_start_handle {
($idmap:expr, $dir:expr, $mode:expr, $qstr:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, $mode, $qstr, ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 2865u32, $nblocks) }
};
}

macro_rules! ext4_new_inode_start_handle__2 {
($idmap:expr, $dir:expr, $mode:expr, $qstr:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, ($mode) as u16, $qstr, ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 3397u32, $nblocks) }
};
}

macro_rules! ext4_new_inode_start_handle__3 {
($idmap:expr, $dir:expr, $mode:expr, $qstr:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, $mode, $qstr, ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 3016u32, $nblocks) }
};
}

macro_rules! ext4_new_inode_start_handle__4 {
($idmap:expr, $dir:expr, $mode:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, $mode, core::ptr::null::<qstr>(), ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 2900u32, ($nblocks) as i32) }
};
}

macro_rules! ext4_new_inode_start_handle__5 {
($idmap:expr, $dir:expr, $mode:expr, $qstr:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, ($mode) as u16, $qstr, ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 3774u32, $nblocks) }
};
}

macro_rules! ext4_new_inode_start_handle__6 {
($idmap:expr, $dir:expr, $mode:expr, $qstr:expr, $goal:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_new_inode($idmap, core::ptr::null_mut::<jbd2_journal_handle>(), $dir, $mode, $qstr, ($goal) as u32, core::ptr::null_mut::<u32>(), 0u32, $type, 2830u32, $nblocks) }
};
}

macro_rules! ext4_read_dirblock {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, ($block) as u32, ($type) as u32, c"ext4_empty_dir".as_ptr() as *const u8, 3087u32)
};
}

macro_rules! ext4_read_dirblock__2 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, ($block) as u32, ($type) as u32, c"dx_probe".as_ptr() as *const u8, 791u32)
};
}

macro_rules! ext4_read_dirblock__3 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"ext4_empty_dir".as_ptr() as *const u8, 3116u32)
};
}

macro_rules! ext4_read_dirblock__4 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"dx_probe".as_ptr() as *const u8, 910u32)
};
}

macro_rules! ext4_read_dirblock__5 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, ($block) as u32, ($type) as u32, c"ext4_get_first_dir_block".as_ptr() as *const u8, 3539u32)
};
}

macro_rules! ext4_read_dirblock__6 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"__ext4_add_entry".as_ptr() as *const u8, 2414u32)
};
}

macro_rules! ext4_read_dirblock__7 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"ext4_htree_next_block".as_ptr() as *const u8, 1021u32)
};
}

macro_rules! ext4_read_dirblock__8 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"ext4_dx_find_entry".as_ptr() as *const u8, 1728u32)
};
}

macro_rules! ext4_read_dirblock__9 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"ext4_dx_add_entry".as_ptr() as *const u8, 2493u32)
};
}

macro_rules! ext4_read_dirblock__10 {
($inode:expr, $block:expr, $type:expr) => {
__ext4_read_dirblock($inode, $block, ($type) as u32, c"htree_dirblock_to_tree".as_ptr() as *const u8, 1051u32)
};
}

macro_rules! ext4_std_error__7 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"make_indexed_dir".as_ptr() as *const u8, 2237u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__19 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"add_dirent_to_buf".as_ptr() as *const u8, 2157u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__20 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_append".as_ptr() as *const u8, 100u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__22 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_setent".as_ptr() as *const u8, 3663u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__26 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_delete_entry".as_ptr() as *const u8, 2746u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__38 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"do_split".as_ptr() as *const u8, 2029u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__39 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_resetent".as_ptr() as *const u8, 3688u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__44 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"add_dirent_to_buf".as_ptr() as *const u8, 2132u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__47 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_dx_add_entry".as_ptr() as *const u8, 2643u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__51 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_rename_dir_finish".as_ptr() as *const u8, 3637u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! fname_len {
($p:expr) => {
unsafe { (*$p).disk_name.len }
};
}

macro_rules! fname_name {
($p:expr) => {
unsafe { (*$p).disk_name.name }
};
}

macro_rules! is_dx {
($dir:expr) => {
{
let __v: bool = ext4_has_feature_dir_index(unsafe { (*$dir).i_sb }) && ext4_test_inode_flag($dir, 12i32) != 0;
__v
}
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, vfs_inode) as u64
};
}

macro_rules! offsetof__45 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dt_checksum) as u64
};
}

macro_rules! offsetof__17 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, name_len) as u64
};
}

macro_rules! smp_mb__before_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
if true {
break;
}
}
};
}

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__2 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

/// swap - swap values of @a and @b
/// @a: first value
/// @b: second value
macro_rules! swap {
($a:expr, $b:expr) => {
loop {
let __tmp: *mut buffer_head = unsafe { $a };
unsafe {
$a = $b;
}
$b = __tmp;
if true {
break;
}
}
};
}

/// swap - swap values of @a and @b
/// @a: first value
/// @b: second value
macro_rules! swap__2 {
($a:expr, $b:expr) => {
loop {
let __tmp: dx_map_entry = unsafe { $a };
unsafe {
$a = $b;
}
unsafe {
$b = __tmp;
}
if true {
break;
}
}
};
}

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit__6 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__3 {
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

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: bool = true && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit_acquire {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
generic_test_bit_acquire(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit_acquire(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
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

macro_rules! warn_no_space_for_csum {
($inode:expr) => {
__warn_no_space_for_csum($inode, c"ext4_dirblock_csum_set".as_ptr() as *const u8, 396u32);
};
}

macro_rules! warn_no_space_for_csum__2 {
($inode:expr) => {
__warn_no_space_for_csum($inode, c"ext4_dirblock_csum_verify".as_ptr() as *const u8, 375u32);
};
}

macro_rules! warn_no_space_for_csum__3 {
($inode:expr) => {
__warn_no_space_for_csum($inode, c"ext4_dx_csum_set".as_ptr() as *const u8, 506u32);
};
}

macro_rules! warn_no_space_for_csum__4 {
($inode:expr) => {
__warn_no_space_for_csum($inode, c"ext4_dx_csum_verify".as_ptr() as *const u8, 477u32);
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__might_resched: static_call_key;
    static mut __cpu_online_mask: cpumask;
    static mut cpu_number: i32;
    static mut __SCK__cond_resched: static_call_key;
    static mut sighand_cachep: *mut kmem_cache;
    static mut dotdot_name: qstr;
    static mut dquot_operations: dquot_operations;
    static mut ext4_dir_operations: file_operations;
    static mut ext4_file_inode_operations: inode_operations;
    static mut ext4_file_operations: file_operations;
    static mut ext4_encrypted_symlink_inode_operations: inode_operations;
    static mut ext4_symlink_inode_operations: inode_operations;
    static mut ext4_fast_symlink_inode_operations: inode_operations;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_ext4_unlink_enter: static_call_key;
    static mut __tracepoint_ext4_unlink_enter: tracepoint;
    static mut __SCK__tp_func_ext4_unlink_exit: static_call_key;
    static mut __tracepoint_ext4_unlink_exit: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strlen(arg0: *const u8) -> u64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __SCT__cond_resched() -> i32;
    fn rcu_is_watching() -> bool;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn d_instantiate(arg0: *mut dentry, arg1: *mut inode);
    fn d_instantiate_new(arg0: *mut dentry, arg1: *mut inode);
    fn d_splice_alias(arg0: *mut inode, arg1: *mut dentry) -> *mut dentry;
    fn d_obtain_alias(arg0: *mut inode) -> *mut dentry;
    fn d_invalidate(arg0: *mut dentry);
    fn d_tmpfile(arg0: *mut file, arg1: *mut inode);
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn inode_set_ctime_current(arg0: *mut inode) -> timespec64;
    fn __compiletime_assert_412();
    fn __compiletime_assert_413();
    fn __mark_inode_dirty(arg0: *mut inode, arg1: i32);
    fn inc_nlink(arg0: *mut inode);
    fn drop_nlink(arg0: *mut inode);
    fn clear_nlink(arg0: *mut inode);
    fn set_nlink(arg0: *mut inode, arg1: u32);
    fn ihold(arg0: *mut inode);
    fn iput(arg0: *mut inode);
    fn finish_open(arg0: *mut file, arg1: *mut dentry, arg2: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>) -> i32;
    fn init_special_inode(arg0: *mut inode, arg1: u16, arg2: u32);
    fn unlock_new_inode(arg0: *mut inode);
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn dquot_initialize(arg0: *mut inode) -> i32;
    fn __wait_on_buffer(arg0: *mut buffer_head);
    fn __brelse(arg0: *mut buffer_head);
    fn inode_maybe_inc_iversion(arg0: *mut inode, arg1: bool) -> bool;
    fn crc32c(arg0: u32, arg1: *const (), arg2: u64) -> u32;
    fn __builtin_unreachable();
    fn ext4_should_retry_alloc(arg0: *mut super_block, arg1: *mut i32) -> i32;
    fn __ext4_check_dir_entry(arg0: *const u8, arg1: u32, arg2: *mut inode, arg3: *mut file, arg4: *mut ext4_dir_entry_2, arg5: *mut buffer_head, arg6: *mut u8, arg7: i32, arg8: u32) -> i32;
    fn ext4_htree_store_dirent(arg0: *mut file, arg1: u32, arg2: u32, arg3: *mut ext4_dir_entry_2, arg4: *mut fscrypt_str) -> i32;
    fn ext4fs_dirhash(arg0: *const inode, arg1: *const u8, arg2: i32, arg3: *mut dx_hash_info) -> i32;
    fn __ext4_new_inode(arg0: *mut mnt_idmap, arg1: *mut jbd2_journal_handle, arg2: *mut inode, arg3: u16, arg4: *const qstr, arg5: u32, arg6: *mut u32, arg7: u32, arg8: i32, arg9: u32, arg10: i32) -> *mut inode;
    fn __ext4_fc_track_unlink(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut dentry);
    fn __ext4_fc_track_link(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut dentry);
    fn ext4_fc_track_unlink(arg0: *mut jbd2_journal_handle, arg1: *mut dentry);
    fn ext4_fc_track_link(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut dentry);
    fn __ext4_fc_track_create(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut dentry);
    fn ext4_fc_track_create(arg0: *mut jbd2_journal_handle, arg1: *mut dentry);
    fn ext4_fc_mark_ineligible(arg0: *mut super_block, arg1: i32, arg2: *mut jbd2_journal_handle);
    fn ext4_bread(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: u32, arg3: i32) -> *mut buffer_head;
    fn ext4_bread_batch(arg0: *mut inode, arg1: u32, arg2: i32, arg3: bool, arg4: *mut *mut buffer_head) -> i32;
    fn __ext4_iget(arg0: *mut super_block, arg1: u64, arg2: u32, arg3: *const u8, arg4: u32) -> *mut inode;
    fn ext4_setattr(arg0: *mut mnt_idmap, arg1: *mut dentry, arg2: *mut iattr) -> i32;
    fn ext4_getattr(arg0: *mut mnt_idmap, arg1: *const path, arg2: *mut kstat, arg3: u32, arg4: u32) -> i32;
    fn ext4_alloc_da_blocks(arg0: *mut inode) -> i32;
    fn ext4_set_aops(arg0: *mut inode);
    fn ext4_fileattr_set(arg0: *mut mnt_idmap, arg1: *mut dentry, arg2: *mut file_kattr) -> i32;
    fn ext4_fileattr_get(arg0: *mut dentry, arg1: *mut file_kattr) -> i32;
    fn __ext4_error_inode(arg0: *mut inode, arg1: *const u8, arg2: u32, arg3: u64, arg4: i32, arg5: *const u8, ...);
    fn __ext4_std_error(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: i32);
    fn __ext4_warning(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: *const u8, ...);
    fn __ext4_warning_inode(arg0: *const inode, arg1: *const u8, arg2: u32, arg3: *const u8, ...);
    fn ext4_update_final_de(arg0: *mut (), arg1: i32, arg2: i32);
    fn ext4_try_add_inline_entry(arg0: *mut jbd2_journal_handle, arg1: *mut ext4_filename, arg2: *mut inode, arg3: *mut inode) -> i32;
    fn ext4_try_create_inline_dir(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut inode) -> i32;
    fn ext4_inlinedir_to_tree(arg0: *mut file, arg1: *mut inode, arg2: u32, arg3: *mut dx_hash_info, arg4: u32, arg5: u32, arg6: *mut i32) -> i32;
    fn ext4_find_inline_entry(arg0: *mut inode, arg1: *mut ext4_filename, arg2: *mut *mut ext4_dir_entry_2, arg3: *mut i32) -> *mut buffer_head;
    fn ext4_delete_inline_entry(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut ext4_dir_entry_2, arg3: *mut buffer_head, arg4: *mut i32) -> i32;
    fn empty_inline_dir(arg0: *mut inode, arg1: *mut i32) -> bool;
    fn ext4_get_first_inline_block(arg0: *mut inode, arg1: *mut *mut ext4_dir_entry_2, arg2: *mut i32) -> *mut buffer_head;
    fn ext4_map_blocks(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut ext4_map_blocks, arg3: i32) -> i32;
    fn ext4_fiemap(arg0: *mut inode, arg1: *mut fiemap_extent_info, arg2: u64, arg3: u64) -> i32;
    fn ext4_orphan_add(arg0: *mut jbd2_journal_handle, arg1: *mut inode) -> i32;
    fn ext4_orphan_del(arg0: *mut jbd2_journal_handle, arg1: *mut inode) -> i32;
    fn __ext4_mark_inode_dirty(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *const u8, arg3: u32) -> i32;
    fn __ext4_journal_get_write_access(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut super_block, arg4: *mut buffer_head, arg5: u32) -> i32;
    fn __ext4_handle_dirty_metadata(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut inode, arg4: *mut buffer_head) -> i32;
    fn __ext4_journal_start_sb(arg0: *mut inode, arg1: *mut super_block, arg2: u32, arg3: i32, arg4: i32, arg5: i32, arg6: i32) -> *mut jbd2_journal_handle;
    fn __ext4_journal_stop(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle) -> i32;
    fn ext4_listxattr(arg0: *mut dentry, arg1: *mut u8, arg2: u64) -> i64;
    fn ext4_get_acl(arg0: *mut inode, arg1: i32, arg2: bool) -> *mut posix_acl;
    fn ext4_set_acl(arg0: *mut mnt_idmap, arg1: *mut dentry, arg2: *mut posix_acl, arg3: i32) -> i32;
    fn __traceiter_ext4_unlink_enter(arg0: *mut (), arg1: *mut inode, arg2: *mut dentry) -> i32;
    fn __SCT__tp_func_ext4_unlink_enter(arg0: *mut (), arg1: *mut inode, arg2: *mut dentry) -> i32;
    fn __compiletime_assert_1060();
    fn __traceiter_ext4_unlink_exit(arg0: *mut (), arg1: *mut dentry, arg2: i32) -> i32;
    fn __SCT__tp_func_ext4_unlink_exit(arg0: *mut (), arg1: *mut dentry, arg2: i32) -> i32;
    fn __compiletime_assert_1067();
    fn __compiletime_assert_1508();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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
struct io_bitmap {
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_0() };
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
__verify_pcpu_ptr__2!();
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
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn generic_test_bit_acquire(nr: u64, addr: *const u64) -> bool {
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mut __v0: u64;
let mut ___p1: u64 = 0;
let mut __v1: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(p as *const u64) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_2() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
1 & __v0 >> (nr & 63u64) != 0
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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

#[inline(always)]
fn arch_test_bit_acquire(nr: u64, addr: *const u64) -> bool {
(variable_test_bit(nr as i64, addr) as i32) != 0
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 3; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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
core::arch::asm!("4: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 4b - ., 4; .popsection\n# {0}", const 4 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
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
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn _test_bit_acquire(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit_acquire(nr, addr)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits121U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn ERR_CAST(ptr: *const ()) -> *mut () {
// cast away the const
ptr as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_unlink_enter, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_unlink_exit, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits128U0,
}


#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
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


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__5!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
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
    __slate_bits_0: __slate_bits::__SlateBits129U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
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
struct llist_node {
    next: *mut llist_node,
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
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
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
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__6!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
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
struct workqueue_struct {
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
struct rseq {
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
struct rb_root {
    rb_node: *mut rb_node,
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
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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
struct k_sigaction {
    sa: sigaction,
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
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits218U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint {
    name: *const u8,
    key: static_key_false,
    static_call_key: *mut static_call_key,
    static_call_tramp: *mut (),
    iterator: *mut (),
    probestub: *mut (),
    funcs: *mut tracepoint_func,
    ext: *mut tracepoint_ext,
}


// If a tracepoint needs to be called from a header file, it is not
// recommended to call it directly, as tracepoints in header files
// may cause side-effects and bloat the kernel. Instead, use
// tracepoint_enabled() to test if the tracepoint is enabled, then if
// it is, call a wrapper function defined in a C file that will then
// call the tracepoint.
//
// For "trace_foo_bar()", you would need to create a wrapper function
// in a C file to call trace_foo_bar():
//   void do_trace_foo_bar(args) { trace_foo_bar(args); }
// Then in the header file, declare the tracepoint:
//   DECLARE_TRACEPOINT(foo_bar);
// And call your wrapper:
//   static inline void some_inlined_function() {
//            [..]
//            if (tracepoint_enabled(foo_bar))
//                    do_trace_foo_bar(args);
//            [..]
//   }
//
// Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//   but is safe to have in headers, where trace_foo_bar_enabled() is not.
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct vm_struct {
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


// Convenience macros for the sake of set_current_state:
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits281U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits72U1,
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
    __slate_bits_2: __slate_bits::__SlateBits72U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
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
// plug has ts that needs updating
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_88: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
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
struct ldt_struct {
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


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
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
// kfree_rcu() - kfree an object after a grace period.
// @ptr: pointer to kfree for double-argument invocations.
// @rhf: the name of the struct rcu_head within the type of @ptr.
//
// Many rcu callbacks functions just call kfree() on the base structure.
// These functions are trivial, but their size adds up, and furthermore
// when they are used in a kernel module, that module must invoke the
// high-latency rcu_barrier() function at module-unload time.
//
// The kfree_rcu() function handles this issue. In order to have a universal
// callback function handling different offsets of rcu_head, the callback needs
// to determine the starting address of the freed object, which can be a large
// kmalloc or vmalloc allocation. To allow simply aligning the pointer down to
// page boundary for those, only offsets up to 4095 bytes can be accommodated.
// If the offset is larger than 4095 bytes, a compile-time error will
// be generated in kvfree_rcu_arg_2(). If this error is triggered, you can
// either fall back to use of call_rcu() or rearrange the structure to
// position the rcu_head structure into the first 4096 bytes.
//
// The object to be freed can be allocated either by kmalloc(),
// kmalloc_nolock(), or kmem_cache_alloc().
//
// Note that the allowable offset might decrease in the future.
//
// The BUILD_BUG_ON check must not involve any function calls, hence the
// checks are done in macros here.
// The BUILD_BUG_ON() makes sure the rcu_head offset can be handled. See the
// comment of kfree_rcu() for details.
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_167() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v15: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v16: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v16 = pfo_val__;
tcp_ptr__ = __v16;
let __v18: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_locks) }) as u64);
tcp_ptr__ = __v18;
__v15 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v13 = __v15;
atomic_long_inc(__v13);
}
// Y, and implicit RCU reader.
// Y, and implicit RCU reader.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
__acquire_shared!();
scp
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


fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v12: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v14: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v15: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v15 = pfo_val__;
tcp_ptr__ = __v15;
let __v17: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v17;
__v14 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v12 = __v14;
atomic_long_inc(__v12);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
}

fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[inline(always)]
fn class_srcu_fast_notrace_constructor(l: *mut srcu_struct) -> __SlateRecord_srcu_h_645 {
let mut _t: __SlateRecord_srcu_h_645 = {
let mut __t0: __SlateRecord_srcu_h_645 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_srcu_h_645 = core::ptr::addr_of_mut!(_t);
unsafe {
(*_T).scp = srcu_read_lock_fast_notrace((*_T).lock);
}
return _t;
}

#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits414U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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
fn sigismember(set: *mut __SlateRecord_signal_h_25, _sig: i32) -> i32 {
let sig: u64 = _sig.wrapping_sub(1) as u64;
if 1i32 == 1 {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) >> sig) as i32;
} else {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset((sig / 64) as isize) }) >> sig % 64) as i32;
}
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
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
struct fs_context {
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
    __slate_bits_0: __slate_bits::__SlateBits284U0,
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
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely__2!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn fatal_signal_pending(p: *mut task_struct) -> i32 {
let __v0: bool = task_sigpending(p) != 0 && __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct __SlateRecord_projid_h_24 {
    val: u32,
}


fn __kprojid_val(projid: __SlateRecord_projid_h_24) -> u32 {
projid.val
}

fn projid_eq(left: __SlateRecord_projid_h_24, right: __SlateRecord_projid_h_24) -> bool {
__kprojid_val(left) == __kprojid_val(right)
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


// dquot modified since read
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
struct io_uring_cmd {
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
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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


fn d_inode(dentry: *const dentry) -> *mut inode {
unsafe { (*dentry).d_inode }
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
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
    __slate_bits_0: __slate_bits::__SlateBits570U0,
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
struct wake_irq {
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
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mapping_metadata_bhs {
    mapping: *mut address_space,
    lock: spinlock,
    list: list_head,
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


fn inode_set_cached_link(inode: *mut inode, link: *mut u8, linklen: i32) {
VFS_WARN_ON_INODE!();
VFS_WARN_ON_INODE!();
unsafe {
(*inode).__slate_anon_47.i_link = link;
}
unsafe {
(*inode).__slate_anon_46.i_linklen = linklen;
}
unsafe {
(*inode).i_opflags = (((*inode).i_opflags as i32) | 64) as u16;
}
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


fn file_inode(f: *const file) -> *mut inode {
unsafe { (*f).f_inode }
}

fn inode_set_mtime_to_ts(inode: *mut inode, ts: timespec64) -> timespec64 {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_412() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*inode).i_mtime_sec, ts.tv_sec);
if true {
break;
}
}
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_413() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*inode).i_mtime_nsec, ts.tv_nsec);
if true {
break;
}
}
return ts;
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
fn mark_inode_dirty(inode: *mut inode) {
unsafe { __mark_inode_dirty(inode, 112i32) };
}

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


fn finish_open_simple(file: *mut file, error: i32) -> i32 {
if error != 0 {
return error;
}
unsafe { finish_open(file, (*file).__slate_anon_10.f_path.dentry, None) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


fn generic_ci_validate_strict_name(dir: *mut inode, name: *const qstr) -> bool {
true
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


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
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
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
    __slate_bits_0: __slate_bits::__SlateBits641U0,
    vm_ops: *const vm_operations_struct,
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
struct disk_stats {
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
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_buffer_head_h_62 {
    b_page: *mut page,
    b_folio: *mut folio,
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
struct bio_alloc_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_report_zones_args {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_queue_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_events {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct badblocks {
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
    __slate_bits_0: __slate_bits::__SlateBits579U0,
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
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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
struct blk_mq_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
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
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct bvec_iter {
    bi_sector: u64,
    bi_size: u32,
    bi_idx: u32,
    bi_offset: u32,
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
struct jbd2_revoke_table_s {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jbd2_h_499 {
    h_transaction: *mut transaction_s,
    h_journal: *mut journal_s,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits667U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits667U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits667U2,
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
    __slate_bits_3: __slate_bits::__SlateBits667U3,
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
struct proc_dir_entry {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_inode_snap {
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
union __SlateRecord_ext4_h_1077 {
    xattr_sem: rw_semaphore,
    i_ea_iput_node: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_1086 {
    i_orphan: list_head,
    i_orphan_idx: u32,
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
struct ext4_locality_group {
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
struct mb_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dax_device {
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
    __slate_bits_0: __slate_bits::__SlateBits664U0,
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


// BLK_STS_AGAIN should only be returned if RQF_NOWAIT is set
// and the bio would block (cf bio_wouldblock_error())
#[repr(C)]
#[derive(Clone, Copy)]
struct bio {
    bi_next: *mut bio,
    bi_bdev: *mut block_device,
    bi_opf: u32,
    bi_flags: u16,
    bi_ioprio: u16,
    bi_write_hint: u8,
    bi_write_stream: u8,
    bi_status: u8,
    bi_bvec_gap_bit: u8,
    __bi_remaining: __SlateRecord_types_h_190,
    bi_io_vec: *mut bio_vec,
    bi_iter: bvec_iter,
    __slate_anon_12: __SlateRecord_blk_types_h_239,
    bi_end_io: Option<unsafe extern "C" fn(*mut bio)>,
    bi_private: *mut (),
    bi_blkg: *mut blkcg_gq,
    issue_time_ns: u64,
    bi_iocost_cost: u64,
    bi_vcnt: u16,
    bi_max_vecs: u16,
    __bi_cnt: __SlateRecord_types_h_190,
    bi_pool: *mut bio_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct buffer_head {
    b_state: u64,
    b_this_page: *mut buffer_head,
    __slate_anon_2: __SlateRecord_buffer_head_h_62,
    b_blocknr: u64,
    b_size: u64,
    b_data: *mut u8,
    b_bdev: *mut block_device,
    b_private: *mut (),
    b_assoc_buffers: list_head,
    b_mmb: *mut mapping_metadata_bhs,
    b_count: __SlateRecord_types_h_190,
    b_uptodate_lock: spinlock,
}


// macro tricks to expand the set_buffer_foo(), clear_buffer_foo()
// and buffer_foo() functions.
// To avoid reset buffer flags that are already set, because that causes
// a costly cache line transition, check the flag first.
// test_set_buffer_foo() and test_clear_buffer_foo()
#[inline(always)]
fn buffer_locked(bh: *const buffer_head) -> i32 {
test_bit__2!(2i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
}

#[inline(always)]
fn set_buffer_uptodate(bh: *mut buffer_head) {
// If somebody else already set this uptodate, they will
// have done the memory barrier, and a reader will thus
// see *some* valid buffer state.
//
// Any other serialization (with IO errors or whatever that
// might clear the bit) has to come from other state (eg BH_Lock).
if test_bit__3!(0i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) != 0 {
return;
}
// make it consistent with folio_mark_uptodate
// pairs with smp_load_acquire in buffer_uptodate
smp_mb__before_atomic!();
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[inline(always)]
fn buffer_uptodate(bh: *const buffer_head) -> i32 {
// make it consistent with folio_test_uptodate
// pairs with smp_mb__before_atomic in set_buffer_uptodate
test_bit_acquire!(0i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
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


fn brelse(bh: *mut buffer_head) {
if !bh.is_null() {
unsafe { __brelse(bh) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_posix_acl_h_23 {
    e_uid: __SlateRecord_uidgid_types_h_9,
    e_gid: __SlateRecord_uidgid_types_h_13,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_posix_acl_h_31 {
    __slate_anon_0: __SlateRecord_posix_acl_h_31_2,
    hdr: posix_acl_hdr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_posix_acl_h_31_2 {
    a_refcount: refcount_struct,
    a_count: u32,
    a_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl_hdr {
    a_refcount: refcount_struct,
    a_count: u32,
    a_rcu: callback_head,
}


fn wait_on_buffer(bh: *mut buffer_head) {
might_sleep!();
if buffer_locked(bh as *const buffer_head) != 0 {
unsafe { __wait_on_buffer(bh) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mempool {
    lock: spinlock,
    min_nr: i32,
    curr_nr: i32,
    elements: *mut *mut (),
    pool_data: *mut (),
    alloc: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>,
    free: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_set {
    bio_slab: *mut kmem_cache,
    front_pad: u32,
    cache: *mut bio_alloc_cache,
    bio_pool: mempool,
    bvec_pool: mempool,
    back_pad: u32,
    rescue_lock: spinlock,
    rescue_list: bio_list,
    rescue_work: work_struct,
    rescue_workqueue: *mut workqueue_struct,
    cpuhp_dead: hlist_node,
}


fn inode_inc_iversion(inode: *mut inode) {
unsafe { inode_maybe_inc_iversion(inode, true) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dx_root_info {
    reserved_zero: u32,
    hash_version: u8,
    /// 8
    info_length: u8,
    indirect_levels: u8,
    unused_flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_list {
    head: *mut request,
    tail: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
    mq_list: rq_list,
    cached_rqs: rq_list,
    cur_ktime: u64,
    nr_ios: u16,
    rq_count: u16,
    multiple_queues: bool,
    has_elevator: bool,
    cb_list: list_head,
}


// Enough for the string representation of any kind of UUID plus NULL.
// EFI UUID is 36 characters. MSDOS UUID is 11 characters.
#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
    uuid: [u8; 37],
    volname: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_integrity {
    flags: u8,
    csum_type: u8,
    metadata_size: u8,
    pi_offset: u8,
    interval_exp: u8,
    tag_size: u8,
    pi_tuple_size: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_namei_c_1602 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
    major: i32,
    first_minor: i32,
    minors: i32,
    disk_name: [u8; 32],
    events: u16,
    event_flags: u16,
    part_tbl: xarray,
    part0: *mut block_device,
    fops: *const block_device_operations,
    queue: *mut request_queue,
    private_data: *mut (),
    bio_split: bio_set,
    flags: i32,
    state: u64,
    open_mutex: mutex,
    open_partitions: u32,
    bdi: *mut backing_dev_info,
    queue_kobj: kobject,
    slave_dir: *mut kobject,
    slave_bdevs: list_head,
    random: *mut timer_rand_state,
    ev: *mut disk_events,
    cdi: *mut cdrom_device_info,
    node_id: i32,
    bb: *mut badblocks,
    lockdep_map: lockdep_map,
    diskseq: u64,
    open_mode: u32,
    ia_ranges: *mut blk_independent_access_ranges,
    rqos_state_mutex: mutex,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits538U0 {
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
    pub struct __SlateBits570U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits281U0 {
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
    pub struct __SlateBits72U0 {
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
    pub struct __SlateBits72U1 {
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
    pub struct __SlateBits284U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits667U0 {
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
    pub struct __SlateBits664U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits667U1 {
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
    pub struct __SlateBits667U2 {
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
    pub struct __SlateBits667U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits121U0 {
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
    pub struct __SlateBits129U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits414U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits641U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits579U0 {
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
    pub struct __SlateBits128U0 {
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
    pub struct __SlateBits218U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits783U0 {
        #[bits(1)]
        pub t_synchronous_commit: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits787U0 {
        #[bits(1)]
        pub h_sync: u8,
        #[bits(1)]
        pub h_reserved: u8,
        #[bits(1)]
        pub h_aborted: u8,
        #[bits(1)]
        pub h_invalid: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
}

// supports a volatile write cache
// supports passing on the FUA bit
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
// is a zoned device
// do not send FLUSH/FUA commands despite advertising a write cache
#[repr(C)]
#[derive(Clone, Copy)]
struct queue_limits {
    features: u32,
    flags: u32,
    seg_boundary_mask: u64,
    virt_boundary_mask: u64,
    max_hw_sectors: u32,
    max_dev_sectors: u32,
    chunk_sectors: u32,
    max_sectors: u32,
    max_user_sectors: u32,
    max_segment_size: u32,
    max_fast_segment_size: u32,
    physical_block_size: u32,
    logical_block_size: u32,
    alignment_offset: u32,
    io_min: u32,
    io_opt: u32,
    max_discard_sectors: u32,
    max_hw_discard_sectors: u32,
    max_user_discard_sectors: u32,
    max_secure_erase_sectors: u32,
    max_write_zeroes_sectors: u32,
    max_wzeroes_unmap_sectors: u32,
    max_hw_wzeroes_unmap_sectors: u32,
    max_user_wzeroes_unmap_sectors: u32,
    max_hw_zone_append_sectors: u32,
    max_zone_append_sectors: u32,
    discard_granularity: u32,
    discard_alignment: u32,
    zone_write_granularity: u32,
    atomic_write_hw_max: u32,
    atomic_write_max_sectors: u32,
    atomic_write_hw_boundary: u32,
    atomic_write_boundary_sectors: u32,
    atomic_write_hw_unit_min: u32,
    atomic_write_unit_min: u32,
    atomic_write_hw_unit_max: u32,
    atomic_write_unit_max: u32,
    max_segments: u16,
    max_integrity_segments: u16,
    max_discard_segments: u16,
    max_write_streams: u16,
    write_stream_granularity: u32,
    max_open_zones: u32,
    max_active_zones: u32,
    dma_alignment: u32,
    dma_pad_mask: u32,
    integrity: blk_integrity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_range {
    kobj: kobject,
    sector: u64,
    nr_sectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_ranges {
    kobj: kobject,
    sysfs_registered: bool,
    nr_ia_ranges: u32,
    ia_range: [blk_independent_access_range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
    queuedata: *mut (),
    elevator: *mut elevator_queue,
    mq_ops: *const blk_mq_ops,
    queue_ctx: *mut blk_mq_ctx,
    queue_flags: u64,
    rq_timeout: u32,
    queue_depth: u32,
    refs: refcount_struct,
    nr_hw_queues: u32,
    queue_hw_ctx: *mut *mut blk_mq_hw_ctx,
    q_usage_counter: percpu_ref,
    io_lock_cls_key: lock_class_key,
    io_lockdep_map: lockdep_map,
    q_lock_cls_key: lock_class_key,
    q_lockdep_map: lockdep_map,
    last_merge: *mut request,
    queue_lock: spinlock,
    quiesce_depth: i32,
    disk: *mut gendisk,
    mq_kobj: *mut kobject,
    limits: queue_limits,
    dev: *mut device,
    rpm_status: i32,
    pm_only: __SlateRecord_types_h_190,
    stats: *mut blk_queue_stats,
    rq_qos: *mut rq_qos,
    rq_qos_mutex: mutex,
    id: i32,
    nr_requests: u32,
    async_depth: u32,
    timeout: timer_list,
    timeout_work: work_struct,
    nr_active_requests_shared_tags: __SlateRecord_types_h_190,
    sched_shared_tags: *mut blk_mq_tags,
    icq_list: list_head,
    blkcg_pols: [u64; 1],
    root_blkg: *mut blkcg_gq,
    blkg_list: list_head,
    blkcg_mutex: mutex,
    node: i32,
    requeue_lock: spinlock,
    requeue_list: list_head,
    requeue_work: delayed_work,
    blk_trace: *mut blk_trace,
    fq: *mut blk_flush_queue,
    flush_list: list_head,
    elevator_lock: mutex,
    sysfs_lock: mutex,
    limits_lock: mutex,
    unused_hctx_list: list_head,
    unused_hctx_lock: spinlock,
    mq_freeze_depth: i32,
    callback_head: callback_head,
    mq_freeze_wq: wait_queue_head,
    mq_freeze_lock: mutex,
    tag_set: *mut blk_mq_tag_set,
    tag_set_list: list_head,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    rqos_debugfs_dir: *mut dentry,
    debugfs_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device_operations {
    submit_bio: Option<unsafe extern "C" fn(*mut bio)>,
    poll_bio: Option<unsafe extern "C" fn(*mut bio, *mut io_comp_batch, u32) -> i32>,
    open: Option<unsafe extern "C" fn(*mut gendisk, u32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut gendisk)>,
    ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut gendisk, u32) -> u32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut gendisk)>,
    getgeo: Option<unsafe extern "C" fn(*mut gendisk, *mut hd_geometry) -> i32>,
    set_read_only: Option<unsafe extern "C" fn(*mut block_device, bool) -> i32>,
    free_disk: Option<unsafe extern "C" fn(*mut gendisk)>,
    swap_slot_free_notify: Option<unsafe extern "C" fn(*mut block_device, u64)>,
    report_zones: Option<unsafe extern "C" fn(*mut gendisk, u64, u32, *mut blk_report_zones_args) -> i32>,
    devnode: Option<unsafe extern "C" fn(*mut gendisk, *mut u16) -> *mut u8>,
    get_unique_id: Option<unsafe extern "C" fn(*mut gendisk, *mut u8, u32) -> i32>,
    owner: *mut module,
    pr_ops: *const pr_ops,
    alternative_gpt_sector: Option<unsafe extern "C" fn(*mut gendisk, *mut u64) -> i32>,
}


// Largest string for a blockdev identifier
#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
    mark_dead: Option<unsafe extern "C" fn(*mut block_device, bool)>,
    sync: Option<unsafe extern "C" fn(*mut block_device)>,
    freeze: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
    req_list: rq_list,
    need_ts: bool,
    complete: Option<unsafe extern "C" fn(*mut io_comp_batch)>,
    poll_ctx: *mut (),
}


// Maximum contiguous integrity buffer allocation.
#[repr(C)]
#[derive(Clone, Copy)]
struct journal_head {
    b_bh: *mut buffer_head,
    b_state_lock: spinlock,
    b_jcount: i32,
    b_jlist: u32,
    b_modified: u32,
    b_frozen_data: *mut u8,
    b_committed_data: *mut u8,
    b_transaction: *mut transaction_s,
    b_next_transaction: *mut transaction_s,
    b_tnext: *mut journal_head,
    b_tprev: *mut journal_head,
    b_cp_transaction: *mut transaction_s,
    b_cpnext: *mut journal_head,
    b_cpprev: *mut journal_head,
    b_triggers: *mut jbd2_buffer_trigger_type,
    b_frozen_triggers: *mut jbd2_buffer_trigger_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct journal_header_s {
    h_magic: u32,
    h_blocktype: u32,
    h_sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct journal_superblock_s {
    s_header: journal_header_s,
    s_blocksize: u32,
    s_maxlen: u32,
    s_first: u32,
    s_sequence: u32,
    s_start: u32,
    s_errno: u32,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_nr_users: u32,
    s_dynsuper: u32,
    s_max_transaction: u32,
    s_max_trans_data: u32,
    s_checksum_type: u8,
    s_padding2: [u8; 3],
    s_num_fc_blks: u32,
    s_head: u32,
    s_padding: [u32; 40],
    s_checksum: u32,
    s_users: [u8; 768],
}


#[inline(always)]
fn set_buffer_verified(bh: *mut buffer_head) {
if test_bit__3!(25i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) == 0 {
set_bit(25i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}
}

#[inline(always)]
fn buffer_verified(bh: *const buffer_head) -> i32 {
test_bit__2!(25i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_inode {
    i_transaction: *mut transaction_s,
    i_next_transaction: *mut transaction_s,
    i_list: list_head,
    i_vfs_inode: *mut inode,
    i_flags: u64,
    i_dirty_start_page: u64,
    i_dirty_end_page: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_journal_handle {
    __slate_anon_0: __SlateRecord_jbd2_h_499,
    h_rsv_handle: *mut jbd2_journal_handle,
    h_total_credits: i32,
    h_revoke_credits: i32,
    h_revoke_credits_requested: i32,
    h_ref: i32,
    h_err: i32,
    __slate_bits_0: __slate_bits::__SlateBits787U0,
    h_type: u8,
    h_line_no: u16,
    h_start_jiffies: u64,
    h_requested_credits: u32,
    saved_alloc_context: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_chp_stats_s {
    cs_chp_time: u64,
    cs_forced_to_close: u32,
    cs_written: u32,
    cs_dropped: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_s {
    t_journal: *mut journal_s,
    t_tid: u32,
    t_state: u32,
    t_log_start: u64,
    t_nr_buffers: i32,
    t_reserved_list: *mut journal_head,
    t_buffers: *mut journal_head,
    t_forget: *mut journal_head,
    t_checkpoint_list: *mut journal_head,
    t_shadow_list: *mut journal_head,
    t_inode_list: list_head,
    t_max_wait: u64,
    t_start: u64,
    t_requested: u64,
    t_chp_stats: transaction_chp_stats_s,
    t_updates: __SlateRecord_types_h_190,
    t_outstanding_credits: __SlateRecord_types_h_190,
    t_outstanding_revokes: __SlateRecord_types_h_190,
    t_handle_count: __SlateRecord_types_h_190,
    t_cpnext: *mut transaction_s,
    t_cpprev: *mut transaction_s,
    t_expires: u64,
    t_start_time: i64,
    __slate_bits_0: __slate_bits::__SlateBits783U0,
    t_need_data_flush: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_run_stats_s {
    rs_wait: u64,
    rs_request_delay: u64,
    rs_running: u64,
    rs_locked: u64,
    rs_flushing: u64,
    rs_logging: u64,
    rs_handle_count: u32,
    rs_blocks: u32,
    rs_blocks_logged: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transaction_stats_s {
    ts_tid: u64,
    ts_requested: u64,
    run: transaction_run_stats_s,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct journal_s {
    j_flags: u64,
    j_errno: i32,
    j_abort_mutex: mutex,
    j_sb_buffer: *mut buffer_head,
    j_superblock: *mut journal_superblock_s,
    j_state_lock: rwlock,
    j_barrier_count: i32,
    j_barrier: mutex,
    j_running_transaction: *mut transaction_s,
    j_committing_transaction: *mut transaction_s,
    j_checkpoint_transactions: *mut transaction_s,
    j_wait_transaction_locked: wait_queue_head,
    j_wait_done_commit: wait_queue_head,
    j_wait_commit: wait_queue_head,
    j_wait_updates: wait_queue_head,
    j_wait_reserved: wait_queue_head,
    j_fc_wait: wait_queue_head,
    j_checkpoint_mutex: mutex,
    j_chkpt_bhs: [*mut buffer_head; 64],
    j_shrinker: *mut shrinker,
    j_checkpoint_jh_count: percpu_counter,
    j_shrink_transaction: *mut transaction_s,
    j_head: u64,
    j_tail: u64,
    j_free: u64,
    j_first: u64,
    j_last: u64,
    j_fc_first: u64,
    j_fc_off: u64,
    j_fc_last: u64,
    j_dev: *mut block_device,
    j_blocksize: i32,
    j_blk_offset: u64,
    j_devname: [u8; 56],
    j_fs_dev: *mut block_device,
    j_fs_dev_wb_err: u32,
    j_total_len: u32,
    j_reserved_credits: __SlateRecord_types_h_190,
    j_list_lock: spinlock,
    j_inode: *mut inode,
    j_tail_sequence: u32,
    j_transaction_sequence: u32,
    j_commit_sequence: u32,
    j_commit_request: u32,
    j_uuid: [u8; 16],
    j_task: *mut task_struct,
    j_max_transaction_buffers: i32,
    j_revoke_records_per_block: i32,
    j_transaction_overhead_buffers: i32,
    j_commit_interval: u64,
    j_commit_timer: timer_list,
    j_revoke_lock: spinlock,
    j_revoke: *mut jbd2_revoke_table_s,
    j_revoke_table: [*mut jbd2_revoke_table_s; 2],
    j_wbuf: *mut *mut buffer_head,
    j_fc_wbuf: *mut *mut buffer_head,
    j_wbufsize: i32,
    j_fc_wbufsize: i32,
    j_last_sync_writer: i32,
    j_average_commit_time: u64,
    j_min_batch_time: u32,
    j_max_batch_time: u32,
    j_commit_callback: Option<unsafe extern "C" fn(*mut journal_s, *mut transaction_s)>,
    j_submit_inode_data_buffers: Option<unsafe extern "C" fn(*mut jbd2_inode) -> i32>,
    j_finish_inode_data_buffers: Option<unsafe extern "C" fn(*mut jbd2_inode) -> i32>,
    j_history_lock: spinlock,
    j_proc_entry: *mut proc_dir_entry,
    j_stats: transaction_stats_s,
    j_failed_commit: u32,
    j_private: *mut (),
    j_csum_seed: u32,
    jbd2_trans_commit_key: lock_class_key,
    j_fc_cleanup_callback: Option<unsafe extern "C" fn(*mut journal_s, i32, u32)>,
    j_fc_replay_callback: Option<unsafe extern "C" fn(*mut journal_s, *mut buffer_head, u32, i32, u32) -> i32>,
    j_bmap: Option<unsafe extern "C" fn(*mut journal_s, *mut u64) -> i32>,
}


// journal feature predicate functions
// Journaling has been aborted for errors.
#[repr(C)]
#[derive(Clone, Copy)]
struct jbd2_buffer_trigger_type {
    t_frozen: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head, *mut (), u64)>,
    t_abort: Option<unsafe extern "C" fn(*mut jbd2_buffer_trigger_type, *mut buffer_head)>,
}


fn journal_current_handle() -> *mut jbd2_journal_handle {
(unsafe { (*get_current()).journal_info }) as *mut jbd2_journal_handle
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct bgl_lock {
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blockgroup_lock {
    locks: [bgl_lock; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent {
    fe_logical: u64,
    fe_physical: u64,
    fe_length: u64,
    fe_reserved64: [u64; 2],
    fe_flags: u32,
    fe_reserved: [u32; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
    fi_flags: u32,
    fi_extents_mapped: u32,
    fi_extents_max: u32,
    fi_extents_start: *mut fiemap_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fscrypt_str {
    name: *mut u8,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fscrypt_name {
    usr_fname: *const qstr,
    disk_name: fscrypt_str,
    hash: u32,
    minor_hash: u32,
    crypto_buf: fscrypt_str,
    is_nokey_name: bool,
}


fn fscrypt_is_nokey_name(dentry: *const dentry) -> bool {
false
}

fn fscrypt_has_permitted_context(parent: *mut inode, child: *mut inode) -> i32 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fscrypt_dummy_policy {
}


fn fscrypt_fname_alloc_buffer(max_encrypted_len: u32, crypto_str: &fscrypt_str) -> i32 {
95i32.wrapping_neg()
}

fn fscrypt_fname_free_buffer(crypto_str: &fscrypt_str) {
return;
}

fn fscrypt_fname_disk_to_usr(inode: *const inode, hash: u32, minor_hash: u32, iname: &fscrypt_str, oname: &fscrypt_str) -> i32 {
95i32.wrapping_neg()
}

fn fscrypt_match_name(fname: &fscrypt_name, de_name: *const u8, de_name_len: u32) -> bool {
// Encryption support disabled; use standard comparison
if de_name_len != unsafe { (*(fname as *const fscrypt_name)).disk_name.len } {
return false;
}
(unsafe { memcmp(de_name as *const (), (*(fname as *const fscrypt_name)).disk_name.name as *const (), (*(fname as *const fscrypt_name)).disk_name.len as u64) }) == 0
}

fn __fscrypt_prepare_link(inode: *mut inode, dir: *mut inode, dentry: *mut dentry) -> i32 {
95i32.wrapping_neg()
}

fn __fscrypt_prepare_rename(old_dir: *mut inode, old_dentry: *mut dentry, new_dir: *mut inode, new_dentry: *mut dentry, flags: u32) -> i32 {
95i32.wrapping_neg()
}

fn __fscrypt_prepare_readdir(dir: *mut inode) -> i32 {
95i32.wrapping_neg()
}

fn fscrypt_prepare_symlink(dir: *mut inode, target: *const u8, len: u32, max_len: u32, disk_link: &mut fscrypt_str) -> i32 {
if IS_ENCRYPTED!(dir) != 0 {
return 95i32.wrapping_neg();
}
unsafe {
(*(disk_link as *mut fscrypt_str)).name = target as *mut u8;
}
unsafe {
(*(disk_link as *mut fscrypt_str)).len = len.wrapping_add(1);
}
if (unsafe { (*(disk_link as *mut fscrypt_str)).len }) > max_len {
return 36i32.wrapping_neg();
}
0
}

fn __fscrypt_encrypt_symlink(inode: *mut inode, target: *const u8, len: u32, disk_link: &fscrypt_str) -> i32 {
95i32.wrapping_neg()
}

fn fscrypt_prepare_link(old_dentry: *mut dentry, dir: *mut inode, dentry: *mut dentry) -> i32 {
if IS_ENCRYPTED!(dir) != 0 {
return __fscrypt_prepare_link(d_inode(old_dentry as *const dentry), dir, dentry);
}
0
}

fn fscrypt_prepare_rename(old_dir: *mut inode, old_dentry: *mut dentry, new_dir: *mut inode, new_dentry: *mut dentry, flags: u32) -> i32 {
if IS_ENCRYPTED!(old_dir) != 0 || IS_ENCRYPTED!(new_dir) != 0 {
return __fscrypt_prepare_rename(old_dir, old_dentry, new_dir, new_dentry, flags);
}
0
}

fn fscrypt_prepare_readdir(dir: *mut inode) -> i32 {
if IS_ENCRYPTED!(dir) != 0 {
return __fscrypt_prepare_readdir(dir);
}
0
}

fn fscrypt_encrypt_symlink(inode: *mut inode, target: *const u8, len: u32, disk_link: &fscrypt_str) -> i32 {
if IS_ENCRYPTED!(inode) != 0 {
return __fscrypt_encrypt_symlink(inode, target, len, disk_link);
}
0
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
struct ext4_map_blocks {
    m_pblk: u64,
    m_lblk: u32,
    m_len: u32,
    m_flags: u32,
    m_seq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_system_blocks {
    root: rb_root,
    rcu: callback_head,
}


// Flags for ext4_io_end->flags
// Root inode
// Maximal count of links to a file
#[repr(C)]
#[derive(Clone, Copy)]
struct flex_groups {
    free_clusters: __SlateRecord_types_h_197,
    free_inodes: __SlateRecord_types_h_190,
    used_dirs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct extent_status {
    rb_node: rb_node,
    es_lblk: u32,
    es_len: u32,
    es_pblk: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_es_tree {
    root: rb_root,
    cache_es: *mut extent_status,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_es_stats {
    es_stats_shrunk: u64,
    es_stats_cache_hits: percpu_counter,
    es_stats_cache_misses: percpu_counter,
    es_stats_scan_time: u64,
    es_stats_max_scan_time: u64,
    es_stats_all_cnt: percpu_counter,
    es_stats_shk_cnt: percpu_counter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_pending_tree {
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_stats {
    fc_ineligible_reason_count: [u32; 13],
    fc_num_commits: u64,
    fc_ineligible_commits: u64,
    fc_failed_commits: u64,
    fc_skipped_commits: u64,
    fc_numblks: u64,
    s_fc_avg_commit_time: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_alloc_region {
    lblk: u32,
    pblk: u64,
    ino: i32,
    len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_replay_state {
    fc_replay_num_tags: i32,
    fc_replay_expected_off: i32,
    fc_current_pass: i32,
    fc_cur_tag: i32,
    fc_crc: i32,
    fc_regions: *mut ext4_fc_alloc_region,
    fc_regions_size: i32,
    fc_regions_used: i32,
    fc_regions_valid: i32,
    fc_modified_inodes: *mut i32,
    fc_modified_inodes_used: i32,
    fc_modified_inodes_size: i32,
}


// Inode flags
// Secure deletion
// Undelete
// Compress file
// Synchronous updates
// Immutable file
// writes to file may only append
// do not dump file
// do not update atime
// Reserved for compression usage...
// One or more compressed clusters
// Don't compress
// nb: was previously EXT2_ECOMPR_FL
// encrypted file
// End compression flags --- maybe not all used
// hash-indexed directory
// AFS directory
// file data should be journaled
// file tail should not be merged
// dirsync behaviour (directories only)
// Top of directory hierarchies
// Set to each huge file
// Inode uses extents
// Verity protected inode
// Inode used for large EA
// Inode has inline data.
// Create with parents projid
// Casefolded directory
// reserved for ext4 lib
// Flags that are appropriate for regular files (all but dir-specific ones).
// Flags that are appropriate for non-directories/regular files.
// Since it's pretty easy to mix up bit numbers and hex values, we use a
// build-time check to make sure that EXT4_XXX_FL is consistent with respect to
// EXT4_INODE_XXX. If all is well, the macros will be dropped, so, it won't cost
// any extra space in the compiled kernel image, otherwise, the build will fail.
// It's important that these values are the same, since we are using
// EXT4_INODE_XXX to test for flag values, but EXT4_XXX_FL must be consistent
// with the values of FS_XXX_FL defined in include/linux/fs.h and the on-disk
// values found in ext2, ext3 and ext4 filesystems, and of course the values
// defined in e2fsprogs.
//
// It's not paranoia if the Murphy's Law really *is* out to get you.  :-)
// Flags used by ext4_map_blocks()
//
// Allocate any needed blocks and/or convert an unwritten
// extent to be an initialized ext4
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_inode_info {
    i_data: [u32; 15],
    i_dtime: u32,
    i_file_acl: u64,
    i_block_group: u32,
    i_dir_start_lookup: u32,
    i_flags: u64,
    __slate_anon_6: __SlateRecord_ext4_h_1077,
    __slate_anon_7: __SlateRecord_ext4_h_1086,
    i_fc_dilist: list_head,
    i_fc_list: list_head,
    i_fc_lblk_start: u32,
    i_fc_lblk_len: u32,
    i_fc_snap: *mut ext4_fc_inode_snap,
    i_raw_lock: spinlock,
    i_fc_lock: spinlock,
    i_disksize: i64,
    i_data_sem: rw_semaphore,
    vfs_inode: inode,
    jinode: *mut jbd2_inode,
    i_metadata_bhs: *mut mapping_metadata_bhs,
    i_crtime: timespec64,
    i_prealloc_active: __SlateRecord_types_h_190,
    i_reserved_data_blocks: u32,
    i_prealloc_node: rb_root,
    i_prealloc_lock: rwlock,
    i_es_tree: ext4_es_tree,
    i_es_lock: rwlock,
    i_es_list: list_head,
    i_es_all_nr: u32,
    i_es_shk_nr: u32,
    i_es_shrink_lblk: u32,
    i_es_seq: u64,
    i_last_alloc_group: u32,
    i_pending_tree: ext4_pending_tree,
    i_extra_isize: u16,
    i_inline_off: u16,
    i_inline_size: u16,
    i_reserved_quota: i64,
    i_block_reservation_lock: spinlock,
    i_completed_io_lock: spinlock,
    i_rsv_conversion_list: list_head,
    i_rsv_conversion_work: work_struct,
    i_sync_tid: u32,
    i_datasync_tid: u32,
    i_dquot: [*mut dquot; 3],
    i_csum_seed: u32,
    i_projid: __SlateRecord_projid_h_24,
}


// Fast commit replay ongoing
// Mode for data writes:
// Write data to journal
// No auto delalloc mapping
// Some quota option set
// Enable support for dio read nolocking
// Delalloc support
// Journal fast commit
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_super_block {
    s_inodes_count: u32,
    s_blocks_count_lo: u32,
    s_r_blocks_count_lo: u32,
    s_free_blocks_count_lo: u32,
    s_free_inodes_count: u32,
    s_first_data_block: u32,
    s_log_block_size: u32,
    s_log_cluster_size: u32,
    s_blocks_per_group: u32,
    s_clusters_per_group: u32,
    s_inodes_per_group: u32,
    s_mtime: u32,
    s_wtime: u32,
    s_mnt_count: u16,
    s_max_mnt_count: u16,
    s_magic: u16,
    s_state: u16,
    s_errors: u16,
    s_minor_rev_level: u16,
    s_lastcheck: u32,
    s_checkinterval: u32,
    s_creator_os: u32,
    s_rev_level: u32,
    s_def_resuid: u16,
    s_def_resgid: u16,
    s_first_ino: u32,
    s_inode_size: u16,
    s_block_group_nr: u16,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_volume_name: [u8; 16],
    s_last_mounted: [u8; 64],
    s_algorithm_usage_bitmap: u32,
    s_prealloc_blocks: u8,
    s_prealloc_dir_blocks: u8,
    s_reserved_gdt_blocks: u16,
    s_journal_uuid: [u8; 16],
    s_journal_inum: u32,
    s_journal_dev: u32,
    s_last_orphan: u32,
    s_hash_seed: [u32; 4],
    s_def_hash_version: u8,
    s_jnl_backup_type: u8,
    s_desc_size: u16,
    s_default_mount_opts: u32,
    s_first_meta_bg: u32,
    s_mkfs_time: u32,
    s_jnl_blocks: [u32; 17],
    s_blocks_count_hi: u32,
    s_r_blocks_count_hi: u32,
    s_free_blocks_count_hi: u32,
    s_min_extra_isize: u16,
    s_want_extra_isize: u16,
    s_flags: u32,
    s_raid_stride: u16,
    s_mmp_update_interval: u16,
    s_mmp_block: u64,
    s_raid_stripe_width: u32,
    s_log_groups_per_flex: u8,
    s_checksum_type: u8,
    s_encryption_level: u8,
    s_reserved_pad: u8,
    s_kbytes_written: u64,
    s_snapshot_inum: u32,
    s_snapshot_id: u32,
    s_snapshot_r_blocks_count: u64,
    s_snapshot_list: u32,
    s_error_count: u32,
    s_first_error_time: u32,
    s_first_error_ino: u32,
    s_first_error_block: u64,
    s_first_error_func: [u8; 32],
    s_first_error_line: u32,
    s_last_error_time: u32,
    s_last_error_ino: u32,
    s_last_error_line: u32,
    s_last_error_block: u64,
    s_last_error_func: [u8; 32],
    s_mount_opts: [u8; 64],
    s_usr_quota_inum: u32,
    s_grp_quota_inum: u32,
    s_overhead_clusters: u32,
    s_backup_bgs: [u32; 2],
    s_encrypt_algos: [u8; 4],
    s_encrypt_pw_salt: [u8; 16],
    s_lpf_ino: u32,
    s_prj_quota_inum: u32,
    s_checksum_seed: u32,
    s_wtime_hi: u8,
    s_mtime_hi: u8,
    s_mkfs_time_hi: u8,
    s_lastcheck_hi: u8,
    s_first_error_time_hi: u8,
    s_last_error_time_hi: u8,
    s_first_error_errcode: u8,
    s_last_error_errcode: u8,
    s_encoding: u16,
    s_encoding_flags: u16,
    s_orphan_file_inum: u32,
    s_def_resuid_hi: u16,
    s_def_resgid_hi: u16,
    s_reserved: [u32; 93],
    s_checksum: u32,
}


// Number of quota types we support
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_journal_trigger {
    tr_triggers: jbd2_buffer_trigger_type,
    sb: *mut super_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_orphan_block {
    ob_free_entries: __SlateRecord_types_h_190,
    ob_bh: *mut buffer_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_orphan_info {
    of_blocks: i32,
    of_csum_seed: u32,
    of_binfo: *mut ext4_orphan_block,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_snap_stats {
    lock_updates_ns_total: __SlateRecord_types_h_197,
    lock_updates_ns_max: __SlateRecord_types_h_197,
    lock_updates_samples: __SlateRecord_types_h_197,
    snap_inodes: __SlateRecord_types_h_197,
    snap_ranges: __SlateRecord_types_h_197,
    snap_fail_es_miss: __SlateRecord_types_h_197,
    snap_fail_es_delayed: __SlateRecord_types_h_197,
    snap_fail_es_other: __SlateRecord_types_h_197,
    snap_fail_inodes_cap: __SlateRecord_types_h_197,
    snap_fail_ranges_cap: __SlateRecord_types_h_197,
    snap_fail_nomem: __SlateRecord_types_h_197,
    snap_fail_inode_loc: __SlateRecord_types_h_197,
    snap_fail_no_snap: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct ext4_sb_info {
    s_desc_size: u64,
    s_inodes_per_block: u64,
    s_blocks_per_group: u64,
    s_clusters_per_group: u64,
    s_inodes_per_group: u64,
    s_itb_per_group: u64,
    s_gdb_count: u64,
    s_desc_per_block: u64,
    s_groups_count: u32,
    s_blockfile_groups: u32,
    s_overhead: u64,
    s_cluster_ratio: u32,
    s_cluster_bits: u32,
    s_bitmap_maxbytes: i64,
    s_sbh: *mut buffer_head,
    s_es: *mut ext4_super_block,
    s_group_desc: *mut *mut buffer_head,
    s_mount_opt: u32,
    s_mount_opt2: u32,
    s_mount_flags: u64,
    s_def_mount_opt: u32,
    s_def_mount_opt2: u32,
    s_sb_block: u64,
    s_resv_clusters: __SlateRecord_types_h_197,
    s_resuid: __SlateRecord_uidgid_types_h_9,
    s_resgid: __SlateRecord_uidgid_types_h_13,
    s_mount_state: u16,
    s_pad: u16,
    s_addr_per_block_bits: i32,
    s_desc_per_block_bits: i32,
    s_inode_size: i32,
    s_first_ino: i32,
    s_inode_readahead_blks: u32,
    s_inode_goal: u32,
    s_hash_seed: [u32; 4],
    s_def_hash_version: i32,
    s_hash_unsigned: i32,
    s_freeclusters_counter: percpu_counter,
    s_freeinodes_counter: percpu_counter,
    s_dirs_counter: percpu_counter,
    s_dirtyclusters_counter: percpu_counter,
    s_sra_exceeded_retry_limit: percpu_counter,
    s_blockgroup_lock: *mut blockgroup_lock,
    s_proc: *mut proc_dir_entry,
    s_kobj: kobject,
    s_kobj_unregister: completion,
    s_error_notify_mutex: mutex,
    s_sb: *mut super_block,
    s_mmp_bh: *mut buffer_head,
    s_journal: *mut journal_s,
    s_ext4_flags: u64,
    s_orphan_lock: mutex,
    s_orphan: list_head,
    s_orphan_info: ext4_orphan_info,
    s_commit_interval: u64,
    s_max_batch_time: u32,
    s_min_batch_time: u32,
    s_journal_bdev_file: *mut file,
    s_qf_names: [*mut u8; 3],
    s_jquota_fmt: i32,
    s_want_extra_isize: u32,
    s_system_blks: *mut ext4_system_blocks,
    s_group_info: *mut *mut *mut ext4_group_info,
    s_buddy_cache: *mut inode,
    s_md_lock: spinlock,
    s_mb_offsets: *mut u16,
    s_mb_maxs: *mut u32,
    s_group_info_size: u32,
    s_mb_free_pending: __SlateRecord_types_h_190,
    s_freed_data_list: [list_head; 2],
    s_discard_list: list_head,
    s_discard_work: work_struct,
    s_retry_alloc_pending: __SlateRecord_types_h_190,
    s_mb_avg_fragment_size: *mut xarray,
    s_mb_largest_free_orders: *mut xarray,
    s_stripe: u64,
    s_mb_max_linear_groups: u32,
    s_mb_stream_request: u32,
    s_mb_max_to_scan: u32,
    s_mb_min_to_scan: u32,
    s_mb_stats: u32,
    s_mb_order2_reqs: u32,
    s_mb_group_prealloc: u32,
    s_max_dir_size_kb: u32,
    s_mb_prefetch: u32,
    s_mb_prefetch_limit: u32,
    s_mb_best_avail_max_trim_order: u32,
    s_sb_update_sec: u32,
    s_sb_update_kb: u32,
    s_mb_last_groups: *mut u32,
    s_mb_nr_global_goals: u32,
    s_bal_reqs: __SlateRecord_types_h_190,
    s_bal_success: __SlateRecord_types_h_190,
    s_bal_allocated: __SlateRecord_types_h_190,
    s_bal_ex_scanned: __SlateRecord_types_h_190,
    s_bal_cX_ex_scanned: [__SlateRecord_types_h_190; 5],
    s_bal_groups_scanned: __SlateRecord_types_h_190,
    s_bal_goals: __SlateRecord_types_h_190,
    s_bal_stream_goals: __SlateRecord_types_h_190,
    s_bal_len_goals: __SlateRecord_types_h_190,
    s_bal_breaks: __SlateRecord_types_h_190,
    s_bal_2orders: __SlateRecord_types_h_190,
    s_bal_cX_groups_considered: [__SlateRecord_types_h_197; 5],
    s_bal_cX_hits: [__SlateRecord_types_h_197; 5],
    s_bal_cX_failed: [__SlateRecord_types_h_197; 5],
    s_mb_buddies_generated: __SlateRecord_types_h_190,
    s_mb_generation_time: __SlateRecord_types_h_197,
    s_mb_lost_chunks: __SlateRecord_types_h_190,
    s_mb_preallocated: __SlateRecord_types_h_190,
    s_mb_discarded: __SlateRecord_types_h_190,
    s_lock_busy: __SlateRecord_types_h_190,
    s_locality_groups: *mut ext4_locality_group,
    s_sectors_written_start: u64,
    s_kbytes_written: u64,
    s_extent_max_zeroout_kb: u32,
    s_log_groups_per_flex: u32,
    s_flex_groups: *mut *mut flex_groups,
    s_flex_groups_allocated: u32,
    rsv_conversion_wq: *mut workqueue_struct,
    s_err_report: timer_list,
    s_err_report_sec: u64,
    s_li_request: *mut ext4_li_request,
    s_li_wait_mult: u32,
    s_mmp_tsk: *mut task_struct,
    s_last_trim_minblks: u64,
    s_min_folio_order: u16,
    s_max_folio_order: u16,
    s_csum_seed: u32,
    s_es_shrinker: *mut shrinker,
    s_es_list: list_head,
    s_es_nr_inode: i64,
    s_es_stats: ext4_es_stats,
    s_ea_block_cache: *mut mb_cache,
    s_ea_inode_cache: *mut mb_cache,
    s_ea_inode_to_free: llist_head,
    s_ea_inode_work: delayed_work,
    __slate_pad_136: [u8; 40],
    s_es_lock: spinlock,
    s_journal_triggers: [ext4_journal_trigger; 1],
    s_err_ratelimit_state: ratelimit_state,
    s_warning_ratelimit_state: ratelimit_state,
    s_msg_ratelimit_state: ratelimit_state,
    s_warning_count: __SlateRecord_types_h_190,
    s_msg_count: __SlateRecord_types_h_190,
    s_dummy_enc_policy: fscrypt_dummy_policy,
    s_writepages_rwsem: percpu_rw_semaphore,
    s_daxdev: *mut dax_device,
    s_dax_part_off: u64,
    s_bdev_wb_err: u32,
    s_bdev_wb_lock: spinlock,
    s_error_lock: spinlock,
    s_add_error_count: i32,
    s_first_error_code: i32,
    s_first_error_line: u32,
    s_first_error_ino: u32,
    s_first_error_block: u64,
    s_first_error_func: *const u8,
    s_first_error_time: i64,
    s_last_error_code: i32,
    s_last_error_line: u32,
    s_last_error_ino: u32,
    s_last_error_block: u64,
    s_last_error_func: *const u8,
    s_last_error_time: i64,
    s_sb_upd_work: work_struct,
    s_awu_min: u32,
    s_awu_max: u32,
    s_fc_subtid: __SlateRecord_types_h_190,
    s_fc_q: [list_head; 2],
    s_fc_dentry_q: [list_head; 2],
    s_fc_bytes: u32,
    s_fc_lock: mutex,
    s_fc_bh: *mut buffer_head,
    s_fc_stats: ext4_fc_stats,
    s_fc_snap_stats: ext4_fc_snap_stats,
    s_fc_ineligible_tid: u32,
    s_fc_replay_state: ext4_fc_replay_state,
}


fn EXT4_SB(sb: *mut super_block) -> *mut ext4_sb_info {
(unsafe { (*sb).s_fs_info }) as *mut ext4_sb_info
}

fn EXT4_I(inode: *mut inode) -> *mut ext4_inode_info {
container_of!(inode, ext4_inode_info)
}

fn ext4_valid_inum(sb: *mut super_block, ino: u64) -> i32 {
let __v0: bool = ino == 2 || ino >= (EXT4_FIRST_INO!(sb) as u64) && ino <= ((unsafe { (*(*EXT4_SB(sb)).s_es).s_inodes_count }) as u64);
__v0 as i32
}

fn ext4_test_mount_flag(sb: *mut super_block, bit: i32) -> i32 {
test_bit__4!(bit, unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_mount_flags) })
}

fn ext4_simulate_fail(sb: *mut super_block, code: u64) -> bool {
false
}

fn ext4_test_inode_flag(inode: *mut inode, bit: i32) -> i32 {
test_bit__4!(bit.wrapping_add(0), unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) })
}

fn ext4_set_inode_flag(inode: *mut inode, bit: i32) {
set_bit(bit.wrapping_add(0) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
}

fn ext4_clear_inode_flag(inode: *mut inode, bit: i32) {
clear_bit(bit.wrapping_add(0) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
}

fn ext4_test_inode_state(inode: *mut inode, bit: i32) -> i32 {
test_bit__4!(bit.wrapping_add(32), unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) })
}

fn ext4_set_inode_state(inode: *mut inode, bit: i32) {
set_bit(bit.wrapping_add(32) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
}

// The reason why "FAST_COMMIT" is a compat feature is that, FS becomes
// incompatible only if fast commit blocks are present in the FS. Since we
// clear the journal (and thus the fast commit blocks), we don't mark FS as
// incompatible. We also have a JBD2 incompat feature, which gets set when
// there are fast commit blocks present in the journal.
// Orphan file exists
// METADATA_CSUM also enables group descriptor checksums (GDT_CSUM).  When
// METADATA_CSUM is set, group descriptor checksums use the same algorithm as
// all other data structures' checksums.  However, the METADATA_CSUM and
// GDT_CSUM bits are mutually exclusive.
// Orphan file may be
// non-empty
// Needs recovery
// Journal device
// extents support
// EA in inode
// data in dirent
// >2GB or 3-lvl htree
// data in inode
fn ext4_has_feature_dir_index(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_compat }) & 32 != 0
}

fn ext4_has_feature_dir_nlink(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 32 != 0
}

fn ext4_has_feature_quota(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 256 != 0
}

fn ext4_has_feature_metadata_csum(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 1024 != 0
}

fn ext4_has_feature_filetype(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 2 != 0
}

fn ext4_has_feature_extents(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 64 != 0
}

fn ext4_has_feature_largedir(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 16384 != 0
}

fn ext4_forced_shutdown(sb: *mut super_block) -> i32 {
test_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_ext4_flags) })
}

fn ext4_emergency_ro(sb: *mut super_block) -> i32 {
test_bit!(3i32, unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_ext4_flags) })
}

fn ext4_emergency_state(sb: *mut super_block) -> i32 {
if unlikely__2!(ext4_forced_shutdown(sb)) != 0 {
return 5i32.wrapping_neg();
}
if unlikely__2!(ext4_emergency_ro(sb)) != 0 {
return 30i32.wrapping_neg();
}
0
}

// Structure of a directory entry
// Base length of the ext4 directory entry excluding the name length
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_dir_entry {
    inode: u32,
    rec_len: u16,
    name_len: u16,
    name: [u8; 255],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_dir_entry_hash {
    hash: u32,
    minor_hash: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_dir_entry_2 {
    inode: u32,
    rec_len: u16,
    name_len: u8,
    file_type: u8,
    name: [u8; 255],
}


fn ext4_hash_in_dirent(inode: *const inode) -> bool {
IS_CASEFOLDED!(inode) != 0 && IS_ENCRYPTED!(inode) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_dir_entry_tail {
    det_reserved_zero1: u32,
    det_rec_len: u16,
    det_reserved_zero2: u8,
    det_reserved_ft: u8,
    det_checksum: u32,
}


// Ext4 directory file types.  Only the low 3 bits are used.  The
// other bits are reserved for now.
fn ext4_dir_rec_len(name_len: u8, dir: *const inode) -> u32 {
let mut rec_len: i32 = (name_len as i32).wrapping_add(8).wrapping_add(3i32);
let __v0: bool = !dir.is_null() && ext4_hash_in_dirent(dir);
if __v0 {
rec_len = (rec_len as u64).wrapping_add(core::mem::size_of::<ext4_dir_entry_hash>() as u64) as i32;
}
(rec_len & -4i32) as u32
}

fn ext4_rec_len_from_disk(dlen: u16, blocksize: u32) -> u32 {
let len: u32 = dlen as u32;
if len == 65535u32 || len == 0 {
return blocksize;
}
len & 65532 | (len & 3) << 16i32
}

fn ext4_rec_len_to_disk(len: u32, blocksize: u32) -> u16 {
loop {
if unlikely!(len > blocksize || blocksize > 262144u32 || len & 3 != 0) != 0 {
loop {
unsafe {
core::arch::asm!("648: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 648b - ., 3; .popsection\n# {0}", const 648 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ext4_rec_len_to_disk_0, sym __slate_asm_string_ext4_rec_len_to_disk_1, const 2621 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if len < 65536 {
return len as u16;
}
if len == blocksize {
if blocksize == 65536 {
return 65535u16;
} else {
return 0;
}
}
(len & 65532 | len >> 16i32 & 3) as u16
}

fn ext4_chksum(crc: u32, address: *const (), length: u32) -> u32 {
unsafe { crc32c(crc, address, length as u64) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dx_hash_info {
    hash: u32,
    minor_hash: u32,
    hash_version: i32,
    seed: *mut u32,
}


// Control parameters used by ext4_htree_next_block
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_filename {
    usr_fname: *const qstr,
    disk_name: fscrypt_str,
    hinfo: dx_hash_info,
}


// Special error return code only used by dx_probe() and its callers.
// htree levels for ext4
fn ext4_dir_htree_level(sb: *mut super_block) -> i32 {
if ext4_has_feature_largedir(sb) { 3 } else { 2 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_li_request {
    lr_super: *mut super_block,
    lr_mode: u32,
    lr_first_not_zeroed: u32,
    lr_next_group: u32,
    lr_request: list_head,
    lr_next_sched: u64,
    lr_timeout: u64,
}


fn ext4_fname_setup_ci_filename(dir: *mut inode, iname: *const qstr, fname: &mut ext4_filename) -> i32 {
0
}

fn ext4_fname_free_ci_filename(fname: &ext4_filename) {
}

fn ext4_fname_setup_filename(dir: *mut inode, iname: *const qstr, lookup: i32, fname: &mut ext4_filename) -> i32 {
unsafe {
(*(fname as *mut ext4_filename)).usr_fname = iname;
}
unsafe {
(*(fname as *mut ext4_filename)).disk_name.name = (*iname).name as *mut u8;
}
unsafe {
(*(fname as *mut ext4_filename)).disk_name.len = (*iname).__slate_anon_0.__slate_anon_0.len;
}
ext4_fname_setup_ci_filename(dir, iname, fname)
}

fn ext4_fname_prepare_lookup(dir: *mut inode, dentry: *mut dentry, fname: &mut ext4_filename) -> i32 {
ext4_fname_setup_filename(dir, unsafe { core::ptr::addr_of!((*dentry).__slate_anon_4.d_name) }, 1i32, fname)
}

fn ext4_fname_free_filename(fname: &ext4_filename) {
ext4_fname_free_ci_filename(fname);
}

fn ext4_update_dx_flag(inode: *mut inode) {
let __v0: bool = !ext4_has_feature_dir_index(unsafe { (*inode).i_sb }) && ext4_test_inode_flag(inode, 12i32) != 0;
if __v0 {
// ext4_iget() should have caught this...
let mut __v1: i64;
let __ret_warn_on: i32 = !(!ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb })) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("649: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 649b - ., 3; .popsection\n# {0}", const 649 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ext4_update_dx_flag_2, sym __slate_asm_string_ext4_update_dx_flag_3, const 3020 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("650: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 650b - ., 4; .popsection\n# {0}", const 650 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
ext4_clear_inode_flag(inode, 12i32);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_group_info {
    bb_state: u64,
    bb_free_root: rb_root,
    bb_first_free: i32,
    bb_free: i32,
    bb_fragments: i32,
    bb_avg_fragment_size_order: i32,
    bb_largest_free_order: i32,
    bb_group: u32,
    bb_prealloc_list: list_head,
    alloc_sem: rw_semaphore,
    bb_counters: [i32; 0],
}


fn ext4_quota_capable(sb: *mut super_block) -> bool {
(unsafe { (*EXT4_SB(sb)).s_mount_opt }) & 262144 != 0 || ext4_has_feature_quota(sb)
}

fn ext4_has_inline_data(inode: *mut inode) -> i32 {
let __v0: bool = ext4_test_inode_flag(inode, 28i32) != 0 && (unsafe { (*EXT4_I(inode)).i_inline_off }) != 0;
__v0 as i32
}

static ext4_type_by_mode: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t0: [u8; 16] = unsafe { core::mem::zeroed() };
__t0[1] = 5u8;
__t0[2] = 3u8;
__t0[4] = 2u8;
__t0[6] = 4u8;
__t0[8] = 1u8;
__t0[10] = 7u8;
__t0[12] = 6u8;
__t0
});

fn ext4_set_de_type(sb: *mut super_block, de: *mut ext4_dir_entry_2, mode: u16) {
if ext4_has_feature_filetype(sb) {
unsafe {
(*de).file_type = ext4_type_by_mode.0[(((mode as i32) & 61440) >> 12i32) as usize];
}
}
}

// Add new method to test whether block and inode bitmaps are properly
// initialized. With uninit_bg reading the block from disk is not enough
// to mark the bitmap uptodate. We need to also zero-out the bitmap
fn ext4_handle_valid(handle: *mut jbd2_journal_handle) -> i32 {
(!handle.is_null() && ((unsafe { (*handle).__slate_bits_0.__get_h_invalid() }) as i32) == 0) as i32
}

fn ext4_handle_sync(handle: *mut jbd2_journal_handle) {
if ext4_handle_valid(handle) != 0 {
unsafe {
(*handle).__slate_bits_0.__set_h_sync(1u8);
}
}
}

fn ext4_free_metadata_revoke_credits(sb: *mut super_block, blocks: i32) -> i32 {
// Freeing each metadata block can result in freeing one cluster
(blocks as u32).wrapping_mul(unsafe { (*EXT4_SB(sb)).s_cluster_ratio }) as i32
}

fn ext4_trans_default_revoke_credits(sb: *mut super_block) -> i32 {
ext4_free_metadata_revoke_credits(sb, 8i32)
}

fn __ext4_journal_start(inode: *mut inode, line: u32, r#type: i32, blocks: i32, rsv_blocks: i32, revoke_creds: i32) -> *mut jbd2_journal_handle {
unsafe { __ext4_journal_start_sb(inode, (*inode).i_sb, line, r#type, blocks, rsv_blocks, revoke_creds) }
}

fn ext4_journal_current_handle() -> *mut jbd2_journal_handle {
journal_current_handle()
}

// Ensure @handle has at least @check_creds credits available. If not,
// transaction will be extended or restarted to contain at least @extend_cred
// credits. Before restarting transaction @fn is executed to allow for cleanup
// before the transaction is restarted.
//
// The return value is < 0 in case of error, 0 in case the handle has enough
// credits or transaction extension succeeded, 1 in case transaction had to be
// restarted.
// Ext4 inode journal modes
// journal data mode
// ordered data mode
// writeback data mode
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
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl_entry {
    e_tag: i16,
    e_perm: u16,
    __slate_anon_2: __SlateRecord_posix_acl_h_23,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
    __slate_anon_0: __SlateRecord_posix_acl_h_31,
    a_entries: [posix_acl_entry; 0],
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
    __slate_bits_0: __slate_bits::__SlateBits538U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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
fn __do_trace_ext4_unlink_enter(parent: *mut inode, dentry: *mut dentry) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1057: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1057) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1058: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1058) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1060() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_unlink_enter)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1059: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1059;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut dentry) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut dentry) -> i32>>(__SCT__tp_func_ext4_unlink_enter as *const ()) };
unsafe { __v12.unwrap()(__data, parent, dentry) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_unlink_enter: __SlateAlign16<[u8; 18]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 117u8, 110u8, 108u8, 105u8, 110u8, 107u8, 95u8, 101u8, 110u8, 116u8, 101u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_unlink_enter_1061: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_unlink_enter) }) as *mut ();

fn trace_ext4_unlink_enter(parent: *mut inode, dentry: *mut dentry) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_unlink_enter.key) }) != 0 {
__do_trace_ext4_unlink_enter(parent, dentry);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1062: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_unlink_exit(dentry: *mut dentry, ret: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_1064: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1064) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1065: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1065) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1067() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_unlink_exit)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1066: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1066;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut dentry, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut dentry, i32) -> i32>>(__SCT__tp_func_ext4_unlink_exit as *const ()) };
unsafe { __v12.unwrap()(__data, dentry, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_unlink_exit: __SlateAlign16<[u8; 17]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 117u8, 110u8, 108u8, 105u8, 110u8, 107u8, 95u8, 101u8, 120u8, 105u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_unlink_exit_1068: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_unlink_exit) }) as *mut ();

fn trace_ext4_unlink_exit(dentry: *mut dentry, ret: i32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_unlink_exit.key) }) != 0 {
__do_trace_ext4_unlink_exit(dentry, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1069: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// define how far ahead to read directories while searching them.
fn ext4_append(handle: *mut jbd2_journal_handle, inode: *mut inode, block: &mut u32) -> *mut buffer_head {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut map: core::mem::MaybeUninit<ext4_map_blocks> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut ext4_map_blocks = core::ptr::addr_of_mut!(map) as *mut ext4_map_blocks;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*EXT4_SB((*inode).i_sb)).s_max_dir_size_kb != 0 {
__v3 = (*inode).i_size >> 10i32 >= ((*EXT4_SB((*inode).i_sb)).s_max_dir_size_kb as i64);
} else {
__v3 = false;
}
if __v3 {
return ERR_PTR(28i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
*(block as *mut u32) = ((*inode).i_size >> ((*(*inode).i_sb).s_blocksize_bits as i32)) as u32;
(*__slate_slot_6).m_lblk = *(block as *mut u32);
(*__slate_slot_6).m_len = 1;
// We're appending new directory block. Make sure the block is not
// allocated yet, otherwise we will end up corrupting the
// directory.
*__slate_slot_4 = ext4_map_blocks(core::ptr::null_mut::<jbd2_journal_handle>(), inode, core::ptr::addr_of_mut!(*__slate_slot_6), 0i32);
if *__slate_slot_4 < 0 {
return ERR_PTR(*__slate_slot_4 as i64) as *mut buffer_head;
} else {
if *__slate_slot_4 != 0 {
__ext4_error_inode(inode, c"ext4_append".as_ptr() as *const u8, 79u32, 0u64, 0i32, c"Logical block already allocated".as_ptr() as *const u8);
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
*__slate_slot_5 = ext4_bread(handle, inode, *(block as *mut u32), 1i32);
if IS_ERR(*__slate_slot_5 as *const ()) {
return *__slate_slot_5;
} else {
(*inode).i_size = ((*inode).i_size as u64).wrapping_add((*(*inode).i_sb).s_blocksize) as i64;
(*EXT4_I!(inode)).i_disksize = (*inode).i_size;
*__slate_slot_4 = ext4_mark_inode_dirty__12!(handle, inode);
if *__slate_slot_4 == 0 {
BUFFER_TRACE!();
*__slate_slot_4 = ext4_journal_get_write_access__9!(handle, unsafe { (*inode).i_sb }, *__slate_slot_5, 1i32);
if *__slate_slot_4 == 0 {
return *__slate_slot_5;
}
}
brelse(*__slate_slot_5);
ext4_std_error__20!(unsafe { (*inode).i_sb }, *__slate_slot_4);
return ERR_PTR(*__slate_slot_4 as i64) as *mut buffer_head;
}
}
}
}
}
}

// Hints to ext4_read_dirblock regarding whether we expect a directory
// block being read to be an index block, or a block containing
// directory entries (and if the latter, whether it was found via a
// logical block in an htree index block).  This is used to control
// what sort of sanity checkinig ext4_read_dirblock() will do on the
// directory block read from the storage device.  EITHER will means
// the caller doesn't know what kind of directory block will be read,
// so no specific verification will be done.
fn __ext4_read_dirblock(inode: *mut inode, block: u32, r#type: u32, func: *const u8, line: u32) -> *mut buffer_head {
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut is_dx_block: i32 = 0;
if (block as i64) >= (unsafe { (*inode).i_size }) >> ((unsafe { (*inode).i_blkbits }) as i32) {
unsafe { __ext4_error_inode(inode, func, line, block as u64, 0i32, c"Attempting to read directory block (%u) that is past i_size (%llu)".as_ptr() as *const u8, block, (*inode).i_size) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
}
if ext4_simulate_fail(unsafe { (*inode).i_sb }, 7u64) {
bh = ERR_PTR(5i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
bh = unsafe { ext4_bread(core::ptr::null_mut::<jbd2_journal_handle>(), inode, block, 0i32) };
}
if IS_ERR(bh as *const ()) {
unsafe { __ext4_warning((*inode).i_sb, func, line, c"inode #%llu: lblock %lu: comm %s: error %pe reading directory block".as_ptr() as *const u8, (*inode).i_ino, block as u64, (*get_current()).comm.as_mut_ptr() as *mut u8, bh) };
return bh;
}
// The first directory block must not be a hole.
if bh.is_null() && ((r#type as u32) == 1 || (r#type as u32) == 3 || block == 0) {
unsafe { __ext4_error_inode(inode, func, line, block as u64, 0i32, c"Directory hole found for htree %s block %u".as_ptr() as *const u8, if (r#type as u32) == 1 { c"index".as_ptr() as *mut u8 } else { c"leaf".as_ptr() as *mut u8 }, block) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
}
if bh.is_null() {
return core::ptr::null_mut::<buffer_head>();
}
let dirent: *mut ext4_dir_entry = (unsafe { (*bh).b_data }) as *mut ext4_dir_entry;
// Determine whether or not we have an index block
if is_dx!(inode) {
if block == 0 {
is_dx_block = 1;
} else {
if (ext4_rec_len_from_disk(unsafe { (*dirent).rec_len }, (unsafe { (*(*inode).i_sb).s_blocksize }) as u32) as u64) == unsafe { (*(*inode).i_sb).s_blocksize } {
is_dx_block = 1;
}
}
}
if is_dx_block == 0 && (r#type as u32) == 1 {
unsafe { __ext4_error_inode(inode, func, line, block as u64, 0i32, c"directory leaf block found instead of index block".as_ptr() as *const u8) };
brelse(bh);
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
}
let __v1: bool = !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) || buffer_verified(bh as *const buffer_head) != 0;
if __v1 {
return bh;
}
// An empty leaf block can get mistaken for a index block; for
// this reason, we can only check the index checksum when the
// caller is sure it should be an index block.
if is_dx_block != 0 && (r#type as u32) == 1 {
let __v2: bool = ext4_dx_csum_verify(inode, dirent) != 0 && !ext4_simulate_fail(unsafe { (*inode).i_sb }, 8u64);
if __v2 {
set_buffer_verified(bh);
} else {
unsafe { __ext4_error_inode(inode, func, line, block as u64, 74i32, c"Directory index failed checksum".as_ptr() as *const u8) };
brelse(bh);
return ERR_PTR(74i32.wrapping_neg() as i64) as *mut buffer_head;
}
}
if is_dx_block == 0 {
let __v3: bool = ext4_dirblock_csum_verify(inode, bh) != 0 && !ext4_simulate_fail(unsafe { (*inode).i_sb }, 8u64);
if __v3 {
set_buffer_verified(bh);
} else {
unsafe { __ext4_error_inode(inode, func, line, block as u64, 74i32, c"Directory block failed checksum".as_ptr() as *const u8) };
brelse(bh);
return ERR_PTR(74i32.wrapping_neg() as i64) as *mut buffer_head;
}
}
bh
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fake_dirent {
    inode: u32,
    rec_len: u16,
    name_len: u8,
    file_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dx_countlimit {
    limit: u16,
    count: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dx_entry {
    hash: u32,
    block: u32,
}


// dx_root_info is laid out so that if it should somehow get overlaid by a
// dirent the two low bits of the hash version will be zero.  Therefore, the
// hash version mod 4 should never be 0.  Sincerely, the paranoia department.
#[repr(C)]
#[derive(Clone, Copy)]
struct dx_root {
    dot: fake_dirent,
    dot_name: [u8; 4],
    dotdot: fake_dirent,
    dotdot_name: [u8; 4],
    info: dx_root_info,
    entries: [dx_entry; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dx_node {
    fake: fake_dirent,
    entries: [dx_entry; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dx_frame {
    bh: *mut buffer_head,
    entries: *mut dx_entry,
    at: *mut dx_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dx_map_entry {
    hash: u32,
    offs: u16,
    size: u16,
}


/// This goes at the end of each htree block.
#[repr(C)]
#[derive(Clone, Copy)]
struct dx_tail {
    dt_reserved: u32,
    /// crc32c(uuid+inum+dirblock)
    dt_checksum: u32,
}


/// checksumming functions
#[unsafe(no_mangle)]
extern "C" fn ext4_initialize_dirent_tail(bh: *mut buffer_head, blocksize: u32) {
let t: *mut ext4_dir_entry_tail = EXT4_DIRENT_TAIL!(unsafe { (*bh).b_data }, blocksize);
unsafe { memset(t as *mut (), 0i32, core::mem::size_of::<ext4_dir_entry_tail>() as u64) };
unsafe {
(*t).det_rec_len = ext4_rec_len_to_disk(core::mem::size_of::<ext4_dir_entry_tail>() as u32, blocksize);
}
unsafe {
(*t).det_reserved_ft = 222;
}
}

/// Walk through a dirent block to find a checksum "dirent" at the tail
fn get_dirent_tail(inode: *mut inode, bh: *mut buffer_head) -> *mut ext4_dir_entry_tail {
let blocksize: i32 = EXT4_BLOCK_SIZE!(unsafe { (*inode).i_sb }) as i32;
let t: *mut ext4_dir_entry_tail = EXT4_DIRENT_TAIL__2!(unsafe { (*bh).b_data }, EXT4_BLOCK_SIZE!(unsafe { (*inode).i_sb }));
let __v0: bool = (unsafe { (*t).det_reserved_zero1 }) != 0 || (ext4_rec_len_from_disk(unsafe { (*t).det_rec_len }, blocksize as u32) as u64) != (core::mem::size_of::<ext4_dir_entry_tail>() as u64);
if __v0 || (unsafe { (*t).det_reserved_zero2 }) != 0 || ((unsafe { (*t).det_reserved_ft }) as i32) != 222 {
return core::ptr::null_mut::<ext4_dir_entry_tail>();
}
t
}

fn ext4_dirblock_csum(inode: *mut inode, dirent: *mut (), size: i32) -> u32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let csum: u32 = ext4_chksum(unsafe { (*ei).i_csum_seed }, dirent as *const (), size as u32);
csum
}

fn __warn_no_space_for_csum(inode: *mut inode, func: *const u8, line: u32) {
unsafe { __ext4_warning_inode(inode as *const inode, func, line, c"No space for directory leaf checksum. Please run e2fsck -D.".as_ptr() as *const u8) };
}

#[unsafe(no_mangle)]
extern "C" fn ext4_dirblock_csum_verify(inode: *mut inode, bh: *mut buffer_head) -> i32 {
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return 1;
}
let t: *mut ext4_dir_entry_tail = get_dirent_tail(inode, bh);
if t.is_null() {
warn_no_space_for_csum__2!(inode);
return 0;
}
if (unsafe { (*t).det_checksum }) != ext4_dirblock_csum(inode, (unsafe { (*bh).b_data }) as *mut (), ((unsafe { (t as *mut u8).offset_from((*bh).b_data as *mut u8) }) as i64) as i32) {
return 0;
}
1
}

fn ext4_dirblock_csum_set(inode: *mut inode, bh: *mut buffer_head) {
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return;
}
let t: *mut ext4_dir_entry_tail = get_dirent_tail(inode, bh);
if t.is_null() {
warn_no_space_for_csum!(inode);
return;
}
unsafe {
(*t).det_checksum = ext4_dirblock_csum(inode, (*bh).b_data as *mut (), ((t as *mut u8).offset_from((*bh).b_data as *mut u8) as i64) as i32);
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_handle_dirty_dirblock(handle: *mut jbd2_journal_handle, inode: *mut inode, bh: *mut buffer_head) -> i32 {
ext4_dirblock_csum_set(inode, bh);
ext4_handle_dirty_metadata__59!(handle, inode, bh)
}

fn get_dx_countlimit(inode: *mut inode, dirent: *mut ext4_dir_entry, offset: *mut i32) -> *mut dx_countlimit {
let mut count_offset: i32 = 0;
let blocksize: i32 = EXT4_BLOCK_SIZE!(unsafe { (*inode).i_sb }) as i32;
let rlen: u32 = ext4_rec_len_from_disk(unsafe { (*dirent).rec_len }, blocksize as u32);
if rlen == (blocksize as u32) {
count_offset = 8;
} else {
if rlen == 12 {
let dp: *mut ext4_dir_entry = ((unsafe { (dirent as *const u8).wrapping_offset(12 as isize) }) as *mut ()) as *mut ext4_dir_entry;
if ext4_rec_len_from_disk(unsafe { (*dp).rec_len }, blocksize as u32) != (blocksize.wrapping_sub(12) as u32) {
return core::ptr::null_mut::<dx_countlimit>();
}
let root: *mut dx_root_info = ((unsafe { (dp as *const u8).wrapping_offset(12 as isize) }) as *mut ()) as *mut dx_root_info;
if (unsafe { (*root).reserved_zero }) != 0 || (((unsafe { (*root).info_length }) as i32) as u64) != (core::mem::size_of::<dx_root_info>() as u64) {
return core::ptr::null_mut::<dx_countlimit>();
}
count_offset = 32;
} else {
return core::ptr::null_mut::<dx_countlimit>();
}
}
if !offset.is_null() {
unsafe {
*offset = count_offset;
}
}
((unsafe { (dirent as *const u8).wrapping_offset(count_offset as isize) }) as *mut ()) as *mut dx_countlimit
}

fn ext4_dx_csum(inode: *mut inode, dirent: *mut ext4_dir_entry, count_offset: i32, count: i32, t: *mut dx_tail) -> u32 {
let ei: *mut ext4_inode_info = EXT4_I!(inode);
let mut csum: u32 = 0;
let mut size: i32 = 0;
let mut dummy_csum: u32 = 0;
let offset: i32 = offsetof__45!(dx_tail) as i32;
size = (count_offset as u64).wrapping_add((count as u64).wrapping_mul(core::mem::size_of::<dx_entry>() as u64)) as i32;
csum = ext4_chksum(unsafe { (*ei).i_csum_seed }, dirent as *const (), size as u32);
csum = ext4_chksum(csum, t as *const (), offset as u32);
csum = ext4_chksum(csum, (core::ptr::addr_of_mut!(dummy_csum) as *mut u8) as *const (), core::mem::size_of::<u32>() as u32);
csum
}

fn ext4_dx_csum_verify(inode: *mut inode, dirent: *mut ext4_dir_entry) -> i32 {
let mut count_offset: i32 = 0;
let mut limit: i32 = 0;
let mut count: i32 = 0;
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return 1;
}
let c: *mut dx_countlimit = get_dx_countlimit(inode, dirent, core::ptr::addr_of_mut!(count_offset));
if c.is_null() {
unsafe { __ext4_error_inode(inode, c"ext4_dx_csum_verify".as_ptr() as *const u8, 470u32, 0u64, 0i32, c"dir seems corrupt?  Run e2fsck -D.".as_ptr() as *const u8) };
return 0;
}
limit = (unsafe { (*c).limit }) as i32;
count = (unsafe { (*c).count }) as i32;
if (count_offset as u64).wrapping_add((limit as u64).wrapping_mul(core::mem::size_of::<dx_entry>() as u64)) > EXT4_BLOCK_SIZE!(unsafe { (*inode).i_sb }).wrapping_sub(core::mem::size_of::<dx_tail>() as u64) {
warn_no_space_for_csum__4!(inode);
return 0;
}
let t: *mut dx_tail = (c as *mut dx_entry).wrapping_offset(limit as isize) as *mut dx_tail;
if (unsafe { (*t).dt_checksum }) != ext4_dx_csum(inode, dirent, count_offset, count, t) {
return 0;
}
1
}

fn ext4_dx_csum_set(inode: *mut inode, dirent: *mut ext4_dir_entry) {
let mut count_offset: i32 = 0;
let mut limit: i32 = 0;
let mut count: i32 = 0;
if !ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
return;
}
let c: *mut dx_countlimit = get_dx_countlimit(inode, dirent, core::ptr::addr_of_mut!(count_offset));
if c.is_null() {
unsafe { __ext4_error_inode(inode, c"ext4_dx_csum_set".as_ptr() as *const u8, 499u32, 0u64, 0i32, c"dir seems corrupt?  Run e2fsck -D.".as_ptr() as *const u8) };
return;
}
limit = (unsafe { (*c).limit }) as i32;
count = (unsafe { (*c).count }) as i32;
if (count_offset as u64).wrapping_add((limit as u64).wrapping_mul(core::mem::size_of::<dx_entry>() as u64)) > EXT4_BLOCK_SIZE!(unsafe { (*inode).i_sb }).wrapping_sub(core::mem::size_of::<dx_tail>() as u64) {
warn_no_space_for_csum__3!(inode);
return;
}
let t: *mut dx_tail = (c as *mut dx_entry).wrapping_offset(limit as isize) as *mut dx_tail;
unsafe {
(*t).dt_checksum = ext4_dx_csum(inode, dirent, count_offset, count, t);
}
}

fn ext4_handle_dirty_dx_node(handle: *mut jbd2_journal_handle, inode: *mut inode, bh: *mut buffer_head) -> i32 {
ext4_dx_csum_set(inode, (unsafe { (*bh).b_data }) as *mut ext4_dir_entry);
ext4_handle_dirty_metadata__28!(handle, inode, bh)
}

/// p is at least 6 bytes before the end of page
fn ext4_next_entry(p: *mut ext4_dir_entry_2, blocksize: u64) -> *mut ext4_dir_entry_2 {
(p as *mut u8).wrapping_offset(ext4_rec_len_from_disk(unsafe { (*p).rec_len }, blocksize as u32) as isize) as *mut ext4_dir_entry_2
}

// Future: use high four bits of block for coalesce-on-delete flags
// Mask them off for now.
fn dx_get_block(entry: *mut dx_entry) -> u32 {
(unsafe { (*entry).block }) & 268435455
}

fn dx_set_block(entry: *mut dx_entry, value: u32) {
unsafe {
(*entry).block = value;
}
}

fn dx_get_hash(entry: *mut dx_entry) -> u32 {
unsafe { (*entry).hash }
}

fn dx_set_hash(entry: *mut dx_entry, value: u32) {
unsafe {
(*entry).hash = value;
}
}

fn dx_get_count(entries: *mut dx_entry) -> u32 {
(unsafe { (*(entries as *mut dx_countlimit)).count }) as u32
}

fn dx_get_limit(entries: *mut dx_entry) -> u32 {
(unsafe { (*(entries as *mut dx_countlimit)).limit }) as u32
}

fn dx_set_count(entries: *mut dx_entry, value: u32) {
unsafe {
(*(entries as *mut dx_countlimit)).count = value as u16;
}
}

fn dx_set_limit(entries: *mut dx_entry, value: u32) {
unsafe {
(*(entries as *mut dx_countlimit)).limit = value as u16;
}
}

fn dx_root_limit(dir: *mut inode, infosize: u32) -> u32 {
let mut entry_space: u32 = unsafe { (*(*dir).i_sb).s_blocksize }.wrapping_sub(ext4_dir_rec_len(1u8, core::ptr::null::<inode>()) as u64).wrapping_sub(ext4_dir_rec_len(2u8, core::ptr::null::<inode>()) as u64).wrapping_sub(infosize as u64) as u32;
if ext4_has_feature_metadata_csum(unsafe { (*dir).i_sb }) {
entry_space = (entry_space as u64).wrapping_sub(core::mem::size_of::<dx_tail>() as u64) as u32;
}
((entry_space as u64) / (core::mem::size_of::<dx_entry>() as u64)) as u32
}

fn dx_node_limit(dir: *mut inode) -> u32 {
let mut entry_space: u32 = unsafe { (*(*dir).i_sb).s_blocksize }.wrapping_sub(ext4_dir_rec_len(0u8, dir as *const inode) as u64) as u32;
if ext4_has_feature_metadata_csum(unsafe { (*dir).i_sb }) {
entry_space = (entry_space as u64).wrapping_sub(core::mem::size_of::<dx_tail>() as u64) as u32;
}
((entry_space as u64) / (core::mem::size_of::<dx_entry>() as u64)) as u32
}

/// Debug
fn htree_rep_invariant_check(at: *mut dx_entry, target: *mut dx_entry, hash: u32, n: u32) {
}

/// Probe for a directory leaf block to search.
///
/// dx_probe can return ERR_BAD_DX_DIR, which means there was a format
/// error in the directory index, and the caller should fall back to
/// searching the directory normally.  The callers of dx_probe **MUST**
/// check for this error code, and make sure it never gets reflected
/// back to userspace.
fn dx_probe(fname: *mut ext4_filename, dir: *mut inode, mut hinfo: *mut dx_hash_info, frame_in: *mut dx_frame) -> *mut dx_frame {
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut blocks: core::mem::MaybeUninit<[u32; 3]> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut [u32; 3] = core::ptr::addr_of_mut!(blocks) as *mut [u32; 3];
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut hash: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(hash) as *mut u32;
let mut ret_err: *mut dx_frame = unsafe { core::mem::zeroed() };
let mut frame: *mut dx_frame = unsafe { core::mem::zeroed() };
let mut root: core::mem::MaybeUninit<*mut dx_root> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut dx_root = core::ptr::addr_of_mut!(root) as *mut *mut dx_root;
let mut m: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut dx_entry = core::ptr::addr_of_mut!(m) as *mut *mut dx_entry;
let mut q: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut dx_entry = core::ptr::addr_of_mut!(q) as *mut *mut dx_entry;
let mut p: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut dx_entry = core::ptr::addr_of_mut!(p) as *mut *mut dx_entry;
let mut entries: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut dx_entry = core::ptr::addr_of_mut!(entries) as *mut *mut dx_entry;
let mut at: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut dx_entry = core::ptr::addr_of_mut!(at) as *mut *mut dx_entry;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut level: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(level) as *mut u32;
let mut indirect: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(indirect) as *mut u32;
let mut count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(count) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
frame = frame_in;
ret_err = ERR_PTR(4094i32.wrapping_neg() as i64) as *mut dx_frame;
let __slate_auto_bytes_10 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
memset(frame_in as *mut (), 0i32, 3u64.wrapping_mul(core::mem::size_of::<dx_frame>() as u64));
(*frame).bh = ext4_read_dirblock__2!(dir, 0i32, 1i32);
if IS_ERR((*frame).bh as *const ()) {
return (*frame).bh as *mut dx_frame;
} else {
'__join_3: {
*__slate_slot_19 = (*(*frame).bh).b_data as *mut dx_root;
if ((*(*__slate_slot_19)).info.hash_version as i32) != 2 && ((*(*__slate_slot_19)).info.hash_version as i32) != 1 && ((*(*__slate_slot_19)).info.hash_version as i32) != 0 && ((*(*__slate_slot_19)).info.hash_version as i32) != 6 {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 801u32, c"Unrecognised inode hash code %u".as_ptr() as *const u8, (*(*__slate_slot_19)).info.hash_version as i32);
} else {
if ext4_hash_in_dirent(dir as *const inode) {
if ((*(*__slate_slot_19)).info.hash_version as i32) != 6 {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 807u32, c"Hash in dirent, but hash is not SIPHASH".as_ptr() as *const u8);
break '__join_3;
}
} else {
if ((*(*__slate_slot_19)).info.hash_version as i32) == 6 {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 813u32, c"Hash code is SIPHASH, but hash not in dirent".as_ptr() as *const u8);
break '__join_3;
}
}
if !fname.is_null() {
hinfo = core::ptr::addr_of_mut!((*fname).hinfo);
}
(*hinfo).hash_version = (*(*__slate_slot_19)).info.hash_version as i32;
if (*hinfo).hash_version <= 2 {
let __v13: i32 = (*EXT4_SB((*dir).i_sb)).s_hash_unsigned;
let __v11: i32 = (*hinfo).hash_version;
(*hinfo).hash_version = __v11.wrapping_add(__v13);
}
(*hinfo).seed = (*EXT4_SB((*dir).i_sb)).s_hash_seed.as_mut_ptr() as *mut u32;
// hash is already computed for encrypted casefolded directory
if !fname.is_null() && !fname_name!(fname).is_null() && !(IS_ENCRYPTED!(dir) != 0 && IS_CASEFOLDED!(dir) != 0) {
ret = ext4fs_dirhash(dir as *const inode, fname_name!(fname) as *const u8, fname_len!(fname) as i32, hinfo);
if ret < 0 {
ret_err = ERR_PTR(ret as i64) as *mut dx_frame;
break '__join_3;
}
}
*__slate_slot_16 = (*hinfo).hash;
if ((*(*__slate_slot_19)).info.unused_flags as i32) & 1 != 0 {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 837u32, c"Unimplemented hash flags: %#06x".as_ptr() as *const u8, (*(*__slate_slot_19)).info.unused_flags as i32);
} else {
*__slate_slot_27 = (*(*__slate_slot_19)).info.indirect_levels as u32;
if *__slate_slot_27 >= (ext4_dir_htree_level((*dir).i_sb) as u32) {
__ext4_warning((*dir).i_sb, c"dx_probe".as_ptr() as *const u8, 846u32, c"Directory (ino: %llu) htree depth %#06x exceedsupported value".as_ptr() as *const u8, (*dir).i_ino, ext4_dir_htree_level((*dir).i_sb));
if ext4_dir_htree_level((*dir).i_sb) < 3 {
__ext4_warning((*dir).i_sb, c"dx_probe".as_ptr() as *const u8, 849u32, c"Enable large directory feature to access it".as_ptr() as *const u8);
}
} else {
*__slate_slot_23 = (core::ptr::addr_of_mut!((*(*__slate_slot_19)).info) as *mut u8).wrapping_offset(((*(*__slate_slot_19)).info.info_length as i32) as isize) as *mut dx_entry;
if dx_get_limit(*__slate_slot_23) != dx_root_limit(dir, (*(*__slate_slot_19)).info.info_length as u32) {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 861u32, c"dx entry: limit %u != root limit %u".as_ptr() as *const u8, dx_get_limit(*__slate_slot_23), dx_root_limit(dir, (*(*__slate_slot_19)).info.info_length as u32));
} else {
*__slate_slot_26 = 0;
(*__slate_slot_14)[0] = 0;
'__join_22: {
'__join_14: {
'__join_8: {
loop {
if true {
*__slate_slot_28 = dx_get_count(*__slate_slot_23);
if *__slate_slot_28 == 0 {
__v8 = true;
} else {
__v8 = *__slate_slot_28 > dx_get_limit(*__slate_slot_23);
}
if __v8 {
break '__join_22;
} else {
*__slate_slot_22 = (*__slate_slot_23).wrapping_offset(1 as isize);
*__slate_slot_21 = (*__slate_slot_23).wrapping_offset(*__slate_slot_28 as isize).wrapping_offset(-(1 as isize));
while *__slate_slot_22 <= *__slate_slot_21 {
*__slate_slot_20 = (*__slate_slot_22).wrapping_offset((((*__slate_slot_21).offset_from(*__slate_slot_22 as *mut dx_entry) as i64) / 2) as isize);
if dx_get_hash(*__slate_slot_20) > *__slate_slot_16 {
*__slate_slot_21 = (*__slate_slot_20).wrapping_offset(-(1 as isize));
} else {
*__slate_slot_22 = (*__slate_slot_20).wrapping_offset(1 as isize);
}
}
htree_rep_invariant_check(*__slate_slot_23, *__slate_slot_22, *__slate_slot_16, (*__slate_slot_28).wrapping_sub(1));
*__slate_slot_24 = (*__slate_slot_22).wrapping_offset(-(1 as isize));
(*frame).entries = *__slate_slot_23;
(*frame).at = *__slate_slot_24;
*__slate_slot_15 = dx_get_block(*__slate_slot_24);
*__slate_slot_25 = 0;
while *__slate_slot_25 <= *__slate_slot_26 {
if (*__slate_slot_14)[*__slate_slot_25 as usize] == *__slate_slot_15 {
break '__join_14;
} else {
let __v7: u32 = *__slate_slot_25;
*__slate_slot_25 = __v7.wrapping_add(1);
}
}
let __v5: u32 = *__slate_slot_26;
let __v4: u32 = __v5.wrapping_add(1);
*__slate_slot_26 = __v4;
if __v4 > *__slate_slot_27 {
return frame;
} else {
(*__slate_slot_14)[*__slate_slot_26 as usize] = *__slate_slot_15;
let __v3: *mut dx_frame = frame;
frame = __v3.wrapping_offset(1 as isize);
(*frame).bh = ext4_read_dirblock__4!(dir, *__slate_slot_15, 1i32);
if IS_ERR((*frame).bh as *const ()) {
break '__join_8;
} else {
*__slate_slot_23 = core::ptr::addr_of_mut!((*((*(*frame).bh).b_data as *mut dx_node)).entries) as *mut dx_entry;
if dx_get_limit(*__slate_slot_23) != dx_node_limit(dir) {
break;
}
}
}
}
} else {
break '__join_3;
}
}
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 922u32, c"dx entry: limit %u != node limit %u".as_ptr() as *const u8, dx_get_limit(*__slate_slot_23), dx_node_limit(dir));
break '__join_3;
}
ret_err = (*frame).bh as *mut dx_frame;
(*frame).bh = core::ptr::null_mut::<buffer_head>();
break '__join_3;
}
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 902u32, c"dx entry: tree cycle block %u points back to block %u".as_ptr() as *const u8, (*__slate_slot_14)[*__slate_slot_26 as usize], *__slate_slot_15);
break '__join_3;
}
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 873u32, c"dx entry: count %u beyond limit %u".as_ptr() as *const u8, *__slate_slot_28, dx_get_limit(*__slate_slot_23));
}
}
}
}
}
while frame >= frame_in {
brelse((*frame).bh);
let __v1: *mut dx_frame = frame;
frame = __v1.wrapping_offset(-(1 as isize));
}
if ret_err == (ERR_PTR(4094i32.wrapping_neg() as i64) as *mut dx_frame) {
__ext4_warning_inode(dir as *const inode, c"dx_probe".as_ptr() as *const u8, 934u32, c"Corrupt directory, running e2fsck is recommended".as_ptr() as *const u8);
}
return ret_err;
}
}
}

fn dx_release(frames: *mut dx_frame) {
let mut i: i32 = 0;
let mut indirect_levels: u32 = 0;
if unsafe { (*frames.wrapping_offset(0 as isize)).bh }.is_null() {
return;
}
let info: *mut dx_root_info = unsafe { core::ptr::addr_of_mut!((*((*(*frames.wrapping_offset(0 as isize)).bh).b_data as *mut dx_root)).info) };
// save local copy, "info" may be freed after brelse()
indirect_levels = (unsafe { (*info).indirect_levels }) as u32;
while (i as u32) <= indirect_levels {
if unsafe { (*frames.wrapping_offset(i as isize)).bh }.is_null() {
break;
}
brelse(unsafe { (*frames.wrapping_offset(i as isize)).bh });
unsafe {
(*frames.wrapping_offset(i as isize)).bh = core::ptr::null_mut::<buffer_head>();
}
i = i.wrapping_add(1);
}
}

/// This function increments the frame pointer to search the next leaf
/// block, and reads in the necessary intervening nodes if the search
/// should be necessary.  Whether or not the search is necessary is
/// controlled by the hash parameter.  If the hash value is even, then
/// the search is only continued if the next block starts with that
/// hash value.  This is used if we are searching for a specific file.
///
/// If the hash value is HASH_NB_ALWAYS, then always go to the next block.
///
/// This function returns 1 if the caller should continue to search,
/// or 0 if it should not.  If there is an error reading one of the
/// index blocks, it will a negative error code.
///
/// If start_hash is non-null, it will be filled in with the starting
/// hash of the next page.
fn ext4_htree_next_block(dir: *mut inode, hash: u32, frame: *mut dx_frame, frames: *mut dx_frame, start_hash: *mut u32) -> i32 {
let mut p: *mut dx_frame = unsafe { core::mem::zeroed() };
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut num_frames: i32 = 0;
let mut bhash: u32 = 0;
p = frame;
// Find the next leaf page by incrementing the frame pointer.
// If we run out of entries in the interior node, loop around and
// increment pointer in the parent node.  When we break out of
// this loop, num_frames indicates the number of interior
// nodes need to be read.
while true {
let __v1: *mut dx_entry = unsafe { (*p).at };
let __v2: *mut dx_entry = __v1.wrapping_offset(1 as isize);
unsafe {
(*p).at = __v2;
}
if __v2 < unsafe { (*p).entries }.wrapping_offset(dx_get_count(unsafe { (*p).entries }) as isize) {
break;
}
if p == frames {
return 0;
}
num_frames = num_frames.wrapping_add(1);
let __v6: *mut dx_frame = p;
p = __v6.wrapping_offset(-(1 as isize));
}
// If the hash is 1, then continue only if the next page has a
// continuation hash of any value.  This is used for readdir
// handling.  Otherwise, check to see if the hash matches the
// desired continuation hash.  If it doesn't, return since
// there's no point to read in the successive index pages.
bhash = dx_get_hash(unsafe { (*p).at });
if !start_hash.is_null() {
unsafe {
*start_hash = bhash;
}
}
if hash & 1 == 0 {
if bhash & 4294967294u32 != hash {
return 0;
}
}
// If the hash is HASH_NB_ALWAYS, we always go to the next
// block so no check is necessary
loop {
let __v11: i32 = num_frames;
num_frames = __v11.wrapping_sub(1);
if __v11 == 0 {
break;
}
bh = ext4_read_dirblock__7!(dir, dx_get_block(unsafe { (*p).at }), 1i32);
if IS_ERR(bh as *const ()) {
return PTR_ERR(bh as *const ()) as i32;
}
let __v8: *mut dx_frame = p;
p = __v8.wrapping_offset(1 as isize);
brelse(unsafe { (*p).bh });
unsafe {
(*p).bh = bh;
}
let __v10: *mut dx_entry = unsafe { core::ptr::addr_of_mut!((*((*bh).b_data as *mut dx_node)).entries) as *mut dx_entry };
unsafe {
(*p).entries = __v10;
}
unsafe {
(*p).at = __v10;
}
}
1
}

/// This function fills a red-black tree with information from a
/// directory block.  It returns the number directory entries loaded
/// into the tree.  If there is an error it is returned in err.
fn htree_dirblock_to_tree(dir_file: *mut file, dir: *mut inode, block: u32, hinfo: *mut dx_hash_info, start_hash: u32, start_minor_hash: u32) -> i32 {
let mut de_name: core::mem::MaybeUninit<fscrypt_str> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut fscrypt_str = core::ptr::addr_of_mut!(de_name) as *mut fscrypt_str;
let mut save_len: i32 = unsafe { core::mem::zeroed() };
let mut csum: i32 = unsafe { core::mem::zeroed() };
let mut tmp_str: core::mem::MaybeUninit<fscrypt_str> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut fscrypt_str = core::ptr::addr_of_mut!(tmp_str) as *mut fscrypt_str;
let mut fname_crypto_str: core::mem::MaybeUninit<fscrypt_str> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut fscrypt_str = core::ptr::addr_of_mut!(fname_crypto_str) as *mut fscrypt_str;
let mut count: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut top: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(top) as *mut *mut ext4_dir_entry_2;
let mut de: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de) as *mut *mut ext4_dir_entry_2;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
unsafe {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
err = 0;
count = 0;
core::ptr::write(__slate_slot_6, {
let mut __t1: fscrypt_str = core::mem::zeroed();
__t1.name = core::ptr::null_mut::<u8>();
__t1.len = 0u32;
__t1
});
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
csum = ext4_has_feature_metadata_csum((*dir).i_sb) as i32;
*__slate_slot_11 = ext4_read_dirblock__10!(dir, block, 3i32);
if IS_ERR(*__slate_slot_11 as *const ()) {
return PTR_ERR(*__slate_slot_11 as *const ()) as i32;
} else {
*__slate_slot_10 = (*(*__slate_slot_11)).b_data as *mut ext4_dir_entry_2;
// csum entries are not larger in the casefolded encrypted case
*__slate_slot_9 = (*__slate_slot_10 as *mut u8).wrapping_offset((*(*dir).i_sb).s_blocksize as isize).wrapping_offset(-(ext4_dir_rec_len(0u8, (if csum != 0 { core::ptr::null_mut::<inode>() } else { dir }) as *const inode) as isize)) as *mut ext4_dir_entry_2;
// Check if the directory is encrypted
if IS_ENCRYPTED!(dir) != 0 {
err = fscrypt_prepare_readdir(dir);
if err < 0 {
brelse(*__slate_slot_11);
return err;
} else {
err = fscrypt_fname_alloc_buffer(255u32, &(*core::ptr::addr_of_mut!(*__slate_slot_6)));
if err < 0 {
brelse(*__slate_slot_11);
return err;
}
}
}
'__join_0: {
'__join_19: {
'__join_4: {
'__join_8: {
'__loop_1: loop {
if *__slate_slot_10 < *__slate_slot_9 {
if ext4_check_dir_entry__9!(dir, *__slate_slot_10, *__slate_slot_11, unsafe { (*(*__slate_slot_11)).b_data }, unsafe { (*(*__slate_slot_11)).b_size }, EXT4_LBLK_TO_B!(dir, block).wrapping_add((*__slate_slot_10 as *mut u8).offset_from((*(*__slate_slot_11)).b_data as *mut u8) as i64)) != 0 {
break '__join_19;
} else {
if ext4_hash_in_dirent(dir as *const inode) {
if (*(*__slate_slot_10)).name_len != 0 && (*(*__slate_slot_10)).inode != 0 {
(*hinfo).hash = EXT4_DIRENT_HASH!(*__slate_slot_10);
(*hinfo).minor_hash = EXT4_DIRENT_MINOR_HASH!(*__slate_slot_10);
} else {
(*hinfo).hash = 0;
(*hinfo).minor_hash = 0;
}
} else {
err = ext4fs_dirhash(dir as *const inode, ((*(*__slate_slot_10)).name.as_mut_ptr() as *mut u8) as *const u8, (*(*__slate_slot_10)).name_len as i32, hinfo);
if err < 0 {
break '__loop_1;
}
}
if !((*hinfo).hash < start_hash || (*hinfo).hash == start_hash && (*hinfo).minor_hash < start_minor_hash) {
if (*(*__slate_slot_10)).inode != 0 {
if IS_ENCRYPTED!(dir) == 0 {
(*__slate_slot_5).name = ((*(*__slate_slot_10)).name.as_mut_ptr() as *mut u8) as *mut u8;
(*__slate_slot_5).len = (*(*__slate_slot_10)).name_len as u32;
err = ext4_htree_store_dirent(dir_file, (*hinfo).hash, (*hinfo).minor_hash, *__slate_slot_10, core::ptr::addr_of_mut!(*__slate_slot_5));
} else {
save_len = (*__slate_slot_6).len as i32;
core::ptr::write(__slate_slot_2, {
let mut __t0: fscrypt_str = core::mem::zeroed();
__t0.name = ((*(*__slate_slot_10)).name.as_mut_ptr() as *mut u8) as *mut u8;
__t0.len = (*(*__slate_slot_10)).name_len as u32;
__t0
});
// Directory is encrypted
err = fscrypt_fname_disk_to_usr(dir as *const inode, (*hinfo).hash, (*hinfo).minor_hash, &(*(core::ptr::addr_of_mut!(*__slate_slot_2) as *const fscrypt_str)), &(*core::ptr::addr_of_mut!(*__slate_slot_6)));
if err != 0 {
break '__join_8;
} else {
err = ext4_htree_store_dirent(dir_file, (*hinfo).hash, (*hinfo).minor_hash, *__slate_slot_10, core::ptr::addr_of_mut!(*__slate_slot_6));
(*__slate_slot_6).len = save_len as u32;
}
}
if err != 0 {
break '__join_4;
} else {
count = count.wrapping_add(1);
}
}
}
*__slate_slot_10 = ext4_next_entry(*__slate_slot_10, (*(*dir).i_sb).s_blocksize);
}
} else {
break '__join_0;
}
}
count = err;
break '__join_0;
}
count = err;
break '__join_0;
}
count = err;
break '__join_0;
}
// silently ignore the rest of the block
}
brelse(*__slate_slot_11);
fscrypt_fname_free_buffer(&(*core::ptr::addr_of_mut!(*__slate_slot_6)));
return count;
}
}
}

/// This function fills a red-black tree with information from a
/// directory.  We start scanning the directory in hash order, starting
/// at start_hash and start_minor_hash.
///
/// This function returns the number of entries inserted into the tree,
/// or a negative error code.
#[unsafe(no_mangle)]
extern "C" fn ext4_htree_fill_tree(dir_file: *mut file, start_hash: u32, start_minor_hash: u32, next_hash: *mut u32) -> i32 {
let mut has_inline_data: i32 = unsafe { core::mem::zeroed() };
let mut tmp_str: core::mem::MaybeUninit<fscrypt_str> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut fscrypt_str = core::ptr::addr_of_mut!(tmp_str) as *mut fscrypt_str;
let mut hashval: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(hashval) as *mut u32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut count: i32 = unsafe { core::mem::zeroed() };
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut dir: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut inode = core::ptr::addr_of_mut!(dir) as *mut *mut inode;
let mut frame: core::mem::MaybeUninit<*mut dx_frame> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut dx_frame = core::ptr::addr_of_mut!(frame) as *mut *mut dx_frame;
let mut frames: core::mem::MaybeUninit<__SlateAlign16<[dx_frame; 3]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut [dx_frame; 3] = core::ptr::addr_of_mut!(frames) as *mut [dx_frame; 3];
let mut de: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de) as *mut *mut ext4_dir_entry_2;
let mut hinfo: core::mem::MaybeUninit<dx_hash_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut dx_hash_info = core::ptr::addr_of_mut!(hinfo) as *mut dx_hash_info;
unsafe {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
count = 0;
let __slate_auto_bytes_6 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
*__slate_slot_16 = file_inode(dir_file as *const file);
if ext4_test_inode_flag(*__slate_slot_16, 12i32) == 0 {
if ext4_hash_in_dirent(*__slate_slot_16 as *const inode) {
(*__slate_slot_20).hash_version = 6;
} else {
(*__slate_slot_20).hash_version = (*EXT4_SB((*(*__slate_slot_16)).i_sb)).s_def_hash_version;
}
if (*__slate_slot_20).hash_version <= 2 {
let __v9: i32 = (*EXT4_SB((*(*__slate_slot_16)).i_sb)).s_hash_unsigned;
let __v8: i32 = (*__slate_slot_20).hash_version;
(*__slate_slot_20).hash_version = __v8.wrapping_add(__v9);
}
(*__slate_slot_20).seed = (*EXT4_SB((*(*__slate_slot_16)).i_sb)).s_hash_seed.as_mut_ptr() as *mut u32;
if ext4_has_inline_data(*__slate_slot_16) != 0 {
has_inline_data = 1;
count = ext4_inlinedir_to_tree(dir_file, *__slate_slot_16, 0u32, core::ptr::addr_of_mut!(*__slate_slot_20), start_hash, start_minor_hash, core::ptr::addr_of_mut!(has_inline_data));
if has_inline_data != 0 {
*next_hash = 4294967295u32;
return count;
}
}
count = htree_dirblock_to_tree(dir_file, *__slate_slot_16, 0u32, core::ptr::addr_of_mut!(*__slate_slot_20), start_hash, start_minor_hash);
*next_hash = 4294967295u32;
return count;
} else {
(*__slate_slot_20).hash = start_hash;
(*__slate_slot_20).minor_hash = 0;
*__slate_slot_17 = dx_probe(core::ptr::null_mut::<ext4_filename>(), *__slate_slot_16, core::ptr::addr_of_mut!(*__slate_slot_20), (*__slate_slot_18).as_mut_ptr() as *mut dx_frame);
if IS_ERR(*__slate_slot_17 as *const ()) {
return PTR_ERR(*__slate_slot_17 as *const ()) as i32;
} else {
'__join_0: {
// Add '.' and '..' from the htree header
if start_hash == 0 && start_minor_hash == 0 {
*__slate_slot_19 = (*(*__slate_slot_18)[0].bh).b_data as *mut ext4_dir_entry_2;
(*__slate_slot_10).name = ((*(*__slate_slot_19)).name.as_mut_ptr() as *mut u8) as *mut u8;
(*__slate_slot_10).len = (*(*__slate_slot_19)).name_len as u32;
*__slate_slot_12 = ext4_htree_store_dirent(dir_file, 0u32, 0u32, *__slate_slot_19, core::ptr::addr_of_mut!(*__slate_slot_10));
if *__slate_slot_12 != 0 {
break '__join_0;
} else {
count = count.wrapping_add(1);
}
}
if start_hash < 2 || start_hash == 2 && start_minor_hash == 0 {
*__slate_slot_19 = (*(*__slate_slot_18)[0].bh).b_data as *mut ext4_dir_entry_2;
*__slate_slot_19 = ext4_next_entry(*__slate_slot_19, (*(*(*__slate_slot_16)).i_sb).s_blocksize);
(*__slate_slot_10).name = ((*(*__slate_slot_19)).name.as_mut_ptr() as *mut u8) as *mut u8;
(*__slate_slot_10).len = (*(*__slate_slot_19)).name_len as u32;
*__slate_slot_12 = ext4_htree_store_dirent(dir_file, 2u32, 0u32, *__slate_slot_19, core::ptr::addr_of_mut!(*__slate_slot_10));
if *__slate_slot_12 != 0 {
break '__join_0;
} else {
count = count.wrapping_add(1);
}
}
'__join_8: {
'__join_6: {
'__join_4: {
while true {
if fatal_signal_pending(get_current()) != 0 {
break '__join_8;
} else {
cond_resched__12!();
*__slate_slot_15 = dx_get_block((*(*__slate_slot_17)).at);
*__slate_slot_13 = htree_dirblock_to_tree(dir_file, *__slate_slot_16, *__slate_slot_15, core::ptr::addr_of_mut!(*__slate_slot_20), start_hash, start_minor_hash);
if *__slate_slot_13 < 0 {
break '__join_6;
} else {
count = count.wrapping_add(*__slate_slot_13);
*__slate_slot_11 = 4294967295u32;
*__slate_slot_13 = ext4_htree_next_block(*__slate_slot_16, 1u32, *__slate_slot_17, (*__slate_slot_18).as_mut_ptr() as *mut dx_frame, core::ptr::addr_of_mut!(*__slate_slot_11));
*next_hash = *__slate_slot_11;
if *__slate_slot_13 < 0 {
break '__join_4;
} else {
// Stop if:  (a) there are no more entries, or
// (b) we have inserted at least one entry and the
// next hash value is not a continuation
if *__slate_slot_13 == 0 || count != 0 && *__slate_slot_11 & 1 == 0 {
break;
}
}
}
}
}
dx_release((*__slate_slot_18).as_mut_ptr() as *mut dx_frame);
return count;
}
*__slate_slot_12 = *__slate_slot_13;
break '__join_0;
}
*__slate_slot_12 = *__slate_slot_13;
break '__join_0;
}
*__slate_slot_12 = 512i32.wrapping_neg();
}
dx_release((*__slate_slot_18).as_mut_ptr() as *mut dx_frame);
return *__slate_slot_12;
}
}
}
}

fn search_dirblock(bh: *mut buffer_head, dir: *mut inode, fname: *mut ext4_filename, offset: u32, res_dir: *mut *mut ext4_dir_entry_2) -> i32 {
ext4_search_dir(bh, unsafe { (*bh).b_data }, (unsafe { (*(*dir).i_sb).s_blocksize }) as i32, dir, fname, offset, res_dir)
}

// Directory block splitting, compacting
/// Create map of hash values, offsets, and sizes, stored at end of block.
/// Returns number of entries mapped.
fn dx_make_map(dir: *mut inode, bh: *mut buffer_head, hinfo: *mut dx_hash_info, mut map_tail: *mut dx_map_entry) -> i32 {
let mut count: i32 = 0;
let mut de: *mut ext4_dir_entry_2 = (unsafe { (*bh).b_data }) as *mut ext4_dir_entry_2;
let mut buflen: u32 = (unsafe { (*bh).b_size }) as u32;
let base: *mut u8 = unsafe { (*bh).b_data };
let mut h: dx_hash_info = unsafe { *hinfo };
let blocksize: i32 = EXT4_BLOCK_SIZE!(unsafe { (*dir).i_sb }) as i32;
if ext4_has_feature_metadata_csum(unsafe { (*dir).i_sb }) {
buflen = (buflen as u64).wrapping_sub(core::mem::size_of::<ext4_dir_entry_tail>() as u64) as u32;
}
while (de as *mut u8) < base.wrapping_offset(buflen as isize) {
if ext4_check_dir_entry__15!(dir, de, bh, base, buflen, (unsafe { (de as *mut u8).offset_from(base as *mut u8) }) as i64) != 0 {
return 117i32.wrapping_neg();
}
if (unsafe { (*de).name_len }) != 0 && (unsafe { (*de).inode }) != 0 {
if ext4_hash_in_dirent(dir as *const inode) {
h.hash = EXT4_DIRENT_HASH!(de);
} else {
let err: i32 = unsafe { ext4fs_dirhash(dir as *const inode, ((*de).name.as_mut_ptr() as *mut u8) as *const u8, (*de).name_len as i32, core::ptr::addr_of_mut!(h)) };
if err < 0 {
return err;
}
}
let __v2: *mut dx_map_entry = map_tail;
map_tail = __v2.wrapping_offset(-(1 as isize));
unsafe {
(*map_tail).hash = h.hash;
}
unsafe {
(*map_tail).offs = (((de as *mut u8).offset_from(base as *mut u8) as i64) >> 2i32) as u16;
}
unsafe {
(*map_tail).size = ext4_rec_len_from_disk((*de).rec_len, blocksize as u32) as u16;
}
count = count.wrapping_add(1);
cond_resched__3!();
}
de = ext4_next_entry(de, blocksize as u64);
}
count
}

/// Sort map by hash value
fn dx_sort_map(map: *mut dx_map_entry, mut count: u32) {
let mut p: *mut dx_map_entry = unsafe { core::mem::zeroed() };
let mut q: *mut dx_map_entry = unsafe { core::mem::zeroed() };
let top: *mut dx_map_entry = map.wrapping_offset(count as isize).wrapping_offset(-(1 as isize));
let mut more: i32 = 0;
// Combsort until bubble sort doesn't suck
while count > 2 {
count = count.wrapping_mul(10) / 13;
if count.wrapping_sub(9) < 2 {
count = 11;
}
// 9, 10 -> 11
p = top;
q = p.wrapping_offset(-(count as isize));
while q >= map {
if (unsafe { (*p).hash }) < unsafe { (*q).hash } {
swap__2!(*p, *q);
}
let __v3: *mut dx_map_entry = p;
p = __v3.wrapping_offset(-(1 as isize));
let __v5: *mut dx_map_entry = q;
q = __v5.wrapping_offset(-(1 as isize));
}
}
// Garden variety bubble sort
loop {
more = 0;
q = top;
loop {
let __v10: *mut dx_map_entry = q;
q = __v10.wrapping_offset(-(1 as isize));
if !(__v10 > map) {
break;
}
if !((unsafe { (*q.wrapping_offset(1 as isize)).hash }) >= unsafe { (*q.wrapping_offset(0 as isize)).hash }) {
swap__2!(*q.wrapping_offset(1 as isize), *q);
more = 1;
}
}
if more == 0 {
break;
}
}
}

fn dx_insert_block(frame: *mut dx_frame, hash: u32, block: u32) {
let entries: *mut dx_entry = unsafe { (*frame).entries };
let old: *mut dx_entry = unsafe { (*frame).at };
let new: *mut dx_entry = old.wrapping_offset(1 as isize);
let count: i32 = dx_get_count(entries) as i32;
loop {
if unlikely!((count as u32) >= dx_get_limit(entries)) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x010Assertion failure in %s() at %s:%d: '%s'\n".as_ptr() as *const u8, c"dx_insert_block".as_ptr() as *mut u8, c"/home/takashi/c-corpus/linux/fs/ext4/namei.c".as_ptr() as *mut u8, 1353i32, c"count < dx_get_limit(entries)".as_ptr() as *mut u8) };
loop {
unsafe {
core::arch::asm!("1504: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1504b - ., 3; .popsection\n# {0}", const 1504 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dx_insert_block_2, sym __slate_asm_string_dx_insert_block_3, const 1353 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!(!(old < entries.wrapping_offset(count as isize))) != 0 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x010Assertion failure in %s() at %s:%d: '%s'\n".as_ptr() as *const u8, c"dx_insert_block".as_ptr() as *mut u8, c"/home/takashi/c-corpus/linux/fs/ext4/namei.c".as_ptr() as *mut u8, 1354i32, c"old < entries + count".as_ptr() as *mut u8) };
loop {
unsafe {
core::arch::asm!("1505: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1505b - ., 3; .popsection\n# {0}", const 1505 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dx_insert_block_9, sym __slate_asm_string_dx_insert_block_10, const 1354 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { memmove(new.wrapping_offset(1 as isize) as *mut (), new as *const (), ((entries.wrapping_offset(count as isize) as *mut u8).offset_from(new as *mut u8) as i64) as u64) };
dx_set_hash(new, hash);
dx_set_block(new, block);
dx_set_count(entries, count.wrapping_add(1) as u32);
}

/// Test whether a directory entry matches the filename being searched for.
///
/// Return: %true if the directory entry matches, otherwise %false.
fn ext4_match(parent: *mut inode, fname: *const ext4_filename, de: *mut ext4_dir_entry_2) -> bool {
let mut f: fscrypt_name = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(f) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 56usize);
}
if (unsafe { (*de).inode }) == 0 {
return false;
}
f.usr_fname = unsafe { (*fname).usr_fname };
f.disk_name = unsafe { (*fname).disk_name };
fscrypt_match_name(unsafe { &(*(core::ptr::addr_of_mut!(f) as *const fscrypt_name)) }, (unsafe { (*de).name.as_mut_ptr() as *mut u8 }) as *const u8, (unsafe { (*de).name_len }) as u32)
}

/// Returns 0 if not found, -EFSCORRUPTED on failure, and 1 on success
#[unsafe(no_mangle)]
extern "C" fn ext4_search_dir(bh: *mut buffer_head, search_buf: *mut u8, buf_size: i32, dir: *mut inode, fname: *mut ext4_filename, mut offset: u32, res_dir: *mut *mut ext4_dir_entry_2) -> i32 {
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut de_len: i32 = 0;
de = search_buf as *mut ext4_dir_entry_2;
let dlimit: *mut u8 = search_buf.wrapping_offset(buf_size as isize);
while (de as *mut u8) < dlimit.wrapping_offset(-((core::mem::size_of::<ext4_dir_entry_2>() as u64).wrapping_sub(255) as isize)) {
// this code is executed quadratically often
//
// do minimal checking `by hand'
let __v0: bool = unsafe { (*de).name.as_mut_ptr() as *mut u8 }.wrapping_offset(((unsafe { (*de).name_len }) as i32) as isize) <= dlimit && (!ext4_hash_in_dirent(dir as *const inode) || (de as *mut u8).wrapping_offset(ext4_dir_rec_len(unsafe { (*de).name_len }, dir as *const inode) as isize) <= dlimit);
let __v2: bool = __v0 && ext4_match(dir, fname as *const ext4_filename, de);
if __v2 {
// found a match - just to be sure, do
// a full check
if ext4_check_dir_entry__16!(dir, de, bh, search_buf, buf_size, offset) != 0 {
return 117i32.wrapping_neg();
}
unsafe {
*res_dir = de;
}
return 1;
}
// prevent looping on a bad block
de_len = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*(*dir).i_sb).s_blocksize }) as u32) as i32;
if de_len <= 0 {
return 117i32.wrapping_neg();
}
offset = offset.wrapping_add(de_len as u32);
de = (de as *mut u8).wrapping_offset(de_len as isize) as *mut ext4_dir_entry_2;
}
0
}

fn is_dx_internal_node(dir: *mut inode, block: u32, de: *mut ext4_dir_entry) -> i32 {
let sb: *mut super_block = unsafe { (*dir).i_sb };
if !is_dx!(dir) {
return 0;
}
if block == 0 {
return 1;
}
let __v1: bool = (unsafe { (*de).inode }) == 0 && (ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*sb).s_blocksize }) as u32) as u64) == unsafe { (*sb).s_blocksize };
if __v1 {
return 1;
}
0
}

// 	__ext4_find_entry()
//
// finds an entry in the specified directory with the wanted name. It
// returns the cache buffer in which the entry was found, and the entry
// itself (as a parameter - res_dir). It does NOT read the inode of the
// entry - you'll have to do that yourself if you want to.
//
// The returned buffer_head has ->b_count elevated.  The caller is expected
// to brelse() it when appropriate.
fn __ext4_find_entry(dir: *mut inode, fname: *mut ext4_filename, res_dir: *mut *mut ext4_dir_entry_2, inlined: *mut i32) -> *mut buffer_head {
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__1507: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1506: u64 = unsafe { core::mem::zeroed() };
// On success, or if the error was file not found,
// return.  Otherwise, fall back to doing a search the
// old fashioned way.
let mut __v12: bool = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut has_inline_data: i32 = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut namelen: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(namelen) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut nblocks: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(nblocks) as *mut u32;
// Current index into readahead
// buffer
let mut ra_ptr: u64 = unsafe { core::mem::zeroed() };
// Number of bh's in the readahead
// buffer, bh_use[]
let mut ra_max: u64 = unsafe { core::mem::zeroed() };
let mut name: *const u8 = unsafe { core::mem::zeroed() };
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(start) as *mut u32;
let mut ret: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut bh_use: core::mem::MaybeUninit<__SlateAlign16<[*mut buffer_head; 8]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut [*mut buffer_head; 8] = core::ptr::addr_of_mut!(bh_use) as *mut [*mut buffer_head; 8];
let mut sb: core::mem::MaybeUninit<*mut super_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut super_block = core::ptr::addr_of_mut!(sb) as *mut *mut super_block;
unsafe {
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 64usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
ret = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
name = (*(*fname).usr_fname).name;
ra_max = 0;
ra_ptr = 0;
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*res_dir = core::ptr::null_mut::<ext4_dir_entry_2>();
*__slate_slot_27 = (*dir).i_sb;
*__slate_slot_16 = (*(*fname).usr_fname).__slate_anon_0.__slate_anon_0.len as i32;
if *__slate_slot_16 > 255 {
return core::ptr::null_mut::<buffer_head>();
} else {
'__join_1: {
if ext4_has_inline_data(dir) != 0 {
has_inline_data = 1;
ret = ext4_find_inline_entry(dir, fname, res_dir, core::ptr::addr_of_mut!(has_inline_data));
if !inlined.is_null() {
*inlined = has_inline_data;
}
if has_inline_data != 0 {
__v13 = true;
} else {
__v13 = IS_ERR(ret as *const ());
}
if __v13 {
break '__join_1;
}
}
if *__slate_slot_16 <= 2 && (*name.wrapping_offset(0 as isize) as i32) == 46 && ((*name.wrapping_offset(1 as isize) as i32) == 46 || (*name.wrapping_offset(1 as isize) as i32) == 0) {
// "." or ".." will only be in the first block
// NFS may look up ".."; "." should be handled by the VFS
*__slate_slot_23 = 0;
*__slate_slot_22 = 0;
*__slate_slot_18 = 1;
} else {
if is_dx!(dir) {
ret = ext4_dx_find_entry(dir, fname, res_dir);
if IS_ERR(ret as *const ()) {
__v12 = PTR_ERR(ret as *const ()) == (4094i32.wrapping_neg() as i64);
} else {
__v12 = false;
}
if __v12 {
ret = core::ptr::null_mut::<buffer_head>();
} else {
break '__join_1;
}
}
*__slate_slot_18 = ((*dir).i_size >> (EXT4_BLOCK_SIZE_BITS!(*__slate_slot_27) as i32)) as u32;
if *__slate_slot_18 == 0 {
ret = core::ptr::null_mut::<buffer_head>();
break '__join_1;
} else {
*__slate_slot_23 = (*EXT4_I!(dir)).i_dir_start_lookup;
if *__slate_slot_23 >= *__slate_slot_18 {
*__slate_slot_23 = 0;
}
*__slate_slot_22 = *__slate_slot_23;
}
}
'__join_23: {
'__join_20: {
'__join_12: {
'__join_8: {
loop {
// We deal with the read-ahead logic here.
cond_resched__10!();
if ra_ptr >= ra_max {
// Refill the readahead buffer
ra_ptr = 0;
if *__slate_slot_22 < *__slate_slot_23 {
ra_max = (*__slate_slot_23).wrapping_sub(*__slate_slot_22) as u64;
} else {
ra_max = (*__slate_slot_18).wrapping_sub(*__slate_slot_22) as u64;
}
__UNIQUE_ID_x__1506 = ra_max;
__UNIQUE_ID_y__1507 = ARRAY_SIZE__28!();
while false {
}
ra_max = if __UNIQUE_ID_x__1506 < __UNIQUE_ID_y__1507 { __UNIQUE_ID_x__1506 } else { __UNIQUE_ID_y__1507 };
*__slate_slot_15 = ext4_bread_batch(dir, *__slate_slot_22, ra_max as i32, false, (*__slate_slot_26).as_mut_ptr() as *mut *mut buffer_head); // wait
if *__slate_slot_15 != 0 {
break '__join_23;
}
}
let __v8: u64 = ra_ptr;
ra_ptr = __v8.wrapping_add(1);
let __v6: *mut buffer_head = (*__slate_slot_26)[__v8 as usize];
*__slate_slot_25 = __v6;
if !__v6.is_null() {
wait_on_buffer(*__slate_slot_25);
if buffer_uptodate(*__slate_slot_25 as *const buffer_head) == 0 {
break '__join_20;
} else {
if buffer_verified(*__slate_slot_25 as *const buffer_head) == 0 {
__v5 = is_dx_internal_node(dir, *__slate_slot_22, (*(*__slate_slot_25)).b_data as *mut ext4_dir_entry) == 0;
} else {
__v5 = false;
}
if __v5 {
__v4 = ext4_dirblock_csum_verify(dir, *__slate_slot_25) == 0;
} else {
__v4 = false;
}
if __v4 {
break '__join_12;
} else {
set_buffer_verified(*__slate_slot_25);
*__slate_slot_17 = search_dirblock(*__slate_slot_25, dir, fname, EXT4_LBLK_TO_B!(dir, *__slate_slot_22) as u32, res_dir);
if *__slate_slot_17 == 1 {
break '__join_8;
} else {
brelse(*__slate_slot_25);
if *__slate_slot_17 < 0 {
break;
}
}
}
}
}
let __v3: u32 = *__slate_slot_22;
let __v2: u32 = __v3.wrapping_add(1);
*__slate_slot_22 = __v2;
if __v2 >= *__slate_slot_18 {
*__slate_slot_22 = 0;
}
if *__slate_slot_22 == *__slate_slot_23 {
// If the directory has grown while we were searching, then
// search the last part of the directory before giving up.
*__slate_slot_22 = *__slate_slot_18;
*__slate_slot_18 = ((*dir).i_size >> (EXT4_BLOCK_SIZE_BITS!(*__slate_slot_27) as i32)) as u32;
if *__slate_slot_22 < *__slate_slot_18 {
*__slate_slot_23 = 0;
} else {
break '__join_1;
}
}
}
ret = ERR_PTR(*__slate_slot_17 as i64) as *mut buffer_head;
break '__join_1;
}
(*EXT4_I!(dir)).i_dir_start_lookup = *__slate_slot_22;
ret = *__slate_slot_25;
break '__join_1;
}
__ext4_error_inode(dir, c"__ext4_find_entry".as_ptr() as *const u8, 1628u32, 0u64, 74i32, c"checksumming directory block %lu".as_ptr() as *const u8, *__slate_slot_22 as u64);
brelse(*__slate_slot_25);
ret = ERR_PTR(74i32.wrapping_neg() as i64) as *mut buffer_head;
break '__join_1;
}
__ext4_error_inode(dir, c"__ext4_find_entry".as_ptr() as *const u8, 1617u32, 0u64, 5i32, c"reading directory lblock %lu".as_ptr() as *const u8, *__slate_slot_22 as u64);
brelse(*__slate_slot_25);
ret = ERR_PTR(5i32.wrapping_neg() as i64) as *mut buffer_head;
break '__join_1;
}
ret = ERR_PTR(*__slate_slot_15 as i64) as *mut buffer_head;
ra_max = 0;
}
while ra_ptr < ra_max {
brelse((*__slate_slot_26)[ra_ptr as usize]);
ra_ptr = ra_ptr.wrapping_add(1);
}
// Clean up the read-ahead blocks
return ret;
}
}
}

fn ext4_find_entry(dir: *mut inode, d_name: *const qstr, res_dir: *mut *mut ext4_dir_entry_2, inlined: *mut i32) -> *mut buffer_head {
let mut err: i32 = 0;
let mut fname: ext4_filename = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fname) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
err = ext4_fname_setup_filename(dir, d_name, 1i32, unsafe { &mut (*core::ptr::addr_of_mut!(fname)) });
if err == 2i32.wrapping_neg() {
return core::ptr::null_mut::<buffer_head>();
}
if err != 0 {
return ERR_PTR(err as i64) as *mut buffer_head;
}
let bh: *mut buffer_head = __ext4_find_entry(dir, core::ptr::addr_of_mut!(fname), res_dir, inlined);
ext4_fname_free_filename(unsafe { &(*core::ptr::addr_of_mut!(fname)) });
bh
}

fn ext4_lookup_entry(dir: *mut inode, dentry: *mut dentry, res_dir: *mut *mut ext4_dir_entry_2) -> *mut buffer_head {
let mut err: i32 = 0;
let mut fname: ext4_filename = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fname) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
err = ext4_fname_prepare_lookup(dir, dentry, unsafe { &mut (*core::ptr::addr_of_mut!(fname)) });
if err == 2i32.wrapping_neg() {
return core::ptr::null_mut::<buffer_head>();
}
if err != 0 {
return ERR_PTR(err as i64) as *mut buffer_head;
}
let bh: *mut buffer_head = __ext4_find_entry(dir, core::ptr::addr_of_mut!(fname), res_dir, core::ptr::null_mut::<i32>());
ext4_fname_free_filename(unsafe { &(*core::ptr::addr_of_mut!(fname)) });
bh
}

fn ext4_dx_find_entry(dir: *mut inode, fname: *mut ext4_filename, res_dir: *mut *mut ext4_dir_entry_2) -> *mut buffer_head {
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut frame: core::mem::MaybeUninit<*mut dx_frame> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut dx_frame = core::ptr::addr_of_mut!(frame) as *mut *mut dx_frame;
let mut frames: core::mem::MaybeUninit<__SlateAlign16<[dx_frame; 3]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut [dx_frame; 3] = core::ptr::addr_of_mut!(frames) as *mut [dx_frame; 3];
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_3 = dx_probe(fname, dir, core::ptr::null_mut::<dx_hash_info>(), (*__slate_slot_4).as_mut_ptr() as *mut dx_frame);
if IS_ERR(*__slate_slot_3 as *const ()) {
return ERR_CAST(*__slate_slot_3 as *const ()) as *mut buffer_head;
} else {
'__join_0: {
'__join_1: {
'__join_6: {
'__join_4: {
loop {
*__slate_slot_1 = dx_get_block((*(*__slate_slot_3)).at);
*__slate_slot_2 = ext4_read_dirblock__8!(dir, *__slate_slot_1, 3i32);
if IS_ERR(*__slate_slot_2 as *const ()) {
break '__join_1;
} else {
*__slate_slot_0 = search_dirblock(*__slate_slot_2, dir, fname, EXT4_LBLK_TO_B!(dir, *__slate_slot_1) as u32, res_dir);
if *__slate_slot_0 == 1 {
break '__join_0;
} else {
brelse(*__slate_slot_2);
if *__slate_slot_0 < 0 {
break '__join_6;
} else {
// Check to see if we should continue to search
*__slate_slot_0 = ext4_htree_next_block(dir, (*fname).hinfo.hash, *__slate_slot_3, (*__slate_slot_4).as_mut_ptr() as *mut dx_frame, core::ptr::null_mut::<u32>());
if *__slate_slot_0 < 0 {
break '__join_4;
} else {
if *__slate_slot_0 != 1 {
break;
}
}
}
}
}
}
*__slate_slot_2 = core::ptr::null_mut::<buffer_head>();
break '__join_1;
}
__ext4_warning_inode(dir as *const inode, c"ext4_dx_find_entry".as_ptr() as *const u8, 1748u32, c"error %d reading directory index block".as_ptr() as *const u8, *__slate_slot_0);
*__slate_slot_2 = ERR_PTR(*__slate_slot_0 as i64) as *mut buffer_head;
break '__join_1;
}
*__slate_slot_2 = ERR_PTR(4094i32.wrapping_neg() as i64) as *mut buffer_head;
}
}
dx_release((*__slate_slot_4).as_mut_ptr() as *mut dx_frame);
return *__slate_slot_2;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_lookup")]
extern "C" fn ext4_lookup(dir: *mut inode, dentry: *mut dentry, flags: u32) -> *mut dentry {
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
let mut de: *mut ext4_dir_entry_2 = core::ptr::null_mut::<ext4_dir_entry_2>();
if (unsafe { (*dentry).__slate_anon_4.d_name.__slate_anon_0.__slate_anon_0.len }) > 255 {
return ERR_PTR(36i32.wrapping_neg() as i64) as *mut dentry;
}
let bh: *mut buffer_head = ext4_lookup_entry(dir, dentry, core::ptr::addr_of_mut!(de));
if IS_ERR(bh as *const ()) {
return ERR_CAST(bh as *const ()) as *mut dentry;
}
inode = core::ptr::null_mut::<inode>();
if !bh.is_null() {
let ino: u32 = unsafe { (*de).inode };
brelse(bh);
if ext4_valid_inum(unsafe { (*dir).i_sb }, ino as u64) == 0 {
unsafe { __ext4_error_inode(dir, c"ext4_lookup".as_ptr() as *const u8, 1779u32, 0u64, 0i32, c"bad inode number: %u".as_ptr() as *const u8, ino) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut dentry;
}
if unlikely!((ino as u64) == unsafe { (*dir).i_ino }) != 0 {
unsafe { __ext4_error_inode(dir, c"ext4_lookup".as_ptr() as *const u8, 1784u32, 0u64, 0i32, c"'%pd' linked to parent dir".as_ptr() as *const u8, dentry) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut dentry;
}
inode = ext4_iget__19!(unsafe { (*dir).i_sb }, ino, 0i32);
if inode == (ERR_PTR(116i32.wrapping_neg() as i64) as *mut inode) {
unsafe { __ext4_error_inode(dir, c"ext4_lookup".as_ptr() as *const u8, 1791u32, 0u64, 0i32, c"deleted inode referenced: %u".as_ptr() as *const u8, ino) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut dentry;
}
let __v0: bool = !IS_ERR(inode as *const ()) && IS_ENCRYPTED!(dir) != 0 && (S_ISDIR!(unsafe { (*inode).i_mode }) || S_ISLNK!(unsafe { (*inode).i_mode })) && fscrypt_has_permitted_context(dir, inode) == 0;
if __v0 {
unsafe { __ext4_warning((*inode).i_sb, c"ext4_lookup".as_ptr() as *const u8, 1799u32, c"Inconsistent encryption contexts: %llu/%llu".as_ptr() as *const u8, (*dir).i_ino, (*inode).i_ino) };
unsafe { iput(inode) };
return ERR_PTR(1i32.wrapping_neg() as i64) as *mut dentry;
}
}
unsafe { d_splice_alias(inode, dentry) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.namei.ext4_get_parent")]
extern "C" fn ext4_get_parent(child: *mut dentry) -> *mut dentry {
let mut ino: u32 = 0;
let mut de: *mut ext4_dir_entry_2 = core::ptr::null_mut::<ext4_dir_entry_2>();
let bh: *mut buffer_head = ext4_find_entry(d_inode(child as *const dentry), unsafe { core::ptr::addr_of!(dotdot_name) }, core::ptr::addr_of_mut!(de), core::ptr::null_mut::<i32>());
if IS_ERR(bh as *const ()) {
return ERR_CAST(bh as *const ()) as *mut dentry;
}
if bh.is_null() {
return ERR_PTR(2i32.wrapping_neg() as i64) as *mut dentry;
}
ino = unsafe { (*de).inode };
brelse(bh);
if ext4_valid_inum(unsafe { (*child).d_sb }, ino as u64) == 0 {
unsafe { __ext4_error_inode(d_inode(child as *const dentry), c"ext4_get_parent".as_ptr() as *const u8, 1834u32, 0u64, 0i32, c"bad parent inode number: %u".as_ptr() as *const u8, ino) };
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut dentry;
}
unsafe { d_obtain_alias(ext4_iget__4!(unsafe { (*child).d_sb }, ino, 0i32)) }
}

/// Move count entries from end of map between two memory locations.
/// Returns pointer to last entry moved.
fn dx_move_dirents(dir: *mut inode, from: *mut u8, mut to: *mut u8, mut map: *mut dx_map_entry, mut count: i32, blocksize: u32) -> *mut ext4_dir_entry_2 {
let mut rec_len: u32 = 0;
loop {
let __v4: i32 = count;
count = __v4.wrapping_sub(1);
if __v4 == 0 {
break;
}
let de: *mut ext4_dir_entry_2 = from.wrapping_offset((((unsafe { (*map).offs }) as i32) << 2i32) as isize) as *mut ext4_dir_entry_2;
rec_len = ext4_dir_rec_len(unsafe { (*de).name_len }, dir as *const inode);
unsafe { memcpy(to as *mut (), de as *const (), rec_len as u64) };
unsafe {
(*(to as *mut ext4_dir_entry_2)).rec_len = ext4_rec_len_to_disk(rec_len, blocksize);
}
// wipe dir_entry excluding the rec_len field
unsafe {
(*de).inode = 0;
}
unsafe { memset(core::ptr::addr_of_mut!((*de).name_len) as *mut (), 0i32, (ext4_rec_len_from_disk((*de).rec_len, blocksize) as u64).wrapping_sub(offsetof__17!(ext4_dir_entry_2))) };
let __v0: *mut dx_map_entry = map;
map = __v0.wrapping_offset(1 as isize);
let __v2: *mut u8 = to;
to = __v2.wrapping_offset(rec_len as isize);
}
to.wrapping_offset(-(rec_len as isize)) as *mut ext4_dir_entry_2
}

/// Compact each dir entry in the range to the minimal rec_len.
/// Returns pointer to last entry in range.
fn dx_pack_dirents(dir: *mut inode, base: *mut u8, blocksize: u32) -> *mut ext4_dir_entry_2 {
let mut to: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut prev: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut de: *mut ext4_dir_entry_2 = base as *mut ext4_dir_entry_2;
let mut rec_len: u32 = 0;
let __v0: *mut ext4_dir_entry_2 = de;
to = __v0;
prev = __v0;
while (de as *mut u8) < base.wrapping_offset(blocksize as isize) {
let next: *mut ext4_dir_entry_2 = ext4_next_entry(de, blocksize as u64);
if (unsafe { (*de).inode }) != 0 && (unsafe { (*de).name_len }) != 0 {
rec_len = ext4_dir_rec_len(unsafe { (*de).name_len }, dir as *const inode);
if de > to {
unsafe { memmove(to as *mut (), de as *const (), rec_len as u64) };
}
unsafe {
(*to).rec_len = ext4_rec_len_to_disk(rec_len, blocksize);
}
prev = to;
to = (to as *mut u8).wrapping_offset(rec_len as isize) as *mut ext4_dir_entry_2;
}
de = next;
}
prev
}

/// Split a full leaf block to make room for a new dir entry.
/// Allocate a new block, and move entries so that they are approx. equally full.
/// Returns pointer to de in block into which the new entry will be inserted.
fn do_split(handle: *mut jbd2_journal_handle, dir: *mut inode, bh: *mut *mut buffer_head, frame: *mut dx_frame, hinfo: *mut dx_hash_info) -> *mut ext4_dir_entry_2 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut csum_size: i32 = unsafe { core::mem::zeroed() };
let mut de2: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de2) as *mut *mut ext4_dir_entry_2;
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut size: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(size) as *mut u32;
let mut r#move: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(r#move) as *mut u32;
let mut split: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(split) as *mut u32;
let mut data2: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut u8 = core::ptr::addr_of_mut!(data2) as *mut *mut u8;
let mut data1: *mut u8 = unsafe { core::mem::zeroed() };
let mut map: core::mem::MaybeUninit<*mut dx_map_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut dx_map_entry = core::ptr::addr_of_mut!(map) as *mut *mut dx_map_entry;
let mut hash2: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(hash2) as *mut u32;
let mut newblock: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(newblock) as *mut u32;
let mut bh2: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh2) as *mut *mut buffer_head;
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut continued: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(continued) as *mut u32;
let mut blocksize: u32 = unsafe { core::mem::zeroed() };
unsafe {
blocksize = (*(*dir).i_sb).s_blocksize as u32;
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
data1 = (*(*bh)).b_data;
let __slate_auto_bytes_6 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
de = core::ptr::null_mut::<ext4_dir_entry_2>();
let __slate_auto_bytes_10 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
csum_size = 0;
err = 0;
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ext4_has_feature_metadata_csum((*dir).i_sb) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
*__slate_slot_24 = ext4_append(handle, dir, &mut (*core::ptr::addr_of_mut!(*__slate_slot_23)));
if IS_ERR(*__slate_slot_24 as *const ()) {
brelse(*bh);
*bh = core::ptr::null_mut::<buffer_head>();
return ERR_CAST(*__slate_slot_24 as *const ()) as *mut ext4_dir_entry_2;
} else {
'__join_0: {
BUFFER_TRACE!();
err = ext4_journal_get_write_access__23!(handle, unsafe { (*dir).i_sb }, unsafe { *bh }, 1i32);
if err == 0 {
BUFFER_TRACE!();
err = ext4_journal_get_write_access__37!(handle, unsafe { (*dir).i_sb }, unsafe { (*frame).bh }, 1i32);
if err == 0 {
*__slate_slot_19 = (*(*__slate_slot_24)).b_data;
// create map in the end of data2 block
*__slate_slot_21 = (*__slate_slot_19).wrapping_offset(blocksize as isize) as *mut dx_map_entry;
*__slate_slot_25 = dx_make_map(dir, *bh, hinfo, *__slate_slot_21);
if *__slate_slot_25 < 0 {
err = *__slate_slot_25;
} else {
let __v10: *mut dx_map_entry = *__slate_slot_21;
*__slate_slot_21 = __v10.wrapping_offset(-(*__slate_slot_25 as isize));
dx_sort_map(*__slate_slot_21, *__slate_slot_25 as u32);
// Ensure that neither split block is over half full
*__slate_slot_16 = 0;
*__slate_slot_17 = 0;
*__slate_slot_11 = (*__slate_slot_25).wrapping_sub(1);
while *__slate_slot_11 >= 0 {
// is more than half of this entry in 2nd half of the block?
if (*__slate_slot_16).wrapping_add((((*(*__slate_slot_21).wrapping_offset(*__slate_slot_11 as isize)).size as i32) / 2) as u32) > blocksize / 2 {
break;
} else {
let __v8: u32 = *__slate_slot_16;
*__slate_slot_16 = __v8.wrapping_add(((*(*__slate_slot_21).wrapping_offset(*__slate_slot_11 as isize)).size as i32) as u32);
let __v6: u32 = *__slate_slot_17;
*__slate_slot_17 = __v6.wrapping_add(1);
let __v4: i32 = *__slate_slot_11;
*__slate_slot_11 = __v4.wrapping_sub(1);
}
}
// map index at which we will split
//
// If the sum of active entries didn't exceed half the block size, just
// split it in half by count; each resulting block will have at least
// half the space free.
if *__slate_slot_11 >= 0 {
*__slate_slot_18 = (*__slate_slot_25 as u32).wrapping_sub(*__slate_slot_17);
} else {
*__slate_slot_18 = (*__slate_slot_25 / 2) as u32;
}
__ret_warn_on = (*__slate_slot_18 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1509: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1509b - ., 3; .popsection\n# {0}", const 1509 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_do_split_30, sym __slate_asm_string_do_split_31, const 1977 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1510: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1510b - ., 4; .popsection\n# {0}", const 1510 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
// Should never happen, but avoid out-of-bounds access below
__ext4_error_inode(dir, c"do_split".as_ptr() as *const u8, 1981u32, (*(*bh)).b_blocknr, 0i32, c"bad indexed directory? hash=%08x:%08x count=%d move=%u".as_ptr() as *const u8, (*hinfo).hash, (*hinfo).minor_hash, *__slate_slot_25, *__slate_slot_17);
err = 117i32.wrapping_neg();
break '__join_0;
} else {
*__slate_slot_22 = (*(*__slate_slot_21).wrapping_offset(*__slate_slot_18 as isize)).hash;
*__slate_slot_26 = (*__slate_slot_22 == (*(*__slate_slot_21).wrapping_offset((*__slate_slot_18).wrapping_sub(1) as isize)).hash) as u32;
// Fancy dance to stay within two buffers
*__slate_slot_14 = dx_move_dirents(dir, data1, *__slate_slot_19, (*__slate_slot_21).wrapping_offset(*__slate_slot_18 as isize), (*__slate_slot_25 as u32).wrapping_sub(*__slate_slot_18) as i32, blocksize);
de = dx_pack_dirents(dir, data1, blocksize);
(*de).rec_len = ext4_rec_len_to_disk((data1.wrapping_offset(blocksize.wrapping_sub(csum_size as u32) as isize).offset_from(de as *mut u8) as i64) as u32, blocksize);
(*(*__slate_slot_14)).rec_len = ext4_rec_len_to_disk(((*__slate_slot_19).wrapping_offset(blocksize.wrapping_sub(csum_size as u32) as isize).offset_from(*__slate_slot_14 as *mut u8) as i64) as u32, blocksize);
if csum_size != 0 {
ext4_initialize_dirent_tail(*bh, blocksize);
ext4_initialize_dirent_tail(*__slate_slot_24, blocksize);
}
// Which block gets the new entry?
if (*hinfo).hash >= *__slate_slot_22 {
swap!(*bh, *__slate_slot_24);
de = *__slate_slot_14;
}
dx_insert_block(frame, (*__slate_slot_22).wrapping_add(*__slate_slot_26), *__slate_slot_23);
err = ext4_handle_dirty_dirblock(handle, dir, *__slate_slot_24);
if err == 0 {
err = ext4_handle_dirty_dx_node(handle, dir, (*frame).bh);
if err == 0 {
brelse(*__slate_slot_24);
return de;
}
}
}
}
}
}
ext4_std_error__38!(unsafe { (*dir).i_sb }, err);
}
brelse(*bh);
brelse(*__slate_slot_24);
*bh = core::ptr::null_mut::<buffer_head>();
return ERR_PTR(err as i64) as *mut ext4_dir_entry_2;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_find_dest_de(dir: *mut inode, bh: *mut buffer_head, buf: *mut (), buf_size: i32, fname: *mut ext4_filename, dest_de: *mut *mut ext4_dir_entry_2) -> i32 {
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let reclen: u16 = ext4_dir_rec_len(fname_len!(fname) as u8, dir as *const inode) as u16;
let mut nlen: i32 = 0;
let mut rlen: i32 = 0;
let mut offset: u32 = 0;
de = buf as *mut ext4_dir_entry_2;
let top: *mut u8 = ((unsafe { (((buf as *const u8).wrapping_offset(buf_size as isize) as *mut ()) as *const u8).wrapping_offset(-((reclen as i32) as isize)) }) as *mut ()) as *mut u8;
while (de as *mut u8) <= top {
if ext4_check_dir_entry__7!(dir, de, bh, buf, buf_size, offset) != 0 {
return 117i32.wrapping_neg();
}
if ext4_match(dir, fname as *const ext4_filename, de) {
return 17i32.wrapping_neg();
}
nlen = ext4_dir_rec_len(unsafe { (*de).name_len }, dir as *const inode) as i32;
rlen = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, buf_size as u32) as i32;
if (if (unsafe { (*de).inode }) != 0 { rlen.wrapping_sub(nlen) } else { rlen }) >= (reclen as i32) {
break;
}
de = (de as *mut u8).wrapping_offset(rlen as isize) as *mut ext4_dir_entry_2;
offset = offset.wrapping_add(rlen as u32);
}
if (de as *mut u8) > top {
return 28i32.wrapping_neg();
}
unsafe {
*dest_de = de;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ext4_insert_dentry(dir: *mut inode, inode: *mut inode, mut de: *mut ext4_dir_entry_2, buf_size: i32, fname: *mut ext4_filename) {
let nlen: i32 = ext4_dir_rec_len(unsafe { (*de).name_len }, dir as *const inode) as i32;
let rlen: i32 = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, buf_size as u32) as i32;
if (unsafe { (*de).inode }) != 0 {
let de1: *mut ext4_dir_entry_2 = (de as *mut u8).wrapping_offset(nlen as isize) as *mut ext4_dir_entry_2;
unsafe {
(*de1).rec_len = ext4_rec_len_to_disk(rlen.wrapping_sub(nlen) as u32, buf_size as u32);
}
unsafe {
(*de).rec_len = ext4_rec_len_to_disk(nlen as u32, buf_size as u32);
}
de = de1;
}
unsafe {
(*de).file_type = 0;
}
unsafe {
(*de).inode = (*inode).i_ino as u32;
}
ext4_set_de_type(unsafe { (*inode).i_sb }, de, unsafe { (*inode).i_mode });
unsafe {
(*de).name_len = fname_len!(fname) as u8;
}
unsafe { memcpy(((*de).name.as_mut_ptr() as *mut u8) as *mut (), fname_name!(fname) as *const (), fname_len!(fname) as u64) };
if ext4_hash_in_dirent(dir as *const inode) {
let hinfo: *mut dx_hash_info = unsafe { core::ptr::addr_of_mut!((*fname).hinfo) };
unsafe {
(*EXT4_DIRENT_HASHES!(de)).hash = (*hinfo).hash;
}
unsafe {
(*EXT4_DIRENT_HASHES!(de)).minor_hash = (*hinfo).minor_hash;
}
}
}

/// Add a new entry into a directory (leaf) block.  If de is non-NULL,
/// it points to a directory entry which is guaranteed to be large
/// enough for new directory entry.  If de is NULL, then
/// add_dirent_to_buf will attempt search the directory block for
/// space.  It will return -ENOSPC if no space is available, and -EIO
/// and -EEXIST if directory entry already exists.
fn add_dirent_to_buf(handle: *mut jbd2_journal_handle, fname: *mut ext4_filename, dir: *mut inode, inode: *mut inode, mut de: *mut ext4_dir_entry_2, bh: *mut buffer_head) -> i32 {
let blocksize: u32 = (unsafe { (*(*dir).i_sb).s_blocksize }) as u32;
let mut csum_size: i32 = 0;
let mut err: i32 = 0;
let mut err2: i32 = 0;
if ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
if de.is_null() {
err = ext4_find_dest_de(dir, bh, (unsafe { (*bh).b_data }) as *mut (), blocksize.wrapping_sub(csum_size as u32) as i32, fname, core::ptr::addr_of_mut!(de));
if err != 0 {
return err;
}
}
BUFFER_TRACE!();
err = ext4_journal_get_write_access__49!(handle, unsafe { (*dir).i_sb }, bh, 1i32);
if err != 0 {
ext4_std_error__44!(unsafe { (*dir).i_sb }, err);
return err;
}
// By now the buffer is marked for journaling
ext4_insert_dentry(dir, inode, de, blocksize as i32, fname);
// XXX shouldn't update any times until successful
// completion of syscall, but too many callers depend
// on this.
//
// XXX similarly, too many callers depend on
// ext4_new_inode() setting the times, but error
// recovery deletes the inode, so the worst that can
// happen is that the times are slightly out of date
// and/or different from the directory change time.
inode_set_mtime_to_ts(dir, unsafe { inode_set_ctime_current(dir) });
ext4_update_dx_flag(dir);
inode_inc_iversion(dir);
err2 = ext4_mark_inode_dirty__50!(handle, dir);
BUFFER_TRACE!();
err = ext4_handle_dirty_dirblock(handle, dir, bh);
if err != 0 {
ext4_std_error__19!(unsafe { (*dir).i_sb }, err);
}
if err != 0 { err } else { err2 }
}

fn ext4_check_dx_root(dir: *mut inode, root: *mut dx_root) -> bool {
let mut blockend: *mut u8 = unsafe { core::mem::zeroed() };
let mut blocksize: u32 = unsafe { core::mem::zeroed() };
let mut rlen: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(rlen) as *mut u32;
let mut error_msg: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *const u8 = core::ptr::addr_of_mut!(error_msg) as *mut *const u8;
let mut fde: core::mem::MaybeUninit<*mut fake_dirent> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut fake_dirent = core::ptr::addr_of_mut!(fde) as *mut *mut fake_dirent;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
blocksize = (*(*dir).i_sb).s_blocksize as u32;
blockend = (root as *mut u8).wrapping_offset((*(*dir).i_sb).s_blocksize as isize);
*__slate_slot_4 = core::ptr::addr_of_mut!((*root).dot);
if unlikely!(((*(*__slate_slot_4)).name_len as i32) != 1) != 0 {
*__slate_slot_3 = c"invalid name_len for '.'".as_ptr() as *const u8;
} else {
if unlikely__2!(strncmp(((*root).dot_name.as_mut_ptr() as *mut u8) as *const u8, c".".as_ptr() as *const u8, (*(*__slate_slot_4)).name_len as u64)) != 0 {
*__slate_slot_3 = c"invalid name for '.'".as_ptr() as *const u8;
} else {
*__slate_slot_2 = ext4_rec_len_from_disk((*(*__slate_slot_4)).rec_len, blocksize);
if unlikely!((*__slate_slot_4 as *mut u8).wrapping_offset(*__slate_slot_2 as isize) >= blockend) != 0 {
*__slate_slot_3 = c"invalid rec_len for '.'".as_ptr() as *const u8;
} else {
*__slate_slot_4 = core::ptr::addr_of_mut!((*root).dotdot);
if unlikely!(((*(*__slate_slot_4)).name_len as i32) != 2) != 0 {
*__slate_slot_3 = c"invalid name_len for '..'".as_ptr() as *const u8;
} else {
if unlikely__2!(strncmp(((*root).dotdot_name.as_mut_ptr() as *mut u8) as *const u8, c"..".as_ptr() as *const u8, (*(*__slate_slot_4)).name_len as u64)) != 0 {
*__slate_slot_3 = c"invalid name for '..'".as_ptr() as *const u8;
} else {
*__slate_slot_2 = ext4_rec_len_from_disk((*(*__slate_slot_4)).rec_len, blocksize);
if unlikely!((*__slate_slot_4 as *mut u8).wrapping_offset(*__slate_slot_2 as isize) >= blockend) != 0 {
*__slate_slot_3 = c"invalid rec_len for '..'".as_ptr() as *const u8;
} else {
return true;
}
}
}
}
}
}
__ext4_error_inode(dir, c"ext4_check_dx_root".as_ptr() as *const u8, 2203u32, 0u64, 0i32, c"Corrupt dir, %s, running e2fsck is recommended".as_ptr() as *const u8, *__slate_slot_3);
return false;
}
}

/// This converts a one block unindexed directory to a 3 block indexed
/// directory, and adds the dentry to the indexed directory.
fn make_indexed_dir(handle: *mut jbd2_journal_handle, fname: *mut ext4_filename, dir: *mut inode, inode: *mut inode, bh: *mut buffer_head) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut csum_size: i32 = unsafe { core::mem::zeroed() };
let mut fde: core::mem::MaybeUninit<*mut fake_dirent> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut fake_dirent = core::ptr::addr_of_mut!(fde) as *mut *mut fake_dirent;
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut blocksize: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(blocksize) as *mut u32;
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut top: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut u8 = core::ptr::addr_of_mut!(top) as *mut *mut u8;
let mut data2: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut u8 = core::ptr::addr_of_mut!(data2) as *mut *mut u8;
let mut de2: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de2) as *mut *mut ext4_dir_entry_2;
let mut de: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de) as *mut *mut ext4_dir_entry_2;
let mut entries: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut dx_entry = core::ptr::addr_of_mut!(entries) as *mut *mut dx_entry;
let mut frame: core::mem::MaybeUninit<*mut dx_frame> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut dx_frame = core::ptr::addr_of_mut!(frame) as *mut *mut dx_frame;
let mut frames: core::mem::MaybeUninit<__SlateAlign16<[dx_frame; 3]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut [dx_frame; 3] = core::ptr::addr_of_mut!(frames) as *mut [dx_frame; 3];
let mut root: core::mem::MaybeUninit<*mut dx_root> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut dx_root = core::ptr::addr_of_mut!(root) as *mut *mut dx_root;
let mut bh2: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh2) as *mut *mut buffer_head;
unsafe {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_12 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
csum_size = 0;
if ext4_has_feature_metadata_csum((*inode).i_sb) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
*__slate_slot_9 = (*(*dir).i_sb).s_blocksize as u32;
BUFFER_TRACE!();
*__slate_slot_10 = ext4_journal_get_write_access__34!(handle, unsafe { (*dir).i_sb }, bh, 1i32);
if *__slate_slot_10 != 0 {
ext4_std_error__7!(unsafe { (*dir).i_sb }, *__slate_slot_10);
brelse(bh);
return *__slate_slot_10;
} else {
*__slate_slot_19 = (*bh).b_data as *mut dx_root;
if !ext4_check_dx_root(dir, *__slate_slot_19) {
brelse(bh);
return 117i32.wrapping_neg();
} else {
// The 0th block becomes the root, move the dirents out
*__slate_slot_7 = core::ptr::addr_of_mut!((*(*__slate_slot_19)).dotdot);
*__slate_slot_15 = (*__slate_slot_7 as *mut u8).wrapping_offset(ext4_rec_len_from_disk((*(*__slate_slot_7)).rec_len, *__slate_slot_9) as isize) as *mut ext4_dir_entry_2;
*__slate_slot_11 = ((*__slate_slot_19 as *mut u8).wrapping_offset((*__slate_slot_9).wrapping_sub(csum_size as u32) as isize).offset_from(*__slate_slot_15 as *mut u8) as i64) as u32;
// Allocate new block for the 0th block's dirents
*__slate_slot_20 = ext4_append(handle, dir, &mut (*core::ptr::addr_of_mut!(*__slate_slot_8)));
if IS_ERR(*__slate_slot_20 as *const ()) {
brelse(bh);
return PTR_ERR(*__slate_slot_20 as *const ()) as i32;
} else {
ext4_set_inode_flag(dir, 12i32);
*__slate_slot_13 = (*(*__slate_slot_20)).b_data;
memcpy(*__slate_slot_13 as *mut (), *__slate_slot_15 as *const (), *__slate_slot_11 as u64);
memset(*__slate_slot_15 as *mut (), 0i32, *__slate_slot_11 as u64); // wipe old data
*__slate_slot_15 = *__slate_slot_13 as *mut ext4_dir_entry_2;
*__slate_slot_12 = (*__slate_slot_13).wrapping_offset(*__slate_slot_11 as isize);
loop {
let __v5: *mut ext4_dir_entry_2 = ext4_next_entry(*__slate_slot_15, *__slate_slot_9 as u64);
*__slate_slot_14 = __v5;
if (__v5 as *mut u8) < *__slate_slot_12 {
if ext4_check_dir_entry__12!(dir, *__slate_slot_15, *__slate_slot_20, *__slate_slot_13, *__slate_slot_11, (*__slate_slot_15 as *mut u8).offset_from(*__slate_slot_13 as *mut u8) as i64) != 0 {
brelse(*__slate_slot_20);
brelse(bh);
return 117i32.wrapping_neg();
} else {
*__slate_slot_15 = *__slate_slot_14;
}
} else {
break;
}
}
(*(*__slate_slot_15)).rec_len = ext4_rec_len_to_disk(((*__slate_slot_13).wrapping_offset((*__slate_slot_9).wrapping_sub(csum_size as u32) as isize).offset_from(*__slate_slot_15 as *mut u8) as i64) as u32, *__slate_slot_9);
if csum_size != 0 {
ext4_initialize_dirent_tail(*__slate_slot_20, *__slate_slot_9);
}
// Initialize the root; the dot dirents already exist
*__slate_slot_15 = core::ptr::addr_of_mut!((*(*__slate_slot_19)).dotdot) as *mut ext4_dir_entry_2;
(*(*__slate_slot_15)).rec_len = ext4_rec_len_to_disk((*__slate_slot_9).wrapping_sub(ext4_dir_rec_len(2u8, core::ptr::null::<inode>())), *__slate_slot_9);
memset(core::ptr::addr_of_mut!((*(*__slate_slot_19)).info) as *mut (), 0i32, core::mem::size_of::<dx_root_info>() as u64);
(*(*__slate_slot_19)).info.info_length = core::mem::size_of::<dx_root_info>() as u8;
if ext4_hash_in_dirent(dir as *const inode) {
(*(*__slate_slot_19)).info.hash_version = 6;
} else {
(*(*__slate_slot_19)).info.hash_version = (*EXT4_SB((*dir).i_sb)).s_def_hash_version as u8;
}
*__slate_slot_16 = core::ptr::addr_of_mut!((*(*__slate_slot_19)).entries) as *mut dx_entry;
dx_set_block(*__slate_slot_16, 1u32);
dx_set_count(*__slate_slot_16, 1u32);
dx_set_limit(*__slate_slot_16, dx_root_limit(dir, core::mem::size_of::<dx_root_info>() as u32));
// Initialize as for dx_probe
(*fname).hinfo.hash_version = (*(*__slate_slot_19)).info.hash_version as i32;
if (*fname).hinfo.hash_version <= 2 {
let __v4: i32 = (*EXT4_SB((*dir).i_sb)).s_hash_unsigned;
let __v2: i32 = (*fname).hinfo.hash_version;
(*fname).hinfo.hash_version = __v2.wrapping_add(__v4);
}
(*fname).hinfo.seed = (*EXT4_SB((*dir).i_sb)).s_hash_seed.as_mut_ptr() as *mut u32;
// casefolded encrypted hashes are computed on fname setup
if !ext4_hash_in_dirent(dir as *const inode) {
err = ext4fs_dirhash(dir as *const inode, fname_name!(fname) as *const u8, fname_len!(fname) as i32, core::ptr::addr_of_mut!((*fname).hinfo));
if err < 0 {
brelse(*__slate_slot_20);
brelse(bh);
return err;
}
}
memset((*__slate_slot_18).as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[dx_frame; 3]>() as u64);
*__slate_slot_17 = (*__slate_slot_18).as_mut_ptr() as *mut dx_frame;
(*(*__slate_slot_17)).entries = *__slate_slot_16;
(*(*__slate_slot_17)).at = *__slate_slot_16;
(*(*__slate_slot_17)).bh = bh;
*__slate_slot_10 = ext4_handle_dirty_dx_node(handle, dir, (*(*__slate_slot_17)).bh);
if *__slate_slot_10 == 0 {
*__slate_slot_10 = ext4_handle_dirty_dirblock(handle, dir, *__slate_slot_20);
if *__slate_slot_10 == 0 {
*__slate_slot_15 = do_split(handle, dir, core::ptr::addr_of_mut!(*__slate_slot_20), *__slate_slot_17, core::ptr::addr_of_mut!((*fname).hinfo));
if IS_ERR(*__slate_slot_15 as *const ()) {
*__slate_slot_10 = PTR_ERR(*__slate_slot_15 as *const ()) as i32;
} else {
*__slate_slot_10 = add_dirent_to_buf(handle, fname, dir, inode, *__slate_slot_15, *__slate_slot_20);
}
}
}
if *__slate_slot_10 != 0 {
ext4_mark_inode_dirty__9!(handle, dir);
}
// Even if the block split failed, we have to properly write
// out all the changes we did so far. Otherwise we can end up
// with corrupted filesystem.
dx_release((*__slate_slot_18).as_mut_ptr() as *mut dx_frame);
brelse(*__slate_slot_20);
return *__slate_slot_10;
}
}
}
}
}

/// 	ext4_add_entry()
///
/// adds a file entry to the specified directory, using the same
/// semantics as ext4_find_entry(). It returns NULL if it failed.
///
/// NOTE!! The inode part of 'de' is left at 0 - which means you
/// may not sleep between calling this and putting something into
/// the entry, as someone else might have used it while you slept.
fn __ext4_add_entry(handle: *mut jbd2_journal_handle, dir: *mut inode, d_name: *const qstr, inode: *mut inode) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut csum_size: i32 = unsafe { core::mem::zeroed() };
let mut blocks: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(blocks) as *mut u32;
let mut block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(block) as *mut u32;
let mut blocksize: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(blocksize) as *mut u32;
let mut dx_fallback: i32 = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut fname: core::mem::MaybeUninit<ext4_filename> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut ext4_filename = core::ptr::addr_of_mut!(fname) as *mut ext4_filename;
let mut sb: core::mem::MaybeUninit<*mut super_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut super_block = core::ptr::addr_of_mut!(sb) as *mut *mut super_block;
let mut de: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de) as *mut *mut ext4_dir_entry_2;
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
unsafe {
bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 48usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
dx_fallback = 0;
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
csum_size = 0;
if ext4_has_feature_metadata_csum((*inode).i_sb) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
*__slate_slot_12 = (*dir).i_sb;
*__slate_slot_8 = (*(*__slate_slot_12)).s_blocksize as u32;
if !generic_ci_validate_strict_name(dir, d_name) {
return 22i32.wrapping_neg();
} else {
*__slate_slot_10 = ext4_fname_setup_filename(dir, d_name, 0i32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_11)));
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
'__join_2: {
if ext4_has_inline_data(dir) != 0 {
*__slate_slot_10 = ext4_try_add_inline_entry(handle, core::ptr::addr_of_mut!(*__slate_slot_11), dir, inode);
if *__slate_slot_10 < 0 {
break '__join_2;
} else {
if *__slate_slot_10 == 1 {
*__slate_slot_10 = 0;
break '__join_2;
}
}
}
if is_dx!(dir) {
*__slate_slot_10 = ext4_dx_add_entry(handle, core::ptr::addr_of_mut!(*__slate_slot_11), dir, inode);
if *__slate_slot_10 == 0 || *__slate_slot_10 != 4094i32.wrapping_neg() {
break '__join_2;
} else {
// Can we just ignore htree data?
if ext4_has_feature_metadata_csum(*__slate_slot_12) {
__ext4_error_inode(dir, c"__ext4_add_entry".as_ptr() as *const u8, 2402u32, 0u64, 0i32, c"Directory has corrupted htree index.".as_ptr() as *const u8);
*__slate_slot_10 = 117i32.wrapping_neg();
break '__join_2;
} else {
ext4_clear_inode_flag(dir, 12i32);
dx_fallback = dx_fallback.wrapping_add(1);
*__slate_slot_10 = ext4_mark_inode_dirty__14!(handle, dir);
if unlikely__2!(*__slate_slot_10) != 0 {
break '__join_2;
}
}
}
}
*__slate_slot_6 = ((*dir).i_size >> ((*(*__slate_slot_12)).s_blocksize_bits as i32)) as u32;
*__slate_slot_7 = 0;
'__join_7: {
'__join_19: {
'__join_17: {
'__join_11: {
while *__slate_slot_7 < *__slate_slot_6 {
bh = ext4_read_dirblock__6!(dir, *__slate_slot_7, 2i32);
if bh.is_null() {
break '__join_19;
} else {
if IS_ERR(bh as *const ()) {
break '__join_17;
} else {
*__slate_slot_10 = add_dirent_to_buf(handle, core::ptr::addr_of_mut!(*__slate_slot_11), dir, inode, core::ptr::null_mut::<ext4_dir_entry_2>(), bh);
if *__slate_slot_10 != 28i32.wrapping_neg() {
break '__join_2;
} else {
if *__slate_slot_6 == 1 && dx_fallback == 0 {
__v2 = ext4_has_feature_dir_index(*__slate_slot_12);
} else {
__v2 = false;
}
if __v2 {
break '__join_11;
} else {
brelse(bh);
let __v1: u32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(1);
}
}
}
}
}
bh = ext4_append(handle, dir, &mut (*core::ptr::addr_of_mut!(*__slate_slot_7)));
break '__join_7;
}
*__slate_slot_10 = make_indexed_dir(handle, core::ptr::addr_of_mut!(*__slate_slot_11), dir, inode, bh);
bh = core::ptr::null_mut::<buffer_head>(); // make_indexed_dir releases bh
break '__join_2;
}
*__slate_slot_10 = PTR_ERR(bh as *const ()) as i32;
bh = core::ptr::null_mut::<buffer_head>();
break '__join_2;
}
bh = ext4_bread(handle, dir, *__slate_slot_7, 1i32);
}
if IS_ERR(bh as *const ()) {
*__slate_slot_10 = PTR_ERR(bh as *const ()) as i32;
bh = core::ptr::null_mut::<buffer_head>();
} else {
*__slate_slot_13 = (*bh).b_data as *mut ext4_dir_entry_2;
(*(*__slate_slot_13)).inode = 0;
(*(*__slate_slot_13)).rec_len = ext4_rec_len_to_disk((*__slate_slot_8).wrapping_sub(csum_size as u32), *__slate_slot_8);
if csum_size != 0 {
ext4_initialize_dirent_tail(bh, *__slate_slot_8);
}
*__slate_slot_10 = add_dirent_to_buf(handle, core::ptr::addr_of_mut!(*__slate_slot_11), dir, inode, *__slate_slot_13, bh);
}
}
ext4_fname_free_filename(&(*core::ptr::addr_of_mut!(*__slate_slot_11)));
brelse(bh);
if *__slate_slot_10 == 0 {
ext4_set_inode_state(inode, 5i32);
}
return *__slate_slot_10;
}
}
}
}

fn ext4_add_entry(handle: *mut jbd2_journal_handle, dentry: *mut dentry, inode: *mut inode) -> i32 {
let dir: *mut inode = d_inode((unsafe { (*dentry).d_parent }) as *const dentry);
if fscrypt_is_nokey_name(dentry as *const dentry) {
return 126i32.wrapping_neg();
}
__ext4_add_entry(handle, dir, unsafe { core::ptr::addr_of!((*dentry).__slate_anon_4.d_name) }, inode)
}

/// Returns 0 for success, or a negative error value
fn ext4_dx_add_entry(handle: *mut jbd2_journal_handle, fname: *mut ext4_filename, dir: *mut inode, inode: *mut inode) -> i32 {
let mut hash2: u32 = unsafe { core::mem::zeroed() };
let mut icount2: u32 = unsafe { core::mem::zeroed() };
let mut icount1: u32 = unsafe { core::mem::zeroed() };
let mut dxroot: core::mem::MaybeUninit<*mut dx_root> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut dx_root = core::ptr::addr_of_mut!(dxroot) as *mut *mut dx_root;
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut bh2: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh2) as *mut *mut buffer_head;
let mut node2: core::mem::MaybeUninit<*mut dx_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut dx_node = core::ptr::addr_of_mut!(node2) as *mut *mut dx_node;
let mut entries2: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut dx_entry = core::ptr::addr_of_mut!(entries2) as *mut *mut dx_entry;
let mut add_level: i32 = unsafe { core::mem::zeroed() };
let mut icount: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(icount) as *mut u32;
let mut levels: i32 = unsafe { core::mem::zeroed() };
let mut newblock: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(newblock) as *mut u32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut restart: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(restart) as *mut i32;
let mut de: core::mem::MaybeUninit<*mut ext4_dir_entry_2> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut ext4_dir_entry_2 = core::ptr::addr_of_mut!(de) as *mut *mut ext4_dir_entry_2;
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut at: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut dx_entry = core::ptr::addr_of_mut!(at) as *mut *mut dx_entry;
let mut entries: core::mem::MaybeUninit<*mut dx_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut dx_entry = core::ptr::addr_of_mut!(entries) as *mut *mut dx_entry;
let mut frame: core::mem::MaybeUninit<*mut dx_frame> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut dx_frame = core::ptr::addr_of_mut!(frame) as *mut *mut dx_frame;
let mut frames: core::mem::MaybeUninit<__SlateAlign16<[dx_frame; 3]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut [dx_frame; 3] = core::ptr::addr_of_mut!(frames) as *mut [dx_frame; 3];
unsafe {
let __slate_auto_bytes_6 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_7 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
sb = (*dir).i_sb;
let __slate_auto_bytes_11 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_12 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
loop {
*__slate_slot_19 = 0;
*__slate_slot_25 = dx_probe(fname, dir, core::ptr::null_mut::<dx_hash_info>(), (*__slate_slot_26).as_mut_ptr() as *mut dx_frame);
if IS_ERR(*__slate_slot_25 as *const ()) {
return PTR_ERR(*__slate_slot_25 as *const ()) as i32;
} else {
'__join_1: {
*__slate_slot_24 = (*(*__slate_slot_25)).entries;
*__slate_slot_23 = (*(*__slate_slot_25)).at;
*__slate_slot_22 = ext4_read_dirblock__9!(dir, dx_get_block((*(*__slate_slot_25)).at), 3i32);
if IS_ERR(*__slate_slot_22 as *const ()) {
*__slate_slot_18 = PTR_ERR(*__slate_slot_22 as *const ()) as i32;
*__slate_slot_22 = core::ptr::null_mut::<buffer_head>();
} else {
'__join_2: {
BUFFER_TRACE!();
*__slate_slot_18 = ext4_journal_get_write_access__59!(handle, sb, *__slate_slot_22, 1i32);
if *__slate_slot_18 == 0 {
*__slate_slot_18 = add_dirent_to_buf(handle, fname, dir, inode, core::ptr::null_mut::<ext4_dir_entry_2>(), *__slate_slot_22);
if *__slate_slot_18 != 28i32.wrapping_neg() {
break '__join_1;
} else {
*__slate_slot_18 = 0;
// Block full, should compress but for now just split
// Need to split index?
if dx_get_count(*__slate_slot_24) == dx_get_limit(*__slate_slot_24) {
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
levels = ((*__slate_slot_25).offset_from((*__slate_slot_26).as_mut_ptr() as *mut dx_frame) as i64).wrapping_add(1) as i32;
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
add_level = 1;
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
'__join_28: {
loop {
if *__slate_slot_25 > ((*__slate_slot_26).as_mut_ptr() as *mut dx_frame) {
if dx_get_count((*(*__slate_slot_25).wrapping_offset(-(1 as isize))).entries) < dx_get_limit((*(*__slate_slot_25).wrapping_offset(-(1 as isize))).entries) {
break;
} else {
let __v10: *mut dx_frame = *__slate_slot_25;
*__slate_slot_25 = __v10.wrapping_offset(-(1 as isize)); // split higher index block
*__slate_slot_23 = (*(*__slate_slot_25)).at;
*__slate_slot_24 = (*(*__slate_slot_25)).entries;
*__slate_slot_19 = 1;
}
} else {
break '__join_28;
}
}
add_level = 0;
}
if add_level != 0 {
__v8 = levels == ext4_dir_htree_level(sb);
} else {
__v8 = false;
}
if __v8 {
__ext4_warning(sb, c"ext4_dx_add_entry".as_ptr() as *const u8, 2537u32, c"Directory (ino: %llu) index full, reach max htree level :%d".as_ptr() as *const u8, (*dir).i_ino, levels);
if ext4_dir_htree_level(sb) < 3 {
__ext4_warning(sb, c"ext4_dx_add_entry".as_ptr() as *const u8, 2541u32, c"Large directory feature is not enabled on this filesystem".as_ptr() as *const u8);
}
*__slate_slot_18 = 28i32.wrapping_neg();
break '__join_1;
} else {
*__slate_slot_15 = dx_get_count(*__slate_slot_24);
*__slate_slot_11 = ext4_append(handle, dir, &mut (*core::ptr::addr_of_mut!(*__slate_slot_17)));
if IS_ERR(*__slate_slot_11 as *const ()) {
*__slate_slot_18 = PTR_ERR(*__slate_slot_11 as *const ()) as i32;
break '__join_1;
} else {
*__slate_slot_12 = (*(*__slate_slot_11)).b_data as *mut dx_node;
*__slate_slot_13 = core::ptr::addr_of_mut!((*(*__slate_slot_12)).entries) as *mut dx_entry;
memset(core::ptr::addr_of_mut!((*(*__slate_slot_12)).fake) as *mut (), 0i32, core::mem::size_of::<fake_dirent>() as u64);
(*(*__slate_slot_12)).fake.rec_len = ext4_rec_len_to_disk((*sb).s_blocksize as u32, (*sb).s_blocksize as u32);
BUFFER_TRACE!();
*__slate_slot_18 = ext4_journal_get_write_access__43!(handle, sb, unsafe { (*(*__slate_slot_25)).bh }, 1i32);
if *__slate_slot_18 != 0 {
brelse(*__slate_slot_11);
break '__join_2;
} else {
if add_level == 0 {
icount1 = *__slate_slot_15 / 2;
icount2 = (*__slate_slot_15).wrapping_sub(icount1);
hash2 = dx_get_hash((*__slate_slot_24).wrapping_offset(icount1 as isize));
BUFFER_TRACE!(); // index root
*__slate_slot_18 = ext4_journal_get_write_access__2!(handle, sb, unsafe { (*(*__slate_slot_25).wrapping_offset(-(1 as isize))).bh }, 1i32);
if *__slate_slot_18 != 0 {
brelse(*__slate_slot_11);
break '__join_2;
} else {
memcpy((*__slate_slot_13 as *mut u8) as *mut (), ((*__slate_slot_24).wrapping_offset(icount1 as isize) as *mut u8) as *const (), (icount2 as u64).wrapping_mul(core::mem::size_of::<dx_entry>() as u64));
dx_set_count(*__slate_slot_24, icount1);
dx_set_count(*__slate_slot_13, icount2);
dx_set_limit(*__slate_slot_13, dx_node_limit(dir));
// Which index block gets the new entry?
if ((*__slate_slot_23).offset_from(*__slate_slot_24 as *mut dx_entry) as i64) >= (icount1 as i64) {
(*(*__slate_slot_25)).at = (*__slate_slot_13).wrapping_offset(((*__slate_slot_23).offset_from(*__slate_slot_24 as *mut dx_entry) as i64).wrapping_sub(icount1 as i64) as isize);
let __v0: *mut dx_entry = *__slate_slot_13;
*__slate_slot_24 = __v0;
(*(*__slate_slot_25)).entries = __v0;
swap!((*(*__slate_slot_25)).bh, *__slate_slot_11);
}
dx_insert_block((*__slate_slot_25).wrapping_offset(-(1 as isize)), hash2, *__slate_slot_17);
*__slate_slot_18 = ext4_handle_dirty_dx_node(handle, dir, *__slate_slot_11);
if *__slate_slot_18 != 0 {
brelse(*__slate_slot_11);
break '__join_2;
} else {
brelse(*__slate_slot_11);
*__slate_slot_18 = ext4_handle_dirty_dx_node(handle, dir, (*(*__slate_slot_25).wrapping_offset(-(1 as isize))).bh);
if *__slate_slot_18 != 0 {
break '__join_2;
} else {
*__slate_slot_18 = ext4_handle_dirty_dx_node(handle, dir, (*(*__slate_slot_25)).bh);
if *__slate_slot_19 != 0 || *__slate_slot_18 != 0 {
break '__join_2;
}
}
}
}
} else {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
memcpy((*__slate_slot_13 as *mut u8) as *mut (), (*__slate_slot_24 as *mut u8) as *const (), (*__slate_slot_15 as u64).wrapping_mul(core::mem::size_of::<dx_entry>() as u64));
dx_set_limit(*__slate_slot_13, dx_node_limit(dir));
// Set up root
dx_set_count(*__slate_slot_24, 1u32);
dx_set_block((*__slate_slot_24).wrapping_offset(0 as isize), *__slate_slot_17);
*__slate_slot_7 = (*(*__slate_slot_26)[0].bh).b_data as *mut dx_root;
let __v6: *mut dx_root = *__slate_slot_7;
(*__v6).info.indirect_levels = ((*__v6).info.indirect_levels as i32).wrapping_add(1) as u8;
*__slate_slot_18 = ext4_handle_dirty_dx_node(handle, dir, (*(*__slate_slot_25)).bh);
if *__slate_slot_18 != 0 {
brelse(*__slate_slot_11);
break '__join_2;
} else {
*__slate_slot_18 = ext4_handle_dirty_dx_node(handle, dir, *__slate_slot_11);
brelse(*__slate_slot_11);
*__slate_slot_19 = 1;
break '__join_2;
}
}
}
}
}
}
*__slate_slot_20 = do_split(handle, dir, core::ptr::addr_of_mut!(*__slate_slot_22), *__slate_slot_25, core::ptr::addr_of_mut!((*fname).hinfo));
if IS_ERR(*__slate_slot_20 as *const ()) {
*__slate_slot_18 = PTR_ERR(*__slate_slot_20 as *const ()) as i32;
break '__join_1;
} else {
*__slate_slot_18 = add_dirent_to_buf(handle, fname, dir, inode, *__slate_slot_20, *__slate_slot_22);
break '__join_1;
}
}
}
}
ext4_std_error__47!(unsafe { (*dir).i_sb }, *__slate_slot_18); // this is a no-op if err == 0
}
}
brelse(*__slate_slot_22);
dx_release((*__slate_slot_26).as_mut_ptr() as *mut dx_frame);
// @restart is true means htree-path has been changed, we need to
// repeat dx_probe() to find out valid htree-path
if !(*__slate_slot_19 != 0 && *__slate_slot_18 == 0) {
break;
}
}
}
return *__slate_slot_18;
}
}

/// ext4_generic_delete_entry deletes a directory entry by merging it
/// with the previous entry
#[unsafe(no_mangle)]
extern "C" fn ext4_generic_delete_entry(dir: *mut inode, de_del: *mut ext4_dir_entry_2, bh: *mut buffer_head, entry_buf: *mut (), buf_size: i32, csum_size: i32) -> i32 {
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut pde: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let blocksize: u32 = (unsafe { (*(*dir).i_sb).s_blocksize }) as u32;
let mut i: i32 = 0;
pde = core::ptr::null_mut::<ext4_dir_entry_2>();
de = entry_buf as *mut ext4_dir_entry_2;
while i < buf_size.wrapping_sub(csum_size) {
if ext4_check_dir_entry__14!(dir, de, bh, entry_buf, buf_size, i) != 0 {
return 117i32.wrapping_neg();
}
if de == de_del {
if !pde.is_null() {
unsafe {
(*pde).rec_len = ext4_rec_len_to_disk(ext4_rec_len_from_disk((*pde).rec_len, blocksize).wrapping_add(ext4_rec_len_from_disk((*de).rec_len, blocksize)), blocksize);
}
// wipe entire dir_entry
unsafe { memset(de as *mut (), 0i32, ext4_rec_len_from_disk((*de).rec_len, blocksize) as u64) };
} else {
// wipe dir_entry excluding the rec_len field
unsafe {
(*de).inode = 0;
}
unsafe { memset(core::ptr::addr_of_mut!((*de).name_len) as *mut (), 0i32, (ext4_rec_len_from_disk((*de).rec_len, blocksize) as u64).wrapping_sub(offsetof__17!(ext4_dir_entry_2))) };
}
inode_inc_iversion(dir);
return 0;
}
let __v0: u32 = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, blocksize);
i = (i as u32).wrapping_add(__v0) as i32;
pde = de;
de = ext4_next_entry(de, blocksize as u64);
}
2i32.wrapping_neg()
}

fn ext4_delete_entry(handle: *mut jbd2_journal_handle, dir: *mut inode, de_del: *mut ext4_dir_entry_2, bh: *mut buffer_head) -> i32 {
let mut has_inline_data: i32 = unsafe { core::mem::zeroed() };
let mut csum_size: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
csum_size = 0;
if ext4_has_inline_data(dir) != 0 {
has_inline_data = 1;
*__slate_slot_2 = ext4_delete_inline_entry(handle, dir, de_del, bh, core::ptr::addr_of_mut!(has_inline_data));
if has_inline_data != 0 {
return *__slate_slot_2;
}
}
if ext4_has_feature_metadata_csum((*dir).i_sb) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
BUFFER_TRACE!();
*__slate_slot_2 = ext4_journal_get_write_access__36!(handle, unsafe { (*dir).i_sb }, bh, 1i32);
if unlikely__2!(*__slate_slot_2) == 0 {
*__slate_slot_2 = ext4_generic_delete_entry(dir, de_del, bh, (*bh).b_data as *mut (), (*(*dir).i_sb).s_blocksize as i32, csum_size);
if *__slate_slot_2 == 0 {
BUFFER_TRACE!();
*__slate_slot_2 = ext4_handle_dirty_dirblock(handle, dir, bh);
if unlikely__2!(*__slate_slot_2) == 0 {
return 0;
}
}
}
if *__slate_slot_2 != 2i32.wrapping_neg() {
ext4_std_error__26!(unsafe { (*dir).i_sb }, *__slate_slot_2);
}
return *__slate_slot_2;
}
}

/// Set directory link count to 1 if nlinks > EXT4_LINK_MAX, or if nlinks == 2
/// since this indicates that nlinks count was previously 1 to avoid overflowing
/// the 16-bit i_links_count field on disk.  Directories with i_nlink == 1 mean
/// that subdirectory link counts are not being maintained accurately.
///
/// The caller has already checked for i_nlink overflow in case the DIR_LINK
/// feature is not enabled and returned -EMLINK.  The is_dx() check is a proxy
/// for checking S_ISDIR(inode) (since the INODE_INDEX feature will not be set
/// on regular files) and to avoid creating huge/slow non-HTREE directories.
fn ext4_inc_count(inode: *mut inode) {
unsafe { inc_nlink(inode) };
if is_dx!(inode) && ((unsafe { (*inode).__slate_anon_12.i_nlink }) > 65000 || (unsafe { (*inode).__slate_anon_12.i_nlink }) == 2) {
unsafe { set_nlink(inode, 1u32) };
}
}

/// If a directory had nlink == 1, then we should let it be 1. This indicates
/// directory has >EXT4_LINK_MAX subdirs.
fn ext4_dec_count(inode: *mut inode) {
if !S_ISDIR!(unsafe { (*inode).i_mode }) || (unsafe { (*inode).__slate_anon_12.i_nlink }) > 2 {
unsafe { drop_nlink(inode) };
}
}

/// Add non-directory inode to a directory. On success, the inode reference is
/// consumed by dentry is instantiation. This is also indicated by clearing of
/// *inodep pointer. On failure, the caller is responsible for dropping the
/// inode reference in the safe context.
fn ext4_add_nondir(handle: *mut jbd2_journal_handle, dentry: *mut dentry, inodep: &mut *mut inode) -> i32 {
let dir: *mut inode = d_inode((unsafe { (*dentry).d_parent }) as *const dentry);
let inode: *mut inode = unsafe { *(inodep as *mut *mut inode) };
let mut err: i32 = ext4_add_entry(handle, dentry, inode);
if err == 0 {
err = ext4_mark_inode_dirty__67!(handle, inode);
if IS_DIRSYNC!(dir) {
ext4_handle_sync(handle);
}
unsafe { d_instantiate_new(dentry, inode) };
unsafe {
*(inodep as *mut *mut inode) = core::ptr::null_mut::<inode>();
}
return err;
}
unsafe { drop_nlink(inode) };
ext4_mark_inode_dirty__49!(handle, inode);
unsafe { ext4_orphan_add(handle, inode) };
unsafe { unlock_new_inode(inode) };
err
}

/// By the time this is called, we already have created
/// the directory cache entry for the new file, but it
/// is so far negative - it has no inode.
///
/// If the create succeeds, we fill in the inode information
/// with d_instantiate().
#[unsafe(link_section = ".text.slate_distinct.namei.ext4_create")]
extern "C" fn ext4_create(idmap: *mut mnt_idmap, dir: *mut inode, dentry: *mut dentry, mode: u16) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
*__slate_slot_3 = dquot_initialize(dir);
if *__slate_slot_3 != 0 {
return *__slate_slot_3;
} else {
*__slate_slot_2 = EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(12).wrapping_add(3) as i32;
loop {
*__slate_slot_4 = ext4_new_inode_start_handle__6!(idmap, dir, mode, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), 0i32, 4i32, *__slate_slot_2);
*__slate_slot_5 = ext4_journal_current_handle();
*__slate_slot_3 = PTR_ERR(*__slate_slot_4 as *const ()) as i32;
if !IS_ERR(*__slate_slot_4 as *const ()) {
(*(*__slate_slot_4)).i_op = core::ptr::addr_of!(ext4_file_inode_operations);
(*(*__slate_slot_4)).__slate_anon_43.i_fop = core::ptr::addr_of!(ext4_file_operations);
ext4_set_aops(*__slate_slot_4);
*__slate_slot_3 = ext4_add_nondir(*__slate_slot_5, dentry, &mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
if *__slate_slot_3 == 0 {
ext4_fc_track_create(*__slate_slot_5, dentry);
}
}
if !(*__slate_slot_5).is_null() {
ext4_journal_stop__21!(*__slate_slot_5);
}
if !IS_ERR_OR_NULL(*__slate_slot_4 as *const ()) {
iput(*__slate_slot_4);
}
if *__slate_slot_3 == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
return *__slate_slot_3;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_mknod")]
extern "C" fn ext4_mknod(idmap: *mut mnt_idmap, dir: *mut inode, dentry: *mut dentry, mode: u16, rdev: u32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
*__slate_slot_3 = dquot_initialize(dir);
if *__slate_slot_3 != 0 {
return *__slate_slot_3;
} else {
*__slate_slot_2 = EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(12).wrapping_add(3) as i32;
loop {
*__slate_slot_4 = ext4_new_inode_start_handle!(idmap, dir, mode, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), 0i32, 4i32, *__slate_slot_2);
*__slate_slot_5 = ext4_journal_current_handle();
*__slate_slot_3 = PTR_ERR(*__slate_slot_4 as *const ()) as i32;
if !IS_ERR(*__slate_slot_4 as *const ()) {
init_special_inode(*__slate_slot_4, (*(*__slate_slot_4)).i_mode, rdev);
(*(*__slate_slot_4)).i_op = core::ptr::addr_of!(ext4_special_inode_operations.0);
*__slate_slot_3 = ext4_add_nondir(*__slate_slot_5, dentry, &mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
if *__slate_slot_3 == 0 {
ext4_fc_track_create(*__slate_slot_5, dentry);
}
}
if !(*__slate_slot_5).is_null() {
ext4_journal_stop__60!(*__slate_slot_5);
}
if !IS_ERR_OR_NULL(*__slate_slot_4 as *const ()) {
iput(*__slate_slot_4);
}
if *__slate_slot_3 == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
return *__slate_slot_3;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_tmpfile")]
extern "C" fn ext4_tmpfile(idmap: *mut mnt_idmap, dir: *mut inode, file: *mut file, mode: u16) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
*__slate_slot_2 = dquot_initialize(dir);
if *__slate_slot_2 != 0 {
return *__slate_slot_2;
} else {
loop {
*__slate_slot_3 = ext4_new_inode_start_handle__4!(idmap, dir, mode, 0i32, 4i32, (EXT4_MAXQUOTAS_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(4) as u32).wrapping_add(6));
*__slate_slot_4 = ext4_journal_current_handle();
*__slate_slot_2 = PTR_ERR(*__slate_slot_3 as *const ()) as i32;
if !IS_ERR(*__slate_slot_3 as *const ()) {
(*(*__slate_slot_3)).i_op = core::ptr::addr_of!(ext4_file_inode_operations);
(*(*__slate_slot_3)).__slate_anon_43.i_fop = core::ptr::addr_of!(ext4_file_operations);
ext4_set_aops(*__slate_slot_3);
d_tmpfile(file, *__slate_slot_3);
*__slate_slot_2 = ext4_orphan_add(*__slate_slot_4, *__slate_slot_3);
if *__slate_slot_2 != 0 {
ext4_journal_stop__15!(*__slate_slot_4);
unlock_new_inode(*__slate_slot_3);
return *__slate_slot_2;
} else {
mark_inode_dirty(*__slate_slot_3);
unlock_new_inode(*__slate_slot_3);
}
}
if !(*__slate_slot_4).is_null() {
ext4_journal_stop__64!(*__slate_slot_4);
}
if *__slate_slot_2 == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
return finish_open_simple(file, *__slate_slot_2);
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_init_dirblock(handle: *mut jbd2_journal_handle, inode: *mut inode, bh: *mut buffer_head, parent_ino: u32, inline_buf: *mut (), inline_size: i32) -> i32 {
let mut de: *mut ext4_dir_entry_2 = (unsafe { (*bh).b_data }) as *mut ext4_dir_entry_2;
let blocksize: u64 = unsafe { (*bh).b_size };
let mut csum_size: i32 = 0;
let mut header_size: i32 = 0;
if ext4_has_feature_metadata_csum(unsafe { (*inode).i_sb }) {
csum_size = core::mem::size_of::<ext4_dir_entry_tail>() as i32;
}
unsafe {
(*de).inode = (*inode).i_ino as u32;
}
unsafe {
(*de).name_len = 1;
}
unsafe {
(*de).rec_len = ext4_rec_len_to_disk(ext4_dir_rec_len((*de).name_len, core::ptr::null::<inode>()), blocksize as u32);
}
unsafe { memcpy(((*de).name.as_mut_ptr() as *mut u8) as *mut (), b".\0".as_ptr() as *const (), 2u64) };
ext4_set_de_type(unsafe { (*inode).i_sb }, de, 16384u16);
de = ext4_next_entry(de, blocksize);
unsafe {
(*de).inode = parent_ino;
}
unsafe {
(*de).name_len = 2;
}
unsafe { memcpy(((*de).name.as_mut_ptr() as *mut u8) as *mut (), b"..\0".as_ptr() as *const (), 3u64) };
ext4_set_de_type(unsafe { (*inode).i_sb }, de, 16384u16);
if !inline_buf.is_null() {
unsafe {
(*de).rec_len = ext4_rec_len_to_disk(ext4_dir_rec_len((*de).name_len, core::ptr::null::<inode>()), blocksize as u32);
}
de = ext4_next_entry(de, blocksize);
header_size = ((unsafe { (de as *mut u8).offset_from((*bh).b_data as *mut u8) }) as i64) as i32;
unsafe { memcpy(de as *mut (), inline_buf as *const (), inline_size as u64) };
unsafe { ext4_update_final_de((*bh).b_data as *mut (), inline_size.wrapping_add(header_size), blocksize.wrapping_sub(csum_size as u64) as i32) };
} else {
unsafe {
(*de).rec_len = ext4_rec_len_to_disk(blocksize.wrapping_sub((csum_size as u32).wrapping_add(ext4_dir_rec_len(1u8, core::ptr::null::<inode>())) as u64) as u32, blocksize as u32);
}
}
if csum_size != 0 {
ext4_initialize_dirent_tail(bh, blocksize as u32);
}
BUFFER_TRACE!();
set_buffer_uptodate(bh);
set_buffer_verified(bh);
ext4_handle_dirty_dirblock(handle, inode, bh)
}

#[unsafe(no_mangle)]
extern "C" fn ext4_init_new_dir(handle: *mut jbd2_journal_handle, dir: *mut inode, inode: *mut inode) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut block: u32 = unsafe { core::mem::zeroed() };
let mut dir_block: *mut buffer_head = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
dir_block = core::ptr::null_mut::<buffer_head>();
block = 0;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ext4_test_inode_state(inode, 6i32) != 0 {
*__slate_slot_0 = ext4_try_create_inline_dir(handle, dir, inode);
if *__slate_slot_0 < 0 && *__slate_slot_0 != 28i32.wrapping_neg() {
break '__join_0;
} else {
if *__slate_slot_0 == 0 {
break '__join_0;
}
}
}
set_nlink(inode, 2u32);
(*inode).i_size = 0;
dir_block = ext4_append(handle, inode, &mut (*core::ptr::addr_of_mut!(block)));
if IS_ERR(dir_block as *const ()) {
return PTR_ERR(dir_block as *const ()) as i32;
} else {
*__slate_slot_0 = ext4_init_dirblock(handle, inode, dir_block, (*dir).i_ino as u32, core::ptr::null_mut::<()>(), 0i32);
}
}
brelse(dir_block);
return *__slate_slot_0;
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_mkdir")]
extern "C" fn ext4_mkdir(idmap: *mut mnt_idmap, dir: *mut inode, dentry: *mut dentry, mode: u16) -> *mut dentry {
let mut __v0: *mut () = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut err2: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
err2 = 0;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
if EXT4_DIR_LINK_MAX!(dir) != 0 {
return ERR_PTR(31i32.wrapping_neg() as i64) as *mut dentry;
} else {
*__slate_slot_5 = dquot_initialize(dir);
if *__slate_slot_5 != 0 {
return ERR_PTR(*__slate_slot_5 as i64) as *mut dentry;
} else {
*__slate_slot_3 = EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(12).wrapping_add(3) as i32;
loop {
'__join_7: {
'__join_9: {
*__slate_slot_6 = ext4_new_inode_start_handle__3!(idmap, dir, mode, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), 0i32, 4i32, *__slate_slot_3);
*__slate_slot_7 = ext4_journal_current_handle();
*__slate_slot_5 = PTR_ERR(*__slate_slot_6 as *const ()) as i32;
if !IS_ERR(*__slate_slot_6 as *const ()) {
(*(*__slate_slot_6)).i_op = core::ptr::addr_of!(ext4_dir_inode_operations.0);
(*(*__slate_slot_6)).__slate_anon_43.i_fop = core::ptr::addr_of!(ext4_dir_operations);
*__slate_slot_5 = ext4_init_new_dir(*__slate_slot_7, dir, *__slate_slot_6);
if *__slate_slot_5 == 0 {
*__slate_slot_5 = ext4_mark_inode_dirty__17!(*__slate_slot_7, *__slate_slot_6);
if *__slate_slot_5 == 0 {
*__slate_slot_5 = ext4_add_entry(*__slate_slot_7, dentry, *__slate_slot_6);
}
if *__slate_slot_5 == 0 {
ext4_inc_count(dir);
ext4_update_dx_flag(dir);
*__slate_slot_5 = ext4_mark_inode_dirty__13!(*__slate_slot_7, dir);
if *__slate_slot_5 == 0 {
d_instantiate_new(dentry, *__slate_slot_6);
ext4_fc_track_create(*__slate_slot_7, dentry);
if IS_DIRSYNC!(dir) {
ext4_handle_sync(*__slate_slot_7);
break '__join_9;
} else {
break '__join_9;
}
}
}
}
clear_nlink(*__slate_slot_6);
ext4_orphan_add(*__slate_slot_7, *__slate_slot_6);
unlock_new_inode(*__slate_slot_6);
err2 = ext4_mark_inode_dirty__34!(*__slate_slot_7, *__slate_slot_6);
if unlikely__2!(err2) != 0 {
*__slate_slot_5 = err2;
}
ext4_journal_stop__7!(*__slate_slot_7);
iput(*__slate_slot_6);
break '__join_7;
}
}
if !(*__slate_slot_7).is_null() {
ext4_journal_stop__30!(*__slate_slot_7);
}
}
if *__slate_slot_5 == 28i32.wrapping_neg() {
__v1 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v1 = false;
}
if !__v1 {
break;
}
}
if *__slate_slot_5 != 0 {
__v0 = ERR_PTR(*__slate_slot_5 as i64);
} else {
__v0 = core::ptr::null_mut::<()>();
}
return __v0 as *mut dentry;
}
}
}
}

/// routine to check that the specified directory is empty (for rmdir)
#[unsafe(no_mangle)]
extern "C" fn ext4_empty_dir(inode: *mut inode) -> bool {
let mut offset: u32 = 0;
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
if ext4_has_inline_data(inode) != 0 {
let mut has_inline_data: i32 = 1;
let ret: i32 = (unsafe { empty_inline_dir(inode, core::ptr::addr_of_mut!(has_inline_data)) }) as i32;
if has_inline_data != 0 {
return ret != 0;
}
}
let sb: *mut super_block = unsafe { (*inode).i_sb };
if (unsafe { (*inode).i_size }) < (ext4_dir_rec_len(1u8, core::ptr::null::<inode>()).wrapping_add(ext4_dir_rec_len(2u8, core::ptr::null::<inode>())) as i64) {
unsafe { __ext4_error_inode(inode, c"ext4_empty_dir".as_ptr() as *const u8, 3084u32, 0u64, 0i32, c"invalid size".as_ptr() as *const u8) };
return false;
}
bh = ext4_read_dirblock!(inode, 0i32, 0i32);
if IS_ERR(bh as *const ()) {
return false;
}
de = (unsafe { (*bh).b_data }) as *mut ext4_dir_entry_2;
if ext4_check_dir_entry__2!(inode, de, bh, unsafe { (*bh).b_data }, unsafe { (*bh).b_size }, 0i32) != 0 || ((unsafe { (*de).inode }) as u64) != unsafe { (*inode).i_ino } || ((unsafe { (*de).name_len }) as i32) != 1 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 46 {
unsafe { __ext4_warning_inode(inode as *const inode, c"ext4_empty_dir".as_ptr() as *const u8, 3096u32, c"directory missing '.'".as_ptr() as *const u8) };
brelse(bh);
return false;
}
offset = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*sb).s_blocksize }) as u32);
de = ext4_next_entry(de, unsafe { (*sb).s_blocksize });
if ext4_check_dir_entry!(inode, de, bh, unsafe { (*bh).b_data }, unsafe { (*bh).b_size }, offset) != 0 || (unsafe { (*de).inode }) == 0 || ((unsafe { (*de).name_len }) as i32) != 2 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 46 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) != 46 {
unsafe { __ext4_warning_inode(inode as *const inode, c"ext4_empty_dir".as_ptr() as *const u8, 3106u32, c"directory missing '..'".as_ptr() as *const u8) };
brelse(bh);
return false;
}
let __v0: u32 = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*sb).s_blocksize }) as u32);
offset = offset.wrapping_add(__v0);
while (offset as i64) < unsafe { (*inode).i_size } {
'__slate_continue_5: {
if (offset as u64) & unsafe { (*sb).s_blocksize }.wrapping_sub(1) == 0 {
let mut lblock: u32 = 0;
brelse(bh);
lblock = offset >> (EXT4_BLOCK_SIZE_BITS!(sb) as i32);
bh = ext4_read_dirblock__3!(inode, lblock, 0i32);
if bh.is_null() {
offset = (offset as u64).wrapping_add(unsafe { (*sb).s_blocksize }) as u32;
break '__slate_continue_5;
}
if IS_ERR(bh as *const ()) {
return false;
}
}
de = unsafe { (*bh).b_data }.wrapping_offset(((offset as u64) & unsafe { (*sb).s_blocksize }.wrapping_sub(1)) as isize) as *mut ext4_dir_entry_2;
if ext4_check_dir_entry__3!(inode, de, bh, unsafe { (*bh).b_data }, unsafe { (*bh).b_size }, offset) != 0 || (unsafe { (*de).inode }) != 0 {
brelse(bh);
return false;
}
let __v6: u32 = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*sb).s_blocksize }) as u32);
offset = offset.wrapping_add(__v6);
}
}
brelse(bh);
true
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_rmdir")]
extern "C" fn ext4_rmdir(dir: *mut inode, dentry: *mut dentry) -> i32 {
let mut handle: *mut jbd2_journal_handle = unsafe { core::mem::zeroed() };
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
de = core::ptr::null_mut::<ext4_dir_entry_2>();
handle = core::ptr::null_mut::<jbd2_journal_handle>();
*__slate_slot_4 = ext4_emergency_state((*dir).i_sb);
if unlikely__2!(*__slate_slot_4) != 0 {
return *__slate_slot_4;
} else {
// Initialize quotas before so that eventual writes go in
// separate transaction
*__slate_slot_4 = dquot_initialize(dir);
if *__slate_slot_4 != 0 {
return *__slate_slot_4;
} else {
*__slate_slot_4 = dquot_initialize(d_inode(dentry as *const dentry));
if *__slate_slot_4 != 0 {
return *__slate_slot_4;
} else {
*__slate_slot_4 = 2i32.wrapping_neg();
*__slate_slot_2 = ext4_find_entry(dir, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), core::ptr::addr_of_mut!(de), core::ptr::null_mut::<i32>());
if IS_ERR(*__slate_slot_2 as *const ()) {
return PTR_ERR(*__slate_slot_2 as *const ()) as i32;
} else {
if !(*__slate_slot_2).is_null() {
*__slate_slot_3 = d_inode(dentry as *const dentry);
*__slate_slot_4 = 117i32.wrapping_neg();
if ((*de).inode as u64) == (*(*__slate_slot_3)).i_ino {
*__slate_slot_4 = 39i32.wrapping_neg();
if ext4_empty_dir(*__slate_slot_3) {
handle = ext4_journal_start__43!(dir, 4i32, EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }));
if IS_ERR(handle as *const ()) {
*__slate_slot_4 = PTR_ERR(handle as *const ()) as i32;
handle = core::ptr::null_mut::<jbd2_journal_handle>();
} else {
if IS_DIRSYNC!(dir) {
ext4_handle_sync(handle);
}
*__slate_slot_4 = ext4_delete_entry(handle, dir, de, *__slate_slot_2);
if *__slate_slot_4 == 0 {
if !EXT4_DIR_LINK_EMPTY!(*__slate_slot_3) {
__ext4_warning_inode(*__slate_slot_3 as *const inode, c"ext4_rmdir".as_ptr() as *const u8, 3194u32, c"empty directory '%.*s' has too many links (%u)".as_ptr() as *const u8, (*dentry).__slate_anon_4.d_name.__slate_anon_0.__slate_anon_0.len, (*dentry).__slate_anon_4.d_name.name, (*(*__slate_slot_3)).__slate_anon_12.i_nlink);
}
inode_inc_iversion(*__slate_slot_3);
clear_nlink(*__slate_slot_3);
// There's no need to set i_disksize: the fact that i_nlink is
// zero will ensure that the right thing happens during any
// recovery.
(*(*__slate_slot_3)).i_size = 0;
ext4_orphan_add(handle, *__slate_slot_3);
inode_set_mtime_to_ts(dir, inode_set_ctime_current(dir));
inode_set_ctime_current(*__slate_slot_3);
*__slate_slot_4 = ext4_mark_inode_dirty__25!(handle, *__slate_slot_3);
if *__slate_slot_4 == 0 {
ext4_dec_count(dir);
ext4_update_dx_flag(dir);
ext4_fc_track_unlink(handle, dentry);
*__slate_slot_4 = ext4_mark_inode_dirty__75!(handle, dir);
// VFS negative dentries are incompatible with Encoding and
// Case-insensitiveness. Eventually we'll want avoid
// invalidating the dentries here, alongside with returning the
// negative dentries at ext4_lookup(), when it is better
// supported by the VFS for the CI case.
}
}
}
}
}
}
brelse(*__slate_slot_2);
if !handle.is_null() {
ext4_journal_stop__69!(handle);
}
return *__slate_slot_4;
}
}
}
}
}
}

/// NULL during fast_commit recovery
#[unsafe(no_mangle)]
extern "C" fn __ext4_unlink(dir: *mut inode, d_name: *const qstr, inode: *mut inode, dentry: *mut dentry) -> i32 {
let mut skip_remove_dentry: i32 = unsafe { core::mem::zeroed() };
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
retval = 2i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
de = core::ptr::null_mut::<ext4_dir_entry_2>();
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
skip_remove_dentry = 0;
// Keep this outside the transaction; it may have to set up the
// directory's encryption key, which isn't GFP_NOFS-safe.
*__slate_slot_3 = ext4_find_entry(dir, d_name, core::ptr::addr_of_mut!(de), core::ptr::null_mut::<i32>());
if IS_ERR(*__slate_slot_3 as *const ()) {
return PTR_ERR(*__slate_slot_3 as *const ()) as i32;
} else {
if (*__slate_slot_3).is_null() {
return 2i32.wrapping_neg();
} else {
'__join_0: {
if ((*de).inode as u64) != (*inode).i_ino {
// It's okay if we find dont find dentry which matches
// the inode. That's because it might have gotten
// renamed to a different inode number
if ((*EXT4_SB((*inode).i_sb)).s_mount_state as i32) & 32 != 0 {
skip_remove_dentry = 1;
} else {
break '__join_0;
}
}
*__slate_slot_1 = ext4_journal_start__48!(dir, 4i32, EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }));
if IS_ERR(*__slate_slot_1 as *const ()) {
retval = PTR_ERR(*__slate_slot_1 as *const ()) as i32;
} else {
if IS_DIRSYNC!(dir) {
ext4_handle_sync(*__slate_slot_1);
}
'__join_1: {
if skip_remove_dentry == 0 {
retval = ext4_delete_entry(*__slate_slot_1, dir, de, *__slate_slot_3);
if retval != 0 {
break '__join_1;
} else {
inode_set_mtime_to_ts(dir, inode_set_ctime_current(dir));
ext4_update_dx_flag(dir);
retval = ext4_mark_inode_dirty__48!(*__slate_slot_1, dir);
if retval != 0 {
break '__join_1;
}
}
} else {
retval = 0;
}
if (*inode).__slate_anon_12.i_nlink == 0 {
__ext4_warning_inode(inode as *const inode, c"__ext4_unlink".as_ptr() as *const u8, 3285u32, c"Deleting file '%.*s' with no links".as_ptr() as *const u8, (*d_name).__slate_anon_0.__slate_anon_0.len, (*d_name).name);
} else {
drop_nlink(inode);
}
if (*inode).__slate_anon_12.i_nlink == 0 {
ext4_orphan_add(*__slate_slot_1, inode);
}
inode_set_ctime_current(inode);
retval = ext4_mark_inode_dirty__77!(*__slate_slot_1, inode);
if !dentry.is_null() && retval == 0 {
ext4_fc_track_unlink(*__slate_slot_1, dentry);
}
}
ext4_journal_stop!(*__slate_slot_1);
}
}
brelse(*__slate_slot_3);
return retval;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_unlink")]
extern "C" fn ext4_unlink(dir: *mut inode, dentry: *mut dentry) -> i32 {
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = ext4_emergency_state((*dir).i_sb);
if unlikely__2!(*__slate_slot_0) != 0 {
return *__slate_slot_0;
} else {
trace_ext4_unlink_enter(dir, dentry);
// Initialize quotas before so that eventual writes go
// in separate transaction
*__slate_slot_0 = dquot_initialize(dir);
if *__slate_slot_0 == 0 {
*__slate_slot_0 = dquot_initialize(d_inode(dentry as *const dentry));
if *__slate_slot_0 == 0 {
*__slate_slot_0 = __ext4_unlink(dir, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), d_inode(dentry as *const dentry), dentry);
// VFS negative dentries are incompatible with Encoding and
// Case-insensitiveness. Eventually we'll want avoid
// invalidating the dentries here, alongside with returning the
// negative dentries at ext4_lookup(), when it is  better
// supported by the VFS for the CI case.
}
}
trace_ext4_unlink_exit(dentry, *__slate_slot_0);
return *__slate_slot_0;
}
}
}

fn ext4_init_symlink_block(handle: *mut jbd2_journal_handle, inode: *mut inode, disk_link: &fscrypt_str) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut kaddr: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut u8 = core::ptr::addr_of_mut!(kaddr) as *mut *mut u8;
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
err = 0;
*__slate_slot_2 = ext4_bread(handle, inode, 0u32, 1i32);
if IS_ERR(*__slate_slot_2 as *const ()) {
return PTR_ERR(*__slate_slot_2 as *const ()) as i32;
} else {
BUFFER_TRACE!();
err = ext4_journal_get_write_access__27!(handle, unsafe { (*inode).i_sb }, *__slate_slot_2, 1i32);
if err == 0 {
*__slate_slot_1 = (*(*__slate_slot_2)).b_data;
memcpy(*__slate_slot_1 as *mut (), (*(disk_link as *const fscrypt_str)).name as *const (), (*(disk_link as *const fscrypt_str)).len as u64);
(*inode).i_size = unsafe { (*(disk_link as *const fscrypt_str)).len }.wrapping_sub(1) as i64;
(*EXT4_I!(inode)).i_disksize = (*inode).i_size;
err = ext4_handle_dirty_metadata__11!(handle, inode, *__slate_slot_2);
}
brelse(*__slate_slot_2);
return err;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_symlink")]
extern "C" fn ext4_symlink(idmap: *mut mnt_idmap, dir: *mut inode, dentry: *mut dentry, symname: *const u8) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut disk_link: core::mem::MaybeUninit<fscrypt_str> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut fscrypt_str = core::ptr::addr_of_mut!(disk_link) as *mut fscrypt_str;
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut len: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
len = strlen(symname) as i32;
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
retries = 0;
*__slate_slot_5 = ext4_emergency_state((*dir).i_sb);
if unlikely__2!(*__slate_slot_5) != 0 {
return *__slate_slot_5;
} else {
*__slate_slot_5 = fscrypt_prepare_symlink(dir, symname, len as u32, (*(*dir).i_sb).s_blocksize as u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
if *__slate_slot_5 != 0 {
return *__slate_slot_5;
} else {
*__slate_slot_5 = dquot_initialize(dir);
if *__slate_slot_5 != 0 {
return *__slate_slot_5;
} else {
// EXT4_INDEX_EXTRA_TRANS_BLOCKS for addition of entry into the
// directory. +3 for inode, inode bitmap, group descriptor allocation.
// EXT4_DATA_TRANS_BLOCKS for the data block allocation and
// modification.
*__slate_slot_3 = EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(12).wrapping_add(3) as i32;
loop {
'__join_6: {
*__slate_slot_6 = ext4_new_inode_start_handle__2!(idmap, dir, 41471i32, core::ptr::addr_of!((*dentry).__slate_anon_4.d_name), 0i32, 4i32, *__slate_slot_3);
*__slate_slot_7 = ext4_journal_current_handle();
if IS_ERR(*__slate_slot_6 as *const ()) {
if !(*__slate_slot_7).is_null() {
ext4_journal_stop__70!(*__slate_slot_7);
}
*__slate_slot_5 = PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
'__join_9: {
if IS_ENCRYPTED!(*__slate_slot_6) != 0 {
*__slate_slot_5 = fscrypt_encrypt_symlink(*__slate_slot_6, symname, len as u32, &(*core::ptr::addr_of_mut!(*__slate_slot_2)));
if *__slate_slot_5 != 0 {
break '__join_9;
} else {
(*(*__slate_slot_6)).i_op = core::ptr::addr_of!(ext4_encrypted_symlink_inode_operations);
}
} else {
if (*__slate_slot_2).len > 60u32 {
(*(*__slate_slot_6)).i_op = core::ptr::addr_of!(ext4_symlink_inode_operations);
} else {
(*(*__slate_slot_6)).i_op = core::ptr::addr_of!(ext4_fast_symlink_inode_operations);
}
}
if (*__slate_slot_2).len > 60u32 {
// alloc symlink block and fill it
*__slate_slot_5 = ext4_init_symlink_block(*__slate_slot_7, *__slate_slot_6, &(*core::ptr::addr_of_mut!(*__slate_slot_2)));
if *__slate_slot_5 != 0 {
break '__join_9;
}
} else {
// clear the extent format for fast symlink
ext4_clear_inode_flag(*__slate_slot_6, 19i32);
memcpy((core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_6)).i_data) as *mut u8) as *mut (), (*__slate_slot_2).name as *const (), (*__slate_slot_2).len as u64);
(*(*__slate_slot_6)).i_size = (*__slate_slot_2).len.wrapping_sub(1) as i64;
(*EXT4_I!(*__slate_slot_6)).i_disksize = (*(*__slate_slot_6)).i_size;
if IS_ENCRYPTED!(*__slate_slot_6) == 0 {
inode_set_cached_link(*__slate_slot_6, core::ptr::addr_of_mut!((*EXT4_I!(*__slate_slot_6)).i_data) as *mut u8, (*(*__slate_slot_6)).i_size as i32);
}
}
*__slate_slot_5 = ext4_add_nondir(*__slate_slot_7, dentry, &mut (*core::ptr::addr_of_mut!(*__slate_slot_6)));
if !(*__slate_slot_7).is_null() {
ext4_journal_stop__84!(*__slate_slot_7);
}
iput(*__slate_slot_6);
break '__join_6;
}
clear_nlink(*__slate_slot_6);
ext4_mark_inode_dirty__26!(*__slate_slot_7, *__slate_slot_6);
ext4_orphan_add(*__slate_slot_7, *__slate_slot_6);
unlock_new_inode(*__slate_slot_6);
if !(*__slate_slot_7).is_null() {
ext4_journal_stop__26!(*__slate_slot_7);
}
iput(*__slate_slot_6);
}
}
if *__slate_slot_5 == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
if (*__slate_slot_2).name != (symname as *mut u8) {
kfree((*__slate_slot_2).name as *const ());
}
return *__slate_slot_5;
}
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn __ext4_link(dir: *mut inode, inode: *mut inode, d_name: *const qstr, dentry: *mut dentry) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
loop {
*__slate_slot_3 = ext4_journal_start__6!(dir, 4i32, EXT4_DATA_TRANS_BLOCKS!(unsafe { (*dir).i_sb }).wrapping_add(12).wrapping_add(1));
if IS_ERR(*__slate_slot_3 as *const ()) {
return PTR_ERR(*__slate_slot_3 as *const ()) as i32;
} else {
if IS_DIRSYNC!(dir) {
ext4_handle_sync(*__slate_slot_3);
}
inode_set_ctime_current(inode);
ext4_inc_count(inode);
*__slate_slot_2 = __ext4_add_entry(*__slate_slot_3, dir, d_name, inode);
if *__slate_slot_2 == 0 {
*__slate_slot_2 = ext4_mark_inode_dirty__78!(*__slate_slot_3, inode);
// this can happen only for tmpfile being
// linked the first time
if (*inode).__slate_anon_12.i_nlink == 1 {
ext4_orphan_del(*__slate_slot_3, inode);
}
if !dentry.is_null() {
ext4_fc_track_link(*__slate_slot_3, inode, dentry);
}
} else {
drop_nlink(inode);
}
ext4_journal_stop__77!(*__slate_slot_3);
if *__slate_slot_2 == 28i32.wrapping_neg() {
__v0 = ext4_should_retry_alloc((*dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break;
}
}
}
return *__slate_slot_2;
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_link")]
extern "C" fn ext4_link(old_dentry: *mut dentry, dir: *mut inode, dentry: *mut dentry) -> i32 {
let inode: *mut inode = d_inode(old_dentry as *const dentry);
let mut err: i32 = 0;
if (unsafe { (*inode).__slate_anon_12.i_nlink }) >= 65000 {
return 31i32.wrapping_neg();
}
err = fscrypt_prepare_link(old_dentry, dir, dentry);
if err != 0 {
return err;
}
let __v0: bool = ext4_test_inode_flag(dir, 29i32) != 0 && !projid_eq(unsafe { (*EXT4_I!(dir)).i_projid }, unsafe { (*EXT4_I!(unsafe { (*old_dentry).d_inode })).i_projid });
if __v0 {
return 18i32.wrapping_neg();
}
err = unsafe { dquot_initialize(dir) };
if err != 0 {
return err;
}
err = __ext4_link(dir, inode, unsafe { core::ptr::addr_of!((*dentry).__slate_anon_4.d_name) }, dentry);
if err == 0 {
unsafe { ihold(inode) };
unsafe { d_instantiate(dentry, inode) };
}
err
}

/// Try to find buffer head where contains the parent block.
/// It should be the inode block if it is inlined or the 1st block
/// if it is a normal dir.
fn ext4_get_first_dir_block(handle: *mut jbd2_journal_handle, inode: *mut inode, retval: *mut i32, parent_de: *mut *mut ext4_dir_entry_2, inlined: *mut i32) -> *mut buffer_head {
if ext4_has_inline_data(inode) == 0 {
let mut de: *mut ext4_dir_entry_2 = unsafe { core::mem::zeroed() };
let mut offset: u32 = 0;
let bh: *mut buffer_head = ext4_read_dirblock__5!(inode, 0i32, 0i32);
if IS_ERR(bh as *const ()) {
unsafe {
*retval = PTR_ERR(bh as *const ()) as i32;
}
return core::ptr::null_mut::<buffer_head>();
}
de = (unsafe { (*bh).b_data }) as *mut ext4_dir_entry_2;
if ext4_check_dir_entry__11!(inode, de, bh, unsafe { (*bh).b_data }, unsafe { (*bh).b_size }, 0i32) != 0 || ((unsafe { (*de).inode }) as u64) != unsafe { (*inode).i_ino } || ((unsafe { (*de).name_len }) as i32) != 1 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 46 {
unsafe { __ext4_error_inode(inode, c"ext4_get_first_dir_block".as_ptr() as *const u8, 3550u32, 0u64, 0i32, c"directory missing '.'".as_ptr() as *const u8) };
brelse(bh);
unsafe {
*retval = 117i32.wrapping_neg();
}
return core::ptr::null_mut::<buffer_head>();
}
offset = ext4_rec_len_from_disk(unsafe { (*de).rec_len }, (unsafe { (*(*inode).i_sb).s_blocksize }) as u32);
de = ext4_next_entry(de, unsafe { (*(*inode).i_sb).s_blocksize });
if ext4_check_dir_entry__5!(inode, de, bh, unsafe { (*bh).b_data }, unsafe { (*bh).b_size }, offset) != 0 || (unsafe { (*de).inode }) == 0 || ((unsafe { (*de).name_len }) as i32) != 2 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 46 || ((unsafe { *((*de).name.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) != 46 {
unsafe { __ext4_error_inode(inode, c"ext4_get_first_dir_block".as_ptr() as *const u8, 3562u32, 0u64, 0i32, c"directory missing '..'".as_ptr() as *const u8) };
brelse(bh);
unsafe {
*retval = 117i32.wrapping_neg();
}
return core::ptr::null_mut::<buffer_head>();
}
unsafe {
*parent_de = de;
}
return bh;
}
unsafe {
*inlined = 1;
}
unsafe { ext4_get_first_inline_block(inode, parent_de, retval) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_renament {
    dir: *mut inode,
    dentry: *mut dentry,
    inode: *mut inode,
    is_dir: bool,
    dir_nlink_delta: i32,
    /// entry for "dentry"
    bh: *mut buffer_head,
    de: *mut ext4_dir_entry_2,
    inlined: i32,
    /// entry for ".." in inode if it's a directory
    dir_bh: *mut buffer_head,
    parent_de: *mut ext4_dir_entry_2,
    dir_inlined: i32,
}


fn ext4_rename_dir_prepare(handle: *mut jbd2_journal_handle, ent: *mut ext4_renament, is_cross: bool) -> i32 {
let mut retval: i32 = 0;
unsafe {
(*ent).is_dir = true;
}
if !is_cross {
return 0;
}
unsafe {
(*ent).dir_bh = ext4_get_first_dir_block(handle, (*ent).inode, core::ptr::addr_of_mut!(retval), core::ptr::addr_of_mut!((*ent).parent_de), core::ptr::addr_of_mut!((*ent).dir_inlined));
}
if unsafe { (*ent).dir_bh }.is_null() {
return retval;
}
if ((unsafe { (*(*ent).parent_de).inode }) as u64) != unsafe { (*(*ent).dir).i_ino } {
return 117i32.wrapping_neg();
}
BUFFER_TRACE!();
ext4_journal_get_write_access__42!(handle, unsafe { (*(*ent).dir).i_sb }, unsafe { (*ent).dir_bh }, 1i32)
}

fn ext4_rename_dir_finish(handle: *mut jbd2_journal_handle, ent: *mut ext4_renament, dir_ino: u32) -> i32 {
let mut retval: i32 = 0;
if unsafe { (*ent).dir_bh }.is_null() {
return 0;
}
unsafe {
(*(*ent).parent_de).inode = dir_ino;
}
BUFFER_TRACE!();
if (unsafe { (*ent).dir_inlined }) == 0 {
if is_dx!(unsafe { (*ent).inode }) {
retval = ext4_handle_dirty_dx_node(handle, unsafe { (*ent).inode }, unsafe { (*ent).dir_bh });
} else {
retval = ext4_handle_dirty_dirblock(handle, unsafe { (*ent).inode }, unsafe { (*ent).dir_bh });
}
} else {
retval = ext4_mark_inode_dirty__71!(handle, unsafe { (*ent).inode });
}
if retval != 0 {
ext4_std_error__51!(unsafe { (*(*ent).dir).i_sb }, retval);
return retval;
}
0
}

fn ext4_setent(handle: *mut jbd2_journal_handle, ent: *mut ext4_renament, ino: u32, file_type: u32) -> i32 {
let mut retval: i32 = 0;
let mut retval2: i32 = 0;
BUFFER_TRACE!();
retval = ext4_journal_get_write_access__3!(handle, unsafe { (*(*ent).dir).i_sb }, unsafe { (*ent).bh }, 1i32);
if retval != 0 {
return retval;
}
unsafe {
(*(*ent).de).inode = ino;
}
if ext4_has_feature_filetype(unsafe { (*(*ent).dir).i_sb }) {
unsafe {
(*(*ent).de).file_type = file_type as u8;
}
}
inode_inc_iversion(unsafe { (*ent).dir });
inode_set_mtime_to_ts(unsafe { (*ent).dir }, unsafe { inode_set_ctime_current((*ent).dir) });
retval = ext4_mark_inode_dirty__38!(handle, unsafe { (*ent).dir });
BUFFER_TRACE!();
if (unsafe { (*ent).inlined }) == 0 {
retval2 = ext4_handle_dirty_dirblock(handle, unsafe { (*ent).dir }, unsafe { (*ent).bh });
if unlikely__2!(retval2) != 0 {
ext4_std_error__22!(unsafe { (*(*ent).dir).i_sb }, retval2);
return retval2;
}
}
retval
}

fn ext4_resetent(handle: *mut jbd2_journal_handle, ent: &ext4_renament, ino: u32, file_type: u32) {
let mut old: ext4_renament = unsafe { *(ent as *const ext4_renament) };
let mut retval: i32 = 0;
// old->de could have moved from under us during make indexed dir,
// so the old->de may no longer valid and need to find it again
// before reset old inode info.
old.bh = ext4_find_entry(old.dir, unsafe { core::ptr::addr_of!((*old.dentry).__slate_anon_4.d_name) }, core::ptr::addr_of_mut!(old.de), core::ptr::addr_of_mut!(old.inlined));
if IS_ERR(old.bh as *const ()) {
retval = PTR_ERR(old.bh as *const ()) as i32;
}
if old.bh.is_null() {
retval = 2i32.wrapping_neg();
}
if retval != 0 {
ext4_std_error__39!(unsafe { (*old.dir).i_sb }, retval);
return;
}
ext4_setent(handle, core::ptr::addr_of_mut!(old), ino, file_type);
brelse(old.bh);
}

fn ext4_find_delete_entry(handle: *mut jbd2_journal_handle, dir: *mut inode, d_name: *const qstr) -> i32 {
let mut retval: i32 = 2i32.wrapping_neg();
let mut de: *mut ext4_dir_entry_2 = core::ptr::null_mut::<ext4_dir_entry_2>();
let bh: *mut buffer_head = ext4_find_entry(dir, d_name, core::ptr::addr_of_mut!(de), core::ptr::null_mut::<i32>());
if IS_ERR(bh as *const ()) {
return PTR_ERR(bh as *const ()) as i32;
}
if !bh.is_null() {
retval = ext4_delete_entry(handle, dir, de, bh);
brelse(bh);
}
retval
}

fn ext4_rename_delete(handle: *mut jbd2_journal_handle, ent: &ext4_renament, force_reread: i32) {
let mut retval: i32 = 0;
// ent->de could have moved from under us during htree split, so make
// sure that we are deleting the right entry.  We might also be pointing
// to a stale entry in the unused part of ent->bh so just checking inum
// and the name isn't enough.
let __v0: bool = ((unsafe { (*(*(ent as *const ext4_renament)).de).inode }) as u64) != unsafe { (*(*(ent as *const ext4_renament)).inode).i_ino } || (((unsafe { (*(*(ent as *const ext4_renament)).de).name_len }) as i32) as u32) != unsafe { (*(*(ent as *const ext4_renament)).dentry).__slate_anon_4.d_name.__slate_anon_0.__slate_anon_0.len } || (unsafe { strncmp(((*(*(ent as *const ext4_renament)).de).name.as_mut_ptr() as *mut u8) as *const u8, (*(*(ent as *const ext4_renament)).dentry).__slate_anon_4.d_name.name as *const u8, (*(*(ent as *const ext4_renament)).de).name_len as u64) }) != 0;
if __v0 || force_reread != 0 {
retval = ext4_find_delete_entry(handle, unsafe { (*(ent as *const ext4_renament)).dir }, unsafe { core::ptr::addr_of!((*(*(ent as *const ext4_renament)).dentry).__slate_anon_4.d_name) });
} else {
retval = ext4_delete_entry(handle, unsafe { (*(ent as *const ext4_renament)).dir }, unsafe { (*(ent as *const ext4_renament)).de }, unsafe { (*(ent as *const ext4_renament)).bh });
if retval == 2i32.wrapping_neg() {
retval = ext4_find_delete_entry(handle, unsafe { (*(ent as *const ext4_renament)).dir }, unsafe { core::ptr::addr_of!((*(*(ent as *const ext4_renament)).dentry).__slate_anon_4.d_name) });
}
}
if retval != 0 {
unsafe { __ext4_warning_inode((*(ent as *const ext4_renament)).dir as *const inode, c"ext4_rename_delete".as_ptr() as *const u8, 3741u32, c"Deleting old file: nlink %d, error=%d".as_ptr() as *const u8, (*(*(ent as *const ext4_renament)).dir).__slate_anon_12.i_nlink, retval) };
}
}

fn ext4_update_dir_count(handle: *mut jbd2_journal_handle, ent: *mut ext4_renament) {
if (unsafe { (*ent).dir_nlink_delta }) != 0 {
if (unsafe { (*ent).dir_nlink_delta }) == 1i32.wrapping_neg() {
ext4_dec_count(unsafe { (*ent).dir });
} else {
ext4_inc_count(unsafe { (*ent).dir });
}
ext4_mark_inode_dirty__28!(handle, unsafe { (*ent).dir });
}
}

fn ext4_whiteout_for_rename(idmap: *mut mnt_idmap, ent: *mut ext4_renament, mut credits: i32, h: &mut *mut jbd2_journal_handle) -> *mut inode {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut wh: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut inode = core::ptr::addr_of_mut!(wh) as *mut *mut inode;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
retries = 0;
// for inode block, sb block, group summaries,
// and inode bitmap
let __v3: u32 = (EXT4_MAXQUOTAS_TRANS_BLOCKS!(unsafe { (*(*ent).dir).i_sb }) as u32).wrapping_add(6).wrapping_add(4);
credits = (credits as u32).wrapping_add(__v3) as i32;
'__join_0: {
loop {
*__slate_slot_6 = ext4_new_inode_start_handle__5!(idmap, unsafe { (*ent).dir }, 8192i32, core::ptr::addr_of!((*(*ent).dentry).__slate_anon_4.d_name), 0i32, 4i32, credits);
*__slate_slot_5 = ext4_journal_current_handle();
if IS_ERR(*__slate_slot_6 as *const ()) {
if !(*__slate_slot_5).is_null() {
ext4_journal_stop__52!(*__slate_slot_5);
}
if PTR_ERR(*__slate_slot_6 as *const ()) == (28i32.wrapping_neg() as i64) {
__v0 = ext4_should_retry_alloc((*(*ent).dir).i_sb, core::ptr::addr_of_mut!(retries)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break '__join_0;
}
} else {
break;
}
}
*(h as *mut *mut jbd2_journal_handle) = *__slate_slot_5;
init_special_inode(*__slate_slot_6, (*(*__slate_slot_6)).i_mode, 0u32);
(*(*__slate_slot_6)).i_op = core::ptr::addr_of!(ext4_special_inode_operations.0);
}
return *__slate_slot_6;
}
}

/// Anybody can rename anything with this: the permission checks are left to the
/// higher-level routines.
///
/// n.b.  old_{dentry,inode) refers to the source dentry/inode
/// while new_{dentry,inode) refers to the destination dentry/inode
/// This comes from rename(const char *oldpath, const char *newpath)
fn ext4_rename(idmap: *mut mnt_idmap, old_dir: *mut inode, old_dentry: *mut dentry, new_dir: *mut inode, new_dentry: *mut dentry, flags: u32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
// If we're renaming a file within an inline_data dir and adding or
// setting the new dirent causes a conversion from inline_data to
// extents/blockmap, we need to force the dirent delete code to
// re-read the directory, or else we end up trying to delete a dirent
// from what is now the extent tree root (or a block map).
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut old_file_type: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(old_file_type) as *mut u8;
let mut credits: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(credits) as *mut i32;
let mut whiteout: *mut inode = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut force_reread: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(force_reread) as *mut i32;
let mut new: core::mem::MaybeUninit<ext4_renament> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut ext4_renament = core::ptr::addr_of_mut!(new) as *mut ext4_renament;
let mut old: core::mem::MaybeUninit<ext4_renament> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut ext4_renament = core::ptr::addr_of_mut!(old) as *mut ext4_renament;
let mut handle: *mut jbd2_journal_handle = unsafe { core::mem::zeroed() };
unsafe {
handle = core::ptr::null_mut::<jbd2_journal_handle>();
core::ptr::write(__slate_slot_13, {
let mut __t0: ext4_renament = core::mem::zeroed();
__t0.dir = old_dir;
__t0.dentry = old_dentry;
__t0.inode = d_inode(old_dentry as *const dentry);
__t0
});
core::ptr::write(__slate_slot_12, {
let mut __t1: ext4_renament = core::mem::zeroed();
__t1.dir = new_dir;
__t1.dentry = new_dentry;
__t1.inode = d_inode(new_dentry as *const dentry);
__t1
});
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
whiteout = core::ptr::null_mut::<inode>();
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if !(*__slate_slot_12).inode.is_null() && (*(*__slate_slot_12).inode).__slate_anon_12.i_nlink == 0 {
__ext4_error_inode((*__slate_slot_12).inode, c"ext4_rename".as_ptr() as *const u8, 3822u32, 0u64, 0i32, c"target of rename is already freed".as_ptr() as *const u8);
return 117i32.wrapping_neg();
} else {
if ext4_test_inode_flag(new_dir, 29i32) != 0 {
__v6 = !projid_eq((*EXT4_I!(new_dir)).i_projid, (*EXT4_I!(unsafe { (*old_dentry).d_inode })).i_projid);
} else {
__v6 = false;
}
if __v6 {
return 18i32.wrapping_neg();
} else {
*__slate_slot_10 = dquot_initialize((*__slate_slot_13).dir);
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
*__slate_slot_10 = dquot_initialize((*__slate_slot_13).inode);
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
*__slate_slot_10 = dquot_initialize((*__slate_slot_12).dir);
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
} else {
// Initialize quotas before so that eventual writes go
// in separate transaction
if !(*__slate_slot_12).inode.is_null() {
*__slate_slot_10 = dquot_initialize((*__slate_slot_12).inode);
if *__slate_slot_10 != 0 {
return *__slate_slot_10;
}
}
(*__slate_slot_13).bh = ext4_find_entry((*__slate_slot_13).dir, core::ptr::addr_of!((*(*__slate_slot_13).dentry).__slate_anon_4.d_name), core::ptr::addr_of_mut!((*__slate_slot_13).de), core::ptr::addr_of_mut!((*__slate_slot_13).inlined));
if IS_ERR((*__slate_slot_13).bh as *const ()) {
return PTR_ERR((*__slate_slot_13).bh as *const ()) as i32;
} else {
'__join_0: {
// Check for inode number is _not_ due to possible IO errors.
// We might rmdir the source, keep it as pwd of some process
// and merrily kill the link to whatever was created under the
// same name. Goodbye sticky bit ;-<
*__slate_slot_10 = 2i32.wrapping_neg();
if !((*__slate_slot_13).bh.is_null() || ((*(*__slate_slot_13).de).inode as u64) != (*(*__slate_slot_13).inode).i_ino) {
(*__slate_slot_12).bh = ext4_find_entry((*__slate_slot_12).dir, core::ptr::addr_of!((*(*__slate_slot_12).dentry).__slate_anon_4.d_name), core::ptr::addr_of_mut!((*__slate_slot_12).de), core::ptr::addr_of_mut!((*__slate_slot_12).inlined));
if IS_ERR((*__slate_slot_12).bh as *const ()) {
*__slate_slot_10 = PTR_ERR((*__slate_slot_12).bh as *const ()) as i32;
(*__slate_slot_12).bh = core::ptr::null_mut::<buffer_head>();
} else {
if !(*__slate_slot_12).bh.is_null() {
if (*__slate_slot_12).inode.is_null() {
brelse((*__slate_slot_12).bh);
(*__slate_slot_12).bh = core::ptr::null_mut::<buffer_head>();
}
}
if !(*__slate_slot_12).inode.is_null() {
__v5 = (*EXT4_SB((*(*__slate_slot_12).dir).i_sb)).s_mount_opt & 65536 == 0;
} else {
__v5 = false;
}
if __v5 {
ext4_alloc_da_blocks((*__slate_slot_13).inode);
}
*__slate_slot_8 = 2u32.wrapping_mul(EXT4_DATA_TRANS_BLOCKS!(unsafe { (*(*__slate_slot_13).dir).i_sb })).wrapping_add(12).wrapping_add(2) as i32;
if flags & 4u32 == 0 {
handle = ext4_journal_start__12!((*__slate_slot_13).dir, 4i32, *__slate_slot_8);
if IS_ERR(handle as *const ()) {
*__slate_slot_10 = PTR_ERR(handle as *const ()) as i32;
break '__join_0;
}
} else {
whiteout = ext4_whiteout_for_rename(idmap, core::ptr::addr_of_mut!(*__slate_slot_13), *__slate_slot_8, &mut (*core::ptr::addr_of_mut!(handle)));
if IS_ERR(whiteout as *const ()) {
*__slate_slot_10 = PTR_ERR(whiteout as *const ()) as i32;
break '__join_0;
}
}
*__slate_slot_7 = (*(*__slate_slot_13).de).file_type;
if IS_DIRSYNC!((*__slate_slot_13).dir) || IS_DIRSYNC!((*__slate_slot_12).dir) {
ext4_handle_sync(handle);
}
'__join_5: {
if S_ISDIR!(unsafe { (*(*__slate_slot_13).inode).i_mode }) {
if !(*__slate_slot_12).inode.is_null() {
*__slate_slot_10 = 39i32.wrapping_neg();
if !ext4_empty_dir((*__slate_slot_12).inode) {
break '__join_5;
}
} else {
*__slate_slot_10 = 31i32.wrapping_neg();
if (*__slate_slot_12).dir != (*__slate_slot_13).dir {
__v4 = EXT4_DIR_LINK_MAX!((*__slate_slot_12).dir) != 0;
} else {
__v4 = false;
}
if __v4 {
break '__join_5;
}
}
*__slate_slot_10 = ext4_rename_dir_prepare(handle, core::ptr::addr_of_mut!(*__slate_slot_13), (*__slate_slot_12).dir != (*__slate_slot_13).dir);
if *__slate_slot_10 != 0 {
break '__join_5;
}
}
if (*(*__slate_slot_12).dir).i_ino == (*(*__slate_slot_13).dir).i_ino {
__v3 = ext4_test_inode_flag((*__slate_slot_12).dir, 28i32) != 0;
} else {
__v3 = false;
}
*__slate_slot_11 = __v3 as i32;
if !whiteout.is_null() {
// Do this before adding a new entry, so the old entry is sure
// to be still pointing to the valid old entry.
*__slate_slot_10 = ext4_setent(handle, core::ptr::addr_of_mut!(*__slate_slot_13), (*whiteout).i_ino as u32, 3u32);
if *__slate_slot_10 != 0 {
break '__join_5;
} else {
*__slate_slot_10 = ext4_mark_inode_dirty__19!(handle, whiteout);
if unlikely__2!(*__slate_slot_10) != 0 {
break '__join_5;
}
}
}
if (*__slate_slot_12).bh.is_null() {
*__slate_slot_10 = ext4_add_entry(handle, (*__slate_slot_12).dentry, (*__slate_slot_13).inode);
if *__slate_slot_10 != 0 {
break '__join_5;
}
} else {
*__slate_slot_10 = ext4_setent(handle, core::ptr::addr_of_mut!(*__slate_slot_12), (*(*__slate_slot_13).inode).i_ino as u32, *__slate_slot_7 as u32);
if *__slate_slot_10 != 0 {
break '__join_5;
}
}
if *__slate_slot_11 != 0 {
*__slate_slot_11 = (ext4_test_inode_flag((*__slate_slot_12).dir, 28i32) == 0) as i32;
}
// Like most other Unix systems, set the ctime for inodes on a
// rename.
inode_set_ctime_current((*__slate_slot_13).inode);
*__slate_slot_10 = ext4_mark_inode_dirty__30!(handle, (*__slate_slot_13).inode);
if unlikely__2!(*__slate_slot_10) == 0 {
if whiteout.is_null() {
// ok, that's it
ext4_rename_delete(handle, &(*core::ptr::addr_of_mut!(*__slate_slot_13)), *__slate_slot_11);
}
if !(*__slate_slot_12).inode.is_null() {
ext4_dec_count((*__slate_slot_12).inode);
inode_set_ctime_current((*__slate_slot_12).inode);
}
inode_set_mtime_to_ts((*__slate_slot_13).dir, inode_set_ctime_current((*__slate_slot_13).dir));
ext4_update_dx_flag((*__slate_slot_13).dir);
if (*__slate_slot_13).is_dir {
*__slate_slot_10 = ext4_rename_dir_finish(handle, core::ptr::addr_of_mut!(*__slate_slot_13), (*(*__slate_slot_12).dir).i_ino as u32);
if *__slate_slot_10 != 0 {
break '__join_5;
} else {
ext4_dec_count((*__slate_slot_13).dir);
if !(*__slate_slot_12).inode.is_null() {
// checked ext4_empty_dir above, can't have another
// parent, ext4_dec_count() won't work for many-linked
// dirs
clear_nlink((*__slate_slot_12).inode);
} else {
ext4_inc_count((*__slate_slot_12).dir);
ext4_update_dx_flag((*__slate_slot_12).dir);
*__slate_slot_10 = ext4_mark_inode_dirty__23!(handle, (*__slate_slot_12).dir);
if unlikely__2!(*__slate_slot_10) != 0 {
break '__join_5;
}
}
}
}
*__slate_slot_10 = ext4_mark_inode_dirty__60!(handle, (*__slate_slot_13).dir);
if unlikely__2!(*__slate_slot_10) == 0 {
if (*__slate_slot_13).is_dir {
// We disable fast commits here that's because the
// replay code is not yet capable of changing dot dot
// dirents in directories.
ext4_fc_mark_ineligible((*(*__slate_slot_13).inode).i_sb, 6i32, handle);
} else {
sb = (*(*__slate_slot_13).inode).i_sb;
if !(*__slate_slot_12).inode.is_null() {
ext4_fc_track_unlink(handle, (*__slate_slot_12).dentry);
}
if (*EXT4_SB(sb)).s_mount_opt2 & 16 != 0 {
__v1 = ((*EXT4_SB(sb)).s_mount_state as i32) & 32 == 0;
} else {
__v1 = false;
}
if __v1 {
__v0 = ext4_test_mount_flag(sb, 1i32) == 0;
} else {
__v0 = false;
}
if __v0 {
__ext4_fc_track_link(handle, (*__slate_slot_13).inode, (*__slate_slot_12).dentry);
__ext4_fc_track_unlink(handle, (*__slate_slot_13).inode, (*__slate_slot_13).dentry);
if !whiteout.is_null() {
__ext4_fc_track_create(handle, whiteout, (*__slate_slot_13).dentry);
}
}
}
if !(*__slate_slot_12).inode.is_null() {
*__slate_slot_10 = ext4_mark_inode_dirty__29!(handle, (*__slate_slot_12).inode);
if unlikely__2!(*__slate_slot_10) != 0 {
break '__join_5;
} else {
if (*(*__slate_slot_12).inode).__slate_anon_12.i_nlink == 0 {
ext4_orphan_add(handle, (*__slate_slot_12).inode);
}
}
}
*__slate_slot_10 = 0;
}
}
}
if !whiteout.is_null() {
if *__slate_slot_10 != 0 {
ext4_resetent(handle, &(*core::ptr::addr_of_mut!(*__slate_slot_13)), (*(*__slate_slot_13).inode).i_ino as u32, *__slate_slot_7 as u32);
drop_nlink(whiteout);
ext4_mark_inode_dirty__68!(handle, whiteout);
ext4_orphan_add(handle, whiteout);
}
unlock_new_inode(whiteout);
ext4_journal_stop__13!(handle);
iput(whiteout);
} else {
ext4_journal_stop__49!(handle);
}
}
}
}
brelse((*__slate_slot_13).dir_bh);
brelse((*__slate_slot_13).bh);
brelse((*__slate_slot_12).bh);
return *__slate_slot_10;
}
}
}
}
}
}
}
}

fn ext4_cross_rename(old_dir: *mut inode, old_dentry: *mut dentry, new_dir: *mut inode, new_dentry: *mut dentry) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut new_file_type: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(new_file_type) as *mut u8;
let mut new: core::mem::MaybeUninit<ext4_renament> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut ext4_renament = core::ptr::addr_of_mut!(new) as *mut ext4_renament;
let mut old: core::mem::MaybeUninit<ext4_renament> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut ext4_renament = core::ptr::addr_of_mut!(old) as *mut ext4_renament;
let mut handle: *mut jbd2_journal_handle = unsafe { core::mem::zeroed() };
unsafe {
handle = core::ptr::null_mut::<jbd2_journal_handle>();
core::ptr::write(__slate_slot_9, {
let mut __t0: ext4_renament = core::mem::zeroed();
__t0.dir = old_dir;
__t0.dentry = old_dentry;
__t0.inode = d_inode(old_dentry as *const dentry);
__t0
});
core::ptr::write(__slate_slot_8, {
let mut __t1: ext4_renament = core::mem::zeroed();
__t1.dir = new_dir;
__t1.dentry = new_dentry;
__t1.inode = d_inode(new_dentry as *const dentry);
__t1
});
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ext4_test_inode_flag(new_dir, 29i32) != 0 {
__v5 = !projid_eq((*EXT4_I!(new_dir)).i_projid, (*EXT4_I!(unsafe { (*old_dentry).d_inode })).i_projid);
} else {
__v5 = false;
}
if __v5 {
__v4 = true;
} else {
if ext4_test_inode_flag(old_dir, 29i32) != 0 {
__v3 = !projid_eq((*EXT4_I!(old_dir)).i_projid, (*EXT4_I!(unsafe { (*new_dentry).d_inode })).i_projid);
} else {
__v3 = false;
}
__v4 = __v3;
}
if __v4 {
return 18i32.wrapping_neg();
} else {
*__slate_slot_6 = dquot_initialize((*__slate_slot_9).dir);
if *__slate_slot_6 != 0 {
return *__slate_slot_6;
} else {
*__slate_slot_6 = dquot_initialize((*__slate_slot_8).dir);
if *__slate_slot_6 != 0 {
return *__slate_slot_6;
} else {
(*__slate_slot_9).bh = ext4_find_entry((*__slate_slot_9).dir, core::ptr::addr_of!((*(*__slate_slot_9).dentry).__slate_anon_4.d_name), core::ptr::addr_of_mut!((*__slate_slot_9).de), core::ptr::addr_of_mut!((*__slate_slot_9).inlined));
if IS_ERR((*__slate_slot_9).bh as *const ()) {
return PTR_ERR((*__slate_slot_9).bh as *const ()) as i32;
} else {
'__join_2: {
// Check for inode number is _not_ due to possible IO errors.
// We might rmdir the source, keep it as pwd of some process
// and merrily kill the link to whatever was created under the
// same name. Goodbye sticky bit ;-<
*__slate_slot_6 = 2i32.wrapping_neg();
if !((*__slate_slot_9).bh.is_null() || ((*(*__slate_slot_9).de).inode as u64) != (*(*__slate_slot_9).inode).i_ino) {
(*__slate_slot_8).bh = ext4_find_entry((*__slate_slot_8).dir, core::ptr::addr_of!((*(*__slate_slot_8).dentry).__slate_anon_4.d_name), core::ptr::addr_of_mut!((*__slate_slot_8).de), core::ptr::addr_of_mut!((*__slate_slot_8).inlined));
if IS_ERR((*__slate_slot_8).bh as *const ()) {
*__slate_slot_6 = PTR_ERR((*__slate_slot_8).bh as *const ()) as i32;
(*__slate_slot_8).bh = core::ptr::null_mut::<buffer_head>();
} else {
// RENAME_EXCHANGE case: old *and* new must both exist
if !((*__slate_slot_8).bh.is_null() || ((*(*__slate_slot_8).de).inode as u64) != (*(*__slate_slot_8).inode).i_ino) {
handle = ext4_journal_start__47!((*__slate_slot_9).dir, 4i32, 2u32.wrapping_mul(EXT4_DATA_TRANS_BLOCKS!(unsafe { (*(*__slate_slot_9).dir).i_sb })).wrapping_add(24u32).wrapping_add(2));
if IS_ERR(handle as *const ()) {
*__slate_slot_6 = PTR_ERR(handle as *const ()) as i32;
handle = core::ptr::null_mut::<jbd2_journal_handle>();
} else {
if IS_DIRSYNC!((*__slate_slot_9).dir) || IS_DIRSYNC!((*__slate_slot_8).dir) {
ext4_handle_sync(handle);
}
if S_ISDIR!(unsafe { (*(*__slate_slot_9).inode).i_mode }) {
*__slate_slot_6 = ext4_rename_dir_prepare(handle, core::ptr::addr_of_mut!(*__slate_slot_9), (*__slate_slot_8).dir != (*__slate_slot_9).dir);
if *__slate_slot_6 != 0 {
break '__join_2;
}
}
if S_ISDIR!(unsafe { (*(*__slate_slot_8).inode).i_mode }) {
*__slate_slot_6 = ext4_rename_dir_prepare(handle, core::ptr::addr_of_mut!(*__slate_slot_8), (*__slate_slot_8).dir != (*__slate_slot_9).dir);
if *__slate_slot_6 != 0 {
break '__join_2;
}
}
// Other than the special case of overwriting a directory, parents'
// nlink only needs to be modified if this is a cross directory rename.
if (*__slate_slot_9).dir != (*__slate_slot_8).dir && ((*__slate_slot_9).is_dir as i32) != ((*__slate_slot_8).is_dir as i32) {
(*__slate_slot_9).dir_nlink_delta = if (*__slate_slot_9).is_dir { 1i32.wrapping_neg() } else { 1 };
(*__slate_slot_8).dir_nlink_delta = (*__slate_slot_9).dir_nlink_delta.wrapping_neg();
*__slate_slot_6 = 31i32.wrapping_neg();
if (*__slate_slot_9).dir_nlink_delta > 0 {
__v2 = EXT4_DIR_LINK_MAX!((*__slate_slot_9).dir) != 0;
} else {
__v2 = false;
}
if __v2 {
__v1 = true;
} else {
if (*__slate_slot_8).dir_nlink_delta > 0 {
__v0 = EXT4_DIR_LINK_MAX!((*__slate_slot_8).dir) != 0;
} else {
__v0 = false;
}
__v1 = __v0;
}
if __v1 {
break '__join_2;
}
}
*__slate_slot_7 = (*(*__slate_slot_8).de).file_type;
*__slate_slot_6 = ext4_setent(handle, core::ptr::addr_of_mut!(*__slate_slot_8), (*(*__slate_slot_9).inode).i_ino as u32, (*(*__slate_slot_9).de).file_type as u32);
if *__slate_slot_6 == 0 {
*__slate_slot_6 = ext4_setent(handle, core::ptr::addr_of_mut!(*__slate_slot_9), (*(*__slate_slot_8).inode).i_ino as u32, *__slate_slot_7 as u32);
if *__slate_slot_6 == 0 {
// Like most other Unix systems, set the ctime for inodes on a
// rename.
inode_set_ctime_current((*__slate_slot_9).inode);
inode_set_ctime_current((*__slate_slot_8).inode);
*__slate_slot_6 = ext4_mark_inode_dirty__73!(handle, (*__slate_slot_9).inode);
if unlikely__2!(*__slate_slot_6) == 0 {
*__slate_slot_6 = ext4_mark_inode_dirty__33!(handle, (*__slate_slot_8).inode);
if unlikely__2!(*__slate_slot_6) == 0 {
ext4_fc_mark_ineligible((*(*__slate_slot_8).inode).i_sb, 1i32, handle);
if !(*__slate_slot_9).dir_bh.is_null() {
*__slate_slot_6 = ext4_rename_dir_finish(handle, core::ptr::addr_of_mut!(*__slate_slot_9), (*(*__slate_slot_8).dir).i_ino as u32);
if *__slate_slot_6 != 0 {
break '__join_2;
}
}
if !(*__slate_slot_8).dir_bh.is_null() {
*__slate_slot_6 = ext4_rename_dir_finish(handle, core::ptr::addr_of_mut!(*__slate_slot_8), (*(*__slate_slot_9).dir).i_ino as u32);
if *__slate_slot_6 != 0 {
break '__join_2;
}
}
ext4_update_dir_count(handle, core::ptr::addr_of_mut!(*__slate_slot_9));
ext4_update_dir_count(handle, core::ptr::addr_of_mut!(*__slate_slot_8));
*__slate_slot_6 = 0;
}
}
}
}
}
}
}
}
}
brelse((*__slate_slot_9).dir_bh);
brelse((*__slate_slot_8).dir_bh);
brelse((*__slate_slot_9).bh);
brelse((*__slate_slot_8).bh);
if !handle.is_null() {
ext4_journal_stop__40!(handle);
}
return *__slate_slot_6;
}
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.namei.ext4_rename2")]
extern "C" fn ext4_rename2(idmap: *mut mnt_idmap, old_dir: *mut inode, old_dentry: *mut dentry, new_dir: *mut inode, new_dentry: *mut dentry, flags: u32) -> i32 {
let mut err: i32 = ext4_emergency_state(unsafe { (*old_dir).i_sb });
if unlikely__2!(err) != 0 {
return err;
}
if flags & 4294967288u32 != 0 {
return 22i32.wrapping_neg();
}
err = fscrypt_prepare_rename(old_dir, old_dentry, new_dir, new_dentry, flags);
if err != 0 {
return err;
}
if flags & 2u32 != 0 {
return ext4_cross_rename(old_dir, old_dentry, new_dir, new_dentry);
}
ext4_rename(idmap, old_dir, old_dentry, new_dir, new_dentry, flags)
}

/// directories can handle most operations...
#[unsafe(no_mangle)]
static ext4_dir_inode_operations: __SlateConst<inode_operations> = __SlateConst({
let mut __t0: inode_operations = unsafe { core::mem::zeroed() };
__t0.lookup = Some(ext4_lookup);
__t0.get_inode_acl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>>(ext4_get_acl as *const ()) };
__t0.create = Some(ext4_create);
__t0.link = Some(ext4_link);
__t0.unlink = Some(ext4_unlink);
__t0.symlink = Some(ext4_symlink);
__t0.mkdir = Some(ext4_mkdir);
__t0.rmdir = Some(ext4_rmdir);
__t0.mknod = Some(ext4_mknod);
__t0.rename = Some(ext4_rename2);
__t0.setattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>>(ext4_setattr as *const ()) };
__t0.getattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>>(ext4_getattr as *const ()) };
__t0.listxattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>>(ext4_listxattr as *const ()) };
__t0.fiemap = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>>(ext4_fiemap as *const ()) };
__t0.tmpfile = Some(ext4_tmpfile);
__t0.set_acl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>>(ext4_set_acl as *const ()) };
__t0.fileattr_set = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>>(ext4_fileattr_set as *const ()) };
__t0.fileattr_get = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>>(ext4_fileattr_get as *const ()) };
__t0
});

#[unsafe(no_mangle)]
static ext4_special_inode_operations: __SlateConst<inode_operations> = __SlateConst({
let mut __t0: inode_operations = unsafe { core::mem::zeroed() };
__t0.get_inode_acl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>>(ext4_get_acl as *const ()) };
__t0.setattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>>(ext4_setattr as *const ()) };
__t0.getattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>>(ext4_getattr as *const ()) };
__t0.listxattr = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>>(ext4_listxattr as *const ()) };
__t0.set_acl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>>(ext4_set_acl as *const ()) };
__t0
});

static __slate_asm_string_do_split_30: [u8; 1] = *b"\0";

static __slate_asm_string_do_split_31: [u8; 45] = *b"/home/takashi/c-corpus/linux/fs/ext4/namei.c\0";

static __slate_asm_string_dx_insert_block_10: [u8; 45] = *b"/home/takashi/c-corpus/linux/fs/ext4/namei.c\0";

static __slate_asm_string_dx_insert_block_2: [u8; 1] = *b"\0";

static __slate_asm_string_dx_insert_block_3: [u8; 45] = *b"/home/takashi/c-corpus/linux/fs/ext4/namei.c\0";

static __slate_asm_string_dx_insert_block_9: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_rec_len_to_disk_0: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_rec_len_to_disk_1: [u8; 44] = *b"/home/takashi/c-corpus/linux/fs/ext4/ext4.h\0";

static __slate_asm_string_ext4_update_dx_flag_2: [u8; 1] = *b"\0";

static __slate_asm_string_ext4_update_dx_flag_3: [u8; 44] = *b"/home/takashi/c-corpus/linux/fs/ext4/ext4.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
