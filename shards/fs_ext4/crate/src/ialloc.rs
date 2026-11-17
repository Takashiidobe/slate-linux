
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
//!  linux/fs/ext4/ialloc.c
//!
//! Copyright (C) 1992, 1993, 1994, 1995
//! Remy Card (card@masi.ibp.fr)
//! Laboratoire MASI - Institut Blaise Pascal
//! Universite Pierre et Marie Curie (Paris VI)
//!
//!  BSD ufs-inspired inode and directory allocation by
//!  Stephen Tweedie (sct@redhat.com), 1993
//!  Big-endian to little-endian byte-swapping/bitmaps by
//!        David S. Miller (davem@caip.rutgers.edu), 1995
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
//! error conditions
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
//! per-IO O_SYNC
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
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! Maximum value for the third parameter of fscrypt_operations.set_context().
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
//! Ext4 handle operation types -- for logging purposes
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
// warning conditions
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
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// No such process
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Text file busy
// No space left on device
// Read-only file system
// Bad CRC detected
// Value too large for defined data type
// Operation not supported on transport endpoint
// Operation now in progress
// Filesystem is corrupted
// Memory page has hardware error
// Operation is not supported
// An exchange-type operation, which takes a value and a pointer, and
// returns the old value.
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

macro_rules! EXT4_CLUSTERS_PER_GROUP {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_clusters_per_group }
};
}

macro_rules! EXT4_FIRST_INO {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_first_ino }
};
}

macro_rules! EXT4_INODES_PER_GROUP {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_inodes_per_group }
};
}

macro_rules! EXT4_INODE_SIZE {
($s:expr) => {
unsafe { (*EXT4_SB($s)).s_inode_size }
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

macro_rules! EXT4_MB_GRP_IBITMAP_CORRUPT {
($grp:expr) => {
test_bit__3!(3i32, unsafe { core::ptr::addr_of_mut!((*$grp).bb_state) })
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_186>() as i32
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
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
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

macro_rules! NEXT_ORPHAN {
($inode:expr) => {
unsafe { (*EXT4_I!($inode)).i_dtime }
};
}

macro_rules! RCU_LOCKDEP_WARN__2 {
($c:expr) => {
while false && $c {
}
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

macro_rules! S_ISDIR__2 {
($m:expr) => {
($m) & 61440 == 16384
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

macro_rules! S_ISREG {
($m:expr) => {
(($m) as i32) & 61440 == 32768
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut *mut flex_groups>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut flex_groups>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut flex_groups>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut flex_groups>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
{
let mut __v: *const cred;
RCU_LOCKDEP_WARN__2!(($c) == 0);
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
macro_rules! __set_bit {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_f217c4720f04f6db($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit($nr, ($addr) as *mut u64);
} else {
___set_bit($nr, ($addr) as *mut u64);
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

macro_rules! __test_and_clear_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___test_and_clear_bit($nr, ($addr) as *mut u64) as i32
} else {
___test_and_clear_bit($nr, ($addr) as *mut u64) as i32
};
__v
}
};
}

macro_rules! __test_and_set_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___test_and_set_bit($nr, ($addr) as *mut u64) as i32
} else {
___test_and_set_bit($nr, ($addr) as *mut u64) as i32
};
__v
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

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
};
}

macro_rules! cond_resched__14 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/fs/ext4/ialloc.c".as_ptr() as *const u8, 1494i32, 0u32);
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

/// current_cred - Access the current task's subjective credentials
///
/// Access the subjective credentials of the current task.  RCU-safe,
/// since nobody else can modify it.
macro_rules! current_cred {
() => {
rcu_dereference_protected__2!(unsafe { (*get_current()).cred }, 1i32)
};
}

macro_rules! current_fsuid {
() => {
{
let mut __v: __SlateRecord_uidgid_types_h_9;
__v = unsafe { (*current_cred!()).fsuid };
__v
}
};
}

/// do_div - returns 2 values: calculate remainder and update new dividend
/// @n: uint64_t dividend (will be updated)
/// @base: uint32_t divisor
///
/// Summary:
/// ``uint32_t remainder = n % base;``
/// ``n = n / base;``
///
/// Return: (uint32_t)remainder
///
/// NOTE: macro parameter @n is evaluated multiple times,
/// beware of side effects!
macro_rules! do_div__3 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __base: u32 = $base;
let __rem: u32 = ((($n) as u64) % (__base as u64)) as u32;
$n = (($n) as u64) / (__base as u64);
__v = __rem;
};
}

macro_rules! ext4_handle_dirty_metadata {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"__ext4_new_inode".as_ptr() as *const u8, 1135u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__5 {
($bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_mark_inode_used".as_ptr() as *const u8, 816u32, core::ptr::null_mut::<jbd2_journal_handle>(), core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__9 {
($bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_mark_inode_used".as_ptr() as *const u8, 869u32, core::ptr::null_mut::<jbd2_journal_handle>(), core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__10 {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"__ext4_new_inode".as_ptr() as *const u8, 1244u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__12 {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_free_inode".as_ptr() as *const u8, 348u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__18 {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"__ext4_new_inode".as_ptr() as *const u8, 1169u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__22 {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_free_inode".as_ptr() as *const u8, 352u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__27 {
($bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_mark_inode_used".as_ptr() as *const u8, 793u32, core::ptr::null_mut::<jbd2_journal_handle>(), core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_handle_dirty_metadata__48 {
($handle:expr, $bh:expr) => {
unsafe { __ext4_handle_dirty_metadata(c"ext4_init_inode_table".as_ptr() as *const u8, 1625u32, $handle, core::ptr::null_mut::<inode>(), $bh) }
};
}

macro_rules! ext4_iget__11 {
($sb:expr, $ino:expr, $flags:expr) => {
unsafe { __ext4_iget($sb, $ino, ($flags) as u32, c"ext4_orphan_get".as_ptr() as *const u8, 1403u32) }
};
}

macro_rules! ext4_journal_get_write_access__10 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"__ext4_new_inode".as_ptr() as *const u8, 1161u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__11 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_free_inode".as_ptr() as *const u8, 315u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__46 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_free_inode".as_ptr() as *const u8, 306u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__47 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"__ext4_new_inode".as_ptr() as *const u8, 1102u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__62 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"__ext4_new_inode".as_ptr() as *const u8, 1143u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_get_write_access__63 {
($handle:expr, $sb:expr, $bh:expr, $trigger_type:expr) => {
unsafe { __ext4_journal_get_write_access(c"ext4_init_inode_table".as_ptr() as *const u8, 1596u32, $handle, $sb, $bh, ($trigger_type) as u32) }
};
}

macro_rules! ext4_journal_start_sb__4 {
($sb:expr, $type:expr, $nblocks:expr) => {
unsafe { __ext4_journal_start_sb(core::ptr::null_mut::<inode>(), $sb, 1546u32, $type, $nblocks, 0i32, ext4_trans_default_revoke_credits($sb)) }
};
}

macro_rules! ext4_journal_stop__61 {
($handle:expr) => {
unsafe { __ext4_journal_stop(c"ext4_init_inode_table".as_ptr() as *const u8, 1629u32, $handle) };
};
}

macro_rules! ext4_mark_inode_dirty__45 {
($__h:expr, $__i:expr) => {
unsafe { __ext4_mark_inode_dirty($__h, $__i, c"__ext4_new_inode".as_ptr() as *const u8, 1353u32) }
};
}

macro_rules! ext4_std_error__2 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1104u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__6 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1355u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__13 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1096u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__14 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_mark_inode_used".as_ptr() as *const u8, 795u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__17 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1137u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__18 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1164u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__27 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_mark_inode_used".as_ptr() as *const u8, 833u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__28 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_free_inode".as_ptr() as *const u8, 363u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__33 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1185u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__37 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"ext4_mark_inode_used".as_ptr() as *const u8, 800u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__40 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1246u32, $errno) };
}
if true {
break;
}
}
};
}

macro_rules! ext4_std_error__48 {
($sb:expr, $errno:expr) => {
loop {
if ($errno) != 0 {
unsafe { __ext4_std_error($sb, c"__ext4_new_inode".as_ptr() as *const u8, 1145u32, $errno) };
}
if true {
break;
}
}
};
}

/// ffz - find first zero bit in word
/// @word: The word to search
///
/// Undefined if no zero exists, so code should check against ~0UL first.
macro_rules! ffz {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ((!$word).trailing_zeros() as i32) as u64 } else { variable_ffz($word) }
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
macro_rules! ilog2__2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
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

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kcsan_release {
() => {
while false {
}
};
}

macro_rules! kcsan_rmb {
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

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
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

macro_rules! offsetof__6 {
() => {
8u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

/// Define the various spin_lock methods.  Note we define these
/// regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
/// various methods are defined as nops in the case they are not
/// required.
macro_rules! raw_spin_trylock {
($lock:expr) => {
unsafe { _raw_spin_trylock($lock) }
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
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
macro_rules! rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__2!($p, $c)
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
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
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

macro_rules! static_branch_unlikely__4 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! test_bit__7 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit($nr, ($addr) as *const u64) as i32
} else {
_test_bit($nr, ($addr) as *const u64) as i32
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

/// time_after32 - compare two 32-bit relative times
/// @a:	the time which may be after @b
/// @b:	the time which may be before @a
///
/// time_after32(a, b) returns true if the time @a is after time @b.
/// time_before32(b, a) returns true if the time @b is before time @a.
///
/// Similar to time_after(), compare two 32-bit timestamps for relative
/// times.  This is useful for comparing 32-bit seconds values that can't
/// be converted to 64-bit values (e.g. due to disk format or wire protocol
/// issues) when it is known that the times are less than 68 years apart.
macro_rules! time_after32 {
($a:expr, $b:expr) => {
((($b) as u32).wrapping_sub(($a) as u32) as i32) < 0
};
}

macro_rules! time_before32 {
($b:expr, $a:expr) => {
time_after32!($a, $b)
};
}

macro_rules! type_max {
() => {
__type_max!()
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
    static mut init_user_ns: user_namespace;
    static mut percpu_counter_batch: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut dquot_operations: dquot_operations;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_ext4_free_inode: static_call_key;
    static mut __tracepoint_ext4_free_inode: tracepoint;
    static mut __SCK__tp_func_ext4_request_inode: static_call_key;
    static mut __tracepoint_ext4_request_inode: tracepoint;
    static mut __SCK__tp_func_ext4_allocate_inode: static_call_key;
    static mut __tracepoint_ext4_allocate_inode: tracepoint;
    static mut __SCK__tp_func_ext4_load_inode_bitmap: static_call_key;
    static mut __tracepoint_ext4_load_inode_bitmap: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn _find_next_zero_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn get_random_u8() -> u8;
    fn get_random_u16() -> u16;
    fn get_random_u32() -> u32;
    fn __get_random_u32_below(arg0: u32) -> u32;
    fn __compiletime_assert_43();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_trylock(arg0: *mut raw_spinlock) -> i32;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn __SCT__cond_resched() -> i32;
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn kvfree_call_rcu(arg0: *mut kvfree_rcu_head, arg1: *mut ());
    fn down_read(arg0: *mut rw_semaphore);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn ktime_get_real_seconds() -> i64;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn percpu_counter_add_batch(arg0: *mut percpu_counter, arg1: i64, arg2: i32);
    fn __compiletime_assert_183();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn from_vfsuid(arg0: *mut mnt_idmap, arg1: *mut user_namespace, arg2: __SlateRecord_mnt_idmapping_h_17) -> __SlateRecord_uidgid_types_h_9;
    fn __compiletime_assert_410();
    fn __compiletime_assert_411();
    fn simple_inode_init_ts(arg0: *mut inode) -> timespec64;
    fn inode_init_owner(arg0: *mut mnt_idmap, arg1: *mut inode, arg2: *const inode, arg3: u16);
    fn clear_nlink(arg0: *mut inode);
    fn iput(arg0: *mut inode);
    fn is_bad_inode(arg0: *mut inode) -> bool;
    fn insert_inode_locked(arg0: *mut inode) -> i32;
    fn unlock_new_inode(arg0: *mut inode);
    fn new_inode(arg0: *mut super_block) -> *mut inode;
    fn dquot_initialize(arg0: *mut inode) -> i32;
    fn dquot_drop(arg0: *mut inode);
    fn dquot_alloc_inode(arg0: *mut inode) -> i32;
    fn dquot_free_inode(arg0: *mut inode);
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn bio_endio_bh(arg0: *mut bio, arg1: *mut *mut buffer_head) -> bool;
    fn __find_get_block(arg0: *mut block_device, arg1: u64, arg2: u32) -> *mut buffer_head;
    fn bdev_getblk(arg0: *mut block_device, arg1: u64, arg2: u32, arg3: u32) -> *mut buffer_head;
    fn __brelse(arg0: *mut buffer_head);
    fn unlock_buffer(arg0: *mut buffer_head);
    fn __lock_buffer(arg0: *mut buffer_head);
    fn sync_dirty_buffer(arg0: *mut buffer_head) -> i32;
    fn blkdev_issue_flush(arg0: *mut block_device) -> i32;
    fn blkdev_issue_zeroout(arg0: *mut block_device, arg1: u64, arg2: u64, arg3: u32, arg4: u32) -> i32;
    fn crc32c(arg0: u32, arg1: *const (), arg2: u64) -> u32;
    fn ext4_inode_bitmap_csum_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: *mut buffer_head);
    fn ext4_inode_bitmap_csum_verify(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: *mut buffer_head) -> i32;
    fn ext4_block_bitmap_csum_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: *mut buffer_head);
    fn ext4_get_group_desc(arg0: *mut super_block, arg1: u32, arg2: *mut *mut buffer_head) -> *mut ext4_group_desc;
    fn ext4_get_group_info(arg0: *mut super_block, arg1: u32) -> *mut ext4_group_info;
    fn ext4_read_block_bitmap(arg0: *mut super_block, arg1: u32) -> *mut buffer_head;
    fn ext4_free_clusters_after_init(arg0: *mut super_block, arg1: u32, arg2: *mut ext4_group_desc) -> u32;
    fn ext4fs_dirhash(arg0: *const inode, arg1: *const u8, arg2: i32, arg3: *mut dx_hash_info) -> i32;
    fn ext4_set_inode_mapping_order(arg0: *mut inode);
    fn __ext4_iget(arg0: *mut super_block, arg1: u64, arg2: u32, arg3: *const u8, arg4: u32) -> *mut inode;
    fn ext4_clear_inode(arg0: *mut inode);
    fn ext4_can_truncate(arg0: *mut inode) -> i32;
    fn ext4_set_inode_flags(arg0: *mut inode, arg1: bool);
    fn ext4_read_bh(arg0: *mut buffer_head, arg1: u32, arg2: Option<unsafe extern "C" fn(*mut bio)>, arg3: bool) -> i32;
    fn ext4_mark_group_bitmap_corrupted(arg0: *mut super_block, arg1: u32, arg2: u32);
    fn __ext4_error(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: bool, arg4: i32, arg5: u64, arg6: *const u8, ...);
    fn __ext4_std_error(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: i32);
    fn __ext4_warning(arg0: *mut super_block, arg1: *const u8, arg2: u32, arg3: *const u8, ...);
    fn __ext4_msg(arg0: *mut super_block, arg1: *const u8, arg2: *const u8, ...);
    fn ext4_inode_bitmap(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u64;
    fn ext4_inode_table(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u64;
    fn ext4_free_group_clusters(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u32;
    fn ext4_free_inodes_count(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u32;
    fn ext4_used_dirs_count(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u32;
    fn ext4_itable_unused_count(arg0: *mut super_block, arg1: *mut ext4_group_desc) -> u32;
    fn ext4_free_group_clusters_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: u32);
    fn ext4_free_inodes_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: u32);
    fn ext4_used_dirs_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: u32);
    fn ext4_itable_unused_set(arg0: *mut super_block, arg1: *mut ext4_group_desc, arg2: u32);
    fn ext4_group_desc_csum_set(arg0: *mut super_block, arg1: u32, arg2: *mut ext4_group_desc);
    fn ext4_ext_tree_init(arg0: *mut jbd2_journal_handle, arg1: *mut inode);
    fn __ext4_mark_inode_dirty(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *const u8, arg3: u32) -> i32;
    fn __ext4_journal_get_write_access(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut super_block, arg4: *mut buffer_head, arg5: u32) -> i32;
    fn __ext4_handle_dirty_metadata(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle, arg3: *mut inode, arg4: *mut buffer_head) -> i32;
    fn __ext4_journal_start_sb(arg0: *mut inode, arg1: *mut super_block, arg2: u32, arg3: i32, arg4: i32, arg5: i32, arg6: i32) -> *mut jbd2_journal_handle;
    fn __ext4_journal_stop(arg0: *const u8, arg1: u32, arg2: *mut jbd2_journal_handle) -> i32;
    fn __ext4_xattr_set_credits(arg0: *mut super_block, arg1: *mut inode, arg2: *mut buffer_head, arg3: u64, arg4: bool) -> i32;
    fn ext4_init_security(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut inode, arg3: *const qstr) -> i32;
    fn __compiletime_assert_677();
    fn get_inode_acl(arg0: *mut inode, arg1: i32) -> *mut posix_acl;
    fn ext4_init_acl(arg0: *mut jbd2_journal_handle, arg1: *mut inode, arg2: *mut inode) -> i32;
    fn __traceiter_ext4_free_inode(arg0: *mut (), arg1: *mut inode) -> i32;
    fn __SCT__tp_func_ext4_free_inode(arg0: *mut (), arg1: *mut inode) -> i32;
    fn __compiletime_assert_696();
    fn __traceiter_ext4_request_inode(arg0: *mut (), arg1: *mut inode, arg2: i32) -> i32;
    fn __SCT__tp_func_ext4_request_inode(arg0: *mut (), arg1: *mut inode, arg2: i32) -> i32;
    fn __compiletime_assert_703();
    fn __traceiter_ext4_allocate_inode(arg0: *mut (), arg1: *mut inode, arg2: *mut inode, arg3: i32) -> i32;
    fn __SCT__tp_func_ext4_allocate_inode(arg0: *mut (), arg1: *mut inode, arg2: *mut inode, arg3: i32) -> i32;
    fn __compiletime_assert_710();
    fn __traceiter_ext4_load_inode_bitmap(arg0: *mut (), arg1: *mut super_block, arg2: u64) -> i32;
    fn __SCT__tp_func_ext4_load_inode_bitmap(arg0: *mut (), arg1: *mut super_block, arg2: u64) -> i32;
    fn __compiletime_assert_1018();
    fn __compiletime_assert_1505();
    fn __compiletime_assert_1507();
    fn __builtin_unreachable();
    fn __compiletime_assert_1510();
    fn __compiletime_assert_1512();
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
struct kvfree_rcu_head {
    next: *mut kvfree_rcu_head,
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
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_test_and_set_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btsq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
arch_test_and_set_bit(nr, addr)
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
struct __SlateRecord_find_h_186 {
}


// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
#[inline(always)]
fn variable_ffz(word: u64) -> u64 {
variable__ffs(!word)
}

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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
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
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn ___test_and_clear_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_clear_bit(nr, addr)
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn _test_bit_acquire(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit_acquire(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


fn test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit_lock(nr, addr)
}

fn test_bit_le(nr: u64, addr: *const ()) -> i32 {
test_bit__7!(nr ^ 0, addr)
}

fn __set_bit_le(nr: u64, addr: *mut ()) {
__set_bit!(nr ^ 0, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


fn __test_and_set_bit_le(nr: u64, addr: *mut ()) -> i32 {
__test_and_set_bit!(nr ^ 0, addr)
}

fn __test_and_clear_bit_le(nr: u64, addr: *mut ()) -> i32 {
__test_and_clear_bit!(nr ^ 0, addr)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
    __slate_bits_0: __slate_bits::__SlateBits125U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
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


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_free_inode, const 0 as u8, label {
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


#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_request_inode, const 0 as u8, label {
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
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_allocate_inode, const 0 as u8, label {
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
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ext4_load_inode_bitmap, const 0 as u8, label {
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
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits132U0,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn arch_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, mut __new: i32) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i32 = unsafe { *old };
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u32;
core::arch::asm!("lock cmpxchgb {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u32;
core::arch::asm!("lock cmpxchgw {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u32;
core::arch::asm!("lock cmpxchgl {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u32;
core::arch::asm!("lock cmpxchgq {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t7 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!success) != 0 {
unsafe {
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn raw_atomic_fetch_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> i32 {
let mut c: i32 = raw_atomic_read(v as *const __SlateRecord_types_h_190);
while unlikely!(c == u) == 0 {
if !(!raw_atomic_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(a))) {
break;
}
}
c
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


#[inline(always)]
fn raw_atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
raw_atomic_fetch_add_unless(v, a, u) != u
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


#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_unless(v, a, u)
}

#[inline(always)]
fn atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_read(v)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[inline(always)]
fn find_next_zero_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) | !GENMASK__2!(size.wrapping_sub(1), offset);
return if val == !0 { size } else { ffz!(val) };
}
unsafe { _find_next_zero_bit(addr, size, offset) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn find_next_zero_bit_le(addr: *const (), size: u64, offset: u64) -> u64 {
find_next_zero_bit(addr as *const u64, size, offset)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
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


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


fn get_random_u32_below(ceil: u32) -> u32 {
if !unsafe { __slate_intrinsic_1c5364fcd9da51fe(ceil) } {
return unsafe { __get_random_u32_below(ceil) };
}
// For the fast path, below, all operations on ceil are precomputed by
// the compiler, so this incurs no overhead for checking pow2, doing
// divisions, or branching based on integer size. The resultant
// algorithm does traditional reciprocal multiplication (typically
// optimized by the compiler into shifts and adds), rejecting samples
// whose lower half would indicate a range indivisible by ceil.
loop {
if ceil == 0 {
unsafe { __compiletime_assert_43() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if ceil <= 1 {
return 0;
}
loop {
if ceil <= 256u32 {
let mult: u32 = ceil.wrapping_mul(((unsafe { get_random_u8() }) as i32) as u32);
if likely!(is_power_of_2(ceil as u64) || (((mult as u8) as i32) as u32) >= 256u32 % ceil) != 0 {
return mult >> 8i32;
}
} else {
if ceil <= 65536u32 {
let mult: u32 = ceil.wrapping_mul(((unsafe { get_random_u16() }) as i32) as u32);
if likely!(is_power_of_2(ceil as u64) || (((mult as u16) as i32) as u32) >= 65536u32 % ceil) != 0 {
return mult >> 16i32;
}
} else {
let mult: u64 = (ceil as u64).wrapping_mul((unsafe { get_random_u32() }) as u64);
if likely!(is_power_of_2(ceil as u64) || (mult as u32) >= ceil.wrapping_neg() % ceil) != 0 {
return (mult >> 32i32) as u32;
}
}
}
}
unsafe { core::mem::zeroed() }
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
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
    __slate_bits_0: __slate_bits::__SlateBits133U0,
    pkru: u32,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
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
struct vm_struct {
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
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits224U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_trylock(lock: *mut spinlock) -> i32 {
raw_spin_trylock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) })
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
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
struct seqcount_spinlock {
    seqcount: seqcount,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
struct key_type {
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


fn __kuid_val(uid: __SlateRecord_uidgid_types_h_9) -> u32 {
uid.val
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn make_kuid(from: *mut user_namespace, uid: u32) -> __SlateRecord_uidgid_types_h_9 {
let mut __slate_compound_0: __SlateRecord_uidgid_types_h_9 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_uidgid_types_h_9 { val: uid };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn make_kgid(from: *mut user_namespace, gid: u32) -> __SlateRecord_uidgid_types_h_13 {
let mut __slate_compound_0: __SlateRecord_uidgid_types_h_13 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_uidgid_types_h_13 { val: gid };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
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


fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
return true;
}
if unlikely!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
fn rcu_read_lock_held() -> i32 {
1
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

#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
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
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
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


#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
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
struct fs_context {
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


fn percpu_counter_add(fbc: *mut percpu_counter, amount: i64) {
unsafe { percpu_counter_add_batch(fbc, amount, percpu_counter_batch) };
}

fn percpu_counter_read_positive(fbc: *mut percpu_counter) -> i64 {
// Prevent reloads of fbc->count
let mut ret: i64 = 0;
let mut __v0: i64;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_183() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*fbc).count) as *const i64) };
ret = __v0;
if ret >= 0 {
return ret;
}
0
}

fn percpu_counter_initialized(fbc: *mut percpu_counter) -> bool {
!unsafe { (*fbc).counters }.is_null()
}

fn percpu_counter_inc(fbc: *mut percpu_counter) {
percpu_counter_add(fbc, 1i64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


fn percpu_counter_dec(fbc: *mut percpu_counter) {
percpu_counter_add(fbc, 1i32.wrapping_neg() as i64);
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits421U0,
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
struct io_uring_cmd {
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


fn make_kprojid(from: *mut user_namespace, projid: u32) -> __SlateRecord_projid_h_24 {
let mut __slate_compound_0: __SlateRecord_projid_h_24 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_projid_h_24 { val: projid };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct disk_stats {
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request {
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
struct bio_alloc_cache {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
fn mapped_fsuid(idmap: *mut mnt_idmap, fs_userns: *mut user_namespace) -> __SlateRecord_uidgid_types_h_9 {
let mut __slate_compound_0: __SlateRecord_mnt_idmapping_h_17 = unsafe { core::mem::zeroed() };
return unsafe { from_vfsuid(idmap, fs_userns, *{
__slate_compound_0 = __SlateRecord_mnt_idmapping_h_17 { val: __kuid_val(current_fsuid!()) };
core::ptr::addr_of_mut!(__slate_compound_0)
}) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits577U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
union __SlateRecord_ext4_h_806 {
    linux1: __SlateRecord_ext4_h_807,
    hurd1: __SlateRecord_ext4_h_810,
    masix1: __SlateRecord_ext4_h_813,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_807 {
    l_i_version: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_810 {
    h_i_translator: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_813 {
    m_i_reserved1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ext4_h_822 {
    linux2: __SlateRecord_ext4_h_823,
    hurd2: __SlateRecord_ext4_h_831,
    masix2: __SlateRecord_ext4_h_838,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_823 {
    l_i_blocks_high: u16,
    l_i_file_acl_high: u16,
    l_i_uid_high: u16,
    l_i_gid_high: u16,
    l_i_checksum_lo: u16,
    l_i_reserved: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_831 {
    h_i_reserved1: u16,
    h_i_mode_high: u16,
    h_i_uid_high: u16,
    h_i_gid_high: u16,
    h_i_author: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ext4_h_838 {
    h_i_reserved1: u16,
    m_i_file_acl_high: u16,
    m_i_reserved2: [u32; 2],
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


fn i_user_ns(inode: *const inode) -> *mut user_namespace {
unsafe { (*(*inode).i_sb).s_user_ns }
}

fn i_uid_write(inode: *mut inode, uid: u32) {
unsafe {
(*inode).i_uid = make_kuid(i_user_ns(inode as *const inode), uid);
}
}

fn i_gid_write(inode: *mut inode, gid: u32) {
unsafe {
(*inode).i_gid = make_kgid(i_user_ns(inode as *const inode), gid);
}
}

fn inode_fsuid_set(inode: *mut inode, idmap: *mut mnt_idmap) {
unsafe {
(*inode).i_uid = mapped_fsuid(idmap, i_user_ns(inode as *const inode));
}
}

fn inode_get_mtime_sec(inode: *const inode) -> i64 {
let mut __v0: i64;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_410() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*inode).i_mtime_sec) as *const i64) };
__v0
}

fn inode_get_mtime_nsec(inode: *const inode) -> i64 {
let mut __v0: u32;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_411() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*inode).i_mtime_nsec) as *const u32) };
__v0 as i64
}

fn inode_get_mtime(inode: *const inode) -> timespec64 {
let ts: timespec64 = timespec64 { tv_sec: inode_get_mtime_sec(inode), tv_nsec: inode_get_mtime_nsec(inode) };
return ts;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_fc_inode_snap {
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


// Inode is not counted to quota
// no suid or xattr security attributes
fn icount_read_once(inode: *const inode) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*inode).i_count) })
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
struct ext4_locality_group {
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
struct mb_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dax_device {
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
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
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
    __slate_bits_0: __slate_bits::__SlateBits648U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits545U0 {
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
    pub struct __SlateBits577U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U0 {
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
    pub struct __SlateBits671U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits674U1 {
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
    pub struct __SlateBits674U2 {
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
    pub struct __SlateBits674U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits125U0 {
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
    pub struct __SlateBits133U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits421U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits648U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits586U0 {
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
    pub struct __SlateBits132U0 {
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
    pub struct __SlateBits224U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits791U0 {
        #[bits(1)]
        pub t_synchronous_commit: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits795U0 {
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
    __slate_bits_0: __slate_bits::__SlateBits586U0,
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits674U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits674U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits674U2,
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
    __slate_bits_3: __slate_bits::__SlateBits674U3,
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
    __slate_bits_0: __slate_bits::__SlateBits671U0,
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


fn mapping_gfp_mask(mapping: *const address_space) -> u32 {
unsafe { (*mapping).gfp_mask }
}

fn mapping_gfp_constraint(mapping: *const address_space, gfp_mask: u32) -> u32 {
mapping_gfp_mask(mapping) & gfp_mask
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
fn buffer_dirty(bh: *const buffer_head) -> i32 {
test_bit__2!(1i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
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

#[inline(always)]
fn buffer_uptodate(bh: *const buffer_head) -> i32 {
// make it consistent with folio_test_uptodate
// pairs with smp_mb__before_atomic in set_buffer_uptodate
test_bit_acquire!(0i32, unsafe { core::ptr::addr_of!((*bh).b_state) })
}

fn put_bh(bh: *mut buffer_head) {
smp_mb__before_atomic!();
atomic_dec(unsafe { core::ptr::addr_of_mut!((*bh).b_count) });
}

fn brelse(bh: *mut buffer_head) {
if !bh.is_null() {
unsafe { __brelse(bh) };
}
}

fn __getblk(bdev: *mut block_device, block: u64, size: u32) -> *mut buffer_head {
let mut gfp: u32 = mapping_gfp_constraint((unsafe { (*bdev).bd_mapping }) as *const address_space, !(BIT!(7i32) as u32));
let __v0: u32 = gfp;
gfp = __v0 | ((BIT!(3i32) as u32) | (BIT!(15i32) as u32));
unsafe { bdev_getblk(bdev, block, size, gfp) }
}

fn sb_getblk(sb: *mut super_block, block: u64) -> *mut buffer_head {
__getblk(unsafe { (*sb).s_bdev }, block, (unsafe { (*sb).s_blocksize }) as u32)
}

fn sb_find_get_block(sb: *mut super_block, block: u64) -> *mut buffer_head {
unsafe { __find_get_block((*sb).s_bdev, block, (*sb).s_blocksize as u32) }
}

fn trylock_buffer(bh: *mut buffer_head) -> i32 {
likely!(!test_and_set_bit_lock(2i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64)) as i32
}

fn lock_buffer(bh: *mut buffer_head) {
might_sleep!();
if trylock_buffer(bh) == 0 {
unsafe { __lock_buffer(bh) };
}
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


fn sb_issue_zeroout(sb: *mut super_block, block: u64, nr_blocks: u64, gfp_mask: u32) -> i32 {
unsafe { blkdev_issue_zeroout((*sb).s_bdev, block << ((*sb).s_blocksize_bits as i32).wrapping_sub(9), nr_blocks << ((*sb).s_blocksize_bits as i32).wrapping_sub(9), gfp_mask, 0u32) }
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
    __slate_bits_0: __slate_bits::__SlateBits795U0,
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
    __slate_bits_0: __slate_bits::__SlateBits791U0,
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


fn is_journal_aborted(journal: *mut journal_s) -> i32 {
((unsafe { (*journal).j_flags }) & 2) as i32
}

fn is_handle_aborted(handle: *mut jbd2_journal_handle) -> i32 {
if ((unsafe { (*handle).__slate_bits_0.__get_h_aborted() }) as i32) != 0 || unsafe { (*handle).__slate_anon_0.h_transaction }.is_null() {
return 1;
}
is_journal_aborted(unsafe { (*(*handle).__slate_anon_0.h_transaction).t_journal })
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


fn bgl_lock_ptr(bgl: *mut blockgroup_lock, block_group: u32) -> *mut spinlock {
unsafe { core::ptr::addr_of_mut!((*((*bgl).locks.as_mut_ptr() as *mut bgl_lock).wrapping_offset((block_group & ((4i32 << ilog2__2!(if false { 64i32 } else { 32i32 })).wrapping_sub(1) as u32)) as isize)).lock) }
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


fn fscrypt_set_context(inode: *mut inode, fs_data: *mut ()) -> i32 {
95i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fscrypt_dummy_policy {
}


fn fscrypt_prepare_new_inode(dir: *mut inode, inode: *mut inode, encrypt_ret: &bool) -> i32 {
if IS_ENCRYPTED!(dir) != 0 {
return 95i32.wrapping_neg();
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
struct ext4_system_blocks {
    root: rb_root,
    rcu: callback_head,
}


// Flags for ext4_io_end->flags
// Root inode
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_group_desc {
    bg_block_bitmap_lo: u32,
    bg_inode_bitmap_lo: u32,
    bg_inode_table_lo: u32,
    bg_free_blocks_count_lo: u16,
    bg_free_inodes_count_lo: u16,
    bg_used_dirs_count_lo: u16,
    bg_flags: u16,
    bg_exclude_bitmap_lo: u32,
    bg_block_bitmap_csum_lo: u16,
    bg_inode_bitmap_csum_lo: u16,
    bg_itable_unused_lo: u16,
    bg_checksum: u16,
    bg_block_bitmap_hi: u32,
    bg_inode_bitmap_hi: u32,
    bg_inode_table_hi: u32,
    bg_free_blocks_count_hi: u16,
    bg_free_inodes_count_hi: u16,
    bg_used_dirs_count_hi: u16,
    bg_itable_unused_hi: u16,
    bg_exclude_bitmap_hi: u32,
    bg_block_bitmap_csum_hi: u16,
    bg_inode_bitmap_csum_hi: u16,
    bg_reserved: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct flex_groups {
    free_clusters: __SlateRecord_types_h_197,
    free_inodes: __SlateRecord_types_h_190,
    used_dirs: __SlateRecord_types_h_190,
}


// Inode table/bitmap not in use
// Block bitmap not in use
// On-disk itable initialized to zero
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
// Inode is DAX
// Inode has inline data.
// Create with parents projid
// Casefolded directory
// reserved for ext4 lib
// Flags that should be inherited by new inodes from their parent.
// Flags that are appropriate for regular files (all but dir-specific ones).
// Flags that are appropriate for non-directories/regular files.
fn ext4_mask_flags(mode: u16, flags: u32) -> u32 {
if S_ISDIR!(mode) {
return flags;
} else {
if S_ISREG!(mode) {
return flags & 2684157951u32;
} else {
return flags & 192u32;
}
}
}

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
#[repr(C)]
#[derive(Clone, Copy)]
struct ext4_inode {
    i_mode: u16,
    i_uid: u16,
    i_size_lo: u32,
    i_atime: u32,
    i_ctime: u32,
    i_mtime: u32,
    i_dtime: u32,
    i_gid: u16,
    i_links_count: u16,
    i_blocks_lo: u32,
    i_flags: u32,
    osd1: __SlateRecord_ext4_h_806,
    i_block: [u32; 15],
    i_generation: u32,
    i_file_acl_lo: u32,
    i_size_high: u32,
    i_obso_faddr: u32,
    osd2: __SlateRecord_ext4_h_822,
    i_extra_isize: u16,
    i_checksum_hi: u16,
    i_ctime_extra: u32,
    i_mtime_extra: u32,
    i_atime_extra: u32,
    i_crtime: u32,
    i_crtime_extra: u32,
    i_version_hi: u32,
    i_projid: u32,
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
// Create files with directory's group
// Mode for data writes:
// Write data to journal
// Some quota option set
// Enable support for dio read nolocking
// Delalloc support
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

// Returns: sbi->field[index]
// Used to access an array element from the following sbi fields which require
// rcu protection to avoid dereferencing an invalid pointer due to reassignment
// - s_group_desc
// - s_group_info
// - s_flex_group
fn ext4_simulate_fail(sb: *mut super_block, code: u64) -> bool {
false
}

fn ext4_test_inode_flag(inode: *mut inode, bit: i32) -> i32 {
test_bit__4!(bit.wrapping_add(0), unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) })
}

fn ext4_set_inode_flag(inode: *mut inode, bit: i32) {
set_bit(bit.wrapping_add(0) as i64, (unsafe { core::ptr::addr_of_mut!((*EXT4_I(inode)).i_flags) }) as *mut u64);
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
fn ext4_has_feature_gdt_csum(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 16 != 0
}

fn ext4_has_feature_metadata_csum(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 1024 != 0
}

fn ext4_has_feature_project(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_ro_compat }) & 8192 != 0
}

fn ext4_has_feature_extents(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 64 != 0
}

fn ext4_has_feature_inline_data(sb: *mut super_block) -> bool {
(unsafe { (*(*EXT4_SB(sb)).s_es).s_feature_incompat }) & 32768 != 0
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

// Default project ID
// Structure of a directory entry
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


// htree levels for ext4
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


#[unsafe(no_mangle)]
extern "C" fn __ext4_new_inode(idmap: *mut mnt_idmap, mut handle: *mut jbd2_journal_handle, dir: *mut inode, mode: u16, qstr: *const qstr, mut goal: u32, owner: *mut u32, mut i_flags: u32, handle_type: i32, line_no: u32, mut nblocks: i32) -> *mut inode {
let mut r#gen: u32 = unsafe { core::mem::zeroed() };
let mut inum: u32 = unsafe { core::mem::zeroed() };
let mut csum: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(csum) as *mut u32;
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut __v11: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_1511: *mut *mut flex_groups = unsafe { core::mem::zeroed() };
let mut _v: core::mem::MaybeUninit<*mut flex_groups> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut flex_groups = core::ptr::addr_of_mut!(_v) as *mut *mut flex_groups;
let mut __v16: *mut flex_groups = unsafe { core::mem::zeroed() };
let mut __v17: bool = unsafe { core::mem::zeroed() };
let mut __v18: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_1509: *mut *mut flex_groups = unsafe { core::mem::zeroed() };
let mut _v: core::mem::MaybeUninit<*mut flex_groups> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut flex_groups = core::ptr::addr_of_mut!(_v) as *mut *mut flex_groups;
let mut __v23: *mut flex_groups = unsafe { core::mem::zeroed() };
let mut f: u32 = unsafe { core::mem::zeroed() };
let mut grp: *mut ext4_group_info = unsafe { core::mem::zeroed() };
let mut free: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut i32 = core::ptr::addr_of_mut!(free) as *mut i32;
let mut block_bitmap_bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut buffer_head = core::ptr::addr_of_mut!(block_bitmap_bh) as *mut *mut buffer_head;
let mut __v40: bool = unsafe { core::mem::zeroed() };
let mut __v41: bool = unsafe { core::mem::zeroed() };
// Skip groups with already-known suspicious inode
// tables
let mut __v42: bool = unsafe { core::mem::zeroed() };
let mut __v47: bool = unsafe { core::mem::zeroed() };
let mut encrypt: bool = unsafe { core::mem::zeroed() };
let mut grp_1: *mut ext4_group_info = unsafe { core::mem::zeroed() };
let mut flex_group: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut u32 = core::ptr::addr_of_mut!(flex_group) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_51: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut ret: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut *mut inode = core::ptr::addr_of_mut!(ret) as *mut *mut inode;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ret2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_54: *mut i32 = core::ptr::addr_of_mut!(ret2) as *mut i32;
let mut sbi: core::mem::MaybeUninit<*mut ext4_sb_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_55: *mut *mut ext4_sb_info = core::ptr::addr_of_mut!(sbi) as *mut *mut ext4_sb_info;
let mut ei: core::mem::MaybeUninit<*mut ext4_inode_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_56: *mut *mut ext4_inode_info = core::ptr::addr_of_mut!(ei) as *mut *mut ext4_inode_info;
let mut gdp: *mut ext4_group_desc = unsafe { core::mem::zeroed() };
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_58: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut ino: u64 = unsafe { core::mem::zeroed() };
let mut group: u32 = unsafe { core::mem::zeroed() };
let mut ngroups: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_61: *mut u32 = core::ptr::addr_of_mut!(ngroups) as *mut u32;
let mut group_desc_bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_62: *mut *mut buffer_head = core::ptr::addr_of_mut!(group_desc_bh) as *mut *mut buffer_head;
let mut inode_bitmap_bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut sb: core::mem::MaybeUninit<*mut super_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_64: *mut *mut super_block = core::ptr::addr_of_mut!(sb) as *mut *mut super_block;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_64 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
inode_bitmap_bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes_2 = __slate_slot_62 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_61 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
group = 0;
ino = 0;
let __slate_auto_bytes_4 = __slate_slot_58 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
gdp = core::ptr::null_mut::<ext4_group_desc>();
let __slate_auto_bytes_5 = __slate_slot_56 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_55 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_54 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_51 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
grp_1 = core::ptr::null_mut::<ext4_group_info>();
encrypt = false;
// Cannot create files in a deleted directory
if dir.is_null() || (*dir).__slate_anon_12.i_nlink == 0 {
return ERR_PTR(1i32.wrapping_neg() as i64) as *mut inode;
} else {
*__slate_slot_64 = (*dir).i_sb;
*__slate_slot_55 = EXT4_SB(*__slate_slot_64);
*__slate_slot_54 = ext4_emergency_state(*__slate_slot_64);
if unlikely__2!(*__slate_slot_54) != 0 {
return ERR_PTR(*__slate_slot_54 as i64) as *mut inode;
} else {
*__slate_slot_61 = ext4_get_groups_count(*__slate_slot_64);
trace_ext4_request_inode(dir, mode as i32);
*__slate_slot_58 = new_inode(*__slate_slot_64);
if (*__slate_slot_58).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut inode;
} else {
*__slate_slot_56 = EXT4_I!(*__slate_slot_58);
// Initialize owners and quota early so that we don't have to account
// for quota initialization worst case in standard inode creating
// transaction
if !owner.is_null() {
(*(*__slate_slot_58)).i_mode = mode;
i_uid_write(*__slate_slot_58, *owner.wrapping_offset(0 as isize));
i_gid_write(*__slate_slot_58, *owner.wrapping_offset(1 as isize));
} else {
if (*EXT4_SB(*__slate_slot_64)).s_mount_opt & 4 != 0 {
(*(*__slate_slot_58)).i_mode = mode;
inode_fsuid_set(*__slate_slot_58, idmap);
(*(*__slate_slot_58)).i_gid = (*dir).i_gid;
} else {
inode_init_owner(idmap, *__slate_slot_58, dir as *const inode, mode);
}
}
if ext4_has_feature_project(*__slate_slot_64) {
__v47 = ext4_test_inode_flag(dir, 29i32) != 0;
} else {
__v47 = false;
}
if __v47 {
(*(*__slate_slot_56)).i_projid = (*EXT4_I!(dir)).i_projid;
} else {
(*(*__slate_slot_56)).i_projid = make_kprojid(core::ptr::addr_of_mut!(init_user_ns), 0u32);
}
'__join_0: {
if i_flags & 2097152 == 0 {
*__slate_slot_53 = fscrypt_prepare_new_inode(dir, *__slate_slot_58, &(*core::ptr::addr_of_mut!(encrypt)));
if *__slate_slot_53 != 0 {
break '__join_0;
} else {
if encrypt {
i_flags |= 2048;
}
}
}
*__slate_slot_53 = dquot_initialize(*__slate_slot_58);
if *__slate_slot_53 == 0 {
if handle.is_null() && !unsafe { (*(*__slate_slot_55)).s_journal }.is_null() && i_flags & 2097152 == 0 {
*__slate_slot_54 = ext4_xattr_credits_for_new_inode(dir, mode as u32, encrypt);
if *__slate_slot_54 < 0 {
*__slate_slot_53 = *__slate_slot_54;
break '__join_0;
} else {
nblocks = nblocks.wrapping_add(*__slate_slot_54);
}
}
if goal == 0 {
goal = (*(*__slate_slot_55)).s_inode_goal;
}
if goal != 0 && goal <= (*(*(*__slate_slot_55)).s_es).s_inodes_count {
group = ((goal.wrapping_sub(1) as u64) / EXT4_INODES_PER_GROUP!(*__slate_slot_64)) as u32;
ino = (goal.wrapping_sub(1) as u64) % EXT4_INODES_PER_GROUP!(*__slate_slot_64);
*__slate_slot_54 = 0;
} else {
if S_ISDIR!(mode) {
*__slate_slot_54 = find_group_orlov(*__slate_slot_64, dir, core::ptr::addr_of_mut!(group), mode, qstr);
} else {
*__slate_slot_54 = find_group_other(*__slate_slot_64, dir, core::ptr::addr_of_mut!(group), mode);
}
}
(*EXT4_I!(dir)).i_last_alloc_group = group;
*__slate_slot_53 = 28i32.wrapping_neg();
if *__slate_slot_54 != 1i32.wrapping_neg() {
// Normally we will only go through one pass of this loop,
// unless we get unlucky and it turns out the group we selected
// had its last inode grabbed by someone else.
*__slate_slot_51 = 0;
'__join_96: {
'__join_92: {
'__join_90: {
'__join_78: {
while *__slate_slot_51 < *__slate_slot_61 {
*__slate_slot_53 = 5i32.wrapping_neg();
gdp = ext4_get_group_desc(*__slate_slot_64, group, core::ptr::addr_of_mut!(*__slate_slot_62));
if gdp.is_null() {
break '__join_0;
} else {
'__join_83: {
// Check free inodes count before loading bitmap.
if ext4_free_inodes_count(*__slate_slot_64, gdp) != 0 {
if ((*(*__slate_slot_55)).s_mount_state as i32) & 32 == 0 {
grp_1 = ext4_get_group_info(*__slate_slot_64, group);
if grp_1.is_null() {
__v42 = true;
} else {
__v42 = EXT4_MB_GRP_IBITMAP_CORRUPT!(grp_1) != 0;
}
if __v42 {
break '__join_83;
}
}
brelse(inode_bitmap_bh);
inode_bitmap_bh = ext4_read_inode_bitmap(*__slate_slot_64, group);
// Skip groups with suspicious inode tables
if IS_ERR(inode_bitmap_bh as *const ()) {
inode_bitmap_bh = core::ptr::null_mut::<buffer_head>();
} else {
if ((*(*__slate_slot_55)).s_mount_state as i32) & 32 == 0 {
__v41 = EXT4_MB_GRP_IBITMAP_CORRUPT!(grp_1) != 0;
} else {
__v41 = false;
}
if !__v41 {
*__slate_slot_54 = find_inode_bit(*__slate_slot_64, group, inode_bitmap_bh, &mut (*core::ptr::addr_of_mut!(ino)));
if *__slate_slot_54 != 0 {
if group == 0 {
__v40 = ino.wrapping_add(1) < (EXT4_FIRST_INO!(*__slate_slot_64) as u64);
} else {
__v40 = false;
}
if __v40 {
__ext4_error(*__slate_slot_64, c"__ext4_new_inode".as_ptr() as *const u8, 1083u32, false, 0i32, 0u64, c"reserved inode found cleared - inode=%lu".as_ptr() as *const u8, ino.wrapping_add(1));
ext4_mark_group_bitmap_corrupted(*__slate_slot_64, group, 8u32);
} else {
if ((*(*__slate_slot_55)).s_mount_state as i32) & 32 == 0 && handle.is_null() {
loop {
if unlikely!(nblocks <= 0) != 0 {
break '__join_96;
} else {
if true {
break;
}
}
}
handle = __ext4_journal_start_sb(core::ptr::null_mut::<inode>(), (*dir).i_sb, line_no, handle_type, nblocks, 0i32, ext4_trans_default_revoke_credits(*__slate_slot_64));
if IS_ERR(handle as *const ()) {
break '__join_92;
}
}
BUFFER_TRACE!();
*__slate_slot_53 = ext4_journal_get_write_access__47!(handle, *__slate_slot_64, inode_bitmap_bh, 1i32);
if *__slate_slot_53 != 0 {
break '__join_90;
} else {
ext4_lock_group(*__slate_slot_64, group);
*__slate_slot_54 = __test_and_set_bit_le(ino, (*inode_bitmap_bh).b_data as *mut ());
if *__slate_slot_54 != 0 {
// Someone already took the bit. Repeat the search
// with lock held.
*__slate_slot_54 = find_inode_bit(*__slate_slot_64, group, inode_bitmap_bh, &mut (*core::ptr::addr_of_mut!(ino)));
if *__slate_slot_54 != 0 {
__set_bit_le(ino, (*inode_bitmap_bh).b_data as *mut ());
*__slate_slot_54 = 0;
} else {
*__slate_slot_54 = 1; // we didn't grab the inode
}
}
ext4_unlock_group(*__slate_slot_64, group);
ino = ino.wrapping_add(1); // the inode bitmap is zero-based
if *__slate_slot_54 == 0 {
break '__join_78;
}
// we grabbed the inode!
}
}
}
}
}
}
}
let __v36: u32 = group.wrapping_add(1);
group = __v36;
if __v36 == *__slate_slot_61 {
group = 0;
}
let __v35: u32 = *__slate_slot_51;
*__slate_slot_51 = __v35.wrapping_add(1);
ino = 0;
}
}
*__slate_slot_53 = 28i32.wrapping_neg();
break '__join_0;
}
BUFFER_TRACE!();
*__slate_slot_53 = ext4_handle_dirty_metadata!(handle, inode_bitmap_bh);
if *__slate_slot_53 != 0 {
ext4_std_error__17!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
} else {
BUFFER_TRACE!();
*__slate_slot_53 = ext4_journal_get_write_access__62!(handle, *__slate_slot_64, *__slate_slot_62, 1i32);
if *__slate_slot_53 != 0 {
ext4_std_error__48!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
} else {
// We may have to initialize the block bitmap if it isn't already
if ext4_has_group_desc_csum(*__slate_slot_64) != 0 && ((*gdp).bg_flags as i32) & 2 != 0 {
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_33 = ext4_read_block_bitmap(*__slate_slot_64, group);
if IS_ERR(*__slate_slot_33 as *const ()) {
*__slate_slot_53 = PTR_ERR(*__slate_slot_33 as *const ()) as i32;
break '__join_0;
} else {
BUFFER_TRACE!();
*__slate_slot_53 = ext4_journal_get_write_access__10!(handle, *__slate_slot_64, *__slate_slot_33, 1i32);
if *__slate_slot_53 != 0 {
brelse(*__slate_slot_33);
ext4_std_error__18!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
} else {
BUFFER_TRACE!();
*__slate_slot_53 = ext4_handle_dirty_metadata__18!(handle, *__slate_slot_33);
// recheck and clear flag under lock if we still need to
ext4_lock_group(*__slate_slot_64, group);
if ext4_has_group_desc_csum(*__slate_slot_64) != 0 && ((*gdp).bg_flags as i32) & 2 != 0 {
let __v30: u16 = (((*gdp).bg_flags as i32) & 65533i32) as u16;
(*gdp).bg_flags = __v30;
ext4_free_group_clusters_set(*__slate_slot_64, gdp, ext4_free_clusters_after_init(*__slate_slot_64, group, gdp));
ext4_block_bitmap_csum_set(*__slate_slot_64, gdp, *__slate_slot_33);
ext4_group_desc_csum_set(*__slate_slot_64, group, gdp);
}
ext4_unlock_group(*__slate_slot_64, group);
brelse(*__slate_slot_33);
if *__slate_slot_53 != 0 {
ext4_std_error__33!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
}
}
}
}
// Update the relevant bg descriptor fields
if ext4_has_group_desc_csum(*__slate_slot_64) != 0 {
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
grp = core::ptr::null_mut::<ext4_group_info>();
if ((*(*__slate_slot_55)).s_mount_state as i32) & 32 == 0 {
grp = ext4_get_group_info(*__slate_slot_64, group);
if grp.is_null() {
*__slate_slot_53 = 117i32.wrapping_neg();
break '__join_0;
} else {
down_read(core::ptr::addr_of_mut!((*grp).alloc_sem));
// protect vs itable
// lazyinit
}
}
ext4_lock_group(*__slate_slot_64, group); // while we modify the bg desc
*__slate_slot_29 = EXT4_INODES_PER_GROUP!(*__slate_slot_64).wrapping_sub(ext4_itable_unused_count(*__slate_slot_64, gdp) as u64) as i32;
if ((*gdp).bg_flags as i32) & 1 != 0 {
let __v25: u16 = (((*gdp).bg_flags as i32) & 65534i32) as u16;
(*gdp).bg_flags = __v25;
*__slate_slot_29 = 0;
}
// Check the relative inode number against the last used
// relative inode number in this group. if it is greater
// we need to update the bg_itable_unused count
if ino > (*__slate_slot_29 as u64) {
ext4_itable_unused_set(*__slate_slot_64, gdp, EXT4_INODES_PER_GROUP!(*__slate_slot_64).wrapping_sub(ino) as u32);
}
if ((*(*__slate_slot_55)).s_mount_state as i32) & 32 == 0 {
up_read(core::ptr::addr_of_mut!((*grp).alloc_sem));
}
} else {
ext4_lock_group(*__slate_slot_64, group);
}
ext4_free_inodes_set(*__slate_slot_64, gdp, ext4_free_inodes_count(*__slate_slot_64, gdp).wrapping_sub(1));
if S_ISDIR!(mode) {
ext4_used_dirs_set(*__slate_slot_64, gdp, ext4_used_dirs_count(*__slate_slot_64, gdp).wrapping_add(1));
if (*(*__slate_slot_55)).s_log_groups_per_flex != 0 {
f = ext4_flex_group(*__slate_slot_55, group);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
while false {
}
let __v19: *mut *mut flex_groups = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_55)).s_flex_groups) as *const *mut *mut flex_groups);
__UNIQUE_ID_rcu_1509 = __v19;
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
let __v21: *mut *mut flex_groups = __UNIQUE_ID_rcu_1509;
*__slate_slot_22 = *__v21.wrapping_offset(f as isize);
rcu_read_unlock();
__v23 = *__slate_slot_22;
atomic_inc(core::ptr::addr_of_mut!((*__v23).used_dirs));
}
}
if ext4_has_group_desc_csum(*__slate_slot_64) != 0 {
ext4_inode_bitmap_csum_set(*__slate_slot_64, gdp, inode_bitmap_bh);
ext4_group_desc_csum_set(*__slate_slot_64, group, gdp);
}
ext4_unlock_group(*__slate_slot_64, group);
BUFFER_TRACE!();
*__slate_slot_53 = ext4_handle_dirty_metadata__10!(handle, *__slate_slot_62);
if *__slate_slot_53 != 0 {
ext4_std_error__40!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
} else {
percpu_counter_dec(core::ptr::addr_of_mut!((*(*__slate_slot_55)).s_freeinodes_counter));
if S_ISDIR!(mode) {
percpu_counter_inc(core::ptr::addr_of_mut!((*(*__slate_slot_55)).s_dirs_counter));
}
if (*(*__slate_slot_55)).s_log_groups_per_flex != 0 {
*__slate_slot_50 = ext4_flex_group(*__slate_slot_55, group);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
while false {
}
let __v12: *mut *mut flex_groups = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_55)).s_flex_groups) as *const *mut *mut flex_groups);
__UNIQUE_ID_rcu_1511 = __v12;
loop {
if false {
if false {
__v10 = true;
} else {
__v10 = rcu_read_lock_held() != 0;
}
__v11 = !__v10;
} else {
__v11 = false;
}
if !__v11 {
break;
}
}
let __v14: *mut *mut flex_groups = __UNIQUE_ID_rcu_1511;
*__slate_slot_15 = *__v14.wrapping_offset(*__slate_slot_50 as isize);
rcu_read_unlock();
__v16 = *__slate_slot_15;
atomic_dec(core::ptr::addr_of_mut!((*__v16).free_inodes));
}
(*(*__slate_slot_58)).i_ino = ino.wrapping_add((group as u64).wrapping_mul(EXT4_INODES_PER_GROUP!(*__slate_slot_64)));
// This is the optimal IO size (for stat), not the fs block size
(*(*__slate_slot_58)).i_blocks = 0;
simple_inode_init_ts(*__slate_slot_58);
(*(*__slate_slot_56)).i_crtime = inode_get_mtime(*__slate_slot_58 as *const inode);
memset(((*(*__slate_slot_56)).i_data.as_mut_ptr() as *mut u32) as *mut (), 0i32, core::mem::size_of::<[u32; 15]>() as u64);
(*(*__slate_slot_56)).i_dir_start_lookup = 0;
(*(*__slate_slot_56)).i_disksize = 0;
// Don't inherit extent flag from directory, amongst others.
(*(*__slate_slot_56)).i_flags = ext4_mask_flags(mode, ((*EXT4_I!(dir)).i_flags & 1644283087u64) as u32) as u64;
let __v9: *mut ext4_inode_info = *__slate_slot_56;
(*__v9).i_flags = (*__v9).i_flags | (i_flags as u64);
(*(*__slate_slot_56)).i_file_acl = 0;
(*(*__slate_slot_56)).i_dtime = 0;
(*(*__slate_slot_56)).i_block_group = group;
(*(*__slate_slot_56)).i_last_alloc_group = 4294967295u32;
ext4_set_inode_flags(*__slate_slot_58, true);
if IS_DIRSYNC!(*__slate_slot_58) {
ext4_handle_sync(handle);
}
if insert_inode_locked(*__slate_slot_58) < 0 {
// Likely a bitmap corruption causing inode to be allocated
// twice.
*__slate_slot_53 = 5i32.wrapping_neg();
__ext4_error(*__slate_slot_64, c"__ext4_new_inode".as_ptr() as *const u8, 1289u32, false, 0i32, 0u64, c"failed to insert inode %llu: doubly allocated?".as_ptr() as *const u8, (*(*__slate_slot_58)).i_ino);
ext4_mark_group_bitmap_corrupted(*__slate_slot_64, group, 8u32);
break '__join_0;
} else {
(*(*__slate_slot_58)).i_generation = get_random_u32();
// Precompute checksum seed for inode metadata
if ext4_has_feature_metadata_csum(*__slate_slot_64) {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
inum = (*(*__slate_slot_58)).i_ino as u32;
r#gen = (*(*__slate_slot_58)).i_generation;
*__slate_slot_6 = ext4_chksum((*(*__slate_slot_55)).s_csum_seed, (core::ptr::addr_of_mut!(inum) as *mut u8) as *const (), core::mem::size_of::<u32>() as u32);
(*(*__slate_slot_56)).i_csum_seed = ext4_chksum(*__slate_slot_6, (core::ptr::addr_of_mut!(r#gen) as *mut u8) as *const (), core::mem::size_of::<u32>() as u32);
}
ext4_set_inode_state(*__slate_slot_58, 0i32);
(*(*__slate_slot_56)).i_extra_isize = (*(*__slate_slot_55)).s_want_extra_isize as u16;
(*(*__slate_slot_56)).i_inline_off = 0;
if ext4_has_feature_inline_data(*__slate_slot_64) && (*(*__slate_slot_56)).i_flags & 2048 == 0 && ((*(*__slate_slot_56)).i_flags & 35651584u64 == 0 || S_ISDIR!(mode)) {
ext4_set_inode_state(*__slate_slot_58, 6i32);
}
// Encrypted inodes cannot have inline data
*__slate_slot_52 = *__slate_slot_58;
*__slate_slot_53 = dquot_alloc_inode(*__slate_slot_58);
if *__slate_slot_53 == 0 {
'__join_2: {
// Since the encryption xattr will always be unique, create it first so
// that it's less likely to end up in an external xattr block and
// prevent its deduplication.
if encrypt {
*__slate_slot_53 = fscrypt_set_context(*__slate_slot_58, handle as *mut ());
if *__slate_slot_53 != 0 {
break '__join_2;
}
}
if (*(*__slate_slot_56)).i_flags & 2097152 == 0 {
*__slate_slot_53 = ext4_init_acl(handle, *__slate_slot_58, dir);
if *__slate_slot_53 != 0 {
break '__join_2;
} else {
*__slate_slot_53 = ext4_init_security(handle, *__slate_slot_58, dir, qstr);
if *__slate_slot_53 != 0 {
break '__join_2;
}
}
}
if ext4_has_feature_extents(*__slate_slot_64) {
// set extent flag only for directory, file and normal symlink
if S_ISDIR!(mode) || S_ISREG!(mode) || S_ISLNK!(mode) {
ext4_set_inode_flag(*__slate_slot_58, 19i32);
ext4_ext_tree_init(handle, *__slate_slot_58);
}
}
ext4_set_inode_mapping_order(*__slate_slot_58);
ext4_update_inode_fsync_trans(handle, *__slate_slot_58, 1i32);
*__slate_slot_53 = ext4_mark_inode_dirty__45!(handle, *__slate_slot_58);
if *__slate_slot_53 != 0 {
ext4_std_error__6!(*__slate_slot_64, *__slate_slot_53);
} else {
trace_ext4_allocate_inode(*__slate_slot_58, dir, mode as i32);
brelse(inode_bitmap_bh);
return *__slate_slot_52;
}
}
dquot_free_inode(*__slate_slot_58);
}
clear_nlink(*__slate_slot_58);
unlock_new_inode(*__slate_slot_58);
break '__join_0;
}
}
}
}
}
ext4_std_error__2!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
}
*__slate_slot_53 = PTR_ERR(handle as *const ()) as i32;
ext4_std_error__13!(*__slate_slot_64, *__slate_slot_53);
break '__join_0;
}
core::arch::asm!("1508: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1508b - ., 3; .popsection\n# {0}", const 1508 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___ext4_new_inode_80, sym __slate_asm_string___ext4_new_inode_81, const 1090 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
dquot_drop(*__slate_slot_58);
let __v2: *mut inode = *__slate_slot_58;
(*__v2).i_flags = (*__v2).i_flags | 32u32;
iput(*__slate_slot_58);
brelse(inode_bitmap_bh);
return ERR_PTR(*__slate_slot_53 as i64) as *mut inode;
}
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_free_inode(handle: *mut jbd2_journal_handle, inode: *mut inode) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_1504: *mut *mut flex_groups = unsafe { core::mem::zeroed() };
let mut _v: core::mem::MaybeUninit<*mut flex_groups> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut flex_groups = core::ptr::addr_of_mut!(_v) as *mut *mut flex_groups;
let mut fg: core::mem::MaybeUninit<*mut flex_groups> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut flex_groups = core::ptr::addr_of_mut!(fg) as *mut *mut flex_groups;
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut grp: core::mem::MaybeUninit<*mut ext4_group_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut ext4_group_info = core::ptr::addr_of_mut!(grp) as *mut *mut ext4_group_info;
let mut cleared: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(cleared) as *mut i32;
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut fatal: i32 = unsafe { core::mem::zeroed() };
let mut sbi: core::mem::MaybeUninit<*mut ext4_sb_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ext4_sb_info = core::ptr::addr_of_mut!(sbi) as *mut *mut ext4_sb_info;
let mut es: core::mem::MaybeUninit<*mut ext4_super_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut ext4_super_block = core::ptr::addr_of_mut!(es) as *mut *mut ext4_super_block;
let mut gdp: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(gdp) as *mut *mut ext4_group_desc;
let mut bit: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u64 = core::ptr::addr_of_mut!(bit) as *mut u64;
let mut block_group: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(block_group) as *mut u32;
let mut bh2: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh2) as *mut *mut buffer_head;
let mut bitmap_bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut ino: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u64 = core::ptr::addr_of_mut!(ino) as *mut u64;
let mut is_directory: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(is_directory) as *mut i32;
let mut sb: *mut super_block = unsafe { core::mem::zeroed() };
unsafe {
sb = (*inode).i_sb;
let __slate_auto_bytes_1 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
bitmap_bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes_3 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
fatal = 0;
let __slate_auto_bytes_9 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_12 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
if sb.is_null() {
while false {
}
_printk(c"\x013EXT4-fs: %s:%d: inode on nonexistent device\n".as_ptr() as *const u8, c"ext4_free_inode".as_ptr() as *mut u8, 254i32);
return;
} else {
if icount_read_once(inode as *const inode) > 1 {
__ext4_msg(sb, c"\x013".as_ptr() as *const u8, c"%s:%d: inode #%llu: count=%d".as_ptr() as *const u8, c"ext4_free_inode".as_ptr() as *mut u8, 259i32, (*inode).i_ino, icount_read_once(inode as *const inode));
return;
} else {
if (*inode).__slate_anon_12.i_nlink != 0 {
__ext4_msg(sb, c"\x013".as_ptr() as *const u8, c"%s:%d: inode #%llu: nlink=%d\n".as_ptr() as *const u8, c"ext4_free_inode".as_ptr() as *mut u8, 265i32, (*inode).i_ino, (*inode).__slate_anon_12.i_nlink);
return;
} else {
'__join_0: {
*__slate_slot_16 = EXT4_SB(sb);
*__slate_slot_23 = (*inode).i_ino;
trace_ext4_free_inode(inode);
dquot_initialize(inode);
dquot_free_inode(inode);
*__slate_slot_24 = S_ISDIR!(unsafe { (*inode).i_mode }) as i32;
// Do this BEFORE marking the inode not in use or returning an error
ext4_clear_inode(inode);
*__slate_slot_17 = (*(*__slate_slot_16)).s_es;
if *__slate_slot_23 < (EXT4_FIRST_INO!(sb) as u64) || *__slate_slot_23 > ((*(*__slate_slot_17)).s_inodes_count as u64) {
__ext4_error(sb, c"ext4_free_inode".as_ptr() as *const u8, 284u32, false, 0i32, 0u64, c"reserved or nonexistent inode %lu".as_ptr() as *const u8, *__slate_slot_23);
} else {
*__slate_slot_20 = ((*__slate_slot_23).wrapping_sub(1) / EXT4_INODES_PER_GROUP!(sb)) as u32;
*__slate_slot_19 = (*__slate_slot_23).wrapping_sub(1) % EXT4_INODES_PER_GROUP!(sb);
bitmap_bh = ext4_read_inode_bitmap(sb, *__slate_slot_20);
// Don't bother if the inode bitmap is corrupt.
if IS_ERR(bitmap_bh as *const ()) {
fatal = PTR_ERR(bitmap_bh as *const ()) as i32;
bitmap_bh = core::ptr::null_mut::<buffer_head>();
} else {
if ((*(*__slate_slot_16)).s_mount_state as i32) & 32 == 0 {
*__slate_slot_11 = ext4_get_group_info(sb, *__slate_slot_20);
if (*__slate_slot_11).is_null() {
__v8 = true;
} else {
__v8 = unlikely__2!(EXT4_MB_GRP_IBITMAP_CORRUPT!(*__slate_slot_11)) != 0;
}
if __v8 {
fatal = 117i32.wrapping_neg();
break '__join_0;
}
}
BUFFER_TRACE!();
fatal = ext4_journal_get_write_access__46!(handle, sb, bitmap_bh, 1i32);
if fatal == 0 {
fatal = 3i32.wrapping_neg();
*__slate_slot_18 = ext4_get_group_desc(sb, *__slate_slot_20, core::ptr::addr_of_mut!(*__slate_slot_21));
if !(*__slate_slot_18).is_null() {
BUFFER_TRACE!();
fatal = ext4_journal_get_write_access__11!(handle, sb, *__slate_slot_21, 1i32);
}
ext4_lock_group(sb, *__slate_slot_20);
*__slate_slot_12 = __test_and_clear_bit_le(*__slate_slot_19, (*bitmap_bh).b_data as *mut ());
if fatal != 0 || *__slate_slot_12 == 0 {
ext4_unlock_group(sb, *__slate_slot_20);
} else {
*__slate_slot_13 = ext4_free_inodes_count(sb, *__slate_slot_18).wrapping_add(1) as i32;
ext4_free_inodes_set(sb, *__slate_slot_18, *__slate_slot_13 as u32);
if *__slate_slot_24 != 0 {
*__slate_slot_13 = ext4_used_dirs_count(sb, *__slate_slot_18).wrapping_sub(1) as i32;
ext4_used_dirs_set(sb, *__slate_slot_18, *__slate_slot_13 as u32);
if percpu_counter_initialized(core::ptr::addr_of_mut!((*(*__slate_slot_16)).s_dirs_counter)) {
percpu_counter_dec(core::ptr::addr_of_mut!((*(*__slate_slot_16)).s_dirs_counter));
}
}
ext4_inode_bitmap_csum_set(sb, *__slate_slot_18, bitmap_bh);
ext4_group_desc_csum_set(sb, *__slate_slot_20, *__slate_slot_18);
ext4_unlock_group(sb, *__slate_slot_20);
if percpu_counter_initialized(core::ptr::addr_of_mut!((*(*__slate_slot_16)).s_freeinodes_counter)) {
percpu_counter_inc(core::ptr::addr_of_mut!((*(*__slate_slot_16)).s_freeinodes_counter));
}
if (*(*__slate_slot_16)).s_log_groups_per_flex != 0 {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rcu_read_lock();
while false {
}
let __v2: *mut *mut flex_groups = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_16)).s_flex_groups) as *const *mut *mut flex_groups);
__UNIQUE_ID_rcu_1504 = __v2;
loop {
if false {
if false {
__v0 = true;
} else {
__v0 = rcu_read_lock_held() != 0;
}
__v1 = !__v0;
} else {
__v1 = false;
}
if !__v1 {
break;
}
}
let __v4: *mut *mut flex_groups = __UNIQUE_ID_rcu_1504;
*__slate_slot_5 = *__v4.wrapping_offset(ext4_flex_group(*__slate_slot_16, *__slate_slot_20) as isize);
rcu_read_unlock();
*__slate_slot_7 = *__slate_slot_5;
atomic_inc(core::ptr::addr_of_mut!((*(*__slate_slot_7)).free_inodes));
if *__slate_slot_24 != 0 {
atomic_dec(core::ptr::addr_of_mut!((*(*__slate_slot_7)).used_dirs));
}
}
BUFFER_TRACE!();
fatal = ext4_handle_dirty_metadata__12!(handle, *__slate_slot_21);
}
if *__slate_slot_12 != 0 {
BUFFER_TRACE!();
*__slate_slot_14 = ext4_handle_dirty_metadata__22!(handle, bitmap_bh);
if fatal == 0 {
fatal = *__slate_slot_14;
}
} else {
__ext4_error(sb, c"ext4_free_inode".as_ptr() as *const u8, 356u32, false, 0i32, 0u64, c"bit already cleared for inode %lu".as_ptr() as *const u8, *__slate_slot_23);
ext4_mark_group_bitmap_corrupted(sb, *__slate_slot_20, 8u32);
}
}
}
}
}
brelse(bitmap_bh);
ext4_std_error__28!(sb, fatal);
}
}
}
}
}

fn ext4_has_group_desc_csum(sb: *mut super_block) -> i32 {
let __v0: bool = ext4_has_feature_gdt_csum(sb) || ext4_has_feature_metadata_csum(sb);
__v0 as i32
}

fn ext4_blocks_count(es: *mut ext4_super_block) -> u64 {
(if (unsafe { (*es).s_feature_incompat }) & 128 != 0 { ((unsafe { (*es).s_blocks_count_hi }) as u64) << 32i32 } else { 0 }) | ((unsafe { (*es).s_blocks_count_lo }) as u64)
}

fn ext4_get_groups_count(sb: *mut super_block) -> u32 {
let ngroups: u32 = unsafe { (*EXT4_SB(sb)).s_groups_count };
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
ngroups
}

fn ext4_flex_group(sbi: *mut ext4_sb_info, block_group: u32) -> u32 {
block_group >> unsafe { (*sbi).s_log_groups_per_flex }
}

fn ext4_flex_bg_size(sbi: *mut ext4_sb_info) -> u32 {
(1i32 << unsafe { (*sbi).s_log_groups_per_flex }) as u32
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


fn ext4_group_lock_ptr(sb: *mut super_block, group: u32) -> *mut spinlock {
bgl_lock_ptr(unsafe { (*EXT4_SB(sb)).s_blockgroup_lock }, group)
}

fn ext4_try_lock_group(sb: *mut super_block, group: u32) -> bool {
if spin_trylock(ext4_group_lock_ptr(sb, group)) == 0 {
return false;
}
// We're able to grab the lock right away, so drop the lock
// contention counter.
atomic_add_unless(unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_lock_busy) }, 1i32.wrapping_neg(), 0i32);
true
}

fn ext4_lock_group(sb: *mut super_block, group: u32) {
if !ext4_try_lock_group(sb, group) {
// The lock is busy, so bump the contention counter,
// and then wait on the spin lock.
atomic_add_unless(unsafe { core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_lock_busy) }, 1i32, 8i32);
spin_lock(ext4_group_lock_ptr(sb, group));
}
}

fn ext4_unlock_group(sb: *mut super_block, group: u32) {
spin_unlock(ext4_group_lock_ptr(sb, group));
}

// Add new method to test whether block and inode bitmaps are properly
// initialized. With uninit_bg reading the block from disk is not enough
// to mark the bitmap uptodate. We need to also zero-out the bitmap
fn bitmap_uptodate(bh: *mut buffer_head) -> i32 {
let __v0: bool = buffer_uptodate(bh as *const buffer_head) != 0 && test_bit__3!(26i32, unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) != 0;
__v0 as i32
}

fn set_bitmap_uptodate(bh: *mut buffer_head) {
set_bit(26i64, (unsafe { core::ptr::addr_of_mut!((*bh).b_state) }) as *mut u64);
}

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

// Ensure @handle has at least @check_creds credits available. If not,
// transaction will be extended or restarted to contain at least @extend_cred
// credits. Before restarting transaction @fn is executed to allow for cleanup
// before the transaction is restarted.
//
// The return value is < 0 in case of error, 0 in case the handle has enough
// credits or transaction extension succeeded, 1 in case transaction had to be
// restarted.
fn ext4_update_inode_fsync_trans(handle: *mut jbd2_journal_handle, inode: *mut inode, datasync: i32) {
let ei: *mut ext4_inode_info = EXT4_I(inode);
let __v0: bool = ext4_handle_valid(handle) != 0 && is_handle_aborted(handle) == 0;
if __v0 {
unsafe {
(*ei).i_sync_tid = (*(*handle).__slate_anon_0.h_transaction).t_tid;
}
if datasync != 0 {
unsafe {
(*ei).i_datasync_tid = (*(*handle).__slate_anon_0.h_transaction).t_tid;
}
}
}
}

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


fn posix_acl_release(acl: *mut posix_acl) {
let __v0: bool = !acl.is_null() && refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*acl).__slate_anon_0.__slate_anon_0.a_refcount) });
if __v0 {
loop {
let ___p: *mut posix_acl = acl;
if !___p.is_null() {
loop {
if offsetof__6!() >= 4096 {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let ___head: *mut kvfree_rcu_head = (unsafe { core::ptr::addr_of_mut!((*___p).__slate_anon_0.__slate_anon_0.a_rcu) }) as *mut kvfree_rcu_head;
unsafe { kvfree_call_rcu(___head, ___p as *mut ()) };
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acl_h_16 {
    e_tag: u16,
    e_perm: u16,
    e_id: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits545U0,
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
fn __do_trace_ext4_free_inode(inode: *mut inode) {
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
let mut __UNIQUE_ID_guard_693: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_693) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_694: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_694) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_696() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_free_inode)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_695: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_695;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode) -> i32>>(__SCT__tp_func_ext4_free_inode as *const ()) };
unsafe { __v12.unwrap()(__data, inode) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_free_inode: __SlateAlign16<[u8; 16]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 102u8, 114u8, 101u8, 101u8, 95u8, 105u8, 110u8, 111u8, 100u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_free_inode_697: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_free_inode) }) as *mut ();

fn trace_ext4_free_inode(inode: *mut inode) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_free_inode.key) }) != 0 {
__do_trace_ext4_free_inode(inode);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_698: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_request_inode(dir: *mut inode, mode: i32) {
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
let mut __UNIQUE_ID_guard_700: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_700) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_701: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_701) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_703() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_request_inode)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_702: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_702;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, i32) -> i32>>(__SCT__tp_func_ext4_request_inode as *const ()) };
unsafe { __v12.unwrap()(__data, dir, mode) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_request_inode: __SlateAlign16<[u8; 19]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 114u8, 101u8, 113u8, 117u8, 101u8, 115u8, 116u8, 95u8, 105u8, 110u8, 111u8, 100u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_request_inode_704: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_request_inode) }) as *mut ();

fn trace_ext4_request_inode(dir: *mut inode, mode: i32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_request_inode.key) }) != 0 {
__do_trace_ext4_request_inode(dir, mode);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_705: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_allocate_inode(inode: *mut inode, dir: *mut inode, mode: i32) {
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
let mut __UNIQUE_ID_guard_707: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_707) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_708: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_708) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_allocate_inode)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_709: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_709;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut inode, *mut inode, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut inode, *mut inode, i32) -> i32>>(__SCT__tp_func_ext4_allocate_inode as *const ()) };
unsafe { __v12.unwrap()(__data, inode, dir, mode) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_allocate_inode: __SlateAlign16<[u8; 20]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 97u8, 116u8, 101u8, 95u8, 105u8, 110u8, 111u8, 100u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_allocate_inode_711: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_allocate_inode) }) as *mut ();

fn trace_ext4_allocate_inode(inode: *mut inode, dir: *mut inode, mode: i32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_allocate_inode.key) }) != 0 {
__do_trace_ext4_allocate_inode(inode, dir, mode);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_712: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ext4_load_inode_bitmap(sb: *mut super_block, group: u64) {
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
let mut __UNIQUE_ID_guard_1015: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1015) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_1016: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1016) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1018() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ext4_load_inode_bitmap)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_1017: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_1017;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut super_block, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut super_block, u64) -> i32>>(__SCT__tp_func_ext4_load_inode_bitmap as *const ()) };
unsafe { __v12.unwrap()(__data, sb, group) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ext4_load_inode_bitmap: __SlateAlign16<[u8; 23]> = __SlateAlign16([101u8, 120u8, 116u8, 52u8, 95u8, 108u8, 111u8, 97u8, 100u8, 95u8, 105u8, 110u8, 111u8, 100u8, 101u8, 95u8, 98u8, 105u8, 116u8, 109u8, 97u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ext4_load_inode_bitmap_1019: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ext4_load_inode_bitmap) }) as *mut ();

fn trace_ext4_load_inode_bitmap(sb: *mut super_block, group: u64) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ext4_load_inode_bitmap.key) }) != 0 {
__do_trace_ext4_load_inode_bitmap(sb, group);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1020: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// ialloc.c contains the inodes allocation and deallocation routines
// The free inodes are managed by bitmaps.  A file system contains several
// blocks groups.  Each group contains 1 bitmap block for blocks, 1 bitmap
// block for inodes, N blocks for the inode table and data blocks.
//
// The file system contains group descriptors which are located after the
// super block.  Each descriptor contains the number of the bitmap block and
// the free blocks count in the block.
/// To avoid calling the atomic setbit hundreds or thousands of times, we only
/// need to use it within a single byte (to ensure we get endianness right).
/// We can use memset for the rest of the bitmap as there are no other users.
#[unsafe(no_mangle)]
extern "C" fn ext4_mark_bitmap_end(start_bit: i32, end_bit: i32, bitmap: *mut u8) {
let mut i: i32 = 0;
if start_bit >= end_bit {
return;
}
let mut __v0: i32;
if false {
unsafe { _printk(c"mark end bits +%d through +%d used\n".as_ptr() as *const u8, start_bit, end_bit) };
}
__v0 = 0;
i = start_bit;
while (i as u64) < (start_bit.wrapping_add(7) as u64) & !7 {
__set_bit_le(i as u64, bitmap as *mut ());
i = i.wrapping_add(1);
}
if i < end_bit {
unsafe { memset(bitmap.wrapping_offset((i >> 3i32) as isize) as *mut (), 255i32, (end_bit.wrapping_sub(i) >> 3i32) as u64) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ialloc.ext4_end_bitmap_read")]
extern "C" fn ext4_end_bitmap_read(bio: *mut bio) {
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let uptodate: bool = unsafe { bio_endio_bh(bio, core::ptr::addr_of_mut!(bh)) };
if uptodate {
set_buffer_uptodate(bh);
set_bitmap_uptodate(bh);
}
unsafe { unlock_buffer(bh) };
}

fn ext4_validate_inode_bitmap(sb: *mut super_block, desc: *mut ext4_group_desc, block_group: u32, bh: *mut buffer_head) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut grp: core::mem::MaybeUninit<*mut ext4_group_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut ext4_group_info = core::ptr::addr_of_mut!(grp) as *mut *mut ext4_group_info;
let mut blk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(blk) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if ((*EXT4_SB(sb)).s_mount_state as i32) & 32 != 0 {
return 0;
} else {
if buffer_verified(bh as *const buffer_head) != 0 {
return 0;
} else {
*__slate_slot_2 = ext4_get_group_info(sb, block_group);
if (*__slate_slot_2).is_null() {
__v1 = true;
} else {
__v1 = EXT4_MB_GRP_IBITMAP_CORRUPT!(*__slate_slot_2) != 0;
}
if __v1 {
return 117i32.wrapping_neg();
} else {
ext4_lock_group(sb, block_group);
if buffer_verified(bh as *const buffer_head) == 0 {
*__slate_slot_3 = ext4_inode_bitmap(sb, desc);
if ext4_inode_bitmap_csum_verify(sb, desc, bh) == 0 {
__v0 = true;
} else {
__v0 = ext4_simulate_fail(sb, 4u64);
}
if __v0 {
ext4_unlock_group(sb, block_group);
__ext4_error(sb, c"ext4_validate_inode_bitmap".as_ptr() as *const u8, 107u32, false, 0i32, 0u64, c"Corrupt inode bitmap - block_group = %u, inode_bitmap = %llu".as_ptr() as *const u8, block_group, *__slate_slot_3);
ext4_mark_group_bitmap_corrupted(sb, block_group, 8u32);
return 74i32.wrapping_neg();
} else {
set_buffer_verified(bh);
}
}
ext4_unlock_group(sb, block_group);
return 0;
}
}
}
}
}

/// Read the inode allocation bitmap for a given block_group, reading
/// into the specified slot in the superblock's bitmap cache.
///
/// Return buffer_head of bitmap on success, or an ERR_PTR on error.
fn ext4_read_inode_bitmap(sb: *mut super_block, block_group: u32) -> *mut buffer_head {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut bitmap_blk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(bitmap_blk) as *mut u64;
let mut bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut desc: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(desc) as *mut *mut ext4_group_desc;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
sbi = EXT4_SB(sb);
bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = ext4_get_group_desc(sb, block_group, core::ptr::null_mut::<*mut buffer_head>());
if (*__slate_slot_5).is_null() {
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
*__slate_slot_2 = ext4_inode_bitmap(sb, *__slate_slot_5);
if *__slate_slot_2 <= ((*(*sbi).s_es).s_first_data_block as u64) {
__v0 = true;
} else {
__v0 = *__slate_slot_2 >= ext4_blocks_count((*sbi).s_es);
}
if __v0 {
__ext4_error(sb, c"ext4_read_inode_bitmap".as_ptr() as *const u8, 141u32, false, 0i32, 0u64, c"Invalid inode bitmap blk %llu in block_group %u".as_ptr() as *const u8, *__slate_slot_2, block_group);
ext4_mark_group_bitmap_corrupted(sb, block_group, 8u32);
return ERR_PTR(117i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
bh = sb_getblk(sb, *__slate_slot_2);
if unlikely!(bh.is_null()) != 0 {
__ext4_warning(sb, c"ext4_read_inode_bitmap".as_ptr() as *const u8, 150u32, c"Cannot read inode bitmap - block_group = %u, inode_bitmap = %llu".as_ptr() as *const u8, block_group, *__slate_slot_2);
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut buffer_head;
} else {
'__join_0: {
if bitmap_uptodate(bh) == 0 {
lock_buffer(bh);
if bitmap_uptodate(bh) != 0 {
unlock_buffer(bh);
} else {
ext4_lock_group(sb, block_group);
if ext4_has_group_desc_csum(sb) != 0 && ((*(*__slate_slot_5)).bg_flags as i32) & 1 != 0 {
if block_group == 0 {
ext4_unlock_group(sb, block_group);
unlock_buffer(bh);
__ext4_error(sb, c"ext4_read_inode_bitmap".as_ptr() as *const u8, 169u32, false, 0i32, 0u64, c"Inode bitmap for bg 0 marked uninitialized".as_ptr() as *const u8);
*__slate_slot_1 = 117i32.wrapping_neg();
break '__join_0;
} else {
memset((*bh).b_data as *mut (), 0i32, EXT4_INODES_PER_GROUP!(sb).wrapping_add(7) / 8);
ext4_mark_bitmap_end(EXT4_INODES_PER_GROUP!(sb) as i32, unsafe { (*sb).s_blocksize }.wrapping_mul(8) as i32, (*bh).b_data);
set_bitmap_uptodate(bh);
set_buffer_uptodate(bh);
set_buffer_verified(bh);
ext4_unlock_group(sb, block_group);
unlock_buffer(bh);
return bh;
}
} else {
ext4_unlock_group(sb, block_group);
if buffer_uptodate(bh as *const buffer_head) != 0 {
// if not uninit if bh is uptodate,
// bitmap is also uptodate
set_bitmap_uptodate(bh);
unlock_buffer(bh);
} else {
// submit the buffer_head for reading
trace_ext4_load_inode_bitmap(sb, block_group as u64);
ext4_read_bh(bh, 12288u32, Some(ext4_end_bitmap_read), ext4_simulate_fail(sb, 3u64));
if buffer_uptodate(bh as *const buffer_head) == 0 {
put_bh(bh);
__ext4_error(sb, c"ext4_read_inode_bitmap".as_ptr() as *const u8, 205u32, false, 5i32, 0u64, c"Cannot read inode bitmap - block_group = %u, inode_bitmap = %llu".as_ptr() as *const u8, block_group, *__slate_slot_2);
ext4_mark_group_bitmap_corrupted(sb, block_group, 8u32);
return ERR_PTR(5i32.wrapping_neg() as i64) as *mut buffer_head;
}
}
}
}
}
*__slate_slot_1 = ext4_validate_inode_bitmap(sb, *__slate_slot_5, block_group, bh);
if *__slate_slot_1 == 0 {
return bh;
}
}
put_bh(bh);
return ERR_PTR(*__slate_slot_1 as i64) as *mut buffer_head;
}
}
}
}
}

// NOTE! When we get the inode, we're the only people
// that have access to it, and as such there are no
// race conditions we have to worry about. The inode
// is not on the hash-lists, and it cannot be reached
// through the filesystem because the directory entry
// has been deleted earlier.
//
// HOWEVER: we must make sure that we get no aliases,
// which means that we have to call "clear_inode()"
// _before_ we mark the inode not in use in the inode
// bitmaps. Otherwise a newly created file might use
// the same inode number (not actually the same pointer
// though), and then we'd have two inodes sharing the
// same inode number and space on the harddisk.
#[repr(C)]
#[derive(Clone, Copy)]
struct orlov_stats {
    free_clusters: u64,
    free_inodes: u32,
    used_dirs: u32,
}


// Helper function for Orlov's allocator; returns critical information
// for a particular block group or flex_bg.  If flex_size is 1, then g
// is a block group number; otherwise it is flex_bg number.
fn get_orlov_stats(sb: *mut super_block, g: u32, flex_size: i32, stats: &mut orlov_stats) {
if flex_size > 1 {
let mut fg: *mut flex_groups = unsafe { core::mem::zeroed() };
let mut __v0: *mut flex_groups;
rcu_read_lock();
let mut __v1: *mut *mut flex_groups;
let mut __v2: *mut *mut flex_groups;
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut *mut flex_groups>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1507() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*EXT4_SB(sb)).s_flex_groups) as *const *mut *mut flex_groups) };
let __UNIQUE_ID_rcu_1506: *mut *mut flex_groups = __v2;
loop {
let __v5: bool = if false {
let __v6: bool = false || rcu_read_lock_held() != 0;
!__v6
} else {
false
};
if !__v5 {
break;
}
}
__v1 = __UNIQUE_ID_rcu_1506;
let _v: *mut flex_groups = unsafe { *__v1.wrapping_offset(g as isize) };
rcu_read_unlock();
__v0 = _v;
fg = __v0;
unsafe {
(*(stats as *mut orlov_stats)).free_inodes = atomic_read(core::ptr::addr_of_mut!((*fg).free_inodes) as *const __SlateRecord_types_h_190) as u32;
}
unsafe {
(*(stats as *mut orlov_stats)).free_clusters = atomic64_read(core::ptr::addr_of_mut!((*fg).free_clusters) as *const __SlateRecord_types_h_197) as u64;
}
unsafe {
(*(stats as *mut orlov_stats)).used_dirs = atomic_read(core::ptr::addr_of_mut!((*fg).used_dirs) as *const __SlateRecord_types_h_190) as u32;
}
return;
}
let desc: *mut ext4_group_desc = unsafe { ext4_get_group_desc(sb, g, core::ptr::null_mut::<*mut buffer_head>()) };
if !desc.is_null() {
unsafe {
(*(stats as *mut orlov_stats)).free_inodes = ext4_free_inodes_count(sb, desc);
}
unsafe {
(*(stats as *mut orlov_stats)).free_clusters = ext4_free_group_clusters(sb, desc) as u64;
}
unsafe {
(*(stats as *mut orlov_stats)).used_dirs = ext4_used_dirs_count(sb, desc);
}
} else {
unsafe {
(*(stats as *mut orlov_stats)).free_inodes = 0;
}
unsafe {
(*(stats as *mut orlov_stats)).free_clusters = 0;
}
unsafe {
(*(stats as *mut orlov_stats)).used_dirs = 0;
}
}
}

// Orlov's allocator for directories.
//
// We always try to spread first-level directories.
//
// If there are blockgroups with both free inodes and free clusters counts
// not worse than average we return one with smallest directory count.
// Otherwise we simply return a random group.
//
// For the rest rules look so:
//
// It's OK to put directory into a group unless
// it has too many directories already (max_dirs) or
// it has too few free inodes left (min_inodes) or
// it has too few free clusters left (min_clusters) or
// Parent's group is preferred, if it doesn't satisfy these
// conditions we search cyclically through the rest. If none
// of the groups look good we just look for a group with more
// free inodes than average (starting at parent's group).
fn find_group_orlov(sb: *mut super_block, parent: *mut inode, group: *mut u32, mode: u16, qstr: *const qstr) -> i32 {
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut best_ndir: i32 = unsafe { core::mem::zeroed() };
let mut __v15: bool = unsafe { core::mem::zeroed() };
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut hinfo: core::mem::MaybeUninit<dx_hash_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut dx_hash_info = core::ptr::addr_of_mut!(hinfo) as *mut dx_hash_info;
let mut flex_size: i32 = unsafe { core::mem::zeroed() };
let mut stats: core::mem::MaybeUninit<orlov_stats> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut orlov_stats = core::ptr::addr_of_mut!(stats) as *mut orlov_stats;
let mut desc: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(desc) as *mut *mut ext4_group_desc;
let mut ngroups: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(ngroups) as *mut u32;
let mut g: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(g) as *mut u32;
let mut grp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u32 = core::ptr::addr_of_mut!(grp) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut min_clusters: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(min_clusters) as *mut i32;
let mut min_inodes: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut i32 = core::ptr::addr_of_mut!(min_inodes) as *mut i32;
let mut max_dirs: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut i32 = core::ptr::addr_of_mut!(max_dirs) as *mut i32;
let mut ndirs: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u32 = core::ptr::addr_of_mut!(ndirs) as *mut u32;
let mut avefreec: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u64 = core::ptr::addr_of_mut!(avefreec) as *mut u64;
let mut freec: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u64 = core::ptr::addr_of_mut!(freec) as *mut u64;
let mut grp_free: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(grp_free) as *mut u32;
let mut avefreei: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u32 = core::ptr::addr_of_mut!(avefreei) as *mut u32;
let mut freei: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u32 = core::ptr::addr_of_mut!(freei) as *mut u32;
let mut inodes_per_group: i32 = unsafe { core::mem::zeroed() };
let mut real_ngroups: u32 = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut parent_group: u32 = unsafe { core::mem::zeroed() };
unsafe {
parent_group = (*EXT4_I!(parent)).i_block_group;
sbi = EXT4_SB(sb);
real_ngroups = ext4_get_groups_count(sb);
inodes_per_group = EXT4_INODES_PER_GROUP!(sb) as i32;
let __slate_auto_bytes = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_12 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_14 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 16usize);
flex_size = ext4_flex_bg_size(sbi) as i32;
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
*__slate_slot_23 = real_ngroups;
if flex_size > 1 {
*__slate_slot_23 = real_ngroups.wrapping_add(flex_size as u32).wrapping_sub(1) >> (*sbi).s_log_groups_per_flex;
parent_group >>= (*sbi).s_log_groups_per_flex;
}
*__slate_slot_35 = percpu_counter_read_positive(core::ptr::addr_of_mut!((*sbi).s_freeinodes_counter)) as u32;
*__slate_slot_34 = *__slate_slot_35 / *__slate_slot_23;
*__slate_slot_32 = percpu_counter_read_positive(core::ptr::addr_of_mut!((*sbi).s_freeclusters_counter)) as u64;
*__slate_slot_31 = *__slate_slot_32;
do_div__3!(*__slate_slot_31, *__slate_slot_23);
*__slate_slot_30 = percpu_counter_read_positive(core::ptr::addr_of_mut!((*sbi).s_dirs_counter)) as u32;
if S_ISDIR!(mode) {
if parent == d_inode((*sb).s_root as *const dentry) {
__v15 = true;
} else {
__v15 = ext4_test_inode_flag(parent, 17i32) != 0;
}
__v16 = __v15;
} else {
__v16 = false;
}
'__join_9: {
'__join_33: {
if __v16 {
best_ndir = inodes_per_group;
ret = 1i32.wrapping_neg();
if !qstr.is_null() {
(*__slate_slot_19).hash_version = 1;
(*__slate_slot_19).seed = (*sbi).s_hash_seed.as_mut_ptr() as *mut u32;
ext4fs_dirhash(parent as *const inode, (*qstr).name as *const u8, (*qstr).__slate_anon_0.__slate_anon_0.len as i32, core::ptr::addr_of_mut!(*__slate_slot_19));
parent_group = (*__slate_slot_19).hash % *__slate_slot_23;
} else {
parent_group = get_random_u32_below(*__slate_slot_23);
}
*__slate_slot_26 = 0;
while *__slate_slot_26 < *__slate_slot_23 {
*__slate_slot_24 = parent_group.wrapping_add(*__slate_slot_26) % *__slate_slot_23;
get_orlov_stats(sb, *__slate_slot_24, flex_size, &mut (*core::ptr::addr_of_mut!(*__slate_slot_21)));
if (*__slate_slot_21).free_inodes != 0 {
if (*__slate_slot_21).used_dirs < (best_ndir as u32) {
if !((*__slate_slot_21).free_inodes < *__slate_slot_34) {
if !((*__slate_slot_21).free_clusters < *__slate_slot_31) {
*__slate_slot_25 = *__slate_slot_24;
ret = 0;
best_ndir = (*__slate_slot_21).used_dirs as i32;
}
}
}
}
let __v12: u32 = *__slate_slot_26;
*__slate_slot_26 = __v12.wrapping_add(1);
}
if ret != 0 {
break '__join_9;
}
} else {
*__slate_slot_29 = (*__slate_slot_30 / *__slate_slot_23).wrapping_add((inodes_per_group.wrapping_mul(flex_size) / 16) as u32) as i32;
*__slate_slot_28 = (*__slate_slot_34).wrapping_sub((inodes_per_group.wrapping_mul(flex_size) / 4) as u32) as i32;
if *__slate_slot_28 < 1 {
*__slate_slot_28 = 1;
}
*__slate_slot_27 = (*__slate_slot_31).wrapping_sub(EXT4_CLUSTERS_PER_GROUP!(sb).wrapping_mul(flex_size as u64) / 4) as i32;
if *__slate_slot_27 < 0 {
*__slate_slot_27 = 0;
}
// Start looking in the flex group where we last allocated an
// inode for this parent directory
if (*EXT4_I!(parent)).i_last_alloc_group != 4294967295u32 {
parent_group = (*EXT4_I!(parent)).i_last_alloc_group;
if flex_size > 1 {
parent_group >>= (*sbi).s_log_groups_per_flex;
}
}
*__slate_slot_26 = 0;
loop {
if *__slate_slot_26 < *__slate_slot_23 {
*__slate_slot_25 = parent_group.wrapping_add(*__slate_slot_26) % *__slate_slot_23;
get_orlov_stats(sb, *__slate_slot_25, flex_size, &mut (*core::ptr::addr_of_mut!(*__slate_slot_21)));
if (*__slate_slot_21).used_dirs < (*__slate_slot_29 as u32) {
if (*__slate_slot_21).free_inodes >= (*__slate_slot_28 as u32) {
if (*__slate_slot_21).free_clusters >= (*__slate_slot_27 as u64) {
break '__join_33;
}
}
}
let __v3: u32 = *__slate_slot_26;
*__slate_slot_26 = __v3.wrapping_add(1);
} else {
break '__join_9;
}
}
}
}
if flex_size == 1 {
*group = *__slate_slot_25;
return 0;
} else {
// We pack inodes at the beginning of the flexgroup's
// inode tables.  Block allocation decisions will do
// something similar, although regular files will
// start at 2nd block group of the flexgroup.  See
// ext4_ext_find_goal() and ext4_find_near().
let __v10: u32 = *__slate_slot_25;
*__slate_slot_25 = __v10.wrapping_mul(flex_size as u32);
*__slate_slot_26 = 0;
loop {
if *__slate_slot_26 < (flex_size as u32) {
if (*__slate_slot_25).wrapping_add(*__slate_slot_26) >= real_ngroups {
break '__join_9;
} else {
*__slate_slot_22 = ext4_get_group_desc(sb, (*__slate_slot_25).wrapping_add(*__slate_slot_26), core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_22).is_null() {
__v8 = ext4_free_inodes_count(sb, *__slate_slot_22) != 0;
} else {
__v8 = false;
}
if __v8 {
break;
} else {
let __v7: u32 = *__slate_slot_26;
*__slate_slot_26 = __v7.wrapping_add(1);
}
}
} else {
break '__join_9;
}
}
*group = (*__slate_slot_25).wrapping_add(*__slate_slot_26);
return 0;
}
}
*__slate_slot_23 = real_ngroups;
*__slate_slot_34 = *__slate_slot_35 / *__slate_slot_23;
loop {
parent_group = (*EXT4_I!(parent)).i_block_group;
*__slate_slot_26 = 0;
while *__slate_slot_26 < *__slate_slot_23 {
*__slate_slot_25 = parent_group.wrapping_add(*__slate_slot_26) % *__slate_slot_23;
*__slate_slot_22 = ext4_get_group_desc(sb, *__slate_slot_25, core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_22).is_null() {
*__slate_slot_33 = ext4_free_inodes_count(sb, *__slate_slot_22);
if *__slate_slot_33 != 0 && *__slate_slot_33 >= *__slate_slot_34 {
*group = *__slate_slot_25;
return 0;
}
}
let __v1: u32 = *__slate_slot_26;
*__slate_slot_26 = __v1.wrapping_add(1);
}
if *__slate_slot_34 != 0 {
// The free-inodes counter is approximate, and for really small
// filesystems the above test can fail to find any blockgroups
*__slate_slot_34 = 0;
} else {
break;
}
}
return 1i32.wrapping_neg();
}
}

fn find_group_other(sb: *mut super_block, parent: *mut inode, group: *mut u32, mode: u16) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut __v17: bool = unsafe { core::mem::zeroed() };
let mut __v18: bool = unsafe { core::mem::zeroed() };
let mut __v21: bool = unsafe { core::mem::zeroed() };
let mut retry: i32 = unsafe { core::mem::zeroed() };
let mut flex_size: i32 = unsafe { core::mem::zeroed() };
let mut desc: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(desc) as *mut *mut ext4_group_desc;
let mut ngroups: u32 = unsafe { core::mem::zeroed() };
let mut last: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(last) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut parent_group: u32 = unsafe { core::mem::zeroed() };
unsafe {
parent_group = (*EXT4_I!(parent)).i_block_group;
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
ngroups = ext4_get_groups_count(sb);
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
flex_size = ext4_flex_bg_size(EXT4_SB(sb)) as i32;
// Try to place the inode is the same flex group as its
// parent.  If we can't find space, use the Orlov algorithm to
// find another flex group, and store that information in the
// parent directory's inode information so that use that flex
// group for future allocations.
if flex_size > 1 {
retry = 0;
loop {
let __v22: u32 = parent_group & (!flex_size.wrapping_sub(1) as u32);
parent_group = __v22;
*__slate_slot_28 = parent_group.wrapping_add(flex_size as u32);
if *__slate_slot_28 > ngroups {
*__slate_slot_28 = ngroups;
}
*__slate_slot_29 = parent_group;
while *__slate_slot_29 < *__slate_slot_28 {
*__slate_slot_26 = ext4_get_group_desc(sb, *__slate_slot_29, core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_26).is_null() {
__v21 = ext4_free_inodes_count(sb, *__slate_slot_26) != 0;
} else {
__v21 = false;
}
if __v21 {
*group = *__slate_slot_29;
return 0;
} else {
let __v20: u32 = *__slate_slot_29;
*__slate_slot_29 = __v20.wrapping_add(1);
}
}
if retry == 0 {
__v18 = (*EXT4_I!(parent)).i_last_alloc_group != 4294967295u32;
} else {
__v18 = false;
}
if __v18 {
retry = 1;
parent_group = (*EXT4_I!(parent)).i_last_alloc_group;
} else {
break;
}
}
// If this didn't work, use the Orlov search algorithm
// to find a new flex group; we pass in the mode to
// avoid the topdir algorithms.
*group = parent_group.wrapping_add(flex_size as u32);
if *group > ngroups {
*group = 0;
}
return find_group_orlov(sb, parent, group, mode, core::ptr::null::<qstr>());
} else {
// Try to place the inode in its parent directory
*group = parent_group;
*__slate_slot_26 = ext4_get_group_desc(sb, *group, core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_26).is_null() {
__v17 = ext4_free_inodes_count(sb, *__slate_slot_26) != 0;
} else {
__v17 = false;
}
if __v17 {
__v16 = ext4_free_group_clusters(sb, *__slate_slot_26) != 0;
} else {
__v16 = false;
}
if __v16 {
return 0;
} else {
// We're going to place this inode in a different blockgroup from its
// parent.  We want to cause files in a common directory to all land in
// the same blockgroup.  But we want files which are in a different
// directory which shares a blockgroup with our parent to land in a
// different blockgroup.
//
// So add our directory's i_ino into the starting point for the hash.
*group = unsafe { *group }.wrapping_add((*parent).i_ino as u32) % ngroups;
// Use a quadratic hash to find a group with a free inode and some free
// blocks.
*__slate_slot_29 = 1;
while *__slate_slot_29 < ngroups {
let __v14: u32 = *group;
*group = __v14.wrapping_add(*__slate_slot_29);
if *group >= ngroups {
let __v11: u32 = *group;
*group = __v11.wrapping_sub(ngroups);
}
*__slate_slot_26 = ext4_get_group_desc(sb, *group, core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_26).is_null() {
__v9 = ext4_free_inodes_count(sb, *__slate_slot_26) != 0;
} else {
__v9 = false;
}
if __v9 {
__v8 = ext4_free_group_clusters(sb, *__slate_slot_26) != 0;
} else {
__v8 = false;
}
if __v8 {
return 0;
} else {
*__slate_slot_29 <<= 1i32;
}
}
// That failed: try linear search for a free inode, even if that group
// has no free blocks.
*group = parent_group;
*__slate_slot_29 = 0;
while *__slate_slot_29 < ngroups {
let __v4: u32 = *group;
let __v3: u32 = __v4.wrapping_add(1);
*group = __v3;
if __v3 >= ngroups {
*group = 0;
}
*__slate_slot_26 = ext4_get_group_desc(sb, *group, core::ptr::null_mut::<*mut buffer_head>());
if !(*__slate_slot_26).is_null() {
__v2 = ext4_free_inodes_count(sb, *__slate_slot_26) != 0;
} else {
__v2 = false;
}
if __v2 {
return 0;
} else {
let __v1: u32 = *__slate_slot_29;
*__slate_slot_29 = __v1.wrapping_add(1);
}
}
return 1i32.wrapping_neg();
}
}
}
}

// In no journal mode, if an inode has recently been deleted, we want
// to avoid reusing it until we're reasonably sure the inode table
// block has been written back to disk.  (Yes, these values are
// somewhat arbitrary...)
fn recently_deleted(sb: *mut super_block, group: u32, ino: i32) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
// Inode was never used in this filesystem?
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut now: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(now) as *mut u32;
let mut dtime: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(dtime) as *mut u32;
let mut recentcy: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut offset: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(offset) as *mut i32;
let mut inodes_per_block: i32 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut buffer_head = core::ptr::addr_of_mut!(bh) as *mut *mut buffer_head;
let mut raw_inode: core::mem::MaybeUninit<*mut ext4_inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ext4_inode = core::ptr::addr_of_mut!(raw_inode) as *mut *mut ext4_inode;
let mut gdp: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(gdp) as *mut *mut ext4_group_desc;
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
inodes_per_block = (*EXT4_SB(sb)).s_inodes_per_block as i32;
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ret = 0;
recentcy = 60;
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_13 = ext4_get_group_desc(sb, group, core::ptr::null_mut::<*mut buffer_head>());
if unlikely!((*__slate_slot_13).is_null()) != 0 {
return 0;
} else {
if ext4_has_group_desc_csum(sb) != 0 {
if ((*(*__slate_slot_13)).bg_flags as i32) & 1 != 0 {
__v3 = true;
} else {
__v3 = (ino as u64) >= EXT4_INODES_PER_GROUP!(sb).wrapping_sub(ext4_itable_unused_count(sb, *__slate_slot_13) as u64);
}
__v4 = __v3;
} else {
__v4 = false;
}
if __v4 {
return 0;
} else {
*__slate_slot_11 = sb_find_get_block(sb, ext4_inode_table(sb, *__slate_slot_13).wrapping_add((ino / inodes_per_block) as u64));
if (*__slate_slot_11).is_null() {
__v2 = true;
} else {
__v2 = buffer_uptodate(*__slate_slot_11 as *const buffer_head) == 0;
}
if !__v2 {
// If the block is not in the buffer cache, then it
// must have been written out, or, most unlikely, is
// being migrated - false failure should be OK here.
*__slate_slot_9 = (ino % inodes_per_block).wrapping_mul(EXT4_INODE_SIZE!(sb));
*__slate_slot_12 = unsafe { (*(*__slate_slot_11)).b_data }.wrapping_offset(*__slate_slot_9 as isize) as *mut ext4_inode;
// i_dtime is only 32 bits on disk, but we only care about relative
// times in the range of a few minutes (i.e. long enough to sync a
// recently-deleted inode to disk), so using the low 32 bits of the
// clock (a 68 year range) is enough, see time_before32()
*__slate_slot_6 = (*(*__slate_slot_12)).i_dtime;
*__slate_slot_5 = ktime_get_real_seconds() as u32;
if buffer_dirty(*__slate_slot_11 as *const buffer_head) != 0 {
recentcy = recentcy.wrapping_add(300);
}
if *__slate_slot_6 != 0 && time_before32!(*__slate_slot_6, *__slate_slot_5) && time_before32!(*__slate_slot_5, (*__slate_slot_6).wrapping_add(recentcy as u32)) {
ret = 1;
}
}
brelse(*__slate_slot_11);
return ret;
}
}
}
}

fn find_inode_bit(sb: *mut super_block, group: u32, bitmap: *mut buffer_head, ino: &mut u64) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut recently_deleted_ino: u64 = unsafe { core::mem::zeroed() };
let mut check_recently_deleted: bool = unsafe { core::mem::zeroed() };
unsafe {
check_recently_deleted = unsafe { (*EXT4_SB(sb)).s_journal }.is_null();
recently_deleted_ino = EXT4_INODES_PER_GROUP!(sb);
'__join_2: {
loop {
*(ino as *mut u64) = find_next_zero_bit_le(((*bitmap).b_data as *mut u64) as *const (), EXT4_INODES_PER_GROUP!(sb), *(ino as *mut u64));
if *(ino as *mut u64) >= EXT4_INODES_PER_GROUP!(sb) {
break '__join_2;
} else {
if check_recently_deleted {
__v0 = recently_deleted(sb, group, *(ino as *mut u64) as i32) != 0;
} else {
__v0 = false;
}
if __v0 {
recently_deleted_ino = *(ino as *mut u64);
*(ino as *mut u64) = unsafe { *(ino as *mut u64) }.wrapping_add(1);
if !(*(ino as *mut u64) < EXT4_INODES_PER_GROUP!(sb)) {
break '__join_2;
}
} else {
break;
}
}
}
return 1;
}
if recently_deleted_ino >= EXT4_INODES_PER_GROUP!(sb) {
return 0;
} else {
// Not reusing recently deleted inodes is mostly a preference. We don't
// want to report ENOSPC or skew allocation patterns because of that.
// So return even recently deleted inode if we could find better in the
// given range.
*(ino as *mut u64) = recently_deleted_ino;
return 1;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_mark_inode_used(sb: *mut super_block, ino: i32) -> i32 {
let mut free: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(free) as *mut i32;
let mut block_bitmap_bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut buffer_head = core::ptr::addr_of_mut!(block_bitmap_bh) as *mut *mut buffer_head;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut bit: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(bit) as *mut i32;
let mut group: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(group) as *mut u32;
let mut gdp: core::mem::MaybeUninit<*mut ext4_group_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut ext4_group_desc = core::ptr::addr_of_mut!(gdp) as *mut *mut ext4_group_desc;
let mut group_desc_bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut inode_bitmap_bh: *mut buffer_head = unsafe { core::mem::zeroed() };
unsafe {
let max_ino: u64 = (*(*EXT4_SB(sb)).s_es).s_inodes_count as u64;
inode_bitmap_bh = core::ptr::null_mut::<buffer_head>();
group_desc_bh = core::ptr::null_mut::<buffer_head>();
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
if ino < EXT4_FIRST_INO!(sb) || (ino as u64) > max_ino {
return 117i32.wrapping_neg();
} else {
*__slate_slot_10 = ((ino.wrapping_sub(1) as u64) / EXT4_INODES_PER_GROUP!(sb)) as u32;
*__slate_slot_9 = ((ino.wrapping_sub(1) as u64) % EXT4_INODES_PER_GROUP!(sb)) as i32;
inode_bitmap_bh = ext4_read_inode_bitmap(sb, *__slate_slot_10);
if IS_ERR(inode_bitmap_bh as *const ()) {
return PTR_ERR(inode_bitmap_bh as *const ()) as i32;
} else {
'__join_0: {
if test_bit_le(*__slate_slot_9 as u64, (*inode_bitmap_bh).b_data as *const ()) != 0 {
*__slate_slot_8 = 0;
} else {
*__slate_slot_11 = ext4_get_group_desc(sb, *__slate_slot_10, core::ptr::addr_of_mut!(group_desc_bh));
if (*__slate_slot_11).is_null() {
*__slate_slot_8 = 22i32.wrapping_neg();
} else {
__set_bit_le(*__slate_slot_9 as u64, (*inode_bitmap_bh).b_data as *mut ());
BUFFER_TRACE!();
*__slate_slot_8 = ext4_handle_dirty_metadata__27!(inode_bitmap_bh);
if *__slate_slot_8 != 0 {
ext4_std_error__14!(sb, *__slate_slot_8);
} else {
*__slate_slot_8 = sync_dirty_buffer(inode_bitmap_bh);
if *__slate_slot_8 != 0 {
ext4_std_error__37!(sb, *__slate_slot_8);
} else {
// We may have to initialize the block bitmap if it isn't already
if ext4_has_group_desc_csum(sb) != 0 && ((*(*__slate_slot_11)).bg_flags as i32) & 2 != 0 {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = ext4_read_block_bitmap(sb, *__slate_slot_10);
if IS_ERR(*__slate_slot_7 as *const ()) {
*__slate_slot_8 = PTR_ERR(*__slate_slot_7 as *const ()) as i32;
break '__join_0;
} else {
BUFFER_TRACE!();
*__slate_slot_8 = ext4_handle_dirty_metadata__5!(*__slate_slot_7);
sync_dirty_buffer(*__slate_slot_7);
// recheck and clear flag under lock if we still need to
ext4_lock_group(sb, *__slate_slot_10);
if ext4_has_group_desc_csum(sb) != 0 && ((*(*__slate_slot_11)).bg_flags as i32) & 2 != 0 {
let __v6: *mut ext4_group_desc = *__slate_slot_11;
let __v4: u16 = (((*__v6).bg_flags as i32) & 65533i32) as u16;
(*__v6).bg_flags = __v4;
ext4_free_group_clusters_set(sb, *__slate_slot_11, ext4_free_clusters_after_init(sb, *__slate_slot_10, *__slate_slot_11));
ext4_block_bitmap_csum_set(sb, *__slate_slot_11, *__slate_slot_7);
ext4_group_desc_csum_set(sb, *__slate_slot_10, *__slate_slot_11);
}
ext4_unlock_group(sb, *__slate_slot_10);
brelse(*__slate_slot_7);
if *__slate_slot_8 != 0 {
ext4_std_error__27!(sb, *__slate_slot_8);
break '__join_0;
}
}
}
// Update the relevant bg descriptor fields
if ext4_has_group_desc_csum(sb) != 0 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ext4_lock_group(sb, *__slate_slot_10); // while we modify the bg desc
*__slate_slot_3 = EXT4_INODES_PER_GROUP!(sb).wrapping_sub(ext4_itable_unused_count(sb, *__slate_slot_11) as u64) as i32;
if ((*(*__slate_slot_11)).bg_flags as i32) & 1 != 0 {
let __v2: *mut ext4_group_desc = *__slate_slot_11;
let __v0: u16 = (((*__v2).bg_flags as i32) & 65534i32) as u16;
(*__v2).bg_flags = __v0;
*__slate_slot_3 = 0;
}
// Check the relative inode number against the last used
// relative inode number in this group. if it is greater
// we need to update the bg_itable_unused count
if *__slate_slot_9 >= *__slate_slot_3 {
ext4_itable_unused_set(sb, *__slate_slot_11, EXT4_INODES_PER_GROUP!(sb).wrapping_sub(*__slate_slot_9 as u64).wrapping_sub(1) as u32);
}
} else {
ext4_lock_group(sb, *__slate_slot_10);
}
ext4_free_inodes_set(sb, *__slate_slot_11, ext4_free_inodes_count(sb, *__slate_slot_11).wrapping_sub(1));
if ext4_has_group_desc_csum(sb) != 0 {
ext4_inode_bitmap_csum_set(sb, *__slate_slot_11, inode_bitmap_bh);
ext4_group_desc_csum_set(sb, *__slate_slot_10, *__slate_slot_11);
}
ext4_unlock_group(sb, *__slate_slot_10);
*__slate_slot_8 = ext4_handle_dirty_metadata__9!(group_desc_bh);
sync_dirty_buffer(group_desc_bh);
}
}
}
}
}
brelse(inode_bitmap_bh);
return *__slate_slot_8;
}
}
}
}

fn ext4_xattr_credits_for_new_inode(dir: *mut inode, mode: u32, encrypt: bool) -> i32 {
let sb: *mut super_block = unsafe { (*dir).i_sb };
let mut nblocks: i32 = 0;
let p: *mut posix_acl = unsafe { get_inode_acl(dir, 16384i32) };
if IS_ERR(p as *const ()) {
return PTR_ERR(p as *const ()) as i32;
}
if !p.is_null() {
let acl_size: i32 = ((unsafe { (*p).__slate_anon_0.__slate_anon_0.a_count }) as u64).wrapping_mul(core::mem::size_of::<__SlateRecord_acl_h_16>() as u64) as i32;
let __v0: i32 = if S_ISDIR__2!(mode) { 2i32 } else { 1i32 }.wrapping_mul(unsafe { __ext4_xattr_set_credits(sb, core::ptr::null_mut::<inode>(), core::ptr::null_mut::<buffer_head>(), acl_size as u64, true) });
nblocks = nblocks.wrapping_add(__v0); // inode
// block_bh
// is_create
posix_acl_release(p);
}
let mut num_security_xattrs: i32 = 1;
num_security_xattrs = num_security_xattrs.wrapping_add(1);
// We assume that security xattrs are never more than 1k.
// In practice they are under 128 bytes.
let __v5: i32 = num_security_xattrs.wrapping_mul(unsafe { __ext4_xattr_set_credits(sb, core::ptr::null_mut::<inode>(), core::ptr::null_mut::<buffer_head>(), 1024u64, true) });
nblocks = nblocks.wrapping_add(__v5); // inode
// block_bh
// is_create
if encrypt {
let __v8: i32 = unsafe { __ext4_xattr_set_credits(sb, core::ptr::null_mut::<inode>(), core::ptr::null_mut::<buffer_head>(), 40u64, true) };
nblocks = nblocks.wrapping_add(__v8);
}
// inode
// block_bh
// is_create
nblocks
}

// There are two policies for allocating an inode.  If the new inode is
// a directory, then a forward search is made for a block group with both
// free space and a low directory-to-inode ratio; if that fails, then of
// the groups with above-average free space, that group with the fewest
// directories already is chosen.
//
// For other inodes, search forward from the parent directory's block
// group to find a free inode.
/// Verify that we are loading a valid orphan from disk
#[unsafe(no_mangle)]
extern "C" fn ext4_orphan_get(sb: *mut super_block, ino: u64) -> *mut inode {
let mut __v5: bool = unsafe { core::mem::zeroed() };
// If the orphans has i_nlinks > 0 then it should be able to
// be truncated, otherwise it won't be removed from the orphan
// list during processing and an infinite loop will result.
// Similarly, it must not be a bad inode.
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
let mut bitmap_bh: *mut buffer_head = unsafe { core::mem::zeroed() };
let mut bit: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(bit) as *mut i32;
let mut block_group: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(block_group) as *mut u32;
let mut max_ino: u64 = unsafe { core::mem::zeroed() };
unsafe {
max_ino = (*(*EXT4_SB(sb)).s_es).s_inodes_count as u64;
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
bitmap_bh = core::ptr::null_mut::<buffer_head>();
inode = core::ptr::null_mut::<inode>();
err = 117i32.wrapping_neg();
if !(ino < (EXT4_FIRST_INO!(sb) as u64) || ino > max_ino) {
*__slate_slot_11 = (ino.wrapping_sub(1) / EXT4_INODES_PER_GROUP!(sb)) as u32;
*__slate_slot_10 = (ino.wrapping_sub(1) % EXT4_INODES_PER_GROUP!(sb)) as i32;
bitmap_bh = ext4_read_inode_bitmap(sb, *__slate_slot_11);
if IS_ERR(bitmap_bh as *const ()) {
return ERR_CAST(bitmap_bh as *const ()) as *mut inode;
} else {
// Having the inode bit set should be a 100% indicator that this
// is a valid orphan (no e2fsck run on fs).  Orphans also include
// inodes that were being truncated, so we can't check i_nlink==0.
if test_bit_le(*__slate_slot_10 as u64, (*bitmap_bh).b_data as *const ()) != 0 {
inode = ext4_iget__11!(sb, ino, 0i32);
if IS_ERR(inode as *const ()) {
err = PTR_ERR(inode as *const ()) as i32;
__ext4_error(sb, c"ext4_orphan_get".as_ptr() as *const u8, 1408u32, false, err.wrapping_neg(), 0u64, c"couldn't read orphan inode %lu (err %d)".as_ptr() as *const u8, ino, err);
brelse(bitmap_bh);
return inode;
} else {
if (*inode).__slate_anon_12.i_nlink != 0 {
__v6 = ext4_can_truncate(inode) == 0;
} else {
__v6 = false;
}
if __v6 {
__v5 = true;
} else {
__v5 = is_bad_inode(inode);
}
if !__v5 {
if (NEXT_ORPHAN!(inode) as u64) <= max_ino {
brelse(bitmap_bh);
return inode;
}
}
}
}
}
}
__ext4_error(sb, c"ext4_orphan_get".as_ptr() as *const u8, 1429u32, false, 0i32, 0u64, c"bad orphan inode %lu".as_ptr() as *const u8, ino);
if !bitmap_bh.is_null() {
while false {
}
_printk(c"\x013ext4_test_bit(bit=%d, block=%llu) = %d\n".as_ptr() as *const u8, *__slate_slot_10, (*bitmap_bh).b_blocknr, test_bit_le(*__slate_slot_10 as u64, (*bitmap_bh).b_data as *const ()));
}
if !inode.is_null() {
while false {
}
_printk(c"\x013is_bad_inode(inode)=%d\n".as_ptr() as *const u8, is_bad_inode(inode) as i32);
while false {
}
_printk(c"\x013NEXT_ORPHAN(inode)=%u\n".as_ptr() as *const u8, NEXT_ORPHAN!(inode));
while false {
}
_printk(c"\x013max_ino=%lu\n".as_ptr() as *const u8, max_ino);
while false {
}
_printk(c"\x013i_nlink=%u\n".as_ptr() as *const u8, (*inode).__slate_anon_12.i_nlink);
// Avoid freeing blocks if we got a bad deleted inode
if (*inode).__slate_anon_12.i_nlink == 0 {
(*inode).i_blocks = 0;
}
iput(inode);
}
brelse(bitmap_bh);
return ERR_PTR(err as i64) as *mut inode;
}
}

#[unsafe(no_mangle)]
extern "C" fn ext4_count_free_inodes(sb: *mut super_block) -> u64 {
let mut desc_count: u64 = 0;
let ngroups: u32 = ext4_get_groups_count(sb);
for i in 0..ngroups {
let gdp: *mut ext4_group_desc = unsafe { ext4_get_group_desc(sb, i, core::ptr::null_mut::<*mut buffer_head>()) };
if !gdp.is_null() {
let __v1: u32 = unsafe { ext4_free_inodes_count(sb, gdp) };
desc_count = desc_count.wrapping_add(__v1 as u64);
cond_resched__14!();
}
}
desc_count
}

/// Called at mount-time, super-block is locked
#[unsafe(no_mangle)]
extern "C" fn ext4_count_dirs(sb: *mut super_block) -> u64 {
let mut count: u64 = 0;
let ngroups: u32 = ext4_get_groups_count(sb);
for i in 0..ngroups {
let gdp: *mut ext4_group_desc = unsafe { ext4_get_group_desc(sb, i, core::ptr::null_mut::<*mut buffer_head>()) };
if !gdp.is_null() {
let __v1: u32 = unsafe { ext4_used_dirs_count(sb, gdp) };
count = count.wrapping_add(__v1 as u64);
}
}
count
}

/// Zeroes not yet zeroed inode table - just write zeroes through the whole
/// inode table. Must be called without any spinlock held. The only place
/// where it is called from on active part of filesystem is ext4lazyinit
/// thread, so we do not need any special locks, however we have to prevent
/// inode allocation from the current group, so we take alloc_sem lock, to
/// block ext4_new_inode() until we are finished.
#[unsafe(no_mangle)]
extern "C" fn ext4_init_inode_table(sb: *mut super_block, group: u32, barrier: i32) -> i32 {
// Are there some uninitialized inodes in the inode table
// before the first normal inode?
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut used_inos: u64 = unsafe { core::mem::zeroed() };
let mut used_blks: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut num: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(num) as *mut i32;
let mut blk: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(blk) as *mut u64;
let mut handle: core::mem::MaybeUninit<*mut jbd2_journal_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut jbd2_journal_handle = core::ptr::addr_of_mut!(handle) as *mut *mut jbd2_journal_handle;
let mut group_desc_bh: core::mem::MaybeUninit<*mut buffer_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut buffer_head = core::ptr::addr_of_mut!(group_desc_bh) as *mut *mut buffer_head;
let mut gdp: *mut ext4_group_desc = unsafe { core::mem::zeroed() };
let mut sbi: *mut ext4_sb_info = unsafe { core::mem::zeroed() };
let mut grp: *mut ext4_group_info = unsafe { core::mem::zeroed() };
unsafe {
grp = ext4_get_group_info(sb, group);
sbi = EXT4_SB(sb);
gdp = core::ptr::null_mut::<ext4_group_desc>();
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 0;
used_blks = 0;
used_inos = 0;
gdp = ext4_get_group_desc(sb, group, core::ptr::addr_of_mut!(*__slate_slot_14));
if !(gdp.is_null() || grp.is_null()) {
// We do not need to lock this, because we are the only one
// handling this flag.
if ((*gdp).bg_flags as i32) & 4 == 0 {
*__slate_slot_13 = ext4_journal_start_sb__4!(sb, 0i32, 1i32);
if IS_ERR(*__slate_slot_13 as *const ()) {
ret = PTR_ERR(*__slate_slot_13 as *const ()) as i32;
} else {
'__join_1: {
down_write(core::ptr::addr_of_mut!((*grp).alloc_sem));
// If inode bitmap was already initialized there may be some
// used inodes so we need to skip blocks with used inodes in
// inode table.
if ((*gdp).bg_flags as i32) & 1 == 0 {
used_inos = EXT4_INODES_PER_GROUP!(sb).wrapping_sub(ext4_itable_unused_count(sb, gdp) as u64);
used_blks = (used_inos.wrapping_add((*sbi).s_inodes_per_block).wrapping_sub(1) / (*sbi).s_inodes_per_block) as i32;
// Bogus inode unused count?
if used_blks < 0 || (used_blks as u64) > (*sbi).s_itb_per_group {
__ext4_error(sb, c"ext4_init_inode_table".as_ptr() as *const u8, 1569u32, false, 0i32, 0u64, c"Something is wrong with group %u: used itable blocks: %d; itable unused count: %u".as_ptr() as *const u8, group, used_blks, ext4_itable_unused_count(sb, gdp));
ret = 1;
break '__join_1;
} else {
let __v7: u64 = (group as u64).wrapping_mul(EXT4_INODES_PER_GROUP!(sb));
used_inos = used_inos.wrapping_add(__v7);
if (used_blks as u64) != (*sbi).s_itb_per_group {
__v4 = used_inos < (EXT4_FIRST_INO!(sb) as u64);
} else {
__v4 = false;
}
if __v4 {
__ext4_error(sb, c"ext4_init_inode_table".as_ptr() as *const u8, 1585u32, false, 0i32, 0u64, c"Something is wrong with group %u: itable unused count: %u; itables initialized count: %ld".as_ptr() as *const u8, group, ext4_itable_unused_count(sb, gdp), used_inos);
ret = 1;
break '__join_1;
}
}
}
*__slate_slot_12 = ext4_inode_table(sb, gdp).wrapping_add(used_blks as u64);
*__slate_slot_11 = unsafe { (*sbi).s_itb_per_group }.wrapping_sub(used_blks as u64) as i32;
BUFFER_TRACE!();
ret = ext4_journal_get_write_access__63!(*__slate_slot_13, sb, *__slate_slot_14, 1i32);
if ret == 0 {
// Skip zeroout if the inode table is full. But we set the ZEROED
// flag anyway, because obviously, when it is full it does not need
// further zeroing.
if unlikely!(*__slate_slot_11 == 0) == 0 {
ret = sb_issue_zeroout(sb, *__slate_slot_12, *__slate_slot_11 as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32));
if ret < 0 {
break '__join_1;
} else {
if barrier != 0 {
blkdev_issue_flush((*sb).s_bdev);
}
}
}
ext4_lock_group(sb, group);
(*gdp).bg_flags = (((*gdp).bg_flags as i32) | 4) as u16;
ext4_group_desc_csum_set(sb, group, gdp);
ext4_unlock_group(sb, group);
BUFFER_TRACE!();
ret = ext4_handle_dirty_metadata__48!(*__slate_slot_13, *__slate_slot_14);
}
}
up_write(core::ptr::addr_of_mut!((*grp).alloc_sem));
ext4_journal_stop__61!(*__slate_slot_13);
}
}
}
return ret;
}
}

static __slate_asm_string___ext4_new_inode_80: [u8; 1] = *b"\0";

static __slate_asm_string___ext4_new_inode_81: [u8; 46] = *b"/home/takashi/c-corpus/linux/fs/ext4/ialloc.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
