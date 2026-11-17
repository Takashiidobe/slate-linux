
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
//! Driver for USB Mass Storage compliant devices
//!
//! Current development and maintenance by:
//!   (c) 1999-2003 Matthew Dharm (mdharm-usb@one-eyed-alien.net)
//!
//! Developed with the assistance of:
//!   (c) 2000 David L. Brown, Jr. (usb-storage@davidb.org)
//!   (c) 2003-2009 Alan Stern (stern@rowland.harvard.edu)
//!
//! Initial work by:
//!   (c) 1999 Michael Gee (michael@linuxspecific.com)
//!
//! usb_device_id support by Adam J. Richter (adam@yggdrasil.com):
//!   (c) 2000 Yggdrasil Computing, Inc.
//!
//! This driver is based on the 'USB Mass Storage Class' document. This
//! describes in detail the protocol used to communicate with such
//! devices.  Clearly, the designers had SCSI and ATAPI commands in
//! mind when they created this document.  The commands are all very
//! similar to commands in the SCSI-II and ATAPI specifications.
//!
//! It is important to note that in a number of cases this class
//! exhibits class-specific exemptions from the USB specification.
//! Notably the usage of NAK, STALL and ACK differs from the norm, in
//! that they are used to communicate wait, failed and OK on commands.
//!
//! Also, for certain devices, the interrupt endpoint is used to convey
//! status of a command.
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
//! New user namespace
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
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
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Helper macro to avoid gfp flags if they are the default one
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
//! Keep unconverted code working
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! The Well Known LUNS (SAM-3) in our int representation of a LUN
//! Generate SD flag indexes
//! Resource type
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! See the comment in asm-generic/io.h about ioremap_np().
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Runtime PM flag argument bits
//! Static flag definitions.  We use this roundabout technique so that the
//! proc_info() routine can automatically display a message for each flag.
//! Typically, flash devices
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// SCHED_ISO: reserved but not implemented yet
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
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
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
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
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// Set the IDR flag and the IDR_FREE tag
// kthread_create - create a kthread on the current node
// @threadfn: the function to run in the thread
// @data: data pointer for @threadfn()
// @namefmt: printf-style format string for the thread name
// @arg: arguments for @namefmt.
//
// This macro will create a kthread on the current node, leaving it in
// the stopped state.  This is just a helper for kthread_create_on_node();
// see the documentation there for more details.
// kthread_run - create and wake a thread.
// @threadfn: the function to run until signal_pending(current).
// @data: data ptr for @threadfn.
// @namefmt: printf-style name for the thread.
//
// Description: Convenient wrapper for kthread_create() followed by
// wake_up_process().  Returns the kthread or ERR_PTR(-ENOMEM).
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Resource extended types
// No address assigned yet
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// allow access to only LUN 0
// unusual_devs entry is necessary
// supports multiple targets
// INQUIRY response needs faking
// READ CAPACITY response too big
// reported residue is wrong
// Uses 32-byte CBW length
// PREVENT/ALLOW not supported
// Need delay after Command phase
// Don't check for write-protect
// Sets max_sectors to 64
// Don't claim device
// sometimes sizes is too big
// Sets max_sectors to arch min
// Ignore tag mismatch in bulk operations
// Sane Sense (> 18 bytes)
// READ CAPACITY response is correct
// Bad Sense (never more than 18 bytes)
// cannot handle READ_DISC_INFO
// cannot handle READ_CAPACITY_16
// Initial READ(10) (and others) must be retried
// Write Cache status is not available
// cannot handle READ_CAPACITY_10
// Device advertises UAS but it is broken
// Cannot handle FUA in WRITE or READ CDBs
// Cannot handle ATA_12 or ATA_16 CDBs
// Cannot handle MI_REPORT_SUPPORTED_OPERATION_CODES
// Sets max_sectors to 240
// Cannot handle REPORT_LUNS
// lies about caching, so always sync
// Cannot handle WRITE_SAME
// Do REQUEST_SENSE after SYNCHRONIZE_CACHE
// CD-ROM
// QIC-157 Tapes
// Floppy
// Removable media
// Transparent
// Use device's value
// Control/Bulk/Interrupt
// Control/Bulk w/o interrupt
// bulk only
// Use device's value
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
((core::mem::size_of::<[us_unusual_dev; 342]>() as u64) / (core::mem::size_of::<us_unusual_dev>() as u64)).wrapping_add(__must_be_array__6!() as u64)
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

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

/// Works only for digits and letters, but small and fast
macro_rules! TOLOWER {
($x:expr) => {
(($x) as i32) | 32
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

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut list_head, $val) };
if true {
break;
}
}
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

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
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

macro_rules! __must_be_array__6 {
() => {
core::mem::size_of::<__SlateRecord_usb_c_1223>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
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
macro_rules! container_of {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { (($ptr) as *const u8).wrapping_offset(-(offsetof__8!() as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! debug_normal_state_change {
() => {
while false {
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
macro_rules! do_div {
($n:expr, $base:expr) => {
{
let mut __v: u32;
let __v2: u32 = ($base) as u32;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
__v
}
};
}

macro_rules! interface_to_usbdev {
($intf:expr) => {
__intf_to_usbdev($intf)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
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

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__4 {
() => {
40u64
};
}

macro_rules! offsetof__42 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, scan_dwork.work) as u64
};
}

macro_rules! offsetof__8 {
() => {
2216u64
};
}

macro_rules! offsetof__2 {
() => {
0u64
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! scsi_lock {
($host:expr) => {
spin_lock_irq(unsafe { (*$host).host_lock });
};
}

/// The scsi_lock() and scsi_unlock() macros protect the sm_state and the
/// single queue element srb for write access
macro_rules! scsi_unlock {
($host:expr) => {
spin_unlock_irq(unsafe { (*$host).host_lock });
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

macro_rules! test_bit__2 {
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

macro_rules! to_scsi_target {
($d:expr) => {
{
let mut __v: *mut scsi_target;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut scsi_target;
__v
}
};
}

macro_rules! to_usb_device__2 {
($__dev:expr) => {
{
let mut __v: *mut usb_device;
__v = ((unsafe { ((($__dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(usb_device) as isize)) }) as *mut ()) as *mut usb_device;
__v as *const usb_device
}
};
}

macro_rules! to_usb_device {
($__dev:expr) => {
{
let mut __v: *mut usb_device;
__v = ((unsafe { ((($__dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(usb_device) as isize)) }) as *mut ()) as *mut usb_device;
__v
}
};
}

macro_rules! trace_set_current_state {
($state_value:expr) => {
loop {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

macro_rules! trace_set_current_state__2 {
($state_value:expr) => {
loop {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
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

macro_rules! usb_rcvbulkpipe {
($dev:expr, $endpoint:expr) => {
3221225472u32 | __create_pipe($dev, ($endpoint) as u32) | 128
};
}

macro_rules! usb_rcvctrlpipe {
($dev:expr, $endpoint:expr) => {
2147483648u32 | __create_pipe($dev, ($endpoint) as u32) | 128
};
}

macro_rules! usb_rcvintpipe {
($dev:expr, $endpoint:expr) => {
1073741824u32 | __create_pipe($dev, ($endpoint) as u32) | 128
};
}

/// use a define to avoid include chaining to get THIS_MODULE & friends
macro_rules! usb_register__2 {
($driver:expr) => {
unsafe { usb_register_driver($driver, core::ptr::null_mut::<module>(), c"usb_storage".as_ptr() as *const u8) }
};
}

macro_rules! usb_sndbulkpipe {
($dev:expr, $endpoint:expr) => {
3221225472u32 | __create_pipe($dev, ($endpoint) as u32)
};
}

/// Create various pipes...
macro_rules! usb_sndctrlpipe {
($dev:expr, $endpoint:expr) => {
2147483648u32 | __create_pipe($dev, ($endpoint) as u32)
};
}

macro_rules! wake_up {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmsg_fops: file_operations;
    static mut cpuinfo_op: seq_operations;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut system_freezable_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_string: kernel_param_ops;
    static mut usb_storage_usb_ids: [usb_device_id; 0];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn skip_spaces(arg0: *const u8) -> *mut u8;
    fn kstrtouint(arg0: *const u8, arg1: u32, arg2: *mut u32) -> i32;
    fn simple_strtoul(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> u64;
    fn int_pow(arg0: u64, arg1: u32) -> u64;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn schedule();
    fn __trace_set_current_state(arg0: i32);
    fn wake_up_process(arg0: *mut task_struct) -> i32;
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion_interruptible(arg0: *mut completion) -> i32;
    fn complete(arg0: *mut completion);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get_mono_fast_ns() -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn kthread_create_on_node(arg0: Option<unsafe extern "C" fn(*mut ()) -> i32>, arg1: *mut (), arg2: i32, arg3: *const u8, ...) -> *mut task_struct;
    fn kthread_stop(arg0: *mut task_struct) -> i32;
    fn kthread_should_stop() -> bool;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_notice(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn scsi_done_direct(arg0: *mut scsi_cmnd);
    fn __compiletime_assert_589();
    fn usb_find_common_endpoints(arg0: *mut usb_host_interface, arg1: *mut *mut usb_endpoint_descriptor, arg2: *mut *mut usb_endpoint_descriptor, arg3: *mut *mut usb_endpoint_descriptor, arg4: *mut *mut usb_endpoint_descriptor) -> i32;
    fn usb_autopm_put_interface(arg0: *mut usb_interface);
    fn usb_autopm_get_interface_no_resume(arg0: *mut usb_interface);
    fn usb_autopm_put_interface_no_suspend(arg0: *mut usb_interface);
    fn usb_register_driver(arg0: *mut usb_driver, arg1: *mut module, arg2: *const u8) -> i32;
    fn usb_deregister(arg0: *mut usb_driver);
    fn usb_alloc_urb(arg0: i32, arg1: u32) -> *mut urb;
    fn usb_free_urb(arg0: *mut urb);
    fn usb_alloc_coherent(arg0: *mut usb_device, arg1: u64, arg2: u32, arg3: *mut u64) -> *mut ();
    fn usb_free_coherent(arg0: *mut usb_device, arg1: u64, arg2: *mut (), arg3: u64);
    fn usb_usual_ignore_device(arg0: *mut usb_interface) -> i32;
    fn scsi_host_alloc(arg0: *const scsi_host_template, arg1: i32) -> *mut Scsi_Host;
    fn scsi_add_host_with_dma(arg0: *mut Scsi_Host, arg1: *mut device, arg2: *mut device) -> i32;
    fn scsi_scan_host(arg0: *mut Scsi_Host);
    fn scsi_remove_host(arg0: *mut Scsi_Host);
    fn scsi_host_put(arg0: *mut Scsi_Host);
    fn usb_stor_report_bus_reset(arg0: *mut us_data);
    fn usb_stor_host_template_init(arg0: *mut scsi_host_template, arg1: *const u8, arg2: *mut module);
    fn usb_stor_CB_transport(arg0: *mut scsi_cmnd, arg1: *mut us_data) -> i32;
    fn usb_stor_CB_reset(arg0: *mut us_data) -> i32;
    fn usb_stor_Bulk_transport(arg0: *mut scsi_cmnd, arg1: *mut us_data) -> i32;
    fn usb_stor_Bulk_max_lun(arg0: *mut us_data) -> i32;
    fn usb_stor_Bulk_reset(arg0: *mut us_data) -> i32;
    fn usb_stor_pad12_command(arg0: *mut scsi_cmnd, arg1: *mut us_data);
    fn usb_stor_ufi_command(arg0: *mut scsi_cmnd, arg1: *mut us_data);
    fn usb_stor_transparent_scsi_command(arg0: *mut scsi_cmnd, arg1: *mut us_data);
    fn usb_stor_set_xfer_buf(arg0: *mut u8, arg1: u32, arg2: *mut scsi_cmnd);
    fn usb_stor_euscsi_init(arg0: *mut us_data) -> i32;
    fn usb_stor_ucr61s2b_init(arg0: *mut us_data) -> i32;
    fn usb_stor_huawei_e220_init(arg0: *mut us_data) -> i32;
    fn sierra_ms_init(arg0: *mut us_data) -> i32;
    fn option_ms_init(arg0: *mut us_data) -> i32;
    fn __compiletime_assert_612();
    fn __compiletime_assert_615();
    fn __compiletime_assert_617();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
struct io_bitmap {
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

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
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


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


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits119U0,
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

// list_entry - get the struct for this entry
// @ptr:	the &struct list_head pointer.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits127U0,
    pkru: u32,
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits222U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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
// plug has ts that needs updating
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
struct user_namespace {
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


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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


fn init_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
}
loop {
unsafe { __init_swait_queue_head(core::ptr::addr_of_mut!((*x).wait), c"&x->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__87)) };
if true {
break;
}
}
}

static mut __key__87: lock_class_key = unsafe { core::mem::zeroed() };

fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
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


fn __init_work(work: *mut work_struct, onstack: i32) {
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


fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
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
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct folio_queue {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits417U0,
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct sock {
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
struct kmem_cache {
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_361() };
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
struct backing_dev_info {
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
union __SlateRecord_iocontext_h_83 {
    q_node: list_head,
    __rcu_icq_cache: *mut kmem_cache,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iocontext_h_87 {
    ioc_node: hlist_node,
    __rcu_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct fiemap_extent_info {
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
struct poll_table_struct {
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


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
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
struct posix_acl {
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_cq {
    q: *mut request_queue,
    ioc: *mut io_context,
    __slate_anon_2: __SlateRecord_iocontext_h_83,
    __slate_anon_3: __SlateRecord_iocontext_h_87,
    flags: u32,
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
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
struct module_param_attrs {
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
struct mnt_namespace {
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
    __slate_bits_0: __slate_bits::__SlateBits645U0,
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
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
    __slate_bits_0: __slate_bits::__SlateBits589U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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


// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
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


// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// module_param_string - a char array parameter
// @name: the name of the parameter
// @string: the string variable
// @len: the maximum length of the string, incl. terminator
// @perm: visibility in sysfs.
//
// This actually copies the string when it's set (unlike type charp).
// @len is usually just sizeof(string).
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
struct latch_tree_node {
    node: [rb_node; 2],
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


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
// What your module does.
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
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct new_utsname {
    sysname: [u8; 65],
    nodename: [u8; 65],
    release: [u8; 65],
    version: [u8; 65],
    machine: [u8; 65],
    domainname: [u8; 65],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
    name: new_utsname,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    ns: ns_common,
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


fn utsname() -> *mut new_utsname {
unsafe { core::ptr::addr_of_mut!((*(*(*get_current()).nsproxy).uts_ns).name) }
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
struct bio_alloc_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits705U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits705U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits705U2,
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
    __slate_bits_3: __slate_bits::__SlateBits705U3,
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
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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
struct blk_mq_ctx {
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
union __SlateRecord_blk_mq_h_125 {
    queuelist: list_head,
    rq_next: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_182 {
    hash: hlist_node,
    ipi_list: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_193 {
    rb_node: rb_node,
    special_vec: bio_vec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_202 {
    icq: *mut io_cq,
    r#priv: [*mut (); 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_207 {
    seq: u32,
    saved_end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_323 {
    lock: spinlock,
    dispatch: list_head,
    state: u64,
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
struct bsg_device {
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
    __slate_bits_0: __slate_bits::__SlateBits702U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device_handler {
}


fn dev_set_drvdata(dev: *mut device, data: *mut ()) {
unsafe {
(*dev).driver_data = data;
}
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
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
    sgl: *mut scatterlist,
    nents: u32,
    orig_nents: u32,
}


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
union __SlateRecord_ch9_h_1010 {
    legacy_padding: u32,
    __slate_anon_1: __SlateRecord_ch9_h_1013,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013 {
    __empty_bmSublinkSpeedAttr: __SlateRecord_ch9_h_1013_2,
    bmSublinkSpeedAttr: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013_2 {
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
struct ep_device {
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbitmap_word {
    word: u64,
    __slate_pad_1: [u8; 56],
    cleared: u64,
    swap_lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap {
    depth: u32,
    shift: u32,
    map_nr: u32,
    round_robin: bool,
    map: *mut sbitmap_word,
    alloc_hint: *mut u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbq_wait_state {
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap_queue {
    sb: sbitmap,
    wake_batch: u32,
    wake_index: __SlateRecord_types_h_190,
    ws: *mut sbq_wait_state,
    ws_active: __SlateRecord_types_h_190,
    min_shallow_depth: u32,
    completion_cnt: __SlateRecord_types_h_190,
    wakeup_cnt: __SlateRecord_types_h_190,
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
struct mon_bus {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
}


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
struct request {
    q: *mut request_queue,
    mq_ctx: *mut blk_mq_ctx,
    mq_hctx: *mut blk_mq_hw_ctx,
    cmd_flags: u32,
    rq_flags: u32,
    tag: i32,
    internal_tag: i32,
    timeout: u32,
    __data_len: u32,
    __sector: u64,
    bio: *mut bio,
    biotail: *mut bio,
    __slate_anon_12: __SlateRecord_blk_mq_h_125,
    part: *mut block_device,
    alloc_time_ns: u64,
    start_time_ns: u64,
    io_start_time_ns: u64,
    stats_sectors: u16,
    nr_phys_segments: u16,
    nr_integrity_segments: u16,
    phys_gap_bit: u8,
    state: u32,
    r#ref: __SlateRecord_types_h_190,
    deadline: u64,
    __slate_anon_24: __SlateRecord_blk_mq_h_182,
    __slate_anon_25: __SlateRecord_blk_mq_h_193,
    elv: __SlateRecord_blk_mq_h_202,
    flush: __SlateRecord_blk_mq_h_207,
    fifo_time: u64,
    end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
    end_io_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
    __slate_anon_0: __SlateRecord_blk_mq_h_323,
    run_work: delayed_work,
    cpumask: [cpumask; 1],
    next_cpu: i32,
    next_cpu_batch: i32,
    flags: u64,
    sched_data: *mut (),
    queue: *mut request_queue,
    fq: *mut blk_flush_queue,
    driver_data: *mut (),
    ctx_map: sbitmap,
    dispatch_from: *mut blk_mq_ctx,
    dispatch_busy: u32,
    r#type: u16,
    nr_ctx: u16,
    ctxs: *mut *mut blk_mq_ctx,
    dispatch_wait_lock: spinlock,
    dispatch_wait: wait_queue_entry,
    wait_index: __SlateRecord_types_h_190,
    tags: *mut blk_mq_tags,
    sched_tags: *mut blk_mq_tags,
    numa_node: i32,
    queue_num: u32,
    nr_active: __SlateRecord_types_h_190,
    cpuhp_online: hlist_node,
    cpuhp_dead: hlist_node,
    kobj: kobject,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    hctx_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_map {
    mq_map: *mut u32,
    nr_queues: u32,
    queue_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
    ops: *const blk_mq_ops,
    map: [blk_mq_queue_map; 3],
    nr_maps: u32,
    nr_hw_queues: u32,
    queue_depth: u32,
    reserved_tags: u32,
    cmd_size: u32,
    numa_node: i32,
    timeout: u32,
    flags: u32,
    driver_data: *mut (),
    tags: *mut *mut blk_mq_tags,
    shared_tags: *mut blk_mq_tags,
    tag_list_lock: mutex,
    tag_list: list_head,
    srcu: *mut srcu_struct,
    tags_srcu: srcu_struct,
    update_nr_hwq_lock: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy_roothub {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gen_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_165_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_169_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_usb_c_1223 {
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
    pub struct __SlateBits577U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits705U0 {
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
    pub struct __SlateBits702U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits705U1 {
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
    pub struct __SlateBits705U2 {
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
    pub struct __SlateBits705U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits645U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits589U0 {
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
    pub struct __SlateBits126U0 {
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
    pub struct __SlateBits417U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits119U0 {
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
    pub struct __SlateBits127U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits222U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits866U0 {
        #[bits(1)]
        pub is_b_host: u32,
        #[bits(1)]
        pub b_hnp_enable: u32,
        #[bits(1)]
        pub no_stop_on_short: u32,
        #[bits(1)]
        pub no_sg_constraint: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits862U0 {
        #[bits(1)]
        pub sysfs_files_created: u32,
        #[bits(1)]
        pub ep_devs_created: u32,
        #[bits(1)]
        pub unregistering: u32,
        #[bits(1)]
        pub needs_remote_wakeup: u32,
        #[bits(1)]
        pub needs_altsetting0: u32,
        #[bits(1)]
        pub needs_binding: u32,
        #[bits(1)]
        pub resetting_device: u32,
        #[bits(1)]
        pub authorized: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits855U0 {
        #[bits(1)]
        pub can_submit: u32,
        #[bits(1)]
        pub persist_enabled: u32,
        #[bits(1)]
        pub reset_in_progress: u32,
        #[bits(1)]
        pub have_langid: u32,
        #[bits(1)]
        pub authorized: u32,
        #[bits(1)]
        pub authenticated: u32,
        #[bits(1)]
        pub lpm_capable: u32,
        #[bits(1)]
        pub lpm_devinit_allow: u32,
        #[bits(1)]
        pub usb2_hw_lpm_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_besl_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_enabled: u32,
        #[bits(1)]
        pub usb2_hw_lpm_allowed: u32,
        #[bits(1)]
        pub usb3_lpm_u1_enabled: u32,
        #[bits(1)]
        pub usb3_lpm_u2_enabled: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits855U1 {
        #[bits(1)]
        pub do_remote_wakeup: u32,
        #[bits(1)]
        pub reset_resume: u32,
        #[bits(1)]
        pub port_is_suspended: u32,
        #[bits(1)]
        pub offload_pm_locked: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits855U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits824U0 {
        #[bits(1)]
        pub create: u32,
        #[bits(1)]
        pub single_lun: u32,
        #[bits(1)]
        pub pdt_1f_for_no_lun: u32,
        #[bits(1)]
        pub no_report_luns: u32,
        #[bits(1)]
        pub expecting_lun_change: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits880U0 {
        #[bits(1)]
        pub tag_alloc_policy_rr: bool,
        #[bits(1)]
        pub track_queue_depth: u32,
        #[bits(2)]
        pub supported_mode: u32,
        #[bits(1)]
        pub emulated: u32,
        #[bits(1)]
        pub skip_settle_delay: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub host_tagset: u32,
        #[bits(1)]
        pub queuecommand_may_block: u32,
        #[bits(7, access = na)]
        pub __slate_pad_8: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits823U0 {
        #[bits(2)]
        pub active_mode: u32,
        #[bits(1)]
        pub host_self_blocked: u32,
        #[bits(1)]
        pub reverse_ordering: u32,
        #[bits(1)]
        pub tmf_in_progress: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub host_tagset: u32,
        #[bits(1)]
        pub queuecommand_may_block: u32,
        #[bits(1)]
        pub short_inquiry: u32,
        #[bits(1)]
        pub no_scsi2_lun_in_cdb: u32,
        #[bits(6, access = na)]
        pub __slate_pad_9: u8,
    }
    #[bitfields::bitfield([u8; 8], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits822U0 {
        #[bits(1)]
        pub manage_system_start_stop: u32,
        #[bits(1)]
        pub manage_runtime_start_stop: u32,
        #[bits(1)]
        pub manage_shutdown: u32,
        #[bits(1)]
        pub manage_restart: u32,
        #[bits(1)]
        pub force_runtime_start_on_system_start: u32,
        #[bits(1)]
        pub is_ata: u32,
        #[bits(1)]
        pub removable: u32,
        #[bits(1)]
        pub changed: u32,
        #[bits(1)]
        pub busy: u32,
        #[bits(1)]
        pub lockable: u32,
        #[bits(1)]
        pub locked: u32,
        #[bits(1)]
        pub borken: u32,
        #[bits(1)]
        pub disconnect: u32,
        #[bits(1)]
        pub soft_reset: u32,
        #[bits(1)]
        pub sdtr: u32,
        #[bits(1)]
        pub wdtr: u32,
        #[bits(1)]
        pub ppr: u32,
        #[bits(1)]
        pub tagged_supported: u32,
        #[bits(1)]
        pub simple_tags: u32,
        #[bits(1)]
        pub was_reset: u32,
        #[bits(1)]
        pub expecting_cc_ua: u32,
        #[bits(1)]
        pub use_10_for_rw: u32,
        #[bits(1)]
        pub use_10_for_ms: u32,
        #[bits(1)]
        pub set_dbd_for_ms: u32,
        #[bits(1)]
        pub read_before_ms: u32,
        #[bits(1)]
        pub no_report_opcodes: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub use_16_for_rw: u32,
        #[bits(1)]
        pub use_16_for_sync: u32,
        #[bits(1)]
        pub skip_ms_page_8: u32,
        #[bits(1)]
        pub skip_ms_page_3f: u32,
        #[bits(1)]
        pub skip_vpd_pages: u32,
        #[bits(1)]
        pub try_vpd_pages: u32,
        #[bits(1)]
        pub use_192_bytes_for_3f: u32,
        #[bits(1)]
        pub no_start_on_add: u32,
        #[bits(1)]
        pub allow_restart: u32,
        #[bits(1)]
        pub start_stop_pwr_cond: u32,
        #[bits(1)]
        pub no_uld_attach: u32,
        #[bits(1)]
        pub select_no_atn: u32,
        #[bits(1)]
        pub fix_capacity: u32,
        #[bits(1)]
        pub guess_capacity: u32,
        #[bits(1)]
        pub retry_hwerror: u32,
        #[bits(1)]
        pub last_sector_bug: u32,
        #[bits(1)]
        pub no_read_disc_info: u32,
        #[bits(1)]
        pub no_read_capacity_16: u32,
        #[bits(1)]
        pub try_rc_10_first: u32,
        #[bits(1)]
        pub security_supported: u32,
        #[bits(1)]
        pub is_visible: u32,
        #[bits(1)]
        pub wce_default_on: u32,
        #[bits(1)]
        pub no_dif: u32,
        #[bits(1)]
        pub broken_fua: u32,
        #[bits(1)]
        pub lun_in_cdb: u32,
        #[bits(1)]
        pub unmap_limit_for_ws: u32,
        #[bits(1)]
        pub rpm_autosuspend: u32,
        #[bits(1)]
        pub ignore_media_change: u32,
        #[bits(1)]
        pub silence_suspend: u32,
        #[bits(1)]
        pub no_vpd_size: u32,
        #[bits(1)]
        pub cdl_supported: u32,
        #[bits(1)]
        pub cdl_enable: u32,
        #[bits(5, access = na)]
        pub __slate_pad_59: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits874U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits856U0 {
        #[bits(1)]
        pub no_dynamic_id: u32,
        #[bits(1)]
        pub supports_autosuspend: u32,
        #[bits(1)]
        pub disable_hub_initiated_lpm: u32,
        #[bits(1)]
        pub soft_unbind: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits891U0 {
        #[bits(1)]
        pub rh_registered: u32,
        #[bits(1)]
        pub rh_pollable: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub skip_phy_initialization: u32,
        #[bits(1)]
        pub uses_new_polling: u32,
        #[bits(1)]
        pub has_tt: u32,
        #[bits(1)]
        pub amd_resume_bug: u32,
        #[bits(1)]
        pub can_do_streams: u32,
        #[bits(1)]
        pub tpl_support: u32,
        #[bits(1)]
        pub cant_recv_wakeups: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_data {
    rq: *mut request,
    last: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ops {
    queue_rq: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *const blk_mq_queue_data) -> u8>,
    commit_rqs: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx)>,
    queue_rqs: Option<unsafe extern "C" fn(*mut rq_list)>,
    get_budget: Option<unsafe extern "C" fn(*mut request_queue) -> i32>,
    put_budget: Option<unsafe extern "C" fn(*mut request_queue, i32)>,
    set_rq_budget_token: Option<unsafe extern "C" fn(*mut request, i32)>,
    get_rq_budget_token: Option<unsafe extern "C" fn(*mut request) -> i32>,
    timeout: Option<unsafe extern "C" fn(*mut request) -> u32>,
    poll: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut io_comp_batch) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut request)>,
    init_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut (), u32) -> i32>,
    exit_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u32)>,
    init_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32, i32) -> i32>,
    exit_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32)>,
    cleanup_rq: Option<unsafe extern "C" fn(*mut request)>,
    busy: Option<unsafe extern "C" fn(*mut request_queue) -> bool>,
    map_queues: Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>,
    show_rq: Option<unsafe extern "C" fn(*mut seq_file, *mut request)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
    nr_tags: u32,
    nr_reserved_tags: u32,
    active_queues: u32,
    bitmap_tags: sbitmap_queue,
    breserved_tags: sbitmap_queue,
    rqs: *mut *mut request,
    static_rqs: *mut *mut request,
    page_list: list_head,
    lock: spinlock,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_vpd {
    rcu: callback_head,
    len: i32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device {
    host: *mut Scsi_Host,
    request_queue: *mut request_queue,
    siblings: list_head,
    same_target_siblings: list_head,
    budget_map: sbitmap,
    device_blocked: __SlateRecord_types_h_190,
    restarts: __SlateRecord_types_h_190,
    list_lock: spinlock,
    starved_entry: list_head,
    queue_depth: u16,
    max_queue_depth: u16,
    last_queue_full_depth: u16,
    last_queue_full_count: u16,
    last_queue_full_time: u64,
    queue_ramp_up_period: u64,
    last_queue_ramp_up: u64,
    id: u32,
    channel: u32,
    lun: u64,
    manufacturer: u32,
    sector_size: u32,
    hostdata: *mut (),
    r#type: u8,
    scsi_level: u8,
    inq_periph_qual: u8,
    inquiry_mutex: mutex,
    inquiry_len: u8,
    inquiry: *mut u8,
    vendor: [u8; 9],
    model: [u8; 17],
    rev: [u8; 5],
    vpd_pg0: *mut scsi_vpd,
    vpd_pg83: *mut scsi_vpd,
    vpd_pg80: *mut scsi_vpd,
    vpd_pg89: *mut scsi_vpd,
    vpd_pgb0: *mut scsi_vpd,
    vpd_pgb1: *mut scsi_vpd,
    vpd_pgb2: *mut scsi_vpd,
    vpd_pgb7: *mut scsi_vpd,
    sdev_target: *mut scsi_target,
    sdev_bflags: u64,
    eh_timeout: u32,
    __slate_bits_0: __slate_bits::__SlateBits822U0,
    queue_stopped: u32,
    offline_already: bool,
    ua_new_media_ctr: __SlateRecord_types_h_190,
    ua_por_ctr: __SlateRecord_types_h_190,
    disk_events_disable_depth: __SlateRecord_types_h_190,
    supported_events: [u64; 1],
    pending_events: [u64; 1],
    event_list: list_head,
    event_work: work_struct,
    max_device_blocked: u32,
    iorequest_cnt: __SlateRecord_types_h_190,
    iodone_cnt: __SlateRecord_types_h_190,
    ioerr_cnt: __SlateRecord_types_h_190,
    iotmo_cnt: __SlateRecord_types_h_190,
    sdev_gendev: device,
    sdev_dev: device,
    requeue_work: work_struct,
    handler: *mut scsi_device_handler,
    handler_data: *mut (),
    dma_drain_len: u64,
    dma_drain_buf: *mut (),
    sg_timeout: u32,
    sg_reserved_size: u32,
    bsg_dev: *mut bsg_device,
    access_state: u8,
    state_mutex: mutex,
    sdev_state: u32,
    quiesced_by: *mut task_struct,
    sdev_data: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_target {
    starget_sdev_user: *mut scsi_device,
    siblings: list_head,
    devices: list_head,
    dev: device,
    reap_ref: kref,
    channel: u32,
    id: u32,
    __slate_bits_0: __slate_bits::__SlateBits824U0,
    target_busy: __SlateRecord_types_h_190,
    target_blocked: __SlateRecord_types_h_190,
    can_queue: u32,
    max_target_blocked: u32,
    scsi_level: u8,
    state: u32,
    hostdata: *mut (),
    starget_data: [u64; 0],
}


fn scsi_target(sdev: *mut scsi_device) -> *mut scsi_target {
to_scsi_target!(unsafe { (*sdev).sdev_gendev.parent })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_data_buffer {
    table: sg_table,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_cmnd {
    device: *mut scsi_device,
    eh_entry: list_head,
    abort_work: delayed_work,
    rcu: callback_head,
    eh_eflags: i32,
    budget_token: i32,
    jiffies_at_alloc: u64,
    retries: i32,
    allowed: i32,
    prot_op: u8,
    prot_type: u8,
    prot_flags: u8,
    submitter: u8,
    cmd_len: u16,
    sc_data_direction: u32,
    cmnd: [u8; 32],
    sdb: scsi_data_buffer,
    prot_sdb: *mut scsi_data_buffer,
    underflow: u32,
    transfersize: u32,
    resid_len: u32,
    sense_len: u32,
    sense_buffer: *mut u8,
    flags: i32,
    state: u64,
    extra_len: u32,
    host_scribble: *mut u8,
    result: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_device_id {
    match_flags: u16,
    idVendor: u16,
    idProduct: u16,
    bcdDevice_lo: u16,
    bcdDevice_hi: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bInterfaceClass: u8,
    bInterfaceSubClass: u8,
    bInterfaceProtocol: u8,
    bInterfaceNumber: u8,
    driver_info: u64,
}


// to device
// to host
// Status Type
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ctrlrequest {
    bRequestType: u8,
    bRequest: u8,
    wValue: u16,
    wIndex: u16,
    wLength: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_device_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bcdUSB: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bMaxPacketSize0: u8,
    idVendor: u16,
    idProduct: u16,
    bcdDevice: u16,
    iManufacturer: u8,
    iProduct: u8,
    iSerialNumber: u8,
    bNumConfigurations: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_config_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumInterfaces: u8,
    bConfigurationValue: u8,
    iConfiguration: u8,
    bmAttributes: u8,
    bMaxPower: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bInterfaceNumber: u8,
    bAlternateSetting: u8,
    bNumEndpoints: u8,
    bInterfaceClass: u8,
    bInterfaceSubClass: u8,
    bInterfaceProtocol: u8,
    iInterface: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_endpoint_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bEndpointAddress: u8,
    bmAttributes: u8,
    wMaxPacketSize: u16,
    bInterval: u8,
    bRefresh: u8,
    bSynchAddress: u8,
}


// Endpoints
// in bEndpointAddress
// in bmAttributes
// The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type.
fn usb_endpoint_num(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bEndpointAddress }) as i32) & 15
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_eusb2_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wMaxPacketSize: u16,
    dwBytesPerInterval: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wReseved: u16,
    dwBytesPerInterval: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bMaxBurst: u8,
    bmAttributes: u8,
    wBytesPerInterval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_assoc_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bFirstInterface: u8,
    bInterfaceCount: u8,
    bFunctionClass: u8,
    bFunctionSubClass: u8,
    bFunctionProtocol: u8,
    iFunction: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_bos_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumDeviceCaps: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ext_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u8,
    wSpeedSupported: u16,
    bFunctionalitySupport: u8,
    bU1devExitLat: u8,
    bU2DevExitLat: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ss_container_id_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    ContainerID: [u8; 16],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    bmAttributes: u32,
    wFunctionalitySupport: u16,
    wReserved: u16,
    __slate_anon_7: __SlateRecord_ch9_h_1010,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
}


// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_mark_last_busy(dev: *mut device) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_589() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*dev).power.last_busy, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_endpoint {
    desc: usb_endpoint_descriptor,
    ss_ep_comp: usb_ss_ep_comp_descriptor,
    ssp_isoc_ep_comp: usb_ssp_isoc_ep_comp_descriptor,
    eusb2_isoc_ep_comp: usb_eusb2_isoc_ep_comp_descriptor,
    urb_list: list_head,
    hcpriv: *mut (),
    ep_dev: *mut ep_device,
    extra: *mut u8,
    extralen: i32,
    enabled: i32,
    streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_interface {
    desc: usb_interface_descriptor,
    extralen: i32,
    extra: *mut u8,
    endpoint: *mut usb_host_endpoint,
    string: *mut u8,
}


fn usb_find_int_in_endpoint(alt: *mut usb_host_interface, int_in: *mut *mut usb_endpoint_descriptor) -> i32 {
unsafe { usb_find_common_endpoints(alt, core::ptr::null_mut::<*mut usb_endpoint_descriptor>(), core::ptr::null_mut::<*mut usb_endpoint_descriptor>(), int_in, core::ptr::null_mut::<*mut usb_endpoint_descriptor>()) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface {
    altsetting: *mut usb_host_interface,
    cur_altsetting: *mut usb_host_interface,
    num_altsetting: u32,
    intf_assoc: *mut usb_interface_assoc_descriptor,
    minor: i32,
    condition: u32,
    __slate_bits_0: __slate_bits::__SlateBits862U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


fn usb_get_intfdata(intf: *mut usb_interface) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*intf).dev) }) as *const device)
}

fn usb_set_intfdata(intf: *mut usb_interface, data: *mut ()) {
dev_set_drvdata(unsafe { core::ptr::addr_of_mut!((*intf).dev) }, data);
}

// this maximum is arbitrary
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_cache {
    num_altsetting: u32,
    r#ref: kref,
    altsetting: [usb_host_interface; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_config {
    desc: usb_config_descriptor,
    string: *mut u8,
    intf_assoc: [*mut usb_interface_assoc_descriptor; 16],
    interface: [*mut usb_interface; 32],
    intf_cache: [*mut usb_interface_cache; 32],
    extra: *mut u8,
    extralen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_bos {
    desc: *mut usb_bos_descriptor,
    ext_cap: *mut usb_ext_cap_descriptor,
    ss_cap: *mut usb_ss_cap_descriptor,
    ssp_cap: *mut usb_ssp_cap_descriptor,
    ss_id: *mut usb_ss_container_id_descriptor,
    ptm_cap: *mut usb_ptm_cap_descriptor,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_suspend: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_resume: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_reset_resume: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_pre_reset: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_post_reset: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_fill_inquiry_response: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_adjust_quirks: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_probe1: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_probe2: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_usb_stor_disconnect: ; .asciz \"GPL\" ; .ascii \"USB_STORAGE\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_usb_storage__622_1258_usb_storage_driver_init6:\t\t\t\n.long\t{10} - .\t\n.previous\t\t\t\t\t\n", sym usb_stor_suspend, sym usb_stor_resume, sym usb_stor_reset_resume, sym usb_stor_pre_reset, sym usb_stor_post_reset, sym fill_inquiry_response, sym usb_stor_adjust_quirks, sym usb_stor_probe1, sym usb_stor_probe2, sym usb_stor_disconnect, sym usb_storage_driver_init, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_bus {
    controller: *mut device,
    sysdev: *mut device,
    busnum: i32,
    bus_name: *const u8,
    uses_pio_for_control: u8,
    otg_port: u8,
    __slate_bits_0: __slate_bits::__SlateBits866U0,
    sg_tablesize: u32,
    devnum_next: i32,
    devnum_next_mutex: mutex,
    devmap: [u64; 2],
    root_hub: *mut usb_device,
    hs_companion: *mut usb_bus,
    bandwidth_allocated: i32,
    bandwidth_int_reqs: i32,
    bandwidth_isoc_reqs: i32,
    resuming_ports: u32,
    mon_bus: *mut mon_bus,
    monitored: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb2_lpm_parameters {
    besl: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb3_lpm_parameters {
    mel: u32,
    pel: u32,
    sel: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_device {
    devnum: i32,
    devpath: [u8; 16],
    route: u32,
    state: u32,
    speed: u32,
    rx_lanes: u32,
    tx_lanes: u32,
    ssp_rate: u32,
    tt: *mut usb_tt,
    ttport: i32,
    toggle: [u32; 2],
    parent: *mut usb_device,
    bus: *mut usb_bus,
    ep0: usb_host_endpoint,
    dev: device,
    descriptor: usb_device_descriptor,
    bos: *mut usb_host_bos,
    config: *mut usb_host_config,
    actconfig: *mut usb_host_config,
    ep_in: [*mut usb_host_endpoint; 16],
    ep_out: [*mut usb_host_endpoint; 16],
    rawdescriptors: *mut *mut u8,
    bus_mA: u16,
    portnum: u8,
    level: u8,
    devaddr: u8,
    __slate_bits_0: __slate_bits::__SlateBits855U0,
    string_langid: i32,
    product: *mut u8,
    manufacturer: *mut u8,
    serial: *mut u8,
    filelist: list_head,
    maxchild: i32,
    quirks: u32,
    urbnum: __SlateRecord_types_h_190,
    active_duration: u64,
    connect_time: u64,
    __slate_bits_1: __slate_bits::__SlateBits855U1,
    offload_usage: i32,
    offload_lock: spinlock,
    tunnel_mode: u32,
    usb4_link: *mut device_link,
    slot_id: i32,
    l1_params: usb2_lpm_parameters,
    u1_params: usb3_lpm_parameters,
    u2_params: usb3_lpm_parameters,
    lpm_disable_count: u32,
    hub_delay: u16,
    __slate_bits_2: __slate_bits::__SlateBits855U2,
}


fn __intf_to_usbdev(intf: *mut usb_interface) -> *mut usb_device {
to_usb_device!(unsafe { (*intf).dev.parent })
}

fn __intf_to_usbdev_const(intf: *const usb_interface) -> *const usb_device {
to_usb_device__2!((unsafe { (*intf).dev.parent }) as *const device)
}

fn usb_mark_last_busy(udev: *mut usb_device) {
pm_runtime_mark_last_busy(unsafe { core::ptr::addr_of_mut!((*udev).dev) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_dynids {
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_driver {
    name: *const u8,
    probe: Option<unsafe extern "C" fn(*mut usb_interface, *const usb_device_id) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut usb_interface)>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut usb_interface, u32, *mut ()) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut usb_interface, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    reset_resume: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    pre_reset: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    post_reset: Option<unsafe extern "C" fn(*mut usb_interface) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_interface)>,
    id_table: *const usb_device_id,
    dev_groups: *mut *const attribute_group,
    dynids: usb_dynids,
    driver: device_driver,
    __slate_bits_0: __slate_bits::__SlateBits856U0,
}


// The following flags are used internally by usbcore and HCDs
// Transfer from device to host
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_iso_packet_descriptor {
    offset: u32,
    length: u32,
    actual_length: u32,
    status: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_anchor {
    urb_list: list_head,
    wait: wait_queue_head,
    lock: spinlock,
    suspend_wakeups: __SlateRecord_types_h_190,
    __slate_bits_0: __slate_bits::__SlateBits874U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct urb {
    kref: kref,
    unlinked: i32,
    hcpriv: *mut (),
    use_count: __SlateRecord_types_h_190,
    reject: __SlateRecord_types_h_190,
    urb_list: list_head,
    anchor_list: list_head,
    anchor: *mut usb_anchor,
    dev: *mut usb_device,
    ep: *mut usb_host_endpoint,
    pipe: u32,
    stream_id: u32,
    status: i32,
    transfer_flags: u32,
    transfer_buffer: *mut (),
    transfer_dma: u64,
    sg: *mut scatterlist,
    sgt: *mut sg_table,
    num_mapped_sgs: i32,
    num_sgs: i32,
    transfer_buffer_length: u32,
    actual_length: u32,
    setup_packet: *mut u8,
    setup_dma: u64,
    start_frame: i32,
    number_of_packets: i32,
    interval: i32,
    error_count: i32,
    context: *mut (),
    complete: Option<unsafe extern "C" fn(*mut urb)>,
    iso_frame_desc: [usb_iso_packet_descriptor; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_sg_request {
    status: i32,
    bytes: u64,
    lock: spinlock,
    dev: *mut usb_device,
    pipe: i32,
    entries: i32,
    urbs: *mut *mut urb,
    count: i32,
    complete: completion,
}


fn __create_pipe(dev: *mut usb_device, endpoint: u32) -> u32 {
(((unsafe { (*dev).devnum }) << 8i32) as u32) | endpoint << 15i32
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
struct scsi_host_template {
    cmd_size: u32,
    queuecommand: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>,
    queue_reserved_command: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>,
    commit_rqs: Option<unsafe extern "C" fn(*mut Scsi_Host, u16)>,
    module: *mut module,
    name: *const u8,
    info: Option<unsafe extern "C" fn(*mut Scsi_Host) -> *const u8>,
    ioctl: Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>,
    init_cmd_priv: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> i32>,
    exit_cmd_priv: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> i32>,
    eh_abort_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_device_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_target_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_bus_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_host_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    sdev_init: Option<unsafe extern "C" fn(*mut scsi_device) -> i32>,
    sdev_configure: Option<unsafe extern "C" fn(*mut scsi_device, *mut queue_limits) -> i32>,
    sdev_destroy: Option<unsafe extern "C" fn(*mut scsi_device)>,
    target_alloc: Option<unsafe extern "C" fn(*mut scsi_target) -> i32>,
    target_destroy: Option<unsafe extern "C" fn(*mut scsi_target)>,
    scan_finished: Option<unsafe extern "C" fn(*mut Scsi_Host, u64) -> i32>,
    scan_start: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
    change_queue_depth: Option<unsafe extern "C" fn(*mut scsi_device, i32) -> i32>,
    map_queues: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
    mq_poll: Option<unsafe extern "C" fn(*mut Scsi_Host, u32) -> i32>,
    dma_need_drain: Option<unsafe extern "C" fn(*mut request) -> bool>,
    bios_param: Option<unsafe extern "C" fn(*mut scsi_device, *mut gendisk, u64, *mut i32) -> i32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut scsi_device)>,
    show_info: Option<unsafe extern "C" fn(*mut seq_file, *mut Scsi_Host) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut u8, i32) -> i32>,
    eh_timed_out: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u32>,
    eh_should_retry_cmd: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> bool>,
    host_reset: Option<unsafe extern "C" fn(*mut Scsi_Host, i32) -> i32>,
    proc_name: *const u8,
    can_queue: i32,
    nr_reserved_cmds: i32,
    this_id: i32,
    sg_tablesize: u16,
    sg_prot_tablesize: u16,
    max_sectors: u32,
    max_segment_size: u32,
    dma_alignment: u32,
    dma_boundary: u64,
    virt_boundary_mask: u64,
    cmd_per_lun: i16,
    __slate_bits_0: __slate_bits::__SlateBits880U0,
    max_host_blocked: u32,
    shost_groups: *mut *const attribute_group,
    sdev_groups: *mut *const attribute_group,
    vendor_id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct Scsi_Host {
    __devices: list_head,
    __targets: list_head,
    starved_list: list_head,
    default_lock: spinlock,
    host_lock: *mut spinlock,
    scan_mutex: mutex,
    eh_abort_list: list_head,
    eh_cmd_q: list_head,
    ehandler: *mut task_struct,
    eh_action: *mut completion,
    host_wait: wait_queue_head,
    hostt: *const scsi_host_template,
    transportt: *mut scsi_transport_template,
    tagset_refcnt: kref,
    tagset_freed: completion,
    tag_set: blk_mq_tag_set,
    host_blocked: __SlateRecord_types_h_190,
    host_failed: u32,
    host_eh_scheduled: u32,
    host_no: u32,
    eh_deadline: i32,
    last_reset: u64,
    max_channel: u32,
    max_id: u32,
    max_lun: u64,
    unique_id: u32,
    max_cmd_len: u16,
    this_id: i32,
    can_queue: i32,
    nr_reserved_cmds: u32,
    cmd_per_lun: i16,
    sg_tablesize: u16,
    sg_prot_tablesize: u16,
    max_sectors: u32,
    opt_sectors: u32,
    max_segment_size: u32,
    dma_alignment: u32,
    dma_boundary: u64,
    virt_boundary_mask: u64,
    nr_hw_queues: u32,
    nr_maps: u32,
    async_scan: bool,
    eh_noresume: bool,
    __slate_bits_0: __slate_bits::__SlateBits823U0,
    work_q: *mut workqueue_struct,
    tmf_work_q: *mut workqueue_struct,
    max_host_blocked: u32,
    prot_capabilities: u32,
    prot_guard_type: u8,
    base: u64,
    io_port: u64,
    n_io_port: u8,
    dma_channel: u8,
    irq: u32,
    shost_state: u32,
    shost_gendev: device,
    shost_dev: device,
    pseudo_sdev: *mut scsi_device,
    shost_data: *mut (),
    dma_dev: *mut device,
    eh_work: work_struct,
    rpm_autosuspend_delay: i32,
    hostdata: [u64; 0],
}


fn scsi_add_host(host: *mut Scsi_Host, dev: *mut device) -> i32 {
unsafe { scsi_add_host_with_dma(host, dev, dev) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct us_unusual_dev {
    vendorName: *const u8,
    productName: *const u8,
    useProtocol: u8,
    useTransport: u8,
    initFunction: Option<unsafe extern "C" fn(*mut us_data) -> i32>,
}


// abort is in progress
// disconnect in progress
// SCSI midlayer timed out
// scanning not yet done
// redo READ(10) command
// Size of the DMA-mapped I/O buffer
#[repr(C)]
#[derive(Clone, Copy)]
struct us_data {
    dev_mutex: mutex,
    pusb_dev: *mut usb_device,
    pusb_intf: *mut usb_interface,
    unusual_dev: *const us_unusual_dev,
    fflags: u64,
    dflags: u64,
    send_bulk_pipe: u32,
    recv_bulk_pipe: u32,
    send_ctrl_pipe: u32,
    recv_ctrl_pipe: u32,
    recv_intr_pipe: u32,
    transport_name: *mut u8,
    protocol_name: *mut u8,
    bcs_signature: u32,
    subclass: u8,
    protocol: u8,
    max_lun: u8,
    ifnum: u8,
    ep_bInterval: u8,
    transport: Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data) -> i32>,
    transport_reset: Option<unsafe extern "C" fn(*mut us_data) -> i32>,
    proto_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>,
    srb: *mut scsi_cmnd,
    tag: u32,
    scsi_name: [u8; 32],
    current_urb: *mut urb,
    cr: *mut usb_ctrlrequest,
    current_sg: usb_sg_request,
    iobuf: *mut u8,
    iobuf_dma: u64,
    ctl_thread: *mut task_struct,
    cmnd_ready: completion,
    notify: completion,
    delay_wait: wait_queue_head,
    scan_dwork: delayed_work,
    extra: *mut (),
    extra_destructor: Option<unsafe extern "C" fn(*mut ())>,
    suspend_resume_hook: Option<unsafe extern "C" fn(*mut us_data, i32)>,
    use_last_sector_hacks: i32,
    last_sector_retries: i32,
}


fn us_to_host(us: *mut us_data) -> *mut Scsi_Host {
container_of__3!(us as *mut (), Scsi_Host)
}

fn host_to_us(host: *mut Scsi_Host) -> *mut us_data {
(unsafe { core::ptr::addr_of_mut!((*host).hostdata) as *mut u64 }) as *mut us_data
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.fill_inquiry_response")]
extern "C" fn fill_inquiry_response(us: *mut us_data, data: *mut u8, data_len: u32) {
if data_len < 36 {
return;
}
// You lose.
unsafe { memset(data.wrapping_offset(8 as isize) as *mut (), 32i32, 28u64) };
if ((unsafe { *data.wrapping_offset(0 as isize) }) as i32) & 32 != 0 {
// USB device currently not connected. Return
// peripheral qualifier 001b ("...however, the
// physical device is not currently connected
// to this logical unit") and leave vendor and
// product identification empty. ("If the target
// does store some of the INQUIRY data on the
// device, it may return zeros or ASCII spaces
// (20h) in those fields until the data is
// available from the device.").
} else {
let bcdDevice: u16 = unsafe { (*(*us).pusb_dev).descriptor.bcdDevice };
let mut n: i32 = (unsafe { strlen((*(*us).unusual_dev).vendorName) }) as i32;
let mut __v0: i32;
let __UNIQUE_ID_x__610: i32 = 8;
let __UNIQUE_ID_y__611: i32 = n;
loop {
if __types_ok__2!(__UNIQUE_ID_x__610, __UNIQUE_ID_y__611) == 0 {
unsafe { __compiletime_assert_612() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__610 < __UNIQUE_ID_y__611 { __UNIQUE_ID_x__610 } else { __UNIQUE_ID_y__611 };
unsafe { memcpy(data.wrapping_offset(8 as isize) as *mut (), (*(*us).unusual_dev).vendorName as *const (), __v0 as u64) };
n = (unsafe { strlen((*(*us).unusual_dev).productName) }) as i32;
let mut __v2: i32;
let __UNIQUE_ID_x__613: i32 = 16;
let __UNIQUE_ID_y__614: i32 = n;
loop {
if __types_ok__2!(__UNIQUE_ID_x__613, __UNIQUE_ID_y__614) == 0 {
unsafe { __compiletime_assert_615() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__613 < __UNIQUE_ID_y__614 { __UNIQUE_ID_x__613 } else { __UNIQUE_ID_y__614 };
unsafe { memcpy(data.wrapping_offset(16 as isize) as *mut (), (*(*us).unusual_dev).productName as *const (), __v2 as u64) };
unsafe {
*data.wrapping_offset(32 as isize) = 48i32.wrapping_add((bcdDevice as i32) >> 12i32 & 15) as u8;
}
unsafe {
*data.wrapping_offset(33 as isize) = 48i32.wrapping_add((bcdDevice as i32) >> 8i32 & 15) as u8;
}
unsafe {
*data.wrapping_offset(34 as isize) = 48i32.wrapping_add((bcdDevice as i32) >> 4i32 & 15) as u8;
}
unsafe {
*data.wrapping_offset(35 as isize) = 48i32.wrapping_add((bcdDevice as i32) & 15) as u8;
}
}
unsafe { usb_stor_set_xfer_buf(data, data_len, (*us).srb) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct giveback_urb_bh {
    running: bool,
    high_prio: bool,
    lock: spinlock,
    head: list_head,
    bh: work_struct,
    completing_ep: *mut usb_host_endpoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_hcd {
    self_: usb_bus,
    kref: kref,
    product_desc: *const u8,
    speed: i32,
    irq_descr: [u8; 24],
    rh_timer: timer_list,
    status_urb: *mut urb,
    wakeup_work: work_struct,
    died_work: work_struct,
    driver: *const hc_driver,
    usb_phy: *mut usb_phy,
    phy_roothub: *mut usb_phy_roothub,
    flags: u64,
    dev_policy: u32,
    __slate_bits_0: __slate_bits::__SlateBits891U0,
    irq: u32,
    regs: *mut (),
    rsrc_start: u64,
    rsrc_len: u64,
    power_budget: u32,
    high_prio_bh: giveback_urb_bh,
    low_prio_bh: giveback_urb_bh,
    address0_mutex: *mut mutex,
    bandwidth_mutex: *mut mutex,
    shared_hcd: *mut usb_hcd,
    primary_hcd: *mut usb_hcd,
    pool: [*mut dma_pool; 4],
    state: i32,
    localmem_pool: *mut gen_pool,
    hcd_priv: [u64; 0],
}


fn bus_to_hcd(bus: *mut usb_bus) -> *mut usb_hcd {
container_of!(bus, usb_hcd)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hc_driver {
    description: *const u8,
    product_desc: *const u8,
    hcd_priv_size: u64,
    irq: Option<unsafe extern "C" fn(*mut usb_hcd) -> u32>,
    flags: i32,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    pci_suspend: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    pci_resume: Option<unsafe extern "C" fn(*mut usb_hcd, pm_message) -> i32>,
    pci_poweroff_late: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    get_frame_number: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    urb_enqueue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    urb_dequeue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
    map_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    unmap_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb)>,
    endpoint_disable: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    endpoint_reset: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    hub_status_data: Option<unsafe extern "C" fn(*mut usb_hcd, *mut u8) -> i32>,
    hub_control: Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>,
    bus_suspend: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    bus_resume: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start_port_reset: Option<unsafe extern "C" fn(*mut usb_hcd, u32) -> i32>,
    get_resuming_ports: Option<unsafe extern "C" fn(*mut usb_hcd) -> u64>,
    relinquish_port: Option<unsafe extern "C" fn(*mut usb_hcd, i32)>,
    port_handed_over: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    clear_tt_buffer_complete: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    alloc_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    free_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    alloc_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32, u32) -> i32>,
    free_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32) -> i32>,
    add_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    drop_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    check_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    reset_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    address_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    enable_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_hub_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_tt, u32) -> i32>,
    reset_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    set_usb2_hw_lpm: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, i32) -> i32>,
    enable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    disable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    find_raw_port_number: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    port_power: Option<unsafe extern "C" fn(*mut usb_hcd, i32, bool) -> i32>,
    submit_single_step_set_feature: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
}


fn hcd_uses_dma(hcd: *mut usb_hcd) -> bool {
IS_ENABLED__2!() != 0 && (unsafe { (*(*hcd).driver).flags }) & 2 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_tt {
    hub: *mut usb_device,
    multi: i32,
    think_time: u32,
    hcpriv: *mut (),
    lock: spinlock,
    clear_list: list_head,
    clear_work: work_struct,
}


unsafe extern "C" fn _usb_stor_dbg(us: *const us_data, fmt: *const u8, __va_args: ...) {
}

/// Some informational data
#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_598: [u8; 65] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 97, 117, 116, 104, 111, 114, 61, 77, 97, 116, 116, 104, 101, 119, 32, 68, 104, 97, 114, 109, 32, 60, 109, 100, 104, 97, 114, 109, 45, 117, 115, 98, 64, 111, 110, 101, 45, 101, 121, 101, 100, 45, 97, 108, 105, 101, 110, 46, 110, 101, 116, 62, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_599: [u8; 58] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 85, 83, 66, 32, 77, 97, 115, 115, 32, 83, 116, 111, 114, 97, 103, 101, 32, 100, 114, 105, 118, 101, 114, 32, 102, 111, 114, 32, 76, 105, 110, 117, 120, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_600: [u8; 49] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 117, 115, 98, 47, 115, 116, 111, 114, 97, 103, 101, 47, 117, 115, 98, 45, 115, 116, 111, 114, 97, 103, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_601: [u8; 24] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static mut delay_use: u32 = 1000;

/// parse_delay_str - parse an unsigned decimal integer delay
/// @str: String to parse.
/// @ndecimals: Number of decimal to scale up.
/// @suffix: Suffix string to parse.
/// @val: Where to store the parsed value.
///
/// Parse an unsigned decimal value in @str, optionally end with @suffix.
/// Stores the parsed value in @val just as it is if @str ends with @suffix.
/// Otherwise store the value scale up by 10^(@ndecimal).
///
/// Returns 0 on success, a negative error code otherwise.
fn parse_delay_str(str: *const u8, ndecimals: i32, suffix: *const u8, val: *mut u32) -> i32 {
let mut n: i32 = 0;
let mut n2: i32 = 0;
let mut l: i32 = 0;
let mut buf: __SlateAlign16<[u8; 16]> = __SlateAlign16([0u8; 16]);
l = (unsafe { strlen(suffix) }) as i32;
n = (unsafe { strlen(str) }) as i32;
if n > 0 && ((unsafe { *str.wrapping_offset(n.wrapping_sub(1) as isize) }) as i32) == 10 {
n = n.wrapping_sub(1);
}
let __v2: bool = n >= l && (unsafe { strncmp(str.wrapping_offset(n.wrapping_sub(l) as isize), suffix, l as u64) }) == 0;
if __v2 {
n = n.wrapping_sub(l);
n2 = 0;
} else {
n2 = ndecimals;
}
if (n.wrapping_add(n2) as u64) > (core::mem::size_of::<[u8; 16]>() as u64).wrapping_sub(1) {
return 22i32.wrapping_neg();
}
unsafe { memcpy(buf.0.as_mut_ptr() as *mut (), str as *const (), n as u64) };
loop {
let __v7: i32 = n2;
n2 = __v7.wrapping_sub(1);
if __v7 <= 0 {
break;
}
let __v5: i32 = n;
n = __v5.wrapping_add(1);
buf.0[__v5 as usize] = 48;
}
buf.0[n as usize] = 0;
unsafe { kstrtouint(buf.0.as_mut_ptr() as *const u8, 10u32, val) }
}

/// format_delay_ms - format an integer value into a delay string
/// @val: The integer value to format, scaled by 10^(@ndecimals).
/// @ndecimals: Number of decimal to scale down.
/// @suffix: Suffix string to format.
/// @str: Where to store the formatted string.
/// @size: The size of buffer for @str.
///
/// Format an integer value in @val scale down by 10^(@ndecimals) without @suffix
/// if @val is divisible by 10^(@ndecimals).
/// Otherwise format a value in @val just as it is with @suffix
///
/// Returns the number of characters written into @str.
fn format_delay_ms(val: u32, ndecimals: i32, suffix: *const u8, str: *mut u8, size: i32) -> i32 {
let mut delay_ms: u64 = val as u64;
let rem: u32 = do_div!(delay_ms, unsafe { int_pow(10u64, ndecimals as u32) });
let ret: i32 = if rem != 0 {
unsafe { scnprintf(str, size as u64, c"%u%s\n".as_ptr() as *const u8, val, suffix) }
} else {
unsafe { scnprintf(str, size as u64, c"%u\n".as_ptr() as *const u8, delay_ms as u32) }
};
ret
}

#[unsafe(link_section = ".text.slate_distinct.storage_usb.delay_use_set")]
extern "C" fn delay_use_set(s: *const u8, kp: *const kernel_param) -> i32 {
let mut delay_ms: u32 = 0;
let ret: i32 = parse_delay_str((unsafe { skip_spaces(s) }) as *const u8, 3i32, c"ms".as_ptr() as *const u8, core::ptr::addr_of_mut!(delay_ms));
if ret < 0 {
return ret;
}
unsafe {
*((*kp).__slate_anon_6.arg as *mut u32) = delay_ms;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.storage_usb.delay_use_get")]
extern "C" fn delay_use_get(s: *mut u8, kp: *const kernel_param) -> i32 {
let delay_ms: u32 = unsafe { *((*kp).__slate_anon_6.arg as *mut u32) };
format_delay_ms(delay_ms, 3i32, c"ms".as_ptr() as *const u8, s, (_AC!() << 12i32) as i32)
}

static delay_use_ops: __SlateConst<kernel_param_ops> = __SlateConst({
let mut __t0: kernel_param_ops = unsafe { core::mem::zeroed() };
__t0.set = Some(delay_use_set);
__t0.get = Some(delay_use_get);
__t0
});

static __param_str_delay_use: __SlateAlign16<[u8; 22]> = __SlateAlign16([117u8, 115u8, 98u8, 95u8, 115u8, 116u8, 111u8, 114u8, 97u8, 103u8, 101u8, 46u8, 100u8, 101u8, 108u8, 97u8, 121u8, 95u8, 117u8, 115u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_delay_use: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_delay_use.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(delay_use_ops.0) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_usb_c_165>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_165_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_165_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_165_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_165_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_165_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(delay_use) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_602: [u8; 67] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 112, 97, 114, 109, 61, 100, 101, 108, 97, 121, 95, 117, 115, 101, 58, 116, 105, 109, 101, 32, 116, 111, 32, 100, 101, 108, 97, 121, 32, 98, 101, 102, 111, 114, 101, 32, 117, 115, 105, 110, 103, 32, 97, 32, 110, 101, 119, 32, 100, 101, 118, 105, 99, 101, 0];

static mut quirks: __SlateAlign16<[u8; 128]> = __SlateAlign16(unsafe { core::mem::zeroed() });

static __param_string_quirks: __SlateConst<kparam_string> = __SlateConst({
let mut __t0: kparam_string = unsafe { core::mem::zeroed() };
__t0.maxlen = (core::mem::size_of::<[u8; 128]>() as u64) as u32;
__t0.string = unsafe { core::ptr::addr_of_mut!(quirks.0) as *mut u8 };
__t0
});

static __param_str_quirks: __SlateAlign16<[u8; 19]> = __SlateAlign16([117u8, 115u8, 98u8, 95u8, 115u8, 116u8, 111u8, 114u8, 97u8, 103u8, 101u8, 46u8, 113u8, 117u8, 105u8, 114u8, 107u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_quirks: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_quirks.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_string) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_usb_c_169>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_169_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_169_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_169_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_169_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_usb_c_169_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { str: unsafe { core::ptr::addr_of!(__param_string_quirks.0) } };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_603: [u8; 35] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 113, 117, 105, 114, 107, 115, 58, 115, 116, 114, 105, 110, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_604: [u8; 73] = [117, 115, 98, 95, 115, 116, 111, 114, 97, 103, 101, 46, 112, 97, 114, 109, 61, 113, 117, 105, 114, 107, 115, 58, 115, 117, 112, 112, 108, 101, 109, 101, 110, 116, 97, 108, 32, 108, 105, 115, 116, 32, 111, 102, 32, 100, 101, 118, 105, 99, 101, 32, 73, 68, 115, 32, 97, 110, 100, 32, 116, 104, 101, 105, 114, 32, 113, 117, 105, 114, 107, 115, 0];

// The entries in this table correspond, line for line,
// with the entries in usb_storage_usb_ids[], defined in usual-tables.c.
// The vendor name should be kept at eight characters or less, and
// the product name should be kept at 16 characters or less. If a device
// has the US_FL_FIX_INQUIRY flag, then the vendor and product names
// normally generated by a device through the INQUIRY response will be
// taken from this list, and this is the reason for the above size
// restriction. However, if the flag is not present, then you
// are free to use as many characters as you like.
/// Terminating entry
static us_unusual_dev_list: __SlateAlign16<[us_unusual_dev; 342]> = __SlateAlign16([{
let mut __t0: us_unusual_dev = unsafe { core::mem::zeroed() };
__t0.vendorName = (b"ATMEL\0".as_ptr() as *mut u8) as *const u8;
__t0.productName = (b"SND1 Storage\0".as_ptr() as *mut u8) as *const u8;
__t0.useProtocol = 255u8;
__t0.useTransport = 255u8;
__t0.initFunction = None;
__t0
}, {
let mut __t1: us_unusual_dev = unsafe { core::mem::zeroed() };
__t1.vendorName = (b"VIA Technologies Inc.\0".as_ptr() as *mut u8) as *const u8;
__t1.productName = (b"Mitsumi multi cardreader\0".as_ptr() as *mut u8) as *const u8;
__t1.useProtocol = 255u8;
__t1.useTransport = 255u8;
__t1.initFunction = None;
__t1
}, {
let mut __t2: us_unusual_dev = unsafe { core::mem::zeroed() };
__t2.vendorName = (b"HP\0".as_ptr() as *mut u8) as *const u8;
__t2.productName = (b"CD-Writer+\0".as_ptr() as *mut u8) as *const u8;
__t2.useProtocol = 5u8;
__t2.useTransport = 1u8;
__t2.initFunction = None;
__t2
}, {
let mut __t3: us_unusual_dev = unsafe { core::mem::zeroed() };
__t3.vendorName = (b"HP\0".as_ptr() as *mut u8) as *const u8;
__t3.productName = (b"Personal Media Drive\0".as_ptr() as *mut u8) as *const u8;
__t3.useProtocol = 255u8;
__t3.useTransport = 255u8;
__t3.initFunction = None;
__t3
}, {
let mut __t4: us_unusual_dev = unsafe { core::mem::zeroed() };
__t4.vendorName = (b"HP\0".as_ptr() as *mut u8) as *const u8;
__t4.productName = (b"PhotoSmart R707\0".as_ptr() as *mut u8) as *const u8;
__t4.useProtocol = 255u8;
__t4.useTransport = 255u8;
__t4.initFunction = None;
__t4
}, {
let mut __t5: us_unusual_dev = unsafe { core::mem::zeroed() };
__t5.vendorName = (b"Adaptec\0".as_ptr() as *mut u8) as *const u8;
__t5.productName = (b"USBConnect 2000\0".as_ptr() as *mut u8) as *const u8;
__t5.useProtocol = 255u8;
__t5.useTransport = 255u8;
__t5.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t5
}, {
let mut __t6: us_unusual_dev = unsafe { core::mem::zeroed() };
__t6.vendorName = (b"NEC\0".as_ptr() as *mut u8) as *const u8;
__t6.productName = (b"NEC USB UF000x\0".as_ptr() as *mut u8) as *const u8;
__t6.useProtocol = 255u8;
__t6.useTransport = 255u8;
__t6.initFunction = None;
__t6
}, {
let mut __t7: us_unusual_dev = unsafe { core::mem::zeroed() };
__t7.vendorName = (b"VIA Technologies Inc.\0".as_ptr() as *mut u8) as *const u8;
__t7.productName = (b"USB 2.0 Card Reader\0".as_ptr() as *mut u8) as *const u8;
__t7.useProtocol = 255u8;
__t7.useTransport = 255u8;
__t7.initFunction = None;
__t7
}, {
let mut __t8: us_unusual_dev = unsafe { core::mem::zeroed() };
__t8.vendorName = (b"Buffalo\0".as_ptr() as *mut u8) as *const u8;
__t8.productName = (b"DUB-P40G HDD\0".as_ptr() as *mut u8) as *const u8;
__t8.useProtocol = 255u8;
__t8.useTransport = 255u8;
__t8.initFunction = None;
__t8
}, {
let mut __t9: us_unusual_dev = unsafe { core::mem::zeroed() };
__t9.vendorName = (b"Samsung Info. Systems America, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t9.productName = (b"MP3 Player\0".as_ptr() as *mut u8) as *const u8;
__t9.useProtocol = 255u8;
__t9.useTransport = 255u8;
__t9.initFunction = None;
__t9
}, {
let mut __t10: us_unusual_dev = unsafe { core::mem::zeroed() };
__t10.vendorName = (b"Samsung\0".as_ptr() as *mut u8) as *const u8;
__t10.productName = (b"MP3 Player\0".as_ptr() as *mut u8) as *const u8;
__t10.useProtocol = 255u8;
__t10.useTransport = 255u8;
__t10.initFunction = None;
__t10
}, {
let mut __t11: us_unusual_dev = unsafe { core::mem::zeroed() };
__t11.vendorName = (b"TrekStor\0".as_ptr() as *mut u8) as *const u8;
__t11.productName = (b"i.Beat 115 2.0\0".as_ptr() as *mut u8) as *const u8;
__t11.useProtocol = 255u8;
__t11.useTransport = 255u8;
__t11.initFunction = None;
__t11
}, {
let mut __t12: us_unusual_dev = unsafe { core::mem::zeroed() };
__t12.vendorName = (b"TrekStor\0".as_ptr() as *mut u8) as *const u8;
__t12.productName = (b"i.Beat Joy 2.0\0".as_ptr() as *mut u8) as *const u8;
__t12.useProtocol = 255u8;
__t12.useTransport = 255u8;
__t12.initFunction = None;
__t12
}, {
let mut __t13: us_unusual_dev = unsafe { core::mem::zeroed() };
__t13.vendorName = (b"GENERIC\0".as_ptr() as *mut u8) as *const u8;
__t13.productName = (b"MP3 PLAYER\0".as_ptr() as *mut u8) as *const u8;
__t13.useProtocol = 255u8;
__t13.useTransport = 255u8;
__t13.initFunction = None;
__t13
}, {
let mut __t14: us_unusual_dev = unsafe { core::mem::zeroed() };
__t14.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t14.productName = (b"Nokia 6288\0".as_ptr() as *mut u8) as *const u8;
__t14.useProtocol = 255u8;
__t14.useTransport = 255u8;
__t14.initFunction = None;
__t14
}, {
let mut __t15: us_unusual_dev = unsafe { core::mem::zeroed() };
__t15.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t15.productName = (b"Nokia 3250\0".as_ptr() as *mut u8) as *const u8;
__t15.useProtocol = 255u8;
__t15.useTransport = 255u8;
__t15.initFunction = None;
__t15
}, {
let mut __t16: us_unusual_dev = unsafe { core::mem::zeroed() };
__t16.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t16.productName = (b"E70\0".as_ptr() as *mut u8) as *const u8;
__t16.useProtocol = 255u8;
__t16.useTransport = 255u8;
__t16.initFunction = None;
__t16
}, {
let mut __t17: us_unusual_dev = unsafe { core::mem::zeroed() };
__t17.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t17.productName = (b"E60\0".as_ptr() as *mut u8) as *const u8;
__t17.useProtocol = 255u8;
__t17.useTransport = 255u8;
__t17.initFunction = None;
__t17
}, {
let mut __t18: us_unusual_dev = unsafe { core::mem::zeroed() };
__t18.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t18.productName = (b"N91\0".as_ptr() as *mut u8) as *const u8;
__t18.useProtocol = 255u8;
__t18.useTransport = 255u8;
__t18.initFunction = None;
__t18
}, {
let mut __t19: us_unusual_dev = unsafe { core::mem::zeroed() };
__t19.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t19.productName = (b"N80\0".as_ptr() as *mut u8) as *const u8;
__t19.useProtocol = 255u8;
__t19.useTransport = 255u8;
__t19.initFunction = None;
__t19
}, {
let mut __t20: us_unusual_dev = unsafe { core::mem::zeroed() };
__t20.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t20.productName = (b"E61\0".as_ptr() as *mut u8) as *const u8;
__t20.useProtocol = 255u8;
__t20.useTransport = 255u8;
__t20.initFunction = None;
__t20
}, {
let mut __t21: us_unusual_dev = unsafe { core::mem::zeroed() };
__t21.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t21.productName = (b"6131\0".as_ptr() as *mut u8) as *const u8;
__t21.useProtocol = 255u8;
__t21.useTransport = 255u8;
__t21.initFunction = None;
__t21
}, {
let mut __t22: us_unusual_dev = unsafe { core::mem::zeroed() };
__t22.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t22.productName = (b"Nokia 6233\0".as_ptr() as *mut u8) as *const u8;
__t22.useProtocol = 255u8;
__t22.useTransport = 255u8;
__t22.initFunction = None;
__t22
}, {
let mut __t23: us_unusual_dev = unsafe { core::mem::zeroed() };
__t23.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t23.productName = (b"6234\0".as_ptr() as *mut u8) as *const u8;
__t23.useProtocol = 255u8;
__t23.useTransport = 255u8;
__t23.initFunction = None;
__t23
}, {
let mut __t24: us_unusual_dev = unsafe { core::mem::zeroed() };
__t24.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t24.productName = (b"5300\0".as_ptr() as *mut u8) as *const u8;
__t24.useProtocol = 255u8;
__t24.useTransport = 255u8;
__t24.initFunction = None;
__t24
}, {
let mut __t25: us_unusual_dev = unsafe { core::mem::zeroed() };
__t25.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t25.productName = (b"305\0".as_ptr() as *mut u8) as *const u8;
__t25.useProtocol = 255u8;
__t25.useTransport = 255u8;
__t25.initFunction = None;
__t25
}, {
let mut __t26: us_unusual_dev = unsafe { core::mem::zeroed() };
__t26.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t26.productName = (b"502\0".as_ptr() as *mut u8) as *const u8;
__t26.useProtocol = 255u8;
__t26.useTransport = 255u8;
__t26.initFunction = None;
__t26
}, {
let mut __t27: us_unusual_dev = unsafe { core::mem::zeroed() };
__t27.vendorName = (b"Nokia\0".as_ptr() as *mut u8) as *const u8;
__t27.productName = (b"Nokia 208\0".as_ptr() as *mut u8) as *const u8;
__t27.useProtocol = 255u8;
__t27.useTransport = 255u8;
__t27.initFunction = None;
__t27
}, {
let mut __t28: us_unusual_dev = unsafe { core::mem::zeroed() };
__t28.vendorName = (b"Microtech\0".as_ptr() as *mut u8) as *const u8;
__t28.productName = (b"CameraMate\0".as_ptr() as *mut u8) as *const u8;
__t28.useProtocol = 6u8;
__t28.useTransport = 1u8;
__t28.initFunction = None;
__t28
}, {
let mut __t29: us_unusual_dev = unsafe { core::mem::zeroed() };
__t29.vendorName = (b"Neuros Audio\0".as_ptr() as *mut u8) as *const u8;
__t29.productName = (b"USB 2.0 HD 2.5\0".as_ptr() as *mut u8) as *const u8;
__t29.useProtocol = 255u8;
__t29.useTransport = 80u8;
__t29.initFunction = None;
__t29
}, {
let mut __t30: us_unusual_dev = unsafe { core::mem::zeroed() };
__t30.vendorName = (b"USBest Technology\0".as_ptr() as *mut u8) as *const u8;
__t30.productName = (b"USB Mass Storage Device\0".as_ptr() as *mut u8) as *const u8;
__t30.useProtocol = 255u8;
__t30.useTransport = 255u8;
__t30.initFunction = None;
__t30
}, {
let mut __t31: us_unusual_dev = unsafe { core::mem::zeroed() };
__t31.vendorName = (b"USB 2.0\0".as_ptr() as *mut u8) as *const u8;
__t31.productName = (b"Flash Disk\0".as_ptr() as *mut u8) as *const u8;
__t31.useProtocol = 255u8;
__t31.useTransport = 255u8;
__t31.initFunction = None;
__t31
}, {
let mut __t32: us_unusual_dev = unsafe { core::mem::zeroed() };
__t32.vendorName = (b"Mitac\0".as_ptr() as *mut u8) as *const u8;
__t32.productName = (b"GPS\0".as_ptr() as *mut u8) as *const u8;
__t32.useProtocol = 255u8;
__t32.useTransport = 255u8;
__t32.initFunction = None;
__t32
}, {
let mut __t33: us_unusual_dev = unsafe { core::mem::zeroed() };
__t33.vendorName = (b"AMI\0".as_ptr() as *mut u8) as *const u8;
__t33.productName = (b"Virtual Floppy\0".as_ptr() as *mut u8) as *const u8;
__t33.useProtocol = 255u8;
__t33.useTransport = 255u8;
__t33.initFunction = None;
__t33
}, {
let mut __t34: us_unusual_dev = unsafe { core::mem::zeroed() };
__t34.vendorName = (b"Toshiba\0".as_ptr() as *mut u8) as *const u8;
__t34.productName = (b"External USB 3.0\0".as_ptr() as *mut u8) as *const u8;
__t34.useProtocol = 255u8;
__t34.useTransport = 255u8;
__t34.initFunction = None;
__t34
}, {
let mut __t35: us_unusual_dev = unsafe { core::mem::zeroed() };
__t35.vendorName = (b"Kyocera\0".as_ptr() as *mut u8) as *const u8;
__t35.productName = (b"Finecam S3x\0".as_ptr() as *mut u8) as *const u8;
__t35.useProtocol = 5u8;
__t35.useTransport = 1u8;
__t35.initFunction = None;
__t35
}, {
let mut __t36: us_unusual_dev = unsafe { core::mem::zeroed() };
__t36.vendorName = (b"Kyocera\0".as_ptr() as *mut u8) as *const u8;
__t36.productName = (b"Finecam S4\0".as_ptr() as *mut u8) as *const u8;
__t36.useProtocol = 5u8;
__t36.useTransport = 1u8;
__t36.initFunction = None;
__t36
}, {
let mut __t37: us_unusual_dev = unsafe { core::mem::zeroed() };
__t37.vendorName = (b"Kyocera\0".as_ptr() as *mut u8) as *const u8;
__t37.productName = (b"Finecam S5\0".as_ptr() as *mut u8) as *const u8;
__t37.useProtocol = 255u8;
__t37.useTransport = 255u8;
__t37.initFunction = None;
__t37
}, {
let mut __t38: us_unusual_dev = unsafe { core::mem::zeroed() };
__t38.vendorName = (b"Kyocera\0".as_ptr() as *mut u8) as *const u8;
__t38.productName = (b"CONTAX SL300R T*\0".as_ptr() as *mut u8) as *const u8;
__t38.useProtocol = 255u8;
__t38.useTransport = 255u8;
__t38.initFunction = None;
__t38
}, {
let mut __t39: us_unusual_dev = unsafe { core::mem::zeroed() };
__t39.vendorName = (b"Hitachi\0".as_ptr() as *mut u8) as *const u8;
__t39.productName = (b"DVD-CAM DZ-MV100A Camcorder\0".as_ptr() as *mut u8) as *const u8;
__t39.useProtocol = 6u8;
__t39.useTransport = 1u8;
__t39.initFunction = None;
__t39
}, {
let mut __t40: us_unusual_dev = unsafe { core::mem::zeroed() };
__t40.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t40.productName = (b"300_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t40.useProtocol = 255u8;
__t40.useTransport = 255u8;
__t40.initFunction = None;
__t40
}, {
let mut __t41: us_unusual_dev = unsafe { core::mem::zeroed() };
__t41.vendorName = (b"NIKON\0".as_ptr() as *mut u8) as *const u8;
__t41.productName = (b"NIKON DSC E2000\0".as_ptr() as *mut u8) as *const u8;
__t41.useProtocol = 255u8;
__t41.useTransport = 255u8;
__t41.initFunction = None;
__t41
}, {
let mut __t42: us_unusual_dev = unsafe { core::mem::zeroed() };
__t42.vendorName = (b"IBM\0".as_ptr() as *mut u8) as *const u8;
__t42.productName = (b"IBM RSA2\0".as_ptr() as *mut u8) as *const u8;
__t42.useProtocol = 255u8;
__t42.useTransport = 1u8;
__t42.initFunction = None;
__t42
}, {
let mut __t43: us_unusual_dev = unsafe { core::mem::zeroed() };
__t43.vendorName = (b"Longmai Technologies\0".as_ptr() as *mut u8) as *const u8;
__t43.productName = (b"USB Key\0".as_ptr() as *mut u8) as *const u8;
__t43.useProtocol = 6u8;
__t43.useTransport = 80u8;
__t43.initFunction = None;
__t43
}, {
let mut __t44: us_unusual_dev = unsafe { core::mem::zeroed() };
__t44.vendorName = (b"Epson\0".as_ptr() as *mut u8) as *const u8;
__t44.productName = (b"875DC Storage\0".as_ptr() as *mut u8) as *const u8;
__t44.useProtocol = 6u8;
__t44.useTransport = 1u8;
__t44.initFunction = None;
__t44
}, {
let mut __t45: us_unusual_dev = unsafe { core::mem::zeroed() };
__t45.vendorName = (b"Epson\0".as_ptr() as *mut u8) as *const u8;
__t45.productName = (b"785EPX Storage\0".as_ptr() as *mut u8) as *const u8;
__t45.useProtocol = 6u8;
__t45.useTransport = 80u8;
__t45.initFunction = None;
__t45
}, {
let mut __t46: us_unusual_dev = unsafe { core::mem::zeroed() };
__t46.vendorName = (b"iODD\0".as_ptr() as *mut u8) as *const u8;
__t46.productName = (b"2531/2541\0".as_ptr() as *mut u8) as *const u8;
__t46.useProtocol = 255u8;
__t46.useTransport = 255u8;
__t46.initFunction = None;
__t46
}, {
let mut __t47: us_unusual_dev = unsafe { core::mem::zeroed() };
__t47.vendorName = (b"Fujifilm\0".as_ptr() as *mut u8) as *const u8;
__t47.productName = (b"FinePix 1400Zoom\0".as_ptr() as *mut u8) as *const u8;
__t47.useProtocol = 4u8;
__t47.useTransport = 255u8;
__t47.initFunction = None;
__t47
}, {
let mut __t48: us_unusual_dev = unsafe { core::mem::zeroed() };
__t48.vendorName = (b"ScanLogic\0".as_ptr() as *mut u8) as *const u8;
__t48.productName = (b"SL11R-IDE\0".as_ptr() as *mut u8) as *const u8;
__t48.useProtocol = 255u8;
__t48.useTransport = 255u8;
__t48.initFunction = None;
__t48
}, {
let mut __t49: us_unusual_dev = unsafe { core::mem::zeroed() };
__t49.vendorName = (b"ScanLogic\0".as_ptr() as *mut u8) as *const u8;
__t49.productName = (b"SL11R-IDE\0".as_ptr() as *mut u8) as *const u8;
__t49.useProtocol = 255u8;
__t49.useTransport = 255u8;
__t49.initFunction = None;
__t49
}, {
let mut __t50: us_unusual_dev = unsafe { core::mem::zeroed() };
__t50.vendorName = (b"Panasonic\0".as_ptr() as *mut u8) as *const u8;
__t50.productName = (b"LS-120 Camera\0".as_ptr() as *mut u8) as *const u8;
__t50.useProtocol = 4u8;
__t50.useTransport = 255u8;
__t50.initFunction = None;
__t50
}, {
let mut __t51: us_unusual_dev = unsafe { core::mem::zeroed() };
__t51.vendorName = (b"Sharp CE-CW05\0".as_ptr() as *mut u8) as *const u8;
__t51.productName = (b"CD-R/RW Drive\0".as_ptr() as *mut u8) as *const u8;
__t51.useProtocol = 5u8;
__t51.useTransport = 1u8;
__t51.initFunction = None;
__t51
}, {
let mut __t52: us_unusual_dev = unsafe { core::mem::zeroed() };
__t52.vendorName = (b"Panasonic\0".as_ptr() as *mut u8) as *const u8;
__t52.productName = (b"DMC-LCx Camera\0".as_ptr() as *mut u8) as *const u8;
__t52.useProtocol = 255u8;
__t52.useTransport = 255u8;
__t52.initFunction = None;
__t52
}, {
let mut __t53: us_unusual_dev = unsafe { core::mem::zeroed() };
__t53.vendorName = (b"LEICA\0".as_ptr() as *mut u8) as *const u8;
__t53.productName = (b"D-LUX Camera\0".as_ptr() as *mut u8) as *const u8;
__t53.useProtocol = 255u8;
__t53.useTransport = 255u8;
__t53.initFunction = None;
__t53
}, {
let mut __t54: us_unusual_dev = unsafe { core::mem::zeroed() };
__t54.vendorName = (b"Matshita\0".as_ptr() as *mut u8) as *const u8;
__t54.productName = (b"LS-120\0".as_ptr() as *mut u8) as *const u8;
__t54.useProtocol = 2u8;
__t54.useTransport = 1u8;
__t54.initFunction = None;
__t54
}, {
let mut __t55: us_unusual_dev = unsafe { core::mem::zeroed() };
__t55.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t55.productName = (b"eUSCSI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t55.useProtocol = 255u8;
__t55.useTransport = 255u8;
__t55.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t55
}, {
let mut __t56: us_unusual_dev = unsafe { core::mem::zeroed() };
__t56.vendorName = (b"SCM Microsystems\0".as_ptr() as *mut u8) as *const u8;
__t56.productName = (b"eUSB CompactFlash Adapter\0".as_ptr() as *mut u8) as *const u8;
__t56.useProtocol = 6u8;
__t56.useTransport = 1u8;
__t56.initFunction = None;
__t56
}, {
let mut __t57: us_unusual_dev = unsafe { core::mem::zeroed() };
__t57.vendorName = (b"SCM Microsystems Inc.\0".as_ptr() as *mut u8) as *const u8;
__t57.productName = (b"eUSB MMC Adapter\0".as_ptr() as *mut u8) as *const u8;
__t57.useProtocol = 6u8;
__t57.useTransport = 1u8;
__t57.initFunction = None;
__t57
}, {
let mut __t58: us_unusual_dev = unsafe { core::mem::zeroed() };
__t58.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t58.productName = (b"eUSB MMC Adapter\0".as_ptr() as *mut u8) as *const u8;
__t58.useProtocol = 6u8;
__t58.useTransport = 255u8;
__t58.initFunction = None;
__t58
}, {
let mut __t59: us_unusual_dev = unsafe { core::mem::zeroed() };
__t59.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t59.productName = (b"Hifd\0".as_ptr() as *mut u8) as *const u8;
__t59.useProtocol = 6u8;
__t59.useTransport = 1u8;
__t59.initFunction = None;
__t59
}, {
let mut __t60: us_unusual_dev = unsafe { core::mem::zeroed() };
__t60.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t60.productName = (b"eUSB ATA/ATAPI Adapter\0".as_ptr() as *mut u8) as *const u8;
__t60.useProtocol = 2u8;
__t60.useTransport = 1u8;
__t60.initFunction = None;
__t60
}, {
let mut __t61: us_unusual_dev = unsafe { core::mem::zeroed() };
__t61.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t61.productName = (b"eUSB CompactFlash Adapter\0".as_ptr() as *mut u8) as *const u8;
__t61.useProtocol = 2u8;
__t61.useTransport = 1u8;
__t61.initFunction = None;
__t61
}, {
let mut __t62: us_unusual_dev = unsafe { core::mem::zeroed() };
__t62.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t62.productName = (b"eUSCSI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t62.useProtocol = 6u8;
__t62.useTransport = 80u8;
__t62.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t62
}, {
let mut __t63: us_unusual_dev = unsafe { core::mem::zeroed() };
__t63.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t63.productName = (b"eUSCSI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t63.useProtocol = 6u8;
__t63.useTransport = 80u8;
__t63.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t63
}, {
let mut __t64: us_unusual_dev = unsafe { core::mem::zeroed() };
__t64.vendorName = (b"SCM Microsystems\0".as_ptr() as *mut u8) as *const u8;
__t64.productName = (b"eUSB SCSI Adapter (Bus Powered)\0".as_ptr() as *mut u8) as *const u8;
__t64.useProtocol = 6u8;
__t64.useTransport = 80u8;
__t64.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t64
}, {
let mut __t65: us_unusual_dev = unsafe { core::mem::zeroed() };
__t65.vendorName = (b"Shuttle\0".as_ptr() as *mut u8) as *const u8;
__t65.productName = (b"CD-RW Device\0".as_ptr() as *mut u8) as *const u8;
__t65.useProtocol = 2u8;
__t65.useTransport = 1u8;
__t65.initFunction = None;
__t65
}, {
let mut __t66: us_unusual_dev = unsafe { core::mem::zeroed() };
__t66.vendorName = (b"Samsung\0".as_ptr() as *mut u8) as *const u8;
__t66.productName = (b"YP-U3\0".as_ptr() as *mut u8) as *const u8;
__t66.useProtocol = 255u8;
__t66.useTransport = 255u8;
__t66.initFunction = None;
__t66
}, {
let mut __t67: us_unusual_dev = unsafe { core::mem::zeroed() };
__t67.vendorName = (b"Samsung\0".as_ptr() as *mut u8) as *const u8;
__t67.productName = (b"YP-CP3\0".as_ptr() as *mut u8) as *const u8;
__t67.useProtocol = 255u8;
__t67.useTransport = 255u8;
__t67.initFunction = None;
__t67
}, {
let mut __t68: us_unusual_dev = unsafe { core::mem::zeroed() };
__t68.vendorName = (b"Samsung\0".as_ptr() as *mut u8) as *const u8;
__t68.productName = (b"YP-Z3\0".as_ptr() as *mut u8) as *const u8;
__t68.useProtocol = 255u8;
__t68.useTransport = 255u8;
__t68.initFunction = None;
__t68
}, {
let mut __t69: us_unusual_dev = unsafe { core::mem::zeroed() };
__t69.vendorName = (b"Kobian Mercury\0".as_ptr() as *mut u8) as *const u8;
__t69.productName = (b"Binocam DCB-132\0".as_ptr() as *mut u8) as *const u8;
__t69.useProtocol = 255u8;
__t69.useTransport = 255u8;
__t69.initFunction = None;
__t69
}, {
let mut __t70: us_unusual_dev = unsafe { core::mem::zeroed() };
__t70.vendorName = (b"Belkin\0".as_ptr() as *mut u8) as *const u8;
__t70.productName = (b"USB SCSI Adaptor\0".as_ptr() as *mut u8) as *const u8;
__t70.useProtocol = 6u8;
__t70.useTransport = 80u8;
__t70.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t70
}, {
let mut __t71: us_unusual_dev = unsafe { core::mem::zeroed() };
__t71.vendorName = (b"Iomega\0".as_ptr() as *mut u8) as *const u8;
__t71.productName = (b"USB Clik! 40\0".as_ptr() as *mut u8) as *const u8;
__t71.useProtocol = 5u8;
__t71.useTransport = 255u8;
__t71.initFunction = None;
__t71
}, {
let mut __t72: us_unusual_dev = unsafe { core::mem::zeroed() };
__t72.vendorName = (b"Linux\0".as_ptr() as *mut u8) as *const u8;
__t72.productName = (b"File-backed Storage Gadget\0".as_ptr() as *mut u8) as *const u8;
__t72.useProtocol = 255u8;
__t72.useTransport = 255u8;
__t72.initFunction = None;
__t72
}, {
let mut __t73: us_unusual_dev = unsafe { core::mem::zeroed() };
__t73.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t73.productName = (b"300_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t73.useProtocol = 255u8;
__t73.useTransport = 255u8;
__t73.initFunction = None;
__t73
}, {
let mut __t74: us_unusual_dev = unsafe { core::mem::zeroed() };
__t74.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t74.productName = (b"300_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t74.useProtocol = 255u8;
__t74.useTransport = 255u8;
__t74.initFunction = None;
__t74
}, {
let mut __t75: us_unusual_dev = unsafe { core::mem::zeroed() };
__t75.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t75.productName = (b"300_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t75.useProtocol = 255u8;
__t75.useTransport = 255u8;
__t75.initFunction = None;
__t75
}, {
let mut __t76: us_unusual_dev = unsafe { core::mem::zeroed() };
__t76.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t76.productName = (b"400_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t76.useProtocol = 255u8;
__t76.useTransport = 255u8;
__t76.initFunction = None;
__t76
}, {
let mut __t77: us_unusual_dev = unsafe { core::mem::zeroed() };
__t77.vendorName = (b"Tekom Technologies, Inc\0".as_ptr() as *mut u8) as *const u8;
__t77.productName = (b"400_CAMERA\0".as_ptr() as *mut u8) as *const u8;
__t77.useProtocol = 255u8;
__t77.useTransport = 255u8;
__t77.initFunction = None;
__t77
}, {
let mut __t78: us_unusual_dev = unsafe { core::mem::zeroed() };
__t78.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t78.productName = (b"DSC-S30/S70/S75/505V/F505/F707/F717/P8\0".as_ptr() as *mut u8) as *const u8;
__t78.useProtocol = 6u8;
__t78.useTransport = 255u8;
__t78.initFunction = None;
__t78
}, {
let mut __t79: us_unusual_dev = unsafe { core::mem::zeroed() };
__t79.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t79.productName = (b"DSC-T1/T5/H5\0".as_ptr() as *mut u8) as *const u8;
__t79.useProtocol = 5u8;
__t79.useTransport = 255u8;
__t79.initFunction = None;
__t79
}, {
let mut __t80: us_unusual_dev = unsafe { core::mem::zeroed() };
__t80.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t80.productName = (b"Memorystick NW-MS7\0".as_ptr() as *mut u8) as *const u8;
__t80.useProtocol = 255u8;
__t80.useTransport = 255u8;
__t80.initFunction = None;
__t80
}, {
let mut __t81: us_unusual_dev = unsafe { core::mem::zeroed() };
__t81.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t81.productName = (b"USB Floppy Drive\0".as_ptr() as *mut u8) as *const u8;
__t81.useProtocol = 255u8;
__t81.useTransport = 255u8;
__t81.initFunction = None;
__t81
}, {
let mut __t82: us_unusual_dev = unsafe { core::mem::zeroed() };
__t82.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t82.productName = (b"Memorystick MSAC-US1\0".as_ptr() as *mut u8) as *const u8;
__t82.useProtocol = 255u8;
__t82.useTransport = 255u8;
__t82.initFunction = None;
__t82
}, {
let mut __t83: us_unusual_dev = unsafe { core::mem::zeroed() };
__t83.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t83.productName = (b"Handycam\0".as_ptr() as *mut u8) as *const u8;
__t83.useProtocol = 6u8;
__t83.useTransport = 255u8;
__t83.initFunction = None;
__t83
}, {
let mut __t84: us_unusual_dev = unsafe { core::mem::zeroed() };
__t84.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t84.productName = (b"Handycam HC-85\0".as_ptr() as *mut u8) as *const u8;
__t84.useProtocol = 4u8;
__t84.useTransport = 255u8;
__t84.initFunction = None;
__t84
}, {
let mut __t85: us_unusual_dev = unsafe { core::mem::zeroed() };
__t85.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t85.productName = (b"Memorystick MSC-U01N\0".as_ptr() as *mut u8) as *const u8;
__t85.useProtocol = 255u8;
__t85.useTransport = 255u8;
__t85.initFunction = None;
__t85
}, {
let mut __t86: us_unusual_dev = unsafe { core::mem::zeroed() };
__t86.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t86.productName = (b"PEG N760c Memorystick\0".as_ptr() as *mut u8) as *const u8;
__t86.useProtocol = 255u8;
__t86.useTransport = 255u8;
__t86.initFunction = None;
__t86
}, {
let mut __t87: us_unusual_dev = unsafe { core::mem::zeroed() };
__t87.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t87.productName = (b"Memorystick MSC-U03\0".as_ptr() as *mut u8) as *const u8;
__t87.useProtocol = 4u8;
__t87.useTransport = 1u8;
__t87.initFunction = None;
__t87
}, {
let mut __t88: us_unusual_dev = unsafe { core::mem::zeroed() };
__t88.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t88.productName = (b"PEG Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t88.useProtocol = 255u8;
__t88.useTransport = 255u8;
__t88.initFunction = None;
__t88
}, {
let mut __t89: us_unusual_dev = unsafe { core::mem::zeroed() };
__t89.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t89.productName = (b"PEG Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t89.useProtocol = 255u8;
__t89.useTransport = 255u8;
__t89.initFunction = None;
__t89
}, {
let mut __t90: us_unusual_dev = unsafe { core::mem::zeroed() };
__t90.vendorName = (b"Sony\0".as_ptr() as *mut u8) as *const u8;
__t90.productName = (b"PEG Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t90.useProtocol = 255u8;
__t90.useTransport = 255u8;
__t90.initFunction = None;
__t90
}, {
let mut __t91: us_unusual_dev = unsafe { core::mem::zeroed() };
__t91.vendorName = (b"Sony Corp.\0".as_ptr() as *mut u8) as *const u8;
__t91.productName = (b"MicroVault Flash Drive\0".as_ptr() as *mut u8) as *const u8;
__t91.useProtocol = 255u8;
__t91.useTransport = 255u8;
__t91.initFunction = None;
__t91
}, {
let mut __t92: us_unusual_dev = unsafe { core::mem::zeroed() };
__t92.vendorName = (b"SAMSUNG\0".as_ptr() as *mut u8) as *const u8;
__t92.productName = (b"SFD-321U [FW 0C]\0".as_ptr() as *mut u8) as *const u8;
__t92.useProtocol = 255u8;
__t92.useTransport = 255u8;
__t92.initFunction = None;
__t92
}, {
let mut __t93: us_unusual_dev = unsafe { core::mem::zeroed() };
__t93.vendorName = (b"Y-E Data\0".as_ptr() as *mut u8) as *const u8;
__t93.productName = (b"Flashbuster-U\0".as_ptr() as *mut u8) as *const u8;
__t93.useProtocol = 255u8;
__t93.useTransport = 1u8;
__t93.initFunction = None;
__t93
}, {
let mut __t94: us_unusual_dev = unsafe { core::mem::zeroed() };
__t94.vendorName = (b"Y-E Data\0".as_ptr() as *mut u8) as *const u8;
__t94.productName = (b"Silicon Media R/W\0".as_ptr() as *mut u8) as *const u8;
__t94.useProtocol = 255u8;
__t94.useTransport = 255u8;
__t94.initFunction = None;
__t94
}, {
let mut __t95: us_unusual_dev = unsafe { core::mem::zeroed() };
__t95.vendorName = (b"JetFlash\0".as_ptr() as *mut u8) as *const u8;
__t95.productName = (b"TS1GJF2A/120\0".as_ptr() as *mut u8) as *const u8;
__t95.useProtocol = 255u8;
__t95.useTransport = 255u8;
__t95.initFunction = None;
__t95
}, {
let mut __t96: us_unusual_dev = unsafe { core::mem::zeroed() };
__t96.vendorName = (b"Fujifilm\0".as_ptr() as *mut u8) as *const u8;
__t96.productName = (b"Digital Camera EX-20 DSC\0".as_ptr() as *mut u8) as *const u8;
__t96.useProtocol = 5u8;
__t96.useTransport = 255u8;
__t96.initFunction = None;
__t96
}, {
let mut __t97: us_unusual_dev = unsafe { core::mem::zeroed() };
__t97.vendorName = (b"Iomega\0".as_ptr() as *mut u8) as *const u8;
__t97.productName = (b"ZIP 100\0".as_ptr() as *mut u8) as *const u8;
__t97.useProtocol = 255u8;
__t97.useTransport = 255u8;
__t97.initFunction = None;
__t97
}, {
let mut __t98: us_unusual_dev = unsafe { core::mem::zeroed() };
__t98.vendorName = (b"Iomega\0".as_ptr() as *mut u8) as *const u8;
__t98.productName = (b"Jaz USB Adapter\0".as_ptr() as *mut u8) as *const u8;
__t98.useProtocol = 255u8;
__t98.useTransport = 255u8;
__t98.initFunction = None;
__t98
}, {
let mut __t99: us_unusual_dev = unsafe { core::mem::zeroed() };
__t99.vendorName = (b"LaCie\0".as_ptr() as *mut u8) as *const u8;
__t99.productName = (b"DVD+-RW\0".as_ptr() as *mut u8) as *const u8;
__t99.useProtocol = 255u8;
__t99.useTransport = 255u8;
__t99.initFunction = None;
__t99
}, {
let mut __t100: us_unusual_dev = unsafe { core::mem::zeroed() };
__t100.vendorName = (b"LaCie\0".as_ptr() as *mut u8) as *const u8;
__t100.productName = (b"External HDD\0".as_ptr() as *mut u8) as *const u8;
__t100.useProtocol = 255u8;
__t100.useTransport = 255u8;
__t100.initFunction = None;
__t100
}, {
let mut __t101: us_unusual_dev = unsafe { core::mem::zeroed() };
__t101.vendorName = (b"In-System\0".as_ptr() as *mut u8) as *const u8;
__t101.productName = (b"PyroGate External CD-ROM Enclosure (FCD-523)\0".as_ptr() as *mut u8) as *const u8;
__t101.useProtocol = 6u8;
__t101.useTransport = 80u8;
__t101.initFunction = None;
__t101
}, {
let mut __t102: us_unusual_dev = unsafe { core::mem::zeroed() };
__t102.vendorName = (b"Apple\0".as_ptr() as *mut u8) as *const u8;
__t102.productName = (b"iPod\0".as_ptr() as *mut u8) as *const u8;
__t102.useProtocol = 255u8;
__t102.useTransport = 255u8;
__t102.initFunction = None;
__t102
}, {
let mut __t103: us_unusual_dev = unsafe { core::mem::zeroed() };
__t103.vendorName = (b"Apple\0".as_ptr() as *mut u8) as *const u8;
__t103.productName = (b"iPod\0".as_ptr() as *mut u8) as *const u8;
__t103.useProtocol = 255u8;
__t103.useTransport = 255u8;
__t103.initFunction = None;
__t103
}, {
let mut __t104: us_unusual_dev = unsafe { core::mem::zeroed() };
__t104.vendorName = (b"Apple\0".as_ptr() as *mut u8) as *const u8;
__t104.productName = (b"iPod\0".as_ptr() as *mut u8) as *const u8;
__t104.useProtocol = 255u8;
__t104.useTransport = 255u8;
__t104.initFunction = None;
__t104
}, {
let mut __t105: us_unusual_dev = unsafe { core::mem::zeroed() };
__t105.vendorName = (b"Apple\0".as_ptr() as *mut u8) as *const u8;
__t105.productName = (b"iPod\0".as_ptr() as *mut u8) as *const u8;
__t105.useProtocol = 255u8;
__t105.useTransport = 255u8;
__t105.initFunction = None;
__t105
}, {
let mut __t106: us_unusual_dev = unsafe { core::mem::zeroed() };
__t106.vendorName = (b"Apple\0".as_ptr() as *mut u8) as *const u8;
__t106.productName = (b"iPod\0".as_ptr() as *mut u8) as *const u8;
__t106.useProtocol = 255u8;
__t106.useTransport = 255u8;
__t106.initFunction = None;
__t106
}, {
let mut __t107: us_unusual_dev = unsafe { core::mem::zeroed() };
__t107.vendorName = (b"Option N.V.\0".as_ptr() as *mut u8) as *const u8;
__t107.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t107.useProtocol = 255u8;
__t107.useTransport = 255u8;
__t107.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(option_ms_init as *const ()) };
__t107
}, {
let mut __t108: us_unusual_dev = unsafe { core::mem::zeroed() };
__t108.vendorName = (b"Lexar\0".as_ptr() as *mut u8) as *const u8;
__t108.productName = (b"USB CF Reader\0".as_ptr() as *mut u8) as *const u8;
__t108.useProtocol = 255u8;
__t108.useTransport = 255u8;
__t108.initFunction = None;
__t108
}, {
let mut __t109: us_unusual_dev = unsafe { core::mem::zeroed() };
__t109.vendorName = (b"Genesys Logic\0".as_ptr() as *mut u8) as *const u8;
__t109.productName = (b"USB to IDE Optical\0".as_ptr() as *mut u8) as *const u8;
__t109.useProtocol = 255u8;
__t109.useTransport = 255u8;
__t109.initFunction = None;
__t109
}, {
let mut __t110: us_unusual_dev = unsafe { core::mem::zeroed() };
__t110.vendorName = (b"Genesys Logic\0".as_ptr() as *mut u8) as *const u8;
__t110.productName = (b"USB to IDE Disk\0".as_ptr() as *mut u8) as *const u8;
__t110.useProtocol = 255u8;
__t110.useTransport = 255u8;
__t110.initFunction = None;
__t110
}, {
let mut __t111: us_unusual_dev = unsafe { core::mem::zeroed() };
__t111.vendorName = (b"Genesys Logic\0".as_ptr() as *mut u8) as *const u8;
__t111.productName = (b"USB to SATA\0".as_ptr() as *mut u8) as *const u8;
__t111.useProtocol = 255u8;
__t111.useTransport = 255u8;
__t111.initFunction = None;
__t111
}, {
let mut __t112: us_unusual_dev = unsafe { core::mem::zeroed() };
__t112.vendorName = (b"Novatek\0".as_ptr() as *mut u8) as *const u8;
__t112.productName = (b"NTK96550-based camera\0".as_ptr() as *mut u8) as *const u8;
__t112.useProtocol = 255u8;
__t112.useTransport = 255u8;
__t112.initFunction = None;
__t112
}, {
let mut __t113: us_unusual_dev = unsafe { core::mem::zeroed() };
__t113.vendorName = (b"Vivitar\0".as_ptr() as *mut u8) as *const u8;
__t113.productName = (b"Vivicam 35Xx\0".as_ptr() as *mut u8) as *const u8;
__t113.useProtocol = 6u8;
__t113.useTransport = 80u8;
__t113.initFunction = None;
__t113
}, {
let mut __t114: us_unusual_dev = unsafe { core::mem::zeroed() };
__t114.vendorName = (b"TEAC\0".as_ptr() as *mut u8) as *const u8;
__t114.productName = (b"Floppy Drive\0".as_ptr() as *mut u8) as *const u8;
__t114.useProtocol = 4u8;
__t114.useTransport = 1u8;
__t114.initFunction = None;
__t114
}, {
let mut __t115: us_unusual_dev = unsafe { core::mem::zeroed() };
__t115.vendorName = (b"SigmaTel\0".as_ptr() as *mut u8) as *const u8;
__t115.productName = (b"USBMSC Audio Player\0".as_ptr() as *mut u8) as *const u8;
__t115.useProtocol = 255u8;
__t115.useTransport = 255u8;
__t115.initFunction = None;
__t115
}, {
let mut __t116: us_unusual_dev = unsafe { core::mem::zeroed() };
__t116.vendorName = (b"Prolific Technology, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t116.productName = (b"Prolific Storage Gadget\0".as_ptr() as *mut u8) as *const u8;
__t116.useProtocol = 255u8;
__t116.useTransport = 255u8;
__t116.initFunction = None;
__t116
}, {
let mut __t117: us_unusual_dev = unsafe { core::mem::zeroed() };
__t117.vendorName = (b"Prolific Technology, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t117.productName = (b"Mass Storage Device\0".as_ptr() as *mut u8) as *const u8;
__t117.useProtocol = 255u8;
__t117.useTransport = 255u8;
__t117.initFunction = None;
__t117
}, {
let mut __t118: us_unusual_dev = unsafe { core::mem::zeroed() };
__t118.vendorName = (b"Prolific Technology Inc.\0".as_ptr() as *mut u8) as *const u8;
__t118.productName = (b"Mass Storage Device\0".as_ptr() as *mut u8) as *const u8;
__t118.useProtocol = 255u8;
__t118.useTransport = 255u8;
__t118.initFunction = None;
__t118
}, {
let mut __t119: us_unusual_dev = unsafe { core::mem::zeroed() };
__t119.vendorName = (b"Prolific Technology Inc.\0".as_ptr() as *mut u8) as *const u8;
__t119.productName = (b"ATAPI-6 Bridge Controller\0".as_ptr() as *mut u8) as *const u8;
__t119.useProtocol = 255u8;
__t119.useTransport = 255u8;
__t119.initFunction = None;
__t119
}, {
let mut __t120: us_unusual_dev = unsafe { core::mem::zeroed() };
__t120.vendorName = (b"Minolta\0".as_ptr() as *mut u8) as *const u8;
__t120.productName = (b"Dimage F300\0".as_ptr() as *mut u8) as *const u8;
__t120.useProtocol = 6u8;
__t120.useTransport = 80u8;
__t120.initFunction = None;
__t120
}, {
let mut __t121: us_unusual_dev = unsafe { core::mem::zeroed() };
__t121.vendorName = (b"Minolta\0".as_ptr() as *mut u8) as *const u8;
__t121.productName = (b"DIMAGE E223\0".as_ptr() as *mut u8) as *const u8;
__t121.useProtocol = 6u8;
__t121.useTransport = 255u8;
__t121.initFunction = None;
__t121
}, {
let mut __t122: us_unusual_dev = unsafe { core::mem::zeroed() };
__t122.vendorName = (b"Hagiwara\0".as_ptr() as *mut u8) as *const u8;
__t122.productName = (b"Flashgate\0".as_ptr() as *mut u8) as *const u8;
__t122.useProtocol = 6u8;
__t122.useTransport = 80u8;
__t122.initFunction = None;
__t122
}, {
let mut __t123: us_unusual_dev = unsafe { core::mem::zeroed() };
__t123.vendorName = (b"Thomson Multimedia Inc.\0".as_ptr() as *mut u8) as *const u8;
__t123.productName = (b"RCA RD1080 MP3 Player\0".as_ptr() as *mut u8) as *const u8;
__t123.useProtocol = 255u8;
__t123.useTransport = 255u8;
__t123.initFunction = None;
__t123
}, {
let mut __t124: us_unusual_dev = unsafe { core::mem::zeroed() };
__t124.vendorName = (b"Newer Technology\0".as_ptr() as *mut u8) as *const u8;
__t124.productName = (b"uSCSI\0".as_ptr() as *mut u8) as *const u8;
__t124.useProtocol = 255u8;
__t124.useTransport = 255u8;
__t124.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t124
}, {
let mut __t125: us_unusual_dev = unsafe { core::mem::zeroed() };
__t125.vendorName = (b"RockChip\0".as_ptr() as *mut u8) as *const u8;
__t125.productName = (b"MP3\0".as_ptr() as *mut u8) as *const u8;
__t125.useProtocol = 255u8;
__t125.useTransport = 255u8;
__t125.initFunction = None;
__t125
}, {
let mut __t126: us_unusual_dev = unsafe { core::mem::zeroed() };
__t126.vendorName = (b"RockChip\0".as_ptr() as *mut u8) as *const u8;
__t126.productName = (b"MTP\0".as_ptr() as *mut u8) as *const u8;
__t126.useProtocol = 255u8;
__t126.useTransport = 255u8;
__t126.initFunction = None;
__t126
}, {
let mut __t127: us_unusual_dev = unsafe { core::mem::zeroed() };
__t127.vendorName = (b"RockChip\0".as_ptr() as *mut u8) as *const u8;
__t127.productName = (b"ROCK MP3\0".as_ptr() as *mut u8) as *const u8;
__t127.useProtocol = 255u8;
__t127.useTransport = 255u8;
__t127.initFunction = None;
__t127
}, {
let mut __t128: us_unusual_dev = unsafe { core::mem::zeroed() };
__t128.vendorName = (b"ATMEL\0".as_ptr() as *mut u8) as *const u8;
__t128.productName = (b"SND1 Storage\0".as_ptr() as *mut u8) as *const u8;
__t128.useProtocol = 255u8;
__t128.useTransport = 255u8;
__t128.initFunction = None;
__t128
}, {
let mut __t129: us_unusual_dev = unsafe { core::mem::zeroed() };
__t129.vendorName = (b"Sandisk\0".as_ptr() as *mut u8) as *const u8;
__t129.productName = (b"ImageMate SDDR-05a\0".as_ptr() as *mut u8) as *const u8;
__t129.useProtocol = 6u8;
__t129.useTransport = 1u8;
__t129.initFunction = None;
__t129
}, {
let mut __t130: us_unusual_dev = unsafe { core::mem::zeroed() };
__t130.vendorName = (b"SanDisk Corporation\0".as_ptr() as *mut u8) as *const u8;
__t130.productName = (b"ImageMate CompactFlash USB\0".as_ptr() as *mut u8) as *const u8;
__t130.useProtocol = 255u8;
__t130.useTransport = 255u8;
__t130.initFunction = None;
__t130
}, {
let mut __t131: us_unusual_dev = unsafe { core::mem::zeroed() };
__t131.vendorName = (b"Sandisk\0".as_ptr() as *mut u8) as *const u8;
__t131.productName = (b"ImageMate SDDR-12\0".as_ptr() as *mut u8) as *const u8;
__t131.useProtocol = 6u8;
__t131.useTransport = 1u8;
__t131.initFunction = None;
__t131
}, {
let mut __t132: us_unusual_dev = unsafe { core::mem::zeroed() };
__t132.vendorName = (b"Freecom Technologies\0".as_ptr() as *mut u8) as *const u8;
__t132.productName = (b"FHD-Classic\0".as_ptr() as *mut u8) as *const u8;
__t132.useProtocol = 255u8;
__t132.useTransport = 255u8;
__t132.initFunction = None;
__t132
}, {
let mut __t133: us_unusual_dev = unsafe { core::mem::zeroed() };
__t133.vendorName = (b"Microtech\0".as_ptr() as *mut u8) as *const u8;
__t133.productName = (b"USB-SCSI-DB25\0".as_ptr() as *mut u8) as *const u8;
__t133.useProtocol = 255u8;
__t133.useTransport = 255u8;
__t133.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t133
}, {
let mut __t134: us_unusual_dev = unsafe { core::mem::zeroed() };
__t134.vendorName = (b"Microtech\0".as_ptr() as *mut u8) as *const u8;
__t134.productName = (b"USB-SCSI-HD50\0".as_ptr() as *mut u8) as *const u8;
__t134.useProtocol = 255u8;
__t134.useTransport = 255u8;
__t134.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t134
}, {
let mut __t135: us_unusual_dev = unsafe { core::mem::zeroed() };
__t135.vendorName = (b"Microtech\0".as_ptr() as *mut u8) as *const u8;
__t135.productName = (b"CameraMate\0".as_ptr() as *mut u8) as *const u8;
__t135.useProtocol = 6u8;
__t135.useTransport = 1u8;
__t135.initFunction = None;
__t135
}, {
let mut __t136: us_unusual_dev = unsafe { core::mem::zeroed() };
__t136.vendorName = (b"Datafab\0".as_ptr() as *mut u8) as *const u8;
__t136.productName = (b"KECF-USB\0".as_ptr() as *mut u8) as *const u8;
__t136.useProtocol = 255u8;
__t136.useTransport = 255u8;
__t136.initFunction = None;
__t136
}, {
let mut __t137: us_unusual_dev = unsafe { core::mem::zeroed() };
__t137.vendorName = (b"Simple Tech/Datafab\0".as_ptr() as *mut u8) as *const u8;
__t137.productName = (b"CF+SM Reader\0".as_ptr() as *mut u8) as *const u8;
__t137.useProtocol = 255u8;
__t137.useTransport = 255u8;
__t137.initFunction = None;
__t137
}, {
let mut __t138: us_unusual_dev = unsafe { core::mem::zeroed() };
__t138.vendorName = (b"Casio\0".as_ptr() as *mut u8) as *const u8;
__t138.productName = (b"QV DigitalCamera\0".as_ptr() as *mut u8) as *const u8;
__t138.useProtocol = 5u8;
__t138.useTransport = 1u8;
__t138.initFunction = None;
__t138
}, {
let mut __t139: us_unusual_dev = unsafe { core::mem::zeroed() };
__t139.vendorName = (b"Casio\0".as_ptr() as *mut u8) as *const u8;
__t139.productName = (b"EX-N1 DigitalCamera\0".as_ptr() as *mut u8) as *const u8;
__t139.useProtocol = 5u8;
__t139.useTransport = 255u8;
__t139.initFunction = None;
__t139
}, {
let mut __t140: us_unusual_dev = unsafe { core::mem::zeroed() };
__t140.vendorName = (b"Samsung\0".as_ptr() as *mut u8) as *const u8;
__t140.productName = (b"Digimax 410\0".as_ptr() as *mut u8) as *const u8;
__t140.useProtocol = 255u8;
__t140.useTransport = 255u8;
__t140.initFunction = None;
__t140
}, {
let mut __t141: us_unusual_dev = unsafe { core::mem::zeroed() };
__t141.vendorName = (b"Argosy\0".as_ptr() as *mut u8) as *const u8;
__t141.productName = (b"Storage\0".as_ptr() as *mut u8) as *const u8;
__t141.useProtocol = 255u8;
__t141.useTransport = 255u8;
__t141.initFunction = None;
__t141
}, {
let mut __t142: us_unusual_dev = unsafe { core::mem::zeroed() };
__t142.vendorName = (b"Argosy\0".as_ptr() as *mut u8) as *const u8;
__t142.productName = (b"Storage\0".as_ptr() as *mut u8) as *const u8;
__t142.useProtocol = 255u8;
__t142.useTransport = 255u8;
__t142.initFunction = None;
__t142
}, {
let mut __t143: us_unusual_dev = unsafe { core::mem::zeroed() };
__t143.vendorName = (b"Argosy\0".as_ptr() as *mut u8) as *const u8;
__t143.productName = (b"Storage\0".as_ptr() as *mut u8) as *const u8;
__t143.useProtocol = 255u8;
__t143.useTransport = 255u8;
__t143.initFunction = None;
__t143
}, {
let mut __t144: us_unusual_dev = unsafe { core::mem::zeroed() };
__t144.vendorName = (b"Castlewood Systems\0".as_ptr() as *mut u8) as *const u8;
__t144.productName = (b"USB to SCSI cable\0".as_ptr() as *mut u8) as *const u8;
__t144.useProtocol = 255u8;
__t144.useTransport = 255u8;
__t144.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t144
}, {
let mut __t145: us_unusual_dev = unsafe { core::mem::zeroed() };
__t145.vendorName = (b"Grandtech\0".as_ptr() as *mut u8) as *const u8;
__t145.productName = (b"DC2MEGA\0".as_ptr() as *mut u8) as *const u8;
__t145.useProtocol = 255u8;
__t145.useTransport = 255u8;
__t145.initFunction = None;
__t145
}, {
let mut __t146: us_unusual_dev = unsafe { core::mem::zeroed() };
__t146.vendorName = (b"MagicPixel\0".as_ptr() as *mut u8) as *const u8;
__t146.productName = (b"FW_Omega2\0".as_ptr() as *mut u8) as *const u8;
__t146.useProtocol = 255u8;
__t146.useTransport = 255u8;
__t146.initFunction = None;
__t146
}, {
let mut __t147: us_unusual_dev = unsafe { core::mem::zeroed() };
__t147.vendorName = (b"PanDigital\0".as_ptr() as *mut u8) as *const u8;
__t147.productName = (b"Photo Frame\0".as_ptr() as *mut u8) as *const u8;
__t147.useProtocol = 255u8;
__t147.useTransport = 255u8;
__t147.initFunction = None;
__t147
}, {
let mut __t148: us_unusual_dev = unsafe { core::mem::zeroed() };
__t148.vendorName = (b"Xircom\0".as_ptr() as *mut u8) as *const u8;
__t148.productName = (b"PortGear USB-SCSI (Mac USB Dock)\0".as_ptr() as *mut u8) as *const u8;
__t148.useProtocol = 255u8;
__t148.useTransport = 255u8;
__t148.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t148
}, {
let mut __t149: us_unusual_dev = unsafe { core::mem::zeroed() };
__t149.vendorName = (b"Xircom\0".as_ptr() as *mut u8) as *const u8;
__t149.productName = (b"PortGear USB to SCSI Converter\0".as_ptr() as *mut u8) as *const u8;
__t149.useProtocol = 255u8;
__t149.useTransport = 255u8;
__t149.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t149
}, {
let mut __t150: us_unusual_dev = unsafe { core::mem::zeroed() };
__t150.vendorName = (b"CITIZEN\0".as_ptr() as *mut u8) as *const u8;
__t150.productName = (b"X1DE-USB\0".as_ptr() as *mut u8) as *const u8;
__t150.useProtocol = 255u8;
__t150.useTransport = 255u8;
__t150.initFunction = None;
__t150
}, {
let mut __t151: us_unusual_dev = unsafe { core::mem::zeroed() };
__t151.vendorName = (b"AIPTEK\0".as_ptr() as *mut u8) as *const u8;
__t151.productName = (b"Aiptek USB Keychain MP3 Player\0".as_ptr() as *mut u8) as *const u8;
__t151.useProtocol = 255u8;
__t151.useTransport = 255u8;
__t151.initFunction = None;
__t151
}, {
let mut __t152: us_unusual_dev = unsafe { core::mem::zeroed() };
__t152.vendorName = (b"Trumpion\0".as_ptr() as *mut u8) as *const u8;
__t152.productName = (b"t33520 USB Flash Card Controller\0".as_ptr() as *mut u8) as *const u8;
__t152.useProtocol = 255u8;
__t152.useTransport = 80u8;
__t152.initFunction = None;
__t152
}, {
let mut __t153: us_unusual_dev = unsafe { core::mem::zeroed() };
__t153.vendorName = (b"Trumpion Microelectronics, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t153.productName = (b"33520 USB Digital Voice Recorder\0".as_ptr() as *mut u8) as *const u8;
__t153.useProtocol = 4u8;
__t153.useTransport = 255u8;
__t153.initFunction = None;
__t153
}, {
let mut __t154: us_unusual_dev = unsafe { core::mem::zeroed() };
__t154.vendorName = (b"Trumpion\0".as_ptr() as *mut u8) as *const u8;
__t154.productName = (b"MP3 player\0".as_ptr() as *mut u8) as *const u8;
__t154.useProtocol = 1u8;
__t154.useTransport = 80u8;
__t154.initFunction = None;
__t154
}, {
let mut __t155: us_unusual_dev = unsafe { core::mem::zeroed() };
__t155.vendorName = (b"Feiya\0".as_ptr() as *mut u8) as *const u8;
__t155.productName = (b"5-in-1 Card Reader\0".as_ptr() as *mut u8) as *const u8;
__t155.useProtocol = 255u8;
__t155.useTransport = 255u8;
__t155.initFunction = None;
__t155
}, {
let mut __t156: us_unusual_dev = unsafe { core::mem::zeroed() };
__t156.vendorName = (b"SMI\0".as_ptr() as *mut u8) as *const u8;
__t156.productName = (b"SM3350 UFS-to-USB-Mass-Storage bridge\0".as_ptr() as *mut u8) as *const u8;
__t156.useProtocol = 255u8;
__t156.useTransport = 255u8;
__t156.initFunction = None;
__t156
}, {
let mut __t157: us_unusual_dev = unsafe { core::mem::zeroed() };
__t157.vendorName = (b"Feiya\0".as_ptr() as *mut u8) as *const u8;
__t157.productName = (b"SD/SDHC Card Reader\0".as_ptr() as *mut u8) as *const u8;
__t157.useProtocol = 255u8;
__t157.useTransport = 255u8;
__t157.initFunction = None;
__t157
}, {
let mut __t158: us_unusual_dev = unsafe { core::mem::zeroed() };
__t158.vendorName = (b"Kingston\0".as_ptr() as *mut u8) as *const u8;
__t158.productName = (b"DT Ultimate G3\0".as_ptr() as *mut u8) as *const u8;
__t158.useProtocol = 255u8;
__t158.useTransport = 255u8;
__t158.initFunction = None;
__t158
}, {
let mut __t159: us_unusual_dev = unsafe { core::mem::zeroed() };
__t159.vendorName = (b"Pentax\0".as_ptr() as *mut u8) as *const u8;
__t159.productName = (b"Optio 2/3/400\0".as_ptr() as *mut u8) as *const u8;
__t159.useProtocol = 255u8;
__t159.useTransport = 255u8;
__t159.initFunction = None;
__t159
}, {
let mut __t160: us_unusual_dev = unsafe { core::mem::zeroed() };
__t160.vendorName = (b"ZyXEL\0".as_ptr() as *mut u8) as *const u8;
__t160.productName = (b"G-220F USB-WLAN Install\0".as_ptr() as *mut u8) as *const u8;
__t160.useProtocol = 255u8;
__t160.useTransport = 255u8;
__t160.initFunction = None;
__t160
}, {
let mut __t161: us_unusual_dev = unsafe { core::mem::zeroed() };
__t161.vendorName = (b"SiteCom\0".as_ptr() as *mut u8) as *const u8;
__t161.productName = (b"WL-117 USB-WLAN Install\0".as_ptr() as *mut u8) as *const u8;
__t161.useProtocol = 255u8;
__t161.useTransport = 255u8;
__t161.initFunction = None;
__t161
}, {
let mut __t162: us_unusual_dev = unsafe { core::mem::zeroed() };
__t162.vendorName = (b"Option N.V.\0".as_ptr() as *mut u8) as *const u8;
__t162.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t162.useProtocol = 255u8;
__t162.useTransport = 255u8;
__t162.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(option_ms_init as *const ()) };
__t162
}, {
let mut __t163: us_unusual_dev = unsafe { core::mem::zeroed() };
__t163.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t163.productName = (b"GI 0401 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t163.useProtocol = 255u8;
__t163.useTransport = 255u8;
__t163.initFunction = None;
__t163
}, {
let mut __t164: us_unusual_dev = unsafe { core::mem::zeroed() };
__t164.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t164.productName = (b"GI 0431 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t164.useProtocol = 255u8;
__t164.useTransport = 255u8;
__t164.initFunction = None;
__t164
}, {
let mut __t165: us_unusual_dev = unsafe { core::mem::zeroed() };
__t165.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t165.productName = (b"GI 0451 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t165.useProtocol = 255u8;
__t165.useTransport = 255u8;
__t165.initFunction = None;
__t165
}, {
let mut __t166: us_unusual_dev = unsafe { core::mem::zeroed() };
__t166.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t166.productName = (b"GI 0451 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t166.useProtocol = 255u8;
__t166.useTransport = 255u8;
__t166.initFunction = None;
__t166
}, {
let mut __t167: us_unusual_dev = unsafe { core::mem::zeroed() };
__t167.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t167.productName = (b"GI 0452 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t167.useProtocol = 255u8;
__t167.useTransport = 255u8;
__t167.initFunction = None;
__t167
}, {
let mut __t168: us_unusual_dev = unsafe { core::mem::zeroed() };
__t168.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t168.productName = (b"GI 0461 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t168.useProtocol = 255u8;
__t168.useTransport = 255u8;
__t168.initFunction = None;
__t168
}, {
let mut __t169: us_unusual_dev = unsafe { core::mem::zeroed() };
__t169.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t169.productName = (b"GI 0461 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t169.useProtocol = 255u8;
__t169.useTransport = 255u8;
__t169.initFunction = None;
__t169
}, {
let mut __t170: us_unusual_dev = unsafe { core::mem::zeroed() };
__t170.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t170.productName = (b"GI 033x SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t170.useProtocol = 255u8;
__t170.useTransport = 255u8;
__t170.initFunction = None;
__t170
}, {
let mut __t171: us_unusual_dev = unsafe { core::mem::zeroed() };
__t171.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t171.productName = (b"GI 033x SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t171.useProtocol = 255u8;
__t171.useTransport = 255u8;
__t171.initFunction = None;
__t171
}, {
let mut __t172: us_unusual_dev = unsafe { core::mem::zeroed() };
__t172.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t172.productName = (b"GI 033x SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t172.useProtocol = 255u8;
__t172.useTransport = 255u8;
__t172.initFunction = None;
__t172
}, {
let mut __t173: us_unusual_dev = unsafe { core::mem::zeroed() };
__t173.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t173.productName = (b"GI 070x SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t173.useProtocol = 255u8;
__t173.useTransport = 255u8;
__t173.initFunction = None;
__t173
}, {
let mut __t174: us_unusual_dev = unsafe { core::mem::zeroed() };
__t174.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t174.productName = (b"GI 1505 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t174.useProtocol = 255u8;
__t174.useTransport = 255u8;
__t174.initFunction = None;
__t174
}, {
let mut __t175: us_unusual_dev = unsafe { core::mem::zeroed() };
__t175.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t175.productName = (b"GI 1509 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t175.useProtocol = 255u8;
__t175.useTransport = 255u8;
__t175.initFunction = None;
__t175
}, {
let mut __t176: us_unusual_dev = unsafe { core::mem::zeroed() };
__t176.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t176.productName = (b"GI 1515 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t176.useProtocol = 255u8;
__t176.useTransport = 255u8;
__t176.initFunction = None;
__t176
}, {
let mut __t177: us_unusual_dev = unsafe { core::mem::zeroed() };
__t177.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t177.productName = (b"GI 1215 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t177.useProtocol = 255u8;
__t177.useTransport = 255u8;
__t177.initFunction = None;
__t177
}, {
let mut __t178: us_unusual_dev = unsafe { core::mem::zeroed() };
__t178.vendorName = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t178.productName = (b"GI 1505 SD-Card\0".as_ptr() as *mut u8) as *const u8;
__t178.useProtocol = 255u8;
__t178.useTransport = 255u8;
__t178.initFunction = None;
__t178
}, {
let mut __t179: us_unusual_dev = unsafe { core::mem::zeroed() };
__t179.vendorName = (b"Seagate\0".as_ptr() as *mut u8) as *const u8;
__t179.productName = (b"Portable HDD\0".as_ptr() as *mut u8) as *const u8;
__t179.useProtocol = 255u8;
__t179.useTransport = 255u8;
__t179.initFunction = None;
__t179
}, {
let mut __t180: us_unusual_dev = unsafe { core::mem::zeroed() };
__t180.vendorName = (b"Seagate\0".as_ptr() as *mut u8) as *const u8;
__t180.productName = (b"FreeAgent Pro\0".as_ptr() as *mut u8) as *const u8;
__t180.useProtocol = 255u8;
__t180.useTransport = 255u8;
__t180.initFunction = None;
__t180
}, {
let mut __t181: us_unusual_dev = unsafe { core::mem::zeroed() };
__t181.vendorName = (b"Seagate\0".as_ptr() as *mut u8) as *const u8;
__t181.productName = (b"External\0".as_ptr() as *mut u8) as *const u8;
__t181.useProtocol = 255u8;
__t181.useTransport = 255u8;
__t181.initFunction = None;
__t181
}, {
let mut __t182: us_unusual_dev = unsafe { core::mem::zeroed() };
__t182.vendorName = (b"Realtek\0".as_ptr() as *mut u8) as *const u8;
__t182.productName = (b"DISK\0".as_ptr() as *mut u8) as *const u8;
__t182.useProtocol = 255u8;
__t182.useTransport = 255u8;
__t182.initFunction = None;
__t182
}, {
let mut __t183: us_unusual_dev = unsafe { core::mem::zeroed() };
__t183.vendorName = (b"Realtek\0".as_ptr() as *mut u8) as *const u8;
__t183.productName = (b"DISK\0".as_ptr() as *mut u8) as *const u8;
__t183.useProtocol = 255u8;
__t183.useTransport = 255u8;
__t183.initFunction = None;
__t183
}, {
let mut __t184: us_unusual_dev = unsafe { core::mem::zeroed() };
__t184.vendorName = (b"Maxtor\0".as_ptr() as *mut u8) as *const u8;
__t184.productName = (b"USB to SATA\0".as_ptr() as *mut u8) as *const u8;
__t184.useProtocol = 255u8;
__t184.useTransport = 255u8;
__t184.initFunction = None;
__t184
}, {
let mut __t185: us_unusual_dev = unsafe { core::mem::zeroed() };
__t185.vendorName = (b"Unknown\0".as_ptr() as *mut u8) as *const u8;
__t185.productName = (b"Unknown\0".as_ptr() as *mut u8) as *const u8;
__t185.useProtocol = 255u8;
__t185.useTransport = 255u8;
__t185.initFunction = None;
__t185
}, {
let mut __t186: us_unusual_dev = unsafe { core::mem::zeroed() };
__t186.vendorName = (b"Medion\0".as_ptr() as *mut u8) as *const u8;
__t186.productName = (b"MD 7425\0".as_ptr() as *mut u8) as *const u8;
__t186.useProtocol = 255u8;
__t186.useTransport = 255u8;
__t186.initFunction = None;
__t186
}, {
let mut __t187: us_unusual_dev = unsafe { core::mem::zeroed() };
__t187.vendorName = (b"Jenoptik\0".as_ptr() as *mut u8) as *const u8;
__t187.productName = (b"JD 5200 z3\0".as_ptr() as *mut u8) as *const u8;
__t187.useProtocol = 255u8;
__t187.useTransport = 255u8;
__t187.initFunction = None;
__t187
}, {
let mut __t188: us_unusual_dev = unsafe { core::mem::zeroed() };
__t188.vendorName = (b"Macpower Technology Co.LTD.\0".as_ptr() as *mut u8) as *const u8;
__t188.productName = (b"USB 2.0 3.5\" DEVICE\0".as_ptr() as *mut u8) as *const u8;
__t188.useProtocol = 255u8;
__t188.useTransport = 255u8;
__t188.initFunction = None;
__t188
}, {
let mut __t189: us_unusual_dev = unsafe { core::mem::zeroed() };
__t189.vendorName = (b"Netac\0".as_ptr() as *mut u8) as *const u8;
__t189.productName = (b"USB-CF-Card\0".as_ptr() as *mut u8) as *const u8;
__t189.useProtocol = 255u8;
__t189.useTransport = 255u8;
__t189.initFunction = None;
__t189
}, {
let mut __t190: us_unusual_dev = unsafe { core::mem::zeroed() };
__t190.vendorName = (b"Netac\0".as_ptr() as *mut u8) as *const u8;
__t190.productName = (b"USB Flash Disk\0".as_ptr() as *mut u8) as *const u8;
__t190.useProtocol = 255u8;
__t190.useTransport = 255u8;
__t190.initFunction = None;
__t190
}, {
let mut __t191: us_unusual_dev = unsafe { core::mem::zeroed() };
__t191.vendorName = (b"WINWARD\0".as_ptr() as *mut u8) as *const u8;
__t191.productName = (b"Music Disk\0".as_ptr() as *mut u8) as *const u8;
__t191.useProtocol = 255u8;
__t191.useTransport = 255u8;
__t191.initFunction = None;
__t191
}, {
let mut __t192: us_unusual_dev = unsafe { core::mem::zeroed() };
__t192.vendorName = (b"PNP_MP3\0".as_ptr() as *mut u8) as *const u8;
__t192.productName = (b"PNP_MP3 PLAYER\0".as_ptr() as *mut u8) as *const u8;
__t192.useProtocol = 255u8;
__t192.useTransport = 255u8;
__t192.initFunction = None;
__t192
}, {
let mut __t193: us_unusual_dev = unsafe { core::mem::zeroed() };
__t193.vendorName = (b"Cowon Systems\0".as_ptr() as *mut u8) as *const u8;
__t193.productName = (b"iAUDIO M5\0".as_ptr() as *mut u8) as *const u8;
__t193.useProtocol = 255u8;
__t193.useTransport = 80u8;
__t193.initFunction = None;
__t193
}, {
let mut __t194: us_unusual_dev = unsafe { core::mem::zeroed() };
__t194.vendorName = (b"USB\0".as_ptr() as *mut u8) as *const u8;
__t194.productName = (b"Solid state disk\0".as_ptr() as *mut u8) as *const u8;
__t194.useProtocol = 255u8;
__t194.useTransport = 255u8;
__t194.initFunction = None;
__t194
}, {
let mut __t195: us_unusual_dev = unsafe { core::mem::zeroed() };
__t195.vendorName = (b"Ours Technology\0".as_ptr() as *mut u8) as *const u8;
__t195.productName = (b"Flash Disk\0".as_ptr() as *mut u8) as *const u8;
__t195.useProtocol = 255u8;
__t195.useTransport = 255u8;
__t195.initFunction = None;
__t195
}, {
let mut __t196: us_unusual_dev = unsafe { core::mem::zeroed() };
__t196.vendorName = (b"USB\0".as_ptr() as *mut u8) as *const u8;
__t196.productName = (b"Flash Disk\0".as_ptr() as *mut u8) as *const u8;
__t196.useProtocol = 255u8;
__t196.useTransport = 255u8;
__t196.initFunction = None;
__t196
}, {
let mut __t197: us_unusual_dev = unsafe { core::mem::zeroed() };
__t197.vendorName = (b"Typhoon\0".as_ptr() as *mut u8) as *const u8;
__t197.productName = (b"My DJ 1820\0".as_ptr() as *mut u8) as *const u8;
__t197.useProtocol = 255u8;
__t197.useTransport = 255u8;
__t197.initFunction = None;
__t197
}, {
let mut __t198: us_unusual_dev = unsafe { core::mem::zeroed() };
__t198.vendorName = (b"Oracom Co., Ltd\0".as_ptr() as *mut u8) as *const u8;
__t198.productName = (b"ORC-200M\0".as_ptr() as *mut u8) as *const u8;
__t198.useProtocol = 255u8;
__t198.useTransport = 255u8;
__t198.initFunction = None;
__t198
}, {
let mut __t199: us_unusual_dev = unsafe { core::mem::zeroed() };
__t199.vendorName = (b"C-MEX\0".as_ptr() as *mut u8) as *const u8;
__t199.productName = (b"A-VOX\0".as_ptr() as *mut u8) as *const u8;
__t199.useProtocol = 255u8;
__t199.useTransport = 255u8;
__t199.initFunction = None;
__t199
}, {
let mut __t200: us_unusual_dev = unsafe { core::mem::zeroed() };
__t200.vendorName = (b"VTech\0".as_ptr() as *mut u8) as *const u8;
__t200.productName = (b"Kidizoom\0".as_ptr() as *mut u8) as *const u8;
__t200.useProtocol = 255u8;
__t200.useTransport = 255u8;
__t200.initFunction = None;
__t200
}, {
let mut __t201: us_unusual_dev = unsafe { core::mem::zeroed() };
__t201.vendorName = (b"Research In Motion\0".as_ptr() as *mut u8) as *const u8;
__t201.productName = (b"BlackBerry Bold 9000\0".as_ptr() as *mut u8) as *const u8;
__t201.useProtocol = 255u8;
__t201.useTransport = 255u8;
__t201.initFunction = None;
__t201
}, {
let mut __t202: us_unusual_dev = unsafe { core::mem::zeroed() };
__t202.vendorName = (b"Sony Ericsson\0".as_ptr() as *mut u8) as *const u8;
__t202.productName = (b"V800-Vodafone 802\0".as_ptr() as *mut u8) as *const u8;
__t202.useProtocol = 255u8;
__t202.useTransport = 255u8;
__t202.initFunction = None;
__t202
}, {
let mut __t203: us_unusual_dev = unsafe { core::mem::zeroed() };
__t203.vendorName = (b"Sony Ericsson\0".as_ptr() as *mut u8) as *const u8;
__t203.productName = (b"MD400\0".as_ptr() as *mut u8) as *const u8;
__t203.useProtocol = 255u8;
__t203.useTransport = 255u8;
__t203.initFunction = None;
__t203
}, {
let mut __t204: us_unusual_dev = unsafe { core::mem::zeroed() };
__t204.vendorName = (b"Sony Ericsson\0".as_ptr() as *mut u8) as *const u8;
__t204.productName = (b"P990i\0".as_ptr() as *mut u8) as *const u8;
__t204.useProtocol = 255u8;
__t204.useTransport = 255u8;
__t204.initFunction = None;
__t204
}, {
let mut __t205: us_unusual_dev = unsafe { core::mem::zeroed() };
__t205.vendorName = (b"Sony Ericsson\0".as_ptr() as *mut u8) as *const u8;
__t205.productName = (b"M600i\0".as_ptr() as *mut u8) as *const u8;
__t205.useProtocol = 255u8;
__t205.useTransport = 255u8;
__t205.initFunction = None;
__t205
}, {
let mut __t206: us_unusual_dev = unsafe { core::mem::zeroed() };
__t206.vendorName = (b"Sony Ericsson\0".as_ptr() as *mut u8) as *const u8;
__t206.productName = (b"P1i\0".as_ptr() as *mut u8) as *const u8;
__t206.useProtocol = 255u8;
__t206.useTransport = 255u8;
__t206.initFunction = None;
__t206
}, {
let mut __t207: us_unusual_dev = unsafe { core::mem::zeroed() };
__t207.vendorName = (b"Desknote\0".as_ptr() as *mut u8) as *const u8;
__t207.productName = (b"UCR-61S2B\0".as_ptr() as *mut u8) as *const u8;
__t207.useProtocol = 255u8;
__t207.useTransport = 255u8;
__t207.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_ucr61s2b_init as *const ()) };
__t207
}, {
let mut __t208: us_unusual_dev = unsafe { core::mem::zeroed() };
__t208.vendorName = (b"Western Digital\0".as_ptr() as *mut u8) as *const u8;
__t208.productName = (b"External HDD\0".as_ptr() as *mut u8) as *const u8;
__t208.useProtocol = 255u8;
__t208.useTransport = 255u8;
__t208.initFunction = None;
__t208
}, {
let mut __t209: us_unusual_dev = unsafe { core::mem::zeroed() };
__t209.vendorName = (b"Western Digital\0".as_ptr() as *mut u8) as *const u8;
__t209.productName = (b"My Passport HDD\0".as_ptr() as *mut u8) as *const u8;
__t209.useProtocol = 255u8;
__t209.useTransport = 255u8;
__t209.initFunction = None;
__t209
}, {
let mut __t210: us_unusual_dev = unsafe { core::mem::zeroed() };
__t210.vendorName = (b"Actions Semiconductor\0".as_ptr() as *mut u8) as *const u8;
__t210.productName = (b"Mtp device\0".as_ptr() as *mut u8) as *const u8;
__t210.useProtocol = 255u8;
__t210.useTransport = 255u8;
__t210.initFunction = None;
__t210
}, {
let mut __t211: us_unusual_dev = unsafe { core::mem::zeroed() };
__t211.vendorName = (b"D-Link\0".as_ptr() as *mut u8) as *const u8;
__t211.productName = (b"USB Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t211.useProtocol = 255u8;
__t211.useTransport = 255u8;
__t211.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(option_ms_init as *const ()) };
__t211
}, {
let mut __t212: us_unusual_dev = unsafe { core::mem::zeroed() };
__t212.vendorName = (b"Sierra Wireless\0".as_ptr() as *mut u8) as *const u8;
__t212.productName = (b"USB MMC Storage\0".as_ptr() as *mut u8) as *const u8;
__t212.useProtocol = 255u8;
__t212.useTransport = 255u8;
__t212.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(sierra_ms_init as *const ()) };
__t212
}, {
let mut __t213: us_unusual_dev = unsafe { core::mem::zeroed() };
__t213.vendorName = (b"Digitech HMG\0".as_ptr() as *mut u8) as *const u8;
__t213.productName = (b"DigiTech Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t213.useProtocol = 255u8;
__t213.useTransport = 255u8;
__t213.initFunction = None;
__t213
}, {
let mut __t214: us_unusual_dev = unsafe { core::mem::zeroed() };
__t214.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t214.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t214.useProtocol = 255u8;
__t214.useTransport = 255u8;
__t214.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t214
}, {
let mut __t215: us_unusual_dev = unsafe { core::mem::zeroed() };
__t215.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t215.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t215.useProtocol = 255u8;
__t215.useTransport = 255u8;
__t215.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t215
}, {
let mut __t216: us_unusual_dev = unsafe { core::mem::zeroed() };
__t216.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t216.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t216.useProtocol = 255u8;
__t216.useTransport = 255u8;
__t216.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t216
}, {
let mut __t217: us_unusual_dev = unsafe { core::mem::zeroed() };
__t217.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t217.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t217.useProtocol = 255u8;
__t217.useTransport = 255u8;
__t217.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t217
}, {
let mut __t218: us_unusual_dev = unsafe { core::mem::zeroed() };
__t218.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t218.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t218.useProtocol = 255u8;
__t218.useTransport = 255u8;
__t218.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t218
}, {
let mut __t219: us_unusual_dev = unsafe { core::mem::zeroed() };
__t219.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t219.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t219.useProtocol = 255u8;
__t219.useTransport = 255u8;
__t219.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t219
}, {
let mut __t220: us_unusual_dev = unsafe { core::mem::zeroed() };
__t220.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t220.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t220.useProtocol = 255u8;
__t220.useTransport = 255u8;
__t220.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t220
}, {
let mut __t221: us_unusual_dev = unsafe { core::mem::zeroed() };
__t221.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t221.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t221.useProtocol = 255u8;
__t221.useTransport = 255u8;
__t221.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t221
}, {
let mut __t222: us_unusual_dev = unsafe { core::mem::zeroed() };
__t222.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t222.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t222.useProtocol = 255u8;
__t222.useTransport = 255u8;
__t222.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t222
}, {
let mut __t223: us_unusual_dev = unsafe { core::mem::zeroed() };
__t223.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t223.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t223.useProtocol = 255u8;
__t223.useTransport = 255u8;
__t223.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t223
}, {
let mut __t224: us_unusual_dev = unsafe { core::mem::zeroed() };
__t224.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t224.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t224.useProtocol = 255u8;
__t224.useTransport = 255u8;
__t224.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t224
}, {
let mut __t225: us_unusual_dev = unsafe { core::mem::zeroed() };
__t225.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t225.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t225.useProtocol = 255u8;
__t225.useTransport = 255u8;
__t225.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t225
}, {
let mut __t226: us_unusual_dev = unsafe { core::mem::zeroed() };
__t226.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t226.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t226.useProtocol = 255u8;
__t226.useTransport = 255u8;
__t226.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t226
}, {
let mut __t227: us_unusual_dev = unsafe { core::mem::zeroed() };
__t227.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t227.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t227.useProtocol = 255u8;
__t227.useTransport = 255u8;
__t227.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t227
}, {
let mut __t228: us_unusual_dev = unsafe { core::mem::zeroed() };
__t228.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t228.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t228.useProtocol = 255u8;
__t228.useTransport = 255u8;
__t228.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t228
}, {
let mut __t229: us_unusual_dev = unsafe { core::mem::zeroed() };
__t229.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t229.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t229.useProtocol = 255u8;
__t229.useTransport = 255u8;
__t229.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t229
}, {
let mut __t230: us_unusual_dev = unsafe { core::mem::zeroed() };
__t230.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t230.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t230.useProtocol = 255u8;
__t230.useTransport = 255u8;
__t230.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t230
}, {
let mut __t231: us_unusual_dev = unsafe { core::mem::zeroed() };
__t231.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t231.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t231.useProtocol = 255u8;
__t231.useTransport = 255u8;
__t231.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t231
}, {
let mut __t232: us_unusual_dev = unsafe { core::mem::zeroed() };
__t232.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t232.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t232.useProtocol = 255u8;
__t232.useTransport = 255u8;
__t232.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t232
}, {
let mut __t233: us_unusual_dev = unsafe { core::mem::zeroed() };
__t233.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t233.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t233.useProtocol = 255u8;
__t233.useTransport = 255u8;
__t233.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t233
}, {
let mut __t234: us_unusual_dev = unsafe { core::mem::zeroed() };
__t234.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t234.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t234.useProtocol = 255u8;
__t234.useTransport = 255u8;
__t234.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t234
}, {
let mut __t235: us_unusual_dev = unsafe { core::mem::zeroed() };
__t235.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t235.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t235.useProtocol = 255u8;
__t235.useTransport = 255u8;
__t235.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t235
}, {
let mut __t236: us_unusual_dev = unsafe { core::mem::zeroed() };
__t236.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t236.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t236.useProtocol = 255u8;
__t236.useTransport = 255u8;
__t236.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t236
}, {
let mut __t237: us_unusual_dev = unsafe { core::mem::zeroed() };
__t237.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t237.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t237.useProtocol = 255u8;
__t237.useTransport = 255u8;
__t237.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t237
}, {
let mut __t238: us_unusual_dev = unsafe { core::mem::zeroed() };
__t238.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t238.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t238.useProtocol = 255u8;
__t238.useTransport = 255u8;
__t238.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t238
}, {
let mut __t239: us_unusual_dev = unsafe { core::mem::zeroed() };
__t239.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t239.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t239.useProtocol = 255u8;
__t239.useTransport = 255u8;
__t239.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t239
}, {
let mut __t240: us_unusual_dev = unsafe { core::mem::zeroed() };
__t240.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t240.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t240.useProtocol = 255u8;
__t240.useTransport = 255u8;
__t240.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t240
}, {
let mut __t241: us_unusual_dev = unsafe { core::mem::zeroed() };
__t241.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t241.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t241.useProtocol = 255u8;
__t241.useTransport = 255u8;
__t241.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t241
}, {
let mut __t242: us_unusual_dev = unsafe { core::mem::zeroed() };
__t242.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t242.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t242.useProtocol = 255u8;
__t242.useTransport = 255u8;
__t242.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t242
}, {
let mut __t243: us_unusual_dev = unsafe { core::mem::zeroed() };
__t243.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t243.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t243.useProtocol = 255u8;
__t243.useTransport = 255u8;
__t243.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t243
}, {
let mut __t244: us_unusual_dev = unsafe { core::mem::zeroed() };
__t244.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t244.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t244.useProtocol = 255u8;
__t244.useTransport = 255u8;
__t244.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t244
}, {
let mut __t245: us_unusual_dev = unsafe { core::mem::zeroed() };
__t245.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t245.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t245.useProtocol = 255u8;
__t245.useTransport = 255u8;
__t245.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t245
}, {
let mut __t246: us_unusual_dev = unsafe { core::mem::zeroed() };
__t246.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t246.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t246.useProtocol = 255u8;
__t246.useTransport = 255u8;
__t246.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t246
}, {
let mut __t247: us_unusual_dev = unsafe { core::mem::zeroed() };
__t247.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t247.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t247.useProtocol = 255u8;
__t247.useTransport = 255u8;
__t247.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t247
}, {
let mut __t248: us_unusual_dev = unsafe { core::mem::zeroed() };
__t248.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t248.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t248.useProtocol = 255u8;
__t248.useTransport = 255u8;
__t248.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t248
}, {
let mut __t249: us_unusual_dev = unsafe { core::mem::zeroed() };
__t249.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t249.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t249.useProtocol = 255u8;
__t249.useTransport = 255u8;
__t249.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t249
}, {
let mut __t250: us_unusual_dev = unsafe { core::mem::zeroed() };
__t250.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t250.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t250.useProtocol = 255u8;
__t250.useTransport = 255u8;
__t250.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t250
}, {
let mut __t251: us_unusual_dev = unsafe { core::mem::zeroed() };
__t251.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t251.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t251.useProtocol = 255u8;
__t251.useTransport = 255u8;
__t251.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t251
}, {
let mut __t252: us_unusual_dev = unsafe { core::mem::zeroed() };
__t252.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t252.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t252.useProtocol = 255u8;
__t252.useTransport = 255u8;
__t252.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t252
}, {
let mut __t253: us_unusual_dev = unsafe { core::mem::zeroed() };
__t253.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t253.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t253.useProtocol = 255u8;
__t253.useTransport = 255u8;
__t253.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t253
}, {
let mut __t254: us_unusual_dev = unsafe { core::mem::zeroed() };
__t254.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t254.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t254.useProtocol = 255u8;
__t254.useTransport = 255u8;
__t254.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t254
}, {
let mut __t255: us_unusual_dev = unsafe { core::mem::zeroed() };
__t255.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t255.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t255.useProtocol = 255u8;
__t255.useTransport = 255u8;
__t255.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t255
}, {
let mut __t256: us_unusual_dev = unsafe { core::mem::zeroed() };
__t256.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t256.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t256.useProtocol = 255u8;
__t256.useTransport = 255u8;
__t256.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t256
}, {
let mut __t257: us_unusual_dev = unsafe { core::mem::zeroed() };
__t257.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t257.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t257.useProtocol = 255u8;
__t257.useTransport = 255u8;
__t257.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t257
}, {
let mut __t258: us_unusual_dev = unsafe { core::mem::zeroed() };
__t258.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t258.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t258.useProtocol = 255u8;
__t258.useTransport = 255u8;
__t258.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t258
}, {
let mut __t259: us_unusual_dev = unsafe { core::mem::zeroed() };
__t259.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t259.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t259.useProtocol = 255u8;
__t259.useTransport = 255u8;
__t259.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t259
}, {
let mut __t260: us_unusual_dev = unsafe { core::mem::zeroed() };
__t260.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t260.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t260.useProtocol = 255u8;
__t260.useTransport = 255u8;
__t260.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t260
}, {
let mut __t261: us_unusual_dev = unsafe { core::mem::zeroed() };
__t261.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t261.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t261.useProtocol = 255u8;
__t261.useTransport = 255u8;
__t261.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t261
}, {
let mut __t262: us_unusual_dev = unsafe { core::mem::zeroed() };
__t262.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t262.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t262.useProtocol = 255u8;
__t262.useTransport = 255u8;
__t262.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t262
}, {
let mut __t263: us_unusual_dev = unsafe { core::mem::zeroed() };
__t263.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t263.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t263.useProtocol = 255u8;
__t263.useTransport = 255u8;
__t263.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t263
}, {
let mut __t264: us_unusual_dev = unsafe { core::mem::zeroed() };
__t264.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t264.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t264.useProtocol = 255u8;
__t264.useTransport = 255u8;
__t264.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t264
}, {
let mut __t265: us_unusual_dev = unsafe { core::mem::zeroed() };
__t265.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t265.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t265.useProtocol = 255u8;
__t265.useTransport = 255u8;
__t265.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t265
}, {
let mut __t266: us_unusual_dev = unsafe { core::mem::zeroed() };
__t266.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t266.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t266.useProtocol = 255u8;
__t266.useTransport = 255u8;
__t266.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t266
}, {
let mut __t267: us_unusual_dev = unsafe { core::mem::zeroed() };
__t267.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t267.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t267.useProtocol = 255u8;
__t267.useTransport = 255u8;
__t267.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t267
}, {
let mut __t268: us_unusual_dev = unsafe { core::mem::zeroed() };
__t268.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t268.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t268.useProtocol = 255u8;
__t268.useTransport = 255u8;
__t268.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t268
}, {
let mut __t269: us_unusual_dev = unsafe { core::mem::zeroed() };
__t269.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t269.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t269.useProtocol = 255u8;
__t269.useTransport = 255u8;
__t269.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t269
}, {
let mut __t270: us_unusual_dev = unsafe { core::mem::zeroed() };
__t270.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t270.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t270.useProtocol = 255u8;
__t270.useTransport = 255u8;
__t270.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t270
}, {
let mut __t271: us_unusual_dev = unsafe { core::mem::zeroed() };
__t271.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t271.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t271.useProtocol = 255u8;
__t271.useTransport = 255u8;
__t271.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t271
}, {
let mut __t272: us_unusual_dev = unsafe { core::mem::zeroed() };
__t272.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t272.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t272.useProtocol = 255u8;
__t272.useTransport = 255u8;
__t272.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t272
}, {
let mut __t273: us_unusual_dev = unsafe { core::mem::zeroed() };
__t273.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t273.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t273.useProtocol = 255u8;
__t273.useTransport = 255u8;
__t273.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t273
}, {
let mut __t274: us_unusual_dev = unsafe { core::mem::zeroed() };
__t274.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t274.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t274.useProtocol = 255u8;
__t274.useTransport = 255u8;
__t274.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t274
}, {
let mut __t275: us_unusual_dev = unsafe { core::mem::zeroed() };
__t275.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t275.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t275.useProtocol = 255u8;
__t275.useTransport = 255u8;
__t275.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t275
}, {
let mut __t276: us_unusual_dev = unsafe { core::mem::zeroed() };
__t276.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t276.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t276.useProtocol = 255u8;
__t276.useTransport = 255u8;
__t276.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t276
}, {
let mut __t277: us_unusual_dev = unsafe { core::mem::zeroed() };
__t277.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t277.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t277.useProtocol = 255u8;
__t277.useTransport = 255u8;
__t277.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t277
}, {
let mut __t278: us_unusual_dev = unsafe { core::mem::zeroed() };
__t278.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t278.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t278.useProtocol = 255u8;
__t278.useTransport = 255u8;
__t278.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t278
}, {
let mut __t279: us_unusual_dev = unsafe { core::mem::zeroed() };
__t279.vendorName = (b"HUAWEI MOBILE\0".as_ptr() as *mut u8) as *const u8;
__t279.productName = (b"Mass Storage\0".as_ptr() as *mut u8) as *const u8;
__t279.useProtocol = 255u8;
__t279.useTransport = 255u8;
__t279.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_huawei_e220_init as *const ()) };
__t279
}, {
let mut __t280: us_unusual_dev = unsafe { core::mem::zeroed() };
__t280.vendorName = (b"Minolta\0".as_ptr() as *mut u8) as *const u8;
__t280.productName = (b"Dimage Z10\0".as_ptr() as *mut u8) as *const u8;
__t280.useProtocol = 255u8;
__t280.useTransport = 255u8;
__t280.initFunction = None;
__t280
}, {
let mut __t281: us_unusual_dev = unsafe { core::mem::zeroed() };
__t281.vendorName = (b"SWISSBIT\0".as_ptr() as *mut u8) as *const u8;
__t281.productName = (b"Black Silver\0".as_ptr() as *mut u8) as *const u8;
__t281.useProtocol = 255u8;
__t281.useTransport = 255u8;
__t281.initFunction = None;
__t281
}, {
let mut __t282: us_unusual_dev = unsafe { core::mem::zeroed() };
__t282.vendorName = (b"Initio Corporation\0".as_ptr() as *mut u8) as *const u8;
__t282.productName = (b"INIC-3619\0".as_ptr() as *mut u8) as *const u8;
__t282.useProtocol = 255u8;
__t282.useTransport = 255u8;
__t282.initFunction = None;
__t282
}, {
let mut __t283: us_unusual_dev = unsafe { core::mem::zeroed() };
__t283.vendorName = (b"Kingston\0".as_ptr() as *mut u8) as *const u8;
__t283.productName = (b"DT 101 G2\0".as_ptr() as *mut u8) as *const u8;
__t283.useProtocol = 255u8;
__t283.useTransport = 255u8;
__t283.initFunction = None;
__t283
}, {
let mut __t284: us_unusual_dev = unsafe { core::mem::zeroed() };
__t284.vendorName = (b"Super Top\0".as_ptr() as *mut u8) as *const u8;
__t284.productName = (b"IDE DEVICE\0".as_ptr() as *mut u8) as *const u8;
__t284.useProtocol = 255u8;
__t284.useTransport = 255u8;
__t284.initFunction = None;
__t284
}, {
let mut __t285: us_unusual_dev = unsafe { core::mem::zeroed() };
__t285.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t285.productName = (b"USB to ATA/ATAPI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t285.useProtocol = 255u8;
__t285.useTransport = 255u8;
__t285.initFunction = None;
__t285
}, {
let mut __t286: us_unusual_dev = unsafe { core::mem::zeroed() };
__t286.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t286.productName = (b"JMS567\0".as_ptr() as *mut u8) as *const u8;
__t286.useProtocol = 255u8;
__t286.useTransport = 255u8;
__t286.initFunction = None;
__t286
}, {
let mut __t287: us_unusual_dev = unsafe { core::mem::zeroed() };
__t287.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t287.productName = (b"USB to ATA/ATAPI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t287.useProtocol = 255u8;
__t287.useTransport = 255u8;
__t287.initFunction = None;
__t287
}, {
let mut __t288: us_unusual_dev = unsafe { core::mem::zeroed() };
__t288.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t288.productName = (b"USB to ATA/ATAPI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t288.useProtocol = 255u8;
__t288.useTransport = 255u8;
__t288.initFunction = None;
__t288
}, {
let mut __t289: us_unusual_dev = unsafe { core::mem::zeroed() };
__t289.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t289.productName = (b"USB to ATA/ATAPI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t289.useProtocol = 255u8;
__t289.useTransport = 255u8;
__t289.initFunction = None;
__t289
}, {
let mut __t290: us_unusual_dev = unsafe { core::mem::zeroed() };
__t290.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t290.productName = (b"JMS56x\0".as_ptr() as *mut u8) as *const u8;
__t290.useProtocol = 255u8;
__t290.useTransport = 255u8;
__t290.initFunction = None;
__t290
}, {
let mut __t291: us_unusual_dev = unsafe { core::mem::zeroed() };
__t291.vendorName = (b"Entrega Technologies\0".as_ptr() as *mut u8) as *const u8;
__t291.productName = (b"USB to SCSI Converter\0".as_ptr() as *mut u8) as *const u8;
__t291.useProtocol = 255u8;
__t291.useTransport = 255u8;
__t291.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t291
}, {
let mut __t292: us_unusual_dev = unsafe { core::mem::zeroed() };
__t292.vendorName = (b"Teac\0".as_ptr() as *mut u8) as *const u8;
__t292.productName = (b"HD-35PUK-B\0".as_ptr() as *mut u8) as *const u8;
__t292.useProtocol = 255u8;
__t292.useTransport = 255u8;
__t292.initFunction = None;
__t292
}, {
let mut __t293: us_unusual_dev = unsafe { core::mem::zeroed() };
__t293.vendorName = (b"ASMedia\0".as_ptr() as *mut u8) as *const u8;
__t293.productName = (b"AS2105\0".as_ptr() as *mut u8) as *const u8;
__t293.useProtocol = 255u8;
__t293.useTransport = 255u8;
__t293.initFunction = None;
__t293
}, {
let mut __t294: us_unusual_dev = unsafe { core::mem::zeroed() };
__t294.vendorName = (b"Yarvik\0".as_ptr() as *mut u8) as *const u8;
__t294.productName = (b"PMP400\0".as_ptr() as *mut u8) as *const u8;
__t294.useProtocol = 255u8;
__t294.useTransport = 255u8;
__t294.initFunction = None;
__t294
}, {
let mut __t295: us_unusual_dev = unsafe { core::mem::zeroed() };
__t295.vendorName = (b"Ariston Technologies\0".as_ptr() as *mut u8) as *const u8;
__t295.productName = (b"iConnect USB to SCSI adapter\0".as_ptr() as *mut u8) as *const u8;
__t295.useProtocol = 255u8;
__t295.useTransport = 255u8;
__t295.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t295
}, {
let mut __t296: us_unusual_dev = unsafe { core::mem::zeroed() };
__t296.vendorName = (b"BUILDWIN\0".as_ptr() as *mut u8) as *const u8;
__t296.productName = (b"Photo Frame\0".as_ptr() as *mut u8) as *const u8;
__t296.useProtocol = 255u8;
__t296.useTransport = 255u8;
__t296.initFunction = None;
__t296
}, {
let mut __t297: us_unusual_dev = unsafe { core::mem::zeroed() };
__t297.vendorName = (b"BUILDWIN\0".as_ptr() as *mut u8) as *const u8;
__t297.productName = (b"Photo Frame\0".as_ptr() as *mut u8) as *const u8;
__t297.useProtocol = 255u8;
__t297.useTransport = 255u8;
__t297.initFunction = None;
__t297
}, {
let mut __t298: us_unusual_dev = unsafe { core::mem::zeroed() };
__t298.vendorName = (b"BUILDWIN\0".as_ptr() as *mut u8) as *const u8;
__t298.productName = (b"Photo Frame\0".as_ptr() as *mut u8) as *const u8;
__t298.useProtocol = 255u8;
__t298.useTransport = 255u8;
__t298.initFunction = None;
__t298
}, {
let mut __t299: us_unusual_dev = unsafe { core::mem::zeroed() };
__t299.vendorName = (b"Amazon\0".as_ptr() as *mut u8) as *const u8;
__t299.productName = (b"Kindle\0".as_ptr() as *mut u8) as *const u8;
__t299.useProtocol = 255u8;
__t299.useTransport = 255u8;
__t299.initFunction = None;
__t299
}, {
let mut __t300: us_unusual_dev = unsafe { core::mem::zeroed() };
__t300.vendorName = (b"ZTE,Incorporated\0".as_ptr() as *mut u8) as *const u8;
__t300.productName = (b"ZTE WCDMA Technologies MSM\0".as_ptr() as *mut u8) as *const u8;
__t300.useProtocol = 255u8;
__t300.useTransport = 255u8;
__t300.initFunction = None;
__t300
}, {
let mut __t301: us_unusual_dev = unsafe { core::mem::zeroed() };
__t301.vendorName = (b"Corsair\0".as_ptr() as *mut u8) as *const u8;
__t301.productName = (b"Padlock v2\0".as_ptr() as *mut u8) as *const u8;
__t301.useProtocol = 255u8;
__t301.useTransport = 255u8;
__t301.initFunction = None;
__t301
}, {
let mut __t302: us_unusual_dev = unsafe { core::mem::zeroed() };
__t302.vendorName = (b"Grain-media Technology Corp.\0".as_ptr() as *mut u8) as *const u8;
__t302.productName = (b"USB3.0 Device GM12U320\0".as_ptr() as *mut u8) as *const u8;
__t302.useProtocol = 255u8;
__t302.useTransport = 255u8;
__t302.initFunction = None;
__t302
}, {
let mut __t303: us_unusual_dev = unsafe { core::mem::zeroed() };
__t303.vendorName = (b"TrekStor GmbH & Co. KG\0".as_ptr() as *mut u8) as *const u8;
__t303.productName = (b"DataStation maxi g.u\0".as_ptr() as *mut u8) as *const u8;
__t303.useProtocol = 255u8;
__t303.useTransport = 255u8;
__t303.initFunction = None;
__t303
}, {
let mut __t304: us_unusual_dev = unsafe { core::mem::zeroed() };
__t304.vendorName = (b"Coby Electronics\0".as_ptr() as *mut u8) as *const u8;
__t304.productName = (b"MP3 Player\0".as_ptr() as *mut u8) as *const u8;
__t304.useProtocol = 255u8;
__t304.useTransport = 255u8;
__t304.initFunction = None;
__t304
}, {
let mut __t305: us_unusual_dev = unsafe { core::mem::zeroed() };
__t305.vendorName = (b"NXP Semiconductors\0".as_ptr() as *mut u8) as *const u8;
__t305.productName = (b"PN7462AU\0".as_ptr() as *mut u8) as *const u8;
__t305.useProtocol = 255u8;
__t305.useTransport = 255u8;
__t305.initFunction = None;
__t305
}, {
let mut __t306: us_unusual_dev = unsafe { core::mem::zeroed() };
__t306.vendorName = (b"Double-H Technology\0".as_ptr() as *mut u8) as *const u8;
__t306.productName = (b"USB to SCSI Intelligent Cable\0".as_ptr() as *mut u8) as *const u8;
__t306.useProtocol = 255u8;
__t306.useTransport = 255u8;
__t306.initFunction = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_euscsi_init as *const ()) };
__t306
}, {
let mut __t307: us_unusual_dev = unsafe { core::mem::zeroed() };
__t307.vendorName = (b"VIA Labs, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t307.productName = (b"VL817 SATA Bridge\0".as_ptr() as *mut u8) as *const u8;
__t307.useProtocol = 255u8;
__t307.useTransport = 255u8;
__t307.initFunction = None;
__t307
}, {
let mut __t308: us_unusual_dev = unsafe { core::mem::zeroed() };
__t308.vendorName = (b"ST\0".as_ptr() as *mut u8) as *const u8;
__t308.productName = (b"2A\0".as_ptr() as *mut u8) as *const u8;
__t308.useProtocol = 255u8;
__t308.useTransport = 255u8;
__t308.initFunction = None;
__t308
}, {
let mut __t309: us_unusual_dev = unsafe { core::mem::zeroed() };
__t309.vendorName = (b"Motorola\0".as_ptr() as *mut u8) as *const u8;
__t309.productName = (b"RAZR V3x\0".as_ptr() as *mut u8) as *const u8;
__t309.useProtocol = 255u8;
__t309.useTransport = 255u8;
__t309.initFunction = None;
__t309
}, {
let mut __t310: us_unusual_dev = unsafe { core::mem::zeroed() };
__t310.vendorName = (b"Motorola\0".as_ptr() as *mut u8) as *const u8;
__t310.productName = (b"MSnc.\0".as_ptr() as *mut u8) as *const u8;
__t310.useProtocol = 255u8;
__t310.useTransport = 255u8;
__t310.initFunction = None;
__t310
}, {
let mut __t311: us_unusual_dev = unsafe { core::mem::zeroed() };
__t311.vendorName = (b"MPIO\0".as_ptr() as *mut u8) as *const u8;
__t311.productName = (b"HS200\0".as_ptr() as *mut u8) as *const u8;
__t311.useProtocol = 255u8;
__t311.useTransport = 255u8;
__t311.initFunction = None;
__t311
}, {
let mut __t312: us_unusual_dev = unsafe { core::mem::zeroed() };
__t312.vendorName = (b"DJI\0".as_ptr() as *mut u8) as *const u8;
__t312.productName = (b"CineSSD\0".as_ptr() as *mut u8) as *const u8;
__t312.useProtocol = 255u8;
__t312.useTransport = 255u8;
__t312.initFunction = None;
__t312
}, {
let mut __t313: us_unusual_dev = unsafe { core::mem::zeroed() };
__t313.vendorName = (b"Mitac\0".as_ptr() as *mut u8) as *const u8;
__t313.productName = (b"Mio DigiWalker USB Sync\0".as_ptr() as *mut u8) as *const u8;
__t313.useProtocol = 255u8;
__t313.useTransport = 255u8;
__t313.initFunction = None;
__t313
}, {
let mut __t314: us_unusual_dev = unsafe { core::mem::zeroed() };
__t314.vendorName = (b"JMicron\0".as_ptr() as *mut u8) as *const u8;
__t314.productName = (b"USB to ATA/ATAPI Bridge\0".as_ptr() as *mut u8) as *const u8;
__t314.useProtocol = 255u8;
__t314.useTransport = 255u8;
__t314.initFunction = None;
__t314
}, {
let mut __t315: us_unusual_dev = unsafe { core::mem::zeroed() };
__t315.vendorName = (b"iRiver\0".as_ptr() as *mut u8) as *const u8;
__t315.productName = (b"MP3 T10\0".as_ptr() as *mut u8) as *const u8;
__t315.useProtocol = 255u8;
__t315.useTransport = 255u8;
__t315.initFunction = None;
__t315
}, {
let mut __t316: us_unusual_dev = unsafe { core::mem::zeroed() };
__t316.vendorName = (b"iRiver\0".as_ptr() as *mut u8) as *const u8;
__t316.productName = (b"P7K\0".as_ptr() as *mut u8) as *const u8;
__t316.useProtocol = 255u8;
__t316.useTransport = 255u8;
__t316.initFunction = None;
__t316
}, {
let mut __t317: us_unusual_dev = unsafe { core::mem::zeroed() };
__t317.vendorName = (b"Iomega\0".as_ptr() as *mut u8) as *const u8;
__t317.productName = (b"Micro Mini 1GB\0".as_ptr() as *mut u8) as *const u8;
__t317.useProtocol = 255u8;
__t317.useTransport = 255u8;
__t317.initFunction = None;
__t317
}, {
let mut __t318: us_unusual_dev = unsafe { core::mem::zeroed() };
__t318.vendorName = (b"SimpleTech\0".as_ptr() as *mut u8) as *const u8;
__t318.productName = (b"External HDD\0".as_ptr() as *mut u8) as *const u8;
__t318.useProtocol = 255u8;
__t318.useTransport = 255u8;
__t318.initFunction = None;
__t318
}, {
let mut __t319: us_unusual_dev = unsafe { core::mem::zeroed() };
__t319.vendorName = (b"Keil Software, Inc.\0".as_ptr() as *mut u8) as *const u8;
__t319.productName = (b"V2M MotherBoard\0".as_ptr() as *mut u8) as *const u8;
__t319.useProtocol = 255u8;
__t319.useTransport = 255u8;
__t319.initFunction = None;
__t319
}, {
let mut __t320: us_unusual_dev = unsafe { core::mem::zeroed() };
__t320.vendorName = (b"ZhuHai JieLi Technology\0".as_ptr() as *mut u8) as *const u8;
__t320.productName = (b"JieLi BR21\0".as_ptr() as *mut u8) as *const u8;
__t320.useProtocol = 255u8;
__t320.useTransport = 255u8;
__t320.initFunction = None;
__t320
}, {
let mut __t321: us_unusual_dev = unsafe { core::mem::zeroed() };
__t321.vendorName = (b"DataStor\0".as_ptr() as *mut u8) as *const u8;
__t321.productName = (b"USB4500 FW1.04\0".as_ptr() as *mut u8) as *const u8;
__t321.useProtocol = 255u8;
__t321.useTransport = 255u8;
__t321.initFunction = None;
__t321
}, {
let mut __t322: us_unusual_dev = unsafe { core::mem::zeroed() };
__t322.vendorName = (b"TGE\0".as_ptr() as *mut u8) as *const u8;
__t322.productName = (b"Digital MP3 Audio Player\0".as_ptr() as *mut u8) as *const u8;
__t322.useProtocol = 255u8;
__t322.useTransport = 255u8;
__t322.initFunction = None;
__t322
}, {
let mut __t323: us_unusual_dev = unsafe { core::mem::zeroed() };
__t323.useProtocol = 1u8;
__t323.useTransport = 1u8;
__t323
}, {
let mut __t324: us_unusual_dev = unsafe { core::mem::zeroed() };
__t324.useProtocol = 2u8;
__t324.useTransport = 1u8;
__t324
}, {
let mut __t325: us_unusual_dev = unsafe { core::mem::zeroed() };
__t325.useProtocol = 3u8;
__t325.useTransport = 1u8;
__t325
}, {
let mut __t326: us_unusual_dev = unsafe { core::mem::zeroed() };
__t326.useProtocol = 4u8;
__t326.useTransport = 1u8;
__t326
}, {
let mut __t327: us_unusual_dev = unsafe { core::mem::zeroed() };
__t327.useProtocol = 5u8;
__t327.useTransport = 1u8;
__t327
}, {
let mut __t328: us_unusual_dev = unsafe { core::mem::zeroed() };
__t328.useProtocol = 6u8;
__t328.useTransport = 1u8;
__t328
}, {
let mut __t329: us_unusual_dev = unsafe { core::mem::zeroed() };
__t329.useProtocol = 1u8;
__t329.useTransport = 0u8;
__t329
}, {
let mut __t330: us_unusual_dev = unsafe { core::mem::zeroed() };
__t330.useProtocol = 2u8;
__t330.useTransport = 0u8;
__t330
}, {
let mut __t331: us_unusual_dev = unsafe { core::mem::zeroed() };
__t331.useProtocol = 3u8;
__t331.useTransport = 0u8;
__t331
}, {
let mut __t332: us_unusual_dev = unsafe { core::mem::zeroed() };
__t332.useProtocol = 4u8;
__t332.useTransport = 0u8;
__t332
}, {
let mut __t333: us_unusual_dev = unsafe { core::mem::zeroed() };
__t333.useProtocol = 5u8;
__t333.useTransport = 0u8;
__t333
}, {
let mut __t334: us_unusual_dev = unsafe { core::mem::zeroed() };
__t334.useProtocol = 6u8;
__t334.useTransport = 0u8;
__t334
}, {
let mut __t335: us_unusual_dev = unsafe { core::mem::zeroed() };
__t335.useProtocol = 1u8;
__t335.useTransport = 80u8;
__t335
}, {
let mut __t336: us_unusual_dev = unsafe { core::mem::zeroed() };
__t336.useProtocol = 2u8;
__t336.useTransport = 80u8;
__t336
}, {
let mut __t337: us_unusual_dev = unsafe { core::mem::zeroed() };
__t337.useProtocol = 3u8;
__t337.useTransport = 80u8;
__t337
}, {
let mut __t338: us_unusual_dev = unsafe { core::mem::zeroed() };
__t338.useProtocol = 4u8;
__t338.useTransport = 80u8;
__t338
}, {
let mut __t339: us_unusual_dev = unsafe { core::mem::zeroed() };
__t339.useProtocol = 5u8;
__t339.useTransport = 80u8;
__t339
}, {
let mut __t340: us_unusual_dev = unsafe { core::mem::zeroed() };
__t340.useProtocol = 6u8;
__t340.useTransport = 80u8;
__t340
}, {
let mut __t341: us_unusual_dev = unsafe { core::mem::zeroed() };
__t341
}]);

static for_dynamic_ids: __SlateConst<us_unusual_dev> = __SlateConst({
let mut __t0: us_unusual_dev = unsafe { core::mem::zeroed() };
__t0.useProtocol = 6u8;
__t0.useTransport = 80u8;
__t0
});

fn us_set_lock_class(mutex: *mut mutex, intf: *mut usb_interface) {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_suspend")]
extern "C" fn usb_stor_suspend(iface: *mut usb_interface, message: pm_message) -> i32 {
let us: *mut us_data = usb_get_intfdata(iface) as *mut us_data;
// Wait until no command is running
unsafe { mutex_lock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
if (unsafe { (*us).suspend_resume_hook }) != None {
unsafe { unsafe { (*us).suspend_resume_hook }.unwrap()(us, 0i32) };
}
// When runtime PM is working, we'll set a flag to indicate
// whether we should autoresume when a SCSI request arrives.
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_suspend_605: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface, pm_message) -> i32>, *mut ()>(Some(usb_stor_suspend)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_resume")]
extern "C" fn usb_stor_resume(iface: *mut usb_interface) -> i32 {
let us: *mut us_data = usb_get_intfdata(iface) as *mut us_data;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
if (unsafe { (*us).suspend_resume_hook }) != None {
unsafe { unsafe { (*us).suspend_resume_hook }.unwrap()(us, 1i32) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_resume_606: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface) -> i32>, *mut ()>(Some(usb_stor_resume)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_reset_resume")]
extern "C" fn usb_stor_reset_resume(iface: *mut usb_interface) -> i32 {
let us: *mut us_data = usb_get_intfdata(iface) as *mut us_data;
// Report the reset to the SCSI core
unsafe { usb_stor_report_bus_reset(us) };
// If any of the subdrivers implemented a reinitialization scheme,
// this is where the callback would be invoked.
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_reset_resume_607: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface) -> i32>, *mut ()>(Some(usb_stor_reset_resume)) };

// The next two routines get called just before and just after
// a USB port reset, whether from this driver or a different one.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_pre_reset")]
extern "C" fn usb_stor_pre_reset(iface: *mut usb_interface) -> i32 {
let us: *mut us_data = usb_get_intfdata(iface) as *mut us_data;
// Make sure no command runs during the reset
unsafe { mutex_lock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_pre_reset_608: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface) -> i32>, *mut ()>(Some(usb_stor_pre_reset)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_post_reset")]
extern "C" fn usb_stor_post_reset(iface: *mut usb_interface) -> i32 {
let us: *mut us_data = usb_get_intfdata(iface) as *mut us_data;
// Report the reset to the SCSI core
unsafe { usb_stor_report_bus_reset(us) };
// If any of the subdrivers implemented a reinitialization scheme,
// this is where the callback would be invoked.
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_post_reset_609: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface) -> i32>, *mut ()>(Some(usb_stor_post_reset)) };

// fill_inquiry_response takes an unsigned char array (which must
// be at least 36 characters) and populates the vendor name,
// product name, and revision fields. Then the array is copied
// into the SCSI command's response buffer (oddly enough
// called request_buffer). data_len contains the length of the
// data array, which again must be at least 36.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fill_inquiry_response_616: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut us_data, *mut u8, u32)>, *mut ()>(Some(fill_inquiry_response)) };

#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_control_thread")]
extern "C" fn usb_stor_control_thread(__us: *mut ()) -> i32 {
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut data_ptr: core::mem::MaybeUninit<__SlateAlign16<[u8; 36]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut [u8; 36] = core::ptr::addr_of_mut!(data_ptr) as *mut [u8; 36];
let mut srb: core::mem::MaybeUninit<*mut scsi_cmnd> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut scsi_cmnd = core::ptr::addr_of_mut!(srb) as *mut *mut scsi_cmnd;
let mut host: *mut Scsi_Host = unsafe { core::mem::zeroed() };
let mut us: *mut us_data = unsafe { core::mem::zeroed() };
unsafe {
us = __us as *mut us_data;
host = us_to_host(us);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
'__join_12: {
loop {
if true {
if wait_for_completion_interruptible(core::ptr::addr_of_mut!((*us).cmnd_ready)) != 0 {
break '__join_12;
} else {
while false {
}
// lock the device pointers
mutex_lock(core::ptr::addr_of_mut!((*us).dev_mutex));
// lock access to the state
scsi_lock!(host);
// When we are called with no command pending, we're done
*__slate_slot_5 = (*us).srb;
if (*__slate_slot_5).is_null() {
break;
} else {
'__join_17: {
// has the command timed out *already* ?
if test_bit__2!(5i32, core::ptr::addr_of_mut!((*us).dflags)) != 0 {
(*(*__slate_slot_5)).result = 327680i32;
} else {
scsi_unlock!(host);
// reject the command if the direction indicator
// is UNKNOWN
if ((*(*__slate_slot_5)).sc_data_direction as u32) == 0 {
while false {
}
(*(*__slate_slot_5)).result = 458752i32;
} else {
if (*(*(*__slate_slot_5)).device).id != 0 && (*us).fflags & 4 == 0 {
while false {
}
(*(*__slate_slot_5)).result = 262144i32;
} else {
if (*(*(*__slate_slot_5)).device).lun > (((*us).max_lun as i32) as u64) {
while false {
}
(*(*__slate_slot_5)).result = 262144i32;
} else {
if (*((*(*__slate_slot_5)).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) == 18 && (*us).fflags & 8 != 0 {
core::ptr::write(__slate_slot_4, {
let mut __t1: [u8; 36] = core::mem::zeroed();
__t1[0] = 0u8;
__t1[1] = 128u8;
__t1[2] = 2u8;
__t1[3] = 2u8;
__t1[4] = 31u8;
__t1[5] = 0u8;
__t1[6] = 0u8;
__t1[7] = 0u8;
__t1
});
while false {
}
fill_inquiry_response(us, (*__slate_slot_4).as_mut_ptr() as *mut u8, 36u32);
(*(*__slate_slot_5)).result = 0;
} else {
unsafe { (*us).proto_handler }.unwrap()(*__slate_slot_5, us);
usb_mark_last_busy((*us).pusb_dev);
}
}
}
}
// reject if target != 0 or if LUN is higher than
// the maximum known LUN
// Handle those devices which need us to fake
// their inquiry data
// we've got a command, let's do it!
// lock access to the state
scsi_lock!(host);
// was the command aborted?
if (*(*__slate_slot_5)).result != 327680i32 {
break '__join_17;
}
}
while false {
}
*__slate_slot_5 = core::ptr::null_mut::<scsi_cmnd>(); // Don't call scsi_done()
}
// If an abort request was received we need to signal that
// the abort has finished.  The proper test for this is
// the TIMED_OUT flag, not srb->result == DID_ABORT, because
// the timeout might have occurred after the command had
// already completed with a different result code.
if test_bit__2!(5i32, core::ptr::addr_of_mut!((*us).dflags)) != 0 {
complete(core::ptr::addr_of_mut!((*us).notify));
// Allow USB transfers to resume
clear_bit(2i64, core::ptr::addr_of_mut!((*us).dflags) as *mut u64);
clear_bit(5i64, core::ptr::addr_of_mut!((*us).dflags) as *mut u64);
}
// finished working on this command
(*us).srb = core::ptr::null_mut::<scsi_cmnd>();
scsi_unlock!(host);
// unlock the device pointers
mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex));
// now that the locks are released, notify the SCSI core
if !(*__slate_slot_5).is_null() {
while false {
}
scsi_done_direct(*__slate_slot_5);
}
}
}
}
}
scsi_unlock!(host);
mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex));
while false {
}
}
// for (;;)
// Wait until we are told to stop
loop {
debug_normal_state_change!();
trace_set_current_state!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_2, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_2 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_0, 1u32);
let __t0: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_0 as u32 => __t0, in(reg) *__slate_slot_2, options(nostack, att_syntax));
*__slate_slot_0 = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
if true {
break;
}
}
if true {
break;
}
}
if true {
if kthread_should_stop() {
break;
} else {
schedule();
}
}
}
loop {
debug_normal_state_change!();
trace_set_current_state__2!(0i32);
loop {
if true {
__WRITE_ONCE__3!((*get_current()).__state, 0i32);
if true {
break;
}
}
}
if true {
break;
}
}
return 0;
}
}

// Device probing and disconnecting
/// Associate our private data with the USB device
fn associate_dev(us: *mut us_data, intf: *mut usb_interface) -> i32 {
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
// Fill in the device-related fields
unsafe {
(*us).pusb_dev = interface_to_usbdev!(intf);
}
unsafe {
(*us).pusb_intf = intf;
}
unsafe {
(*us).ifnum = (*(*intf).cur_altsetting).desc.bInterfaceNumber;
}
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"Vendor: 0x%04x, Product: 0x%04x, Revision: 0x%04x\n".as_ptr() as *const u8, (*(*us).pusb_dev).descriptor.idVendor as i32, (*(*us).pusb_dev).descriptor.idProduct as i32, (*(*us).pusb_dev).descriptor.bcdDevice as i32) };
}
if true {
break;
}
}
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"Interface Subclass: 0x%02x, Protocol: 0x%02x\n".as_ptr() as *const u8, (*(*intf).cur_altsetting).desc.bInterfaceSubClass as i32, (*(*intf).cur_altsetting).desc.bInterfaceProtocol as i32) };
}
if true {
break;
}
}
// Store our private data in the interface
usb_set_intfdata(intf, us as *mut ());
// Allocate the control/setup and DMA-mapped buffers
let mut __v2: *mut usb_ctrlrequest;
let __obj_size: u64 = size_mul(core::mem::size_of::<usb_ctrlrequest>() as u64, 1u64);
let mut __v3: *mut ();
let mut __v4: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
}
__v4 = _res;
__v3 = __v4;
__v2 = __v3 as *mut usb_ctrlrequest;
unsafe {
(*us).cr = __v2;
}
if unsafe { (*us).cr }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*us).iobuf = usb_alloc_coherent((*us).pusb_dev, 64u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), core::ptr::addr_of_mut!((*us).iobuf_dma)) as *mut u8;
}
if unsafe { (*us).iobuf }.is_null() {
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"I/O buffer allocation failed\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
return 12i32.wrapping_neg();
}
0
}

/// Adjust device flags based on the "quirks=" module parameter
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_adjust_quirks")]
extern "C" fn usb_stor_adjust_quirks(udev: *mut usb_device, fflags: *mut u64) {
let mut p: *mut u8 = unsafe { core::mem::zeroed() };
let vid: u16 = unsafe { (*udev).descriptor.idVendor };
let pid: u16 = unsafe { (*udev).descriptor.idProduct };
let mut f: u64 = 0;
let mask: u64 = 2126487217u64;
p = unsafe { core::ptr::addr_of_mut!(quirks.0) as *mut u8 };
while (unsafe { *p }) != 0 {
// Each entry consists of VID:PID:flags
let __v0: bool = ((vid as i32) as u64) == unsafe { simple_strtoul(p as *const u8, core::ptr::addr_of_mut!(p), 16u32) } && ((unsafe { *p }) as i32) == 58 && ((pid as i32) as u64) == unsafe { simple_strtoul(p.wrapping_offset(1 as isize) as *const u8, core::ptr::addr_of_mut!(p), 16u32) };
if __v0 && ((unsafe { *p }) as i32) == 58 {
break;
}
// Move forward to the next entry
while (unsafe { *p }) != 0 {
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(1 as isize);
if ((unsafe { *__v2 }) as i32) == 44 {
break;
}
}
}
if (unsafe { *p }) == 0 {
return;
}
// No match
// Collect the flags
loop {
let __v50: *mut u8 = p;
let __v51: *mut u8 = __v50.wrapping_offset(1 as isize);
p = __v51;
if !((unsafe { *__v51 }) != 0 && ((unsafe { *p }) as i32) != 44) {
break;
}
'__slate_break_7: {
match TOLOWER!(unsafe { *p }) {
97 => {
f |= 32768;
}
98 => {
f |= 131072;
}
99 => {
f |= 16;
}
100 => {
f |= 262144;
}
101 => {
f |= 524288;
}
102 => {
f |= 67108864;
}
103 => {
f |= 134217728;
}
104 => {
f |= 4096;
}
105 => {
f |= 2048;
}
106 => {
f |= 268435456;
}
107 => {
f |= 1073741824;
}
108 => {
f |= 128;
}
109 => {
f |= 1024;
}
110 => {
f |= 1048576;
}
111 => {
f |= 65536;
}
112 => {
f |= 2097152;
}
114 => {
f |= 32;
}
115 => {
f |= 1;
}
116 => {
f |= 33554432;
}
117 => {
f |= 8388608;
}
119 => {
f |= 512;
}
121 => {
f |= 536870912;
break '__slate_break_7;
// Ignore unrecognized flag characters
}
_ => {
}
}
}
}
unsafe {
*fflags = *fflags & !mask | f;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_adjust_quirks_618: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_device, *mut u64)>, *mut ()>(Some(usb_stor_adjust_quirks)) };

/// Get the unusual_devs entries and the string descriptors
fn get_device_info(us: *mut us_data, id: *const usb_device_id, unusual_dev: *const us_unusual_dev) -> i32 {
let dev: *mut usb_device = unsafe { (*us).pusb_dev };
let idesc: *mut usb_interface_descriptor = unsafe { core::ptr::addr_of_mut!((*(*(*us).pusb_intf).cur_altsetting).desc) };
let pdev: *mut device = unsafe { core::ptr::addr_of_mut!((*(*us).pusb_intf).dev) };
// Store the entries
unsafe {
(*us).unusual_dev = unusual_dev;
}
unsafe {
(*us).subclass = (if ((*unusual_dev).useProtocol as i32) == 255 { (*idesc).bInterfaceSubClass as i32 } else { (*unusual_dev).useProtocol as i32 }) as u8;
}
unsafe {
(*us).protocol = (if ((*unusual_dev).useTransport as i32) == 255 { (*idesc).bInterfaceProtocol as i32 } else { (*unusual_dev).useTransport as i32 }) as u8;
}
unsafe {
(*us).fflags = (*id).driver_info;
}
usb_stor_adjust_quirks(unsafe { (*us).pusb_dev }, unsafe { core::ptr::addr_of_mut!((*us).fflags) });
if (unsafe { (*us).fflags }) & 2048 != 0 {
while false {
}
unsafe { _dev_info(pdev as *const device, c"device ignored\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
// This flag is only needed when we're in high-speed, so let's
// disable it if we're in full-speed
if ((unsafe { (*dev).speed }) as u32) != 3 {
let __v3: u64 = (unsafe { (*us).fflags }) & 4294967039u64;
unsafe {
(*us).fflags = __v3;
}
}
if (unsafe { (*us).fflags }) != 0 {
while false {
}
unsafe { _dev_info(pdev as *const device, c"Quirks match for vid %04x pid %04x: %llx\n".as_ptr() as *const u8, (*dev).descriptor.idVendor as i32, (*dev).descriptor.idProduct as i32, (*us).fflags) };
}
// Log a message if a non-generic unusual_dev entry contains an
// unnecessary subclass or protocol override.  This may stimulate
// reports from users that will help us remove unneeded entries
// from the unusual_devs.h table.
if (unsafe { (*id).idVendor }) != 0 || (unsafe { (*id).idProduct }) != 0 {
let ddesc: *mut usb_device_descriptor = unsafe { core::ptr::addr_of_mut!((*dev).descriptor) };
let mut msg: i32 = 1i32.wrapping_neg();
if ((unsafe { (*unusual_dev).useProtocol }) as i32) != 255 && ((unsafe { (*us).subclass }) as i32) == ((unsafe { (*idesc).bInterfaceSubClass }) as i32) {
msg = msg.wrapping_add(1);
}
if ((unsafe { (*unusual_dev).useTransport }) as i32) != 255 && ((unsafe { (*us).protocol }) as i32) == ((unsafe { (*idesc).bInterfaceProtocol }) as i32) {
msg = msg.wrapping_add(2);
}
if msg >= 0 && (unsafe { (*us).fflags }) & 2 == 0 {
while false {
}
unsafe { _dev_notice(pdev as *const device, c"This device (%04x,%04x,%04x S %02x P %02x) has %s in unusual_devs.h (kernel %s)\n   Please send a copy of this message to <linux-usb@vger.kernel.org> and <usb-storage@lists.one-eyed-alien.net>\n".as_ptr() as *const u8, (*ddesc).idVendor as i32, (*ddesc).idProduct as i32, (*ddesc).bcdDevice as i32, (*idesc).bInterfaceSubClass as i32, (*idesc).bInterfaceProtocol as i32, msgs.0[msg as usize], (*utsname()).release.as_mut_ptr() as *mut u8) };
}
}
0
}

static mut msgs: __SlateAlign16<[*const u8; 3]> = __SlateAlign16([(b"an unneeded SubClass entry\0".as_ptr() as *mut u8) as *const u8, (b"an unneeded Protocol entry\0".as_ptr() as *mut u8) as *const u8, (b"unneeded SubClass and Protocol entries\0".as_ptr() as *mut u8) as *const u8]);

/// Get the transport settings
fn get_transport(us: *mut us_data) {
match (unsafe { (*us).protocol }) as i32 {
1 => {
unsafe {
(*us).transport_name = c"Control/Bulk".as_ptr() as *mut u8;
}
unsafe {
(*us).transport = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data) -> i32>>(usb_stor_CB_transport as *const ()) };
}
unsafe {
(*us).transport_reset = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_CB_reset as *const ()) };
}
unsafe {
(*us).max_lun = 7;
}
}
0 => {
unsafe {
(*us).transport_name = c"Control/Bulk/Interrupt".as_ptr() as *mut u8;
}
unsafe {
(*us).transport = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data) -> i32>>(usb_stor_CB_transport as *const ()) };
}
unsafe {
(*us).transport_reset = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_CB_reset as *const ()) };
}
unsafe {
(*us).max_lun = 7;
}
}
80 => {
unsafe {
(*us).transport_name = c"Bulk".as_ptr() as *mut u8;
}
unsafe {
(*us).transport = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data) -> i32>>(usb_stor_Bulk_transport as *const ()) };
}
unsafe {
(*us).transport_reset = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut us_data) -> i32>>(usb_stor_Bulk_reset as *const ()) };
}
}
_ => {
}
}
}

/// Get the protocol settings
fn get_protocol(us: *mut us_data) {
match (unsafe { (*us).subclass }) as i32 {
1 => {
unsafe {
(*us).protocol_name = c"Reduced Block Commands (RBC)".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_transparent_scsi_command as *const ()) };
}
}
2 => {
unsafe {
(*us).protocol_name = c"8020i".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_pad12_command as *const ()) };
}
unsafe {
(*us).max_lun = 0;
}
}
3 => {
unsafe {
(*us).protocol_name = c"QIC-157".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_pad12_command as *const ()) };
}
unsafe {
(*us).max_lun = 0;
}
}
5 => {
unsafe {
(*us).protocol_name = c"8070i".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_pad12_command as *const ()) };
}
unsafe {
(*us).max_lun = 0;
}
}
6 => {
unsafe {
(*us).protocol_name = c"Transparent SCSI".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_transparent_scsi_command as *const ()) };
}
}
4 => {
unsafe {
(*us).protocol_name = c"Uniform Floppy Interface (UFI)".as_ptr() as *mut u8;
}
unsafe {
(*us).proto_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data)>>(usb_stor_ufi_command as *const ()) };
}
}
_ => {
}
}
}

/// Get the pipe settings
fn get_pipes(us: *mut us_data) -> i32 {
let alt: *mut usb_host_interface = unsafe { (*(*us).pusb_intf).cur_altsetting };
let mut ep_in: *mut usb_endpoint_descriptor = unsafe { core::mem::zeroed() };
let mut ep_out: *mut usb_endpoint_descriptor = unsafe { core::mem::zeroed() };
let mut ep_int: *mut usb_endpoint_descriptor = unsafe { core::mem::zeroed() };
let mut res: i32 = 0;
// Find the first endpoint of each type we need.
// We are expecting a minimum of 2 endpoints - in and out (bulk).
// An optional interrupt-in is OK (necessary for CBI protocol).
// We will ignore any others.
res = unsafe { usb_find_common_endpoints(alt, core::ptr::addr_of_mut!(ep_in), core::ptr::addr_of_mut!(ep_out), core::ptr::null_mut::<*mut usb_endpoint_descriptor>(), core::ptr::null_mut::<*mut usb_endpoint_descriptor>()) };
if res != 0 {
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"bulk endpoints not found\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
return res;
}
res = usb_find_int_in_endpoint(alt, core::ptr::addr_of_mut!(ep_int));
if res != 0 && ((unsafe { (*us).protocol }) as i32) == 0 {
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"interrupt endpoint not found\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
return res;
}
// Calculate and store the pipe values
unsafe {
(*us).send_ctrl_pipe = usb_sndctrlpipe!(unsafe { (*us).pusb_dev }, 0i32);
}
unsafe {
(*us).recv_ctrl_pipe = usb_rcvctrlpipe!(unsafe { (*us).pusb_dev }, 0i32);
}
unsafe {
(*us).send_bulk_pipe = usb_sndbulkpipe!(unsafe { (*us).pusb_dev }, usb_endpoint_num(ep_out as *const usb_endpoint_descriptor));
}
unsafe {
(*us).recv_bulk_pipe = usb_rcvbulkpipe!(unsafe { (*us).pusb_dev }, usb_endpoint_num(ep_in as *const usb_endpoint_descriptor));
}
if !ep_int.is_null() {
unsafe {
(*us).recv_intr_pipe = usb_rcvintpipe!(unsafe { (*us).pusb_dev }, usb_endpoint_num(ep_int as *const usb_endpoint_descriptor));
}
unsafe {
(*us).ep_bInterval = (*ep_int).bInterval;
}
}
0
}

/// Initialize all the dynamic resources we need
fn usb_stor_acquire_resources(us: *mut us_data) -> i32 {
let mut p: i32 = 0;
unsafe {
(*us).current_urb = usb_alloc_urb(0i32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
if unsafe { (*us).current_urb }.is_null() {
return 12i32.wrapping_neg();
}
// Just before we start our control thread, initialize
// the device if it needs initialization
if (unsafe { (*(*us).unusual_dev).initFunction }) != None {
p = unsafe { unsafe { (*(*us).unusual_dev).initFunction }.unwrap()(us) };
if p != 0 {
return p;
}
}
// Start up our control thread
let mut __v0: *mut task_struct;
let __k: *mut task_struct = unsafe { kthread_create_on_node(Some(usb_stor_control_thread), us as *mut (), 1i32.wrapping_neg(), c"usb-storage".as_ptr() as *const u8) };
if !IS_ERR(__k as *const ()) {
unsafe { wake_up_process(__k) };
}
__v0 = __k;
let th: *mut task_struct = __v0;
if IS_ERR(th as *const ()) {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*us).pusb_intf).dev) as *const device, c"Unable to start control thread\n".as_ptr() as *const u8) };
return PTR_ERR(th as *const ()) as i32;
}
unsafe {
(*us).ctl_thread = th;
}
0
}

/// Release all our dynamic resources
fn usb_stor_release_resources(us: *mut us_data) {
// Tell the control thread to exit.  The SCSI host must
// already have been removed and the DISCONNECTING flag set
// so that we won't accept any more commands.
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"-- sending exit command to thread\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
unsafe { complete(core::ptr::addr_of_mut!((*us).cmnd_ready)) };
if !unsafe { (*us).ctl_thread }.is_null() {
unsafe { kthread_stop((*us).ctl_thread) };
}
// Call the destructor routine, if it exists
if (unsafe { (*us).extra_destructor }) != None {
loop {
if false {
unsafe { _usb_stor_dbg(us as *const us_data, c"-- calling extra_destructor()\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
unsafe { unsafe { (*us).extra_destructor }.unwrap()((*us).extra) };
}
// Free the extra data and the URB
unsafe { kfree((*us).extra as *const ()) };
unsafe { usb_free_urb((*us).current_urb) };
}

/// Dissociate from the USB device
fn dissociate_dev(us: *mut us_data) {
// Free the buffers
unsafe { kfree((*us).cr as *const ()) };
unsafe { usb_free_coherent((*us).pusb_dev, 64u64, (*us).iobuf as *mut (), (*us).iobuf_dma) };
// Remove our private data from the interface
usb_set_intfdata(unsafe { (*us).pusb_intf }, core::ptr::null_mut::<()>());
}

/// First stage of disconnect processing: stop SCSI scanning,
/// remove the host, and stop accepting new commands
fn quiesce_and_remove_host(us: *mut us_data) {
let host: *mut Scsi_Host = us_to_host(us);
// If the device is really gone, cut short reset delays
if ((unsafe { (*(*us).pusb_dev).state }) as u32) == 0 {
set_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*us).dflags) }) as *mut u64);
wake_up!(unsafe { core::ptr::addr_of_mut!((*us).delay_wait) });
}
// Prevent SCSI scanning (if it hasn't started yet)
// or wait for the SCSI-scanning routine to stop.
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*us).scan_dwork)) };
// Balance autopm calls if scanning was cancelled
if test_bit__2!(6i32, unsafe { core::ptr::addr_of_mut!((*us).dflags) }) != 0 {
unsafe { usb_autopm_put_interface_no_suspend((*us).pusb_intf) };
}
// Removing the host will perform an orderly shutdown: caches
// synchronized, disks spun down, etc.
unsafe { scsi_remove_host(host) };
// Prevent any new commands from being accepted and cut short
// reset delays.
scsi_lock!(host);
set_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*us).dflags) }) as *mut u64);
scsi_unlock!(host);
wake_up!(unsafe { core::ptr::addr_of_mut!((*us).delay_wait) });
}

/// Second stage of disconnect processing: deallocate all resources
fn release_everything(us: *mut us_data) {
usb_stor_release_resources(us);
dissociate_dev(us);
// Drop our reference to the host; the SCSI core will free it
// (and "us" along with it) when the refcount becomes 0.
unsafe { scsi_host_put(us_to_host(us)) };
}

/// Delayed-work routine to carry out SCSI-device scanning
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_scan_dwork")]
extern "C" fn usb_stor_scan_dwork(work: *mut work_struct) {
let mut us: *mut us_data = unsafe { core::mem::zeroed() };
let mut __v0: *mut us_data;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__42!(us_data) as isize)) }) as *mut ()) as *mut us_data;
us = __v0;
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*(*us).pusb_intf).dev) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, dev as *const device, c"starting scan\n".as_ptr() as *const u8) };
}
// For bulk-only devices, determine the max LUN value
if ((unsafe { (*us).protocol }) as i32) == 80 && (unsafe { (*us).fflags }) & 1 == 0 && (unsafe { (*us).fflags }) & 4 == 0 {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
unsafe {
(*us).max_lun = usb_stor_Bulk_max_lun(us) as u8;
}
// Allow proper scanning of devices that present more than 8 LUNs
// While not affecting other devices that may need the previous
// behavior
if ((unsafe { (*us).max_lun }) as i32) >= 8 {
unsafe {
(*us_to_host(us)).max_lun = ((*us).max_lun as i32).wrapping_add(1) as u64;
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*us).dev_mutex)) };
}
unsafe { scsi_scan_host(us_to_host(us)) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, dev as *const device, c"scan complete\n".as_ptr() as *const u8) };
}
// Should we unbind if no devices were detected?
unsafe { usb_autopm_put_interface((*us).pusb_intf) };
clear_bit(6i64, (unsafe { core::ptr::addr_of_mut!((*us).dflags) }) as *mut u64);
}

fn usb_stor_sg_tablesize(intf: *mut usb_interface) -> u32 {
let usb_dev: *mut usb_device = interface_to_usbdev!(intf);
if (unsafe { (*(*usb_dev).bus).sg_tablesize }) != 0 {
return unsafe { (*(*usb_dev).bus).sg_tablesize };
}
128
}

/// First part of general USB mass-storage probing
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_probe1")]
extern "C" fn usb_stor_probe1(pus: *mut *mut us_data, intf: *mut usb_interface, id: *const usb_device_id, unusual_dev: *const us_unusual_dev, sht: *const scsi_host_template) -> i32 {
let mut __slate_compound_6: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Some USB host controllers can't do DMA: They have to use PIO, or they
// have to use a small dedicated local memory area, or they have other
// restrictions on addressable memory.
//
// We can't support these controllers on highmem systems as we don't
// kmap or bounce buffer.
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut us: core::mem::MaybeUninit<*mut us_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut us_data = core::ptr::addr_of_mut!(us) as *mut *mut us_data;
let mut host: core::mem::MaybeUninit<*mut Scsi_Host> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut Scsi_Host = core::ptr::addr_of_mut!(host) as *mut *mut Scsi_Host;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
while false {
}
_dev_info(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"USB Mass Storage device detected\n".as_ptr() as *const u8);
// Ask the SCSI layer to allocate a host structure, with extra
// space at the end for our private us_data structure.
*__slate_slot_5 = scsi_host_alloc(sht, core::mem::size_of::<us_data>() as i32);
if (*__slate_slot_5).is_null() {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*intf).dev) as *const device, c"Unable to allocate the scsi host\n".as_ptr() as *const u8);
return 12i32.wrapping_neg();
} else {
// Allow 16-byte CDBs and thus > 2TB
(*(*__slate_slot_5)).max_cmd_len = 16;
(*(*__slate_slot_5)).sg_tablesize = usb_stor_sg_tablesize(intf) as u16;
let __v2: *mut us_data = host_to_us(*__slate_slot_5);
*__slate_slot_4 = __v2;
*pus = __v2;
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_4)).dev_mutex), c"&(us->dev_mutex)".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1046));
if true {
break;
}
}
us_set_lock_class(core::ptr::addr_of_mut!((*(*__slate_slot_4)).dev_mutex), intf);
init_completion(core::ptr::addr_of_mut!((*(*__slate_slot_4)).cmnd_ready));
init_completion(core::ptr::addr_of_mut!((*(*__slate_slot_4)).notify));
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*(*__slate_slot_4)).delay_wait), c"&us->delay_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1050));
if true {
break;
}
}
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_4)).scan_dwork)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_4)).scan_dwork)).work)).data = *{
__slate_compound_6 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_6)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_4)).scan_dwork)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_4)).scan_dwork)).work)).func = Some(usb_stor_scan_dwork);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_4)).scan_dwork)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
// Associate the us_data structure with the USB device
*__slate_slot_3 = associate_dev(*__slate_slot_4, intf);
if *__slate_slot_3 == 0 {
if IS_ENABLED!() != 0 {
if !hcd_uses_dma(bus_to_hcd((*(*(*__slate_slot_4)).pusb_dev).bus)) {
__v0 = true;
} else {
__v0 = !unsafe { (*bus_to_hcd((*(*(*__slate_slot_4)).pusb_dev).bus)).localmem_pool }.is_null();
}
__v1 = __v0;
} else {
__v1 = false;
}
// Get the unusual_devs entries and the descriptors
*__slate_slot_3 = get_device_info(*__slate_slot_4, id, unusual_dev);
if *__slate_slot_3 == 0 {
// Get standard transport and protocol settings
get_transport(*__slate_slot_4);
get_protocol(*__slate_slot_4);
// Give the caller a chance to fill in specialized transport
// or protocol settings.
return 0;
}
}
while false {
}
release_everything(*__slate_slot_4);
return *__slate_slot_3;
}
}
}

static mut __key__1046: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1050: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1051: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_probe1_619: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut us_data, *mut usb_interface, *const usb_device_id, *const us_unusual_dev, *const scsi_host_template) -> i32>, *mut ()>(Some(usb_stor_probe1)) };

/// Second part of general USB mass-storage probing
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_probe2")]
extern "C" fn usb_stor_probe2(us: *mut us_data) -> i32 {
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
dev = core::ptr::addr_of_mut!((*(*us).pusb_intf).dev);
// Make sure the transport and protocol have both been set
if (*us).transport == None || (*us).proto_handler == None {
*__slate_slot_1 = 6i32.wrapping_neg();
} else {
while false {
}
while false {
}
if (*us).fflags & 4 != 0 {
// SCM eUSCSI bridge devices can have different numbers
// of LUNs on different targets; allow all to be probed.
(*us).max_lun = 7;
// The eUSCSI itself has ID 7, so avoid scanning that
(*us_to_host(us)).this_id = 7;
// max_id is 8 initially, so no need to set it here
} else {
// In the normal case there is only a single target
(*us_to_host(us)).max_id = 1;
// Like Windows, we won't store the LUN bits in CDB[1] for
// SCSI-2 devices using the Bulk-Only transport (even though
// this violates the SCSI spec).
if (*us).transport == unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd, *mut us_data) -> i32>>(usb_stor_Bulk_transport as *const ()) } {
(*us_to_host(us)).__slate_bits_0.__set_no_scsi2_lun_in_cdb(1u32);
}
}
// fix for single-lun devices
if (*us).fflags & 1 != 0 {
(*us).max_lun = 0;
}
// Find the endpoints and calculate pipe values
*__slate_slot_1 = get_pipes(us);
if *__slate_slot_1 == 0 {
// If the device returns invalid data for the first READ(10)
// command, indicate the command should be retried.
if (*us).fflags & 1048576 != 0 {
set_bit(7i64, core::ptr::addr_of_mut!((*us).dflags) as *mut u64);
}
// Acquire all the other resources and add the host
*__slate_slot_1 = usb_stor_acquire_resources(us);
if *__slate_slot_1 == 0 {
usb_autopm_get_interface_no_resume((*us).pusb_intf);
snprintf((*us).scsi_name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"usb-storage %s".as_ptr() as *const u8, dev_name(core::ptr::addr_of_mut!((*(*us).pusb_intf).dev) as *const device));
*__slate_slot_1 = scsi_add_host(us_to_host(us), dev);
if *__slate_slot_1 != 0 {
while false {
}
_dev_warn(dev as *const device, c"Unable to add the scsi host\n".as_ptr() as *const u8);
usb_autopm_put_interface_no_suspend((*us).pusb_intf);
} else {
// Submit the delayed_work for SCSI-device scanning
set_bit(6i64, core::ptr::addr_of_mut!((*us).dflags) as *mut u64);
queue_delayed_work(system_freezable_wq, core::ptr::addr_of_mut!((*us).scan_dwork), msecs_to_jiffies(delay_use));
return 0;
}
}
}
}
while false {
}
release_everything(us);
return *__slate_slot_1;
}
// We come here if there are any problems
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_probe2_620: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut us_data) -> i32>, *mut ()>(Some(usb_stor_probe2)) };

/// Handle a USB mass-storage disconnect
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.storage_usb.usb_stor_disconnect")]
extern "C" fn usb_stor_disconnect(intf: *mut usb_interface) {
let us: *mut us_data = usb_get_intfdata(intf) as *mut us_data;
quiesce_and_remove_host(us);
release_everything(us);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_stor_disconnect_621: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_interface)>, *mut ()>(Some(usb_stor_disconnect)) };

static mut usb_stor_host_template: scsi_host_template = unsafe { core::mem::zeroed() };

/// The main probe routine for standard devices
#[unsafe(link_section = ".text.slate_distinct.storage_usb.storage_probe")]
extern "C" fn storage_probe(intf: *mut usb_interface, id: *const usb_device_id) -> i32 {
let mut unusual_dev: *const us_unusual_dev = unsafe { core::mem::zeroed() };
let mut us: *mut us_data = unsafe { core::mem::zeroed() };
let mut result: i32 = 0;
let mut size: i32 = 0;
// If uas is enabled and this device can do uas then ignore it.
// If the device isn't standard (is handled by a subdriver
// module) then don't accept it.
if (unsafe { usb_usual_ignore_device(intf) }) != 0 {
return 6i32.wrapping_neg();
}
// Call the general probe procedures.
//
// The unusual_dev_list array is parallel to the usb_storage_usb_ids
// table, so we use the index of the id entry to find the
// corresponding unusual_devs entry.
size = ARRAY_SIZE!() as i32;
if id >= unsafe { core::ptr::addr_of!(usb_storage_usb_ids) as *const usb_device_id } && id < unsafe { core::ptr::addr_of!(usb_storage_usb_ids) as *const usb_device_id }.wrapping_offset(size as isize) {
unusual_dev = unsafe { core::ptr::addr_of!(us_unusual_dev_list.0) as *const us_unusual_dev }.wrapping_offset(((unsafe { id.offset_from((core::ptr::addr_of!(usb_storage_usb_ids) as *const usb_device_id) as *const usb_device_id) }) as i64) as isize);
} else {
unusual_dev = unsafe { core::ptr::addr_of!(for_dynamic_ids.0) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*intf).dev) as *const device, c"Use Bulk-Only transport with the Transparent SCSI protocol for dynamic id: 0x%04x 0x%04x\n".as_ptr() as *const u8, (*id).idVendor as i32, (*id).idProduct as i32) };
}
}
result = usb_stor_probe1(core::ptr::addr_of_mut!(us), intf, id, unusual_dev, (unsafe { core::ptr::addr_of_mut!(usb_stor_host_template) }) as *const scsi_host_template);
if result != 0 {
return result;
}
// No special transport or protocol settings in the main module
result = usb_stor_probe2(us);
result
}

static mut usb_storage_driver: usb_driver = {
let mut __t0: usb_driver = unsafe { core::mem::zeroed() };
__t0.name = (b"usb-storage\0".as_ptr() as *mut u8) as *const u8;
__t0.probe = Some(storage_probe);
__t0.disconnect = Some(usb_stor_disconnect);
__t0.suspend = Some(usb_stor_suspend);
__t0.resume = Some(usb_stor_resume);
__t0.reset_resume = Some(usb_stor_reset_resume);
__t0.pre_reset = Some(usb_stor_pre_reset);
__t0.post_reset = Some(usb_stor_post_reset);
__t0.id_table = unsafe { core::ptr::addr_of!(usb_storage_usb_ids) as *const usb_device_id };
__t0.__slate_bits_0.__set_supports_autosuspend(1u32);
__t0.__slate_bits_0.__set_soft_unbind(1u32);
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn usb_storage_driver_init() -> i32 {
unsafe { usb_stor_host_template_init(core::ptr::addr_of_mut!(usb_stor_host_template), c"usb-storage".as_ptr() as *const u8, core::ptr::null_mut::<module>()) };
usb_register__2!(unsafe { core::ptr::addr_of_mut!(usb_storage_driver) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_usb_storage_driver_init_623: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(usb_storage_driver_init)) };

#[unsafe(link_section = ".exit.text")]
extern "C" fn usb_storage_driver_exit() {
unsafe { usb_deregister(core::ptr::addr_of_mut!(usb_storage_driver)) };
}

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_usb_storage_driver_exit: Option<unsafe extern "C" fn()> = Some(usb_storage_driver_exit);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
