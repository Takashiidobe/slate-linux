
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
//! xHCI host controller driver
//!
//! Copyright (C) 2008 Intel Corp.
//!
//! Author: Sarah Sharp
//! Some code borrowed from the Linux EHCI driver.
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
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
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
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! Internal kernel timer frequency
//! Sign Flag
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
//! Out of streams resources
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
//! SCHED_ISO: reserved but not implemented yet
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
//! The first word is the work queue pointer and the flags rolled into
//! one
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
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
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! Bits set in the VMA until the stack is in its final location
//! return the first ptr or next element in an RCU protected list
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! See the comment in asm-generic/io.h about ioremap_np().
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
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! Runtime PM flag argument bits
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
//! HCCPARAMS offset from PCI base address
//! PORTSC - Port Status and Control Register - port_status_base bitmasks
//!
//! true: device connected
//! HCSPARAMS2 - hcs_params2 - bitmasks
//!
//! bits 3:0 - Isochronous Scheduling Threshold, frames or uframes that SW
//! needs to queue transactions ahead of the HW to meet periodic deadlines.
//! - Bits 2:0: Threshold value
//! - Bit 3: Unit indicator
//!   - '1': Threshold in Frames
//!   - '0': Threshold in Microframes (uframes)
//! Note: 1 Frame = 8 Microframes
//! xHCI specification section 5.3.4.
//! max buffer size for trace and debug messages
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! The maximum length of a string descriptor is 255, because the bLength
//! field in the usb_string_descriptor struct is __u8.  In practice the
//! maximum length is 254, because a string descriptor consists of a 2 byte
//! header followed by UTF-16 characters (2 bytes each). This allows for
//! only 126 characters (code points) in the string, which is where
//! USB_MAX_STRING_LEN comes from.
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
// Overflow Flag
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
// Cross-device link
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Broken pipe
// Communication error on send
// Protocol error
// Value too large for defined data type
// Illegal byte sequence
// Operation not supported on transport endpoint
// Cannot send after transport endpoint shutdown
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Implementation specific max size
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
// HCCPARAMS contains the first extended capability pointer
// Extended capability register fields
// true: port enabled
// bit 2 reserved and zeroed
//
// true: port has an over-current condition
// true: port reset signaling asserted
// Port Link State - bits 5:8
// A read gives the current link PM state of the port,
// a write with Link State Write Strobe set sets the link state.
// true: port has power (see HCC_PPC)
// Bits 20:23 in the Slot Context are the speed for the device
// Port Link State Write Strobe - set this when changing link state
// true: connect status change
// true: port enable change
// true: warm reset for a USB 3.0 device is done.  A "hot" reset puts the port
// into an enabled state, and the device into the default state.  A "warm" reset
// also resets the link, forcing the device through the link training sequence.
// SW can also look at the Port Reset register to see when warm reset is done.
// true: over-current change
// true: reset change - 1 to 0 transition of PORT_RESET
// port link status change - set on some port link state transitions:
// Transition				Reason
// - U3 to Resume			Wakeup signaling from a device
// - Resume to Recovery to U0		USB 3.0 device resume
// - Resume to U0			USB 2.0 device resume
// - U3 to Recovery to U0		Software resume of USB 3.0 device complete
// - U3 to U0				Software resume of USB 2.0 device complete
// - U2 to U0				L1 resume of USB 2.1 device complete
// - U0 to U0 (???)			L1 entry rejection by USB 2.1 device
// - U0 to disabled			L1 entry error with USB 2.1 device
// - Any state to inactive		Error on USB 3.0 port
// port configure error change - port failed to configure its link partner
// Cold Attach Status - xHC can set this bit to report device attached during
// Sx state. Warm port reset should be perfomed to clear this bit and move port
// to connected state.
// wake on connect (enable)
// wake on disconnect (enable)
// wake on over-current (enable)
// bits 28:29 reserved
//
// true: device is non-removable - for USB 3.0 roothub emulation
// Initiate a warm port reset - complete when PORT_WRC is '1'
// We mark duplicate entries with -1
// bit 10 - Stopped EDTLA Capability
//
// bit 11 - Contiguous Frame ID Capability
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

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
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

/// status bitmasks
/// Command Completion Parameter
macro_rules! COMP_PARAM {
($p:expr) => {
($p) & 16777215
};
}

macro_rules! CTX_TO_EP_TYPE {
($p:expr) => {
($p) >> 3i32 & 7
};
}

macro_rules! DB_VALUE {
($ep:expr, $stream:expr) => {
$ep.wrapping_add(1) & 255 | ($stream) << 16i32
};
}

/// bits 19:16 are the dev speed
macro_rules! DEV_SPEED_FOR_TRB {
($p:expr) => {
(($p) as i32) << 16i32
};
}

macro_rules! DEV_SUPERSPEED_ANY {
($p:expr) => {
($p) & 15360u32 >= 4096u32
};
}

/// DIV_ROUND_UP_POW2 - divide and round up
/// @n: numerator
/// @d: denominator (must be a power of 2)
///
/// Divides @n by @d and rounds up to next multiple of @d (which must be a power
/// of 2). Avoids integer overflows that may occur with __KERNEL_DIV_ROUND_UP().
/// Performance is roughly equivalent to __KERNEL_DIV_ROUND_UP().
macro_rules! DIV_ROUND_UP_POW2 {
($n:expr, $d:expr) => {
(($n) / $d).wrapping_add((($n) & $d.wrapping_sub(1) != 0) as i32)
};
}

macro_rules! EP_INDEX_FOR_TRB {
($p:expr) => {
($p.wrapping_add(1) & 31) << 16i32
};
}

macro_rules! EP_INTERVAL_TO_UFRAMES {
($p:expr) => {
1i32 << (($p) >> 16i32 & 255)
};
}

/// Transfer event TRB length bit mask
macro_rules! EVENT_TRB_LEN {
($p:expr) => {
($p) & 16777215
};
}

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_3060>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_1169>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__7 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_1496>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__9 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_3060_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__15 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_3165>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__16 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_1080>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__20 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_1512>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__21 {
() => {
core::mem::size_of::<__SlateRecord_xhci_ring_c_3065>() as i32
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
(GENMASK_INPUT_CHECK__16!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__7 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__15!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__12 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__4!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__16 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__9!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__17 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__21!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__20 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__20!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__23 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__7!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__12!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__8 {
($h:expr, $l:expr) => {
GENMASK_TYPE__17!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__10 {
($h:expr, $l:expr) => {
GENMASK_TYPE__20!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__11 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__14 {
($h:expr, $l:expr) => {
GENMASK_TYPE__23!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__16 {
($h:expr, $l:expr) => {
GENMASK_TYPE__16!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__19 {
($h:expr, $l:expr) => {
GENMASK_TYPE__7!(u64, $h, $l)
};
}

macro_rules! GET_COMP_CODE {
($p:expr) => {
(($p) & 4278190080u32) >> 24i32
};
}

macro_rules! GET_EP_CTX_STATE {
($ctx:expr) => {
(unsafe { (*$ctx).ep_info }) & 7
};
}

/// Port Status Change Event TRB fields
///
/// Port ID - bits 31:24
macro_rules! GET_PORT_ID {
($p:expr) => {
(($p) & 4278190080u32) >> 24i32
};
}

macro_rules! GET_SLOT_STATE {
($p:expr) => {
(($p) & 4160749568u32) >> 27i32
};
}

macro_rules! HCD_HW_ACCESSIBLE {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 1u64
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

/// Bits [13:3] of the microframe index equals the 1ms frame index
macro_rules! MFINDEX_TO_FRAME {
($p:expr) => {
($p) >> 3i32 & 2047
};
}

macro_rules! NEC_FW_MAJOR {
($p:expr) => {
($p) >> 8i32 & 255
};
}

macro_rules! NEC_FW_MINOR {
($p:expr) => {
($p) >> 0i32 & 255
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SCT_FOR_TRB {
($p:expr) => {
(($p) & 7) << 1i32
};
}

macro_rules! SLOT_ID_FOR_TRB {
($p:expr) => {
(($p) & 255) << 24i32
};
}

macro_rules! SLOT_ID_FOR_TRB__2 {
($p:expr) => {
(($p) & 255) << 24i32
};
}

macro_rules! STREAM_ID_FOR_TRB {
($p:expr) => {
(($p) & 65535) << 16i32
};
}

/// Stop Endpoint TRB - ep_index to endpoint ID for this TRB
macro_rules! SUSPEND_PORT_FOR_TRB {
($p:expr) => {
(($p) & 1) << 23i32
};
}

/// How much data is left before the 64KB boundary?
macro_rules! TRB_BUFF_LEN_UP_TO_BOUNDARY {
($addr:expr) => {
65536u64.wrapping_sub(($addr) & 65535u64)
};
}

macro_rules! TRB_FIELD_TO_TYPE {
($p:expr) => {
(($p) & 64512) >> 10i32
};
}

macro_rules! TRB_FRAME_ID {
($p:expr) => {
(($p) & 2047) << 20i32
};
}

/// Interrupter Target - which MSI-X vector to target the completion event at
macro_rules! TRB_INTR_TARGET {
($p:expr) => {
(($p) & 1023) << 22i32
};
}

/// Normal TRB fields
///
/// transfer_len bitmasks - bits 0:16
macro_rules! TRB_LEN {
($p:expr) => {
($p) & 131071
};
}

/// Normal TRB fields
///
/// transfer_len bitmasks - bits 0:16
macro_rules! TRB_LEN__2 {
($p:expr) => {
($p) & 131071
};
}

/// Total burst count field, Rsvdz on xhci 1.1 with Extended TBC enabled (ETE)
macro_rules! TRB_TBC {
($p:expr) => {
(($p) & 3) << 7i32
};
}

macro_rules! TRB_TLBPC {
($p:expr) => {
(($p) & 15) << 16i32
};
}

/// Endpoint ID 1 - 31
macro_rules! TRB_TO_EP_ID {
($p:expr) => {
($p) >> 16i32 & 31
};
}

/// Endpoint index 0 - 30
macro_rules! TRB_TO_EP_INDEX {
($p:expr) => {
TRB_TO_EP_ID!($p).wrapping_sub(1)
};
}

/// Transfer event flags bitfield, also for select command completion events
macro_rules! TRB_TO_SLOT_ID {
($p:expr) => {
($p) >> 24i32 & 255
};
}

/// Set TR Dequeue Pointer command TRB fields, 6.4.3.9
macro_rules! TRB_TO_STREAM_ID {
($p:expr) => {
(($p) & 4294901760u32) >> 16i32
};
}

macro_rules! TRB_TO_SUSPEND_PORT {
($p:expr) => {
(($p) & 8388608u32) >> 23i32
};
}

macro_rules! TRB_TX_TYPE {
($p:expr) => {
($p) << 16i32
};
}

/// Above, but for __le32 types -- can avoid work by swapping constants:
macro_rules! TRB_TYPE_LINK_LE32 {
($x:expr) => {
($x) & 64512 == (TRB_TYPE!(6i32) as u32)
};
}

macro_rules! TRB_TYPE_NOOP_LE32 {
($x:expr) => {
($x) & 64512 == (TRB_TYPE!(8i32) as u32)
};
}

macro_rules! TRB_TYPE {
($p:expr) => {
($p) << 10i32
};
}

macro_rules! ULL {
() => {
_ULL!()
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

macro_rules! _AC__3 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _ULL {
() => {
_AC!()
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

macro_rules! ____is_defined {
() => {
0i32
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u32
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u64
};
}

macro_rules! __cmp_once__3 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
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

macro_rules! __le16_to_cpu {
($x:expr) => {
($x) as u16
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
macro_rules! __native_word__7 {
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
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __pa {
($x:expr) => {
__phys_addr!(($x) as u64)
};
}

macro_rules! __phys_addr {
($x:expr) => {
__phys_addr_nodebug($x)
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__11 {
($ptr:expr) => {
{
let mut __v: *mut xhci_td;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__20!() as isize)) }) as *mut ()) as *mut xhci_td;
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
macro_rules! container_of__5 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__12!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__26 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__70!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__43 {
($ptr:expr) => {
{
let mut __v: *mut xhci_command;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__75!() as isize)) }) as *mut ()) as *mut xhci_command;
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
macro_rules! container_of__49 {
($ptr:expr) => {
{
let mut __v: *mut xhci_td;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__65!() as isize)) }) as *mut ()) as *mut xhci_td;
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
macro_rules! container_of__52 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__10!() as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// delayed_work_pending - Find out whether a delayable work item is currently
/// pending
/// @w: The work item in question
macro_rules! delayed_work_pending {
($w:expr) => {
work_pending!(unsafe { core::ptr::addr_of_mut!((*$w).work) })
};
}

macro_rules! dma_map_single {
($d:expr, $a:expr, $s:expr, $r:expr) => {
dma_map_single_attrs($d, $a, ($s) as u64, ($r) as u32, 0u64)
};
}

macro_rules! dma_unmap_single {
($d:expr, $a:expr, $s:expr, $r:expr) => {
dma_unmap_single_attrs($d, $a, ($s) as u64, ($r) as u32, 0u64);
};
}

/// in_range - Determine if a value lies within a range.
/// @val: Value to test.
/// @start: First value in range.
/// @len: Number of values in range.
///
/// This is more efficient than "if (start <= val && val < (start + len))".
/// It also gives a different answer if @start + @len overflows the size of
/// the type by a sufficient amount to encompass @val.  Decide for yourself
/// which behaviour you want, or prove that start + len never overflow.
/// Do not blindly replace one form with the other.
macro_rules! in_range__2 {
($val:expr, $start:expr, $len:expr) => {
{
let __v: i32 = if (core::mem::size_of::<u64>() as u64) | (core::mem::size_of::<i32>() as u64) | (core::mem::size_of::<u64>() as u64) <= (core::mem::size_of::<u32>() as u64) {
in_range32(($val) as u32, ($start) as u32, ($len) as u32) as i32
} else {
in_range64($val, $start, ($len) as u64) as i32
};
__v
}
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

macro_rules! kcsan_rmb {
() => {
while false {
}
};
}

macro_rules! kcsan_wmb {
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__25 {
($ptr:expr, $type:ty) => {
container_of__26!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__26 {
($ptr:expr) => {
container_of__43!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__27 {
($ptr:expr) => {
container_of__11!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr, $type:ty) => {
container_of__4!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__33 {
($ptr:expr) => {
container_of__49!($ptr)
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
macro_rules! list_first_entry__4 {
($ptr:expr, $type:ty) => {
list_entry__3!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__33!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__11 {
($ptr:expr) => {
list_entry__26!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__12 {
($ptr:expr, $type:ty) => {
list_entry__25!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__13 {
($ptr:expr) => {
list_entry__27!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__33!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__9 {
($pos:expr, $member:ident) => {
list_entry__26!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__10 {
($pos:expr, $member:ident) => {
list_entry__27!(unsafe { (*$pos).$member.next })
};
}

/// lower_32_bits - return bits 0-31 of a number
/// @n: the number we're accessing
macro_rules! lower_32_bits {
($n:expr) => {
(($n) & 4294967295) as u32
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__4 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__3!($type, $x, $y)
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof__20 {
() => {
core::mem::offset_of!(xhci_td, td_list) as u64
};
}

macro_rules! offsetof__10 {
() => {
184u64
};
}

macro_rules! offsetof__11 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, td_list) as u64
};
}

macro_rules! offsetof__12 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__65 {
() => {
core::mem::offset_of!(xhci_td, cancelled_td_list) as u64
};
}

macro_rules! offsetof__70 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, cmd_list) as u64
};
}

macro_rules! offsetof__75 {
() => {
core::mem::offset_of!(xhci_command, cmd_list) as u64
};
}

macro_rules! offsetof__2 {
() => {
0u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
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

/// roundup - round up to the next specified multiple
/// @x: the value to up
/// @y: multiple to round up to
///
/// Rounds @x up to next multiple of @y. If @y will always be a power
/// of 2, consider using the faster round_up().
macro_rules! roundup {
($x:expr, $y:expr) => {
{
let mut __v: u32;
let __v2: i32 = $y;
__v = ($x.wrapping_add(__v2.wrapping_sub(1) as u32) / (__v2 as u32)).wrapping_mul(__v2 as u32);
__v
}
};
}

/// These macros should be used after a dma_map_sg call has been done
/// to get bus addresses of each of the SG entries and their lengths.
/// You should only work with the number of sg entries dma_map_sg
/// returns, or alternatively stop on the first sg_dma_len(sg) which
/// is 0.
macro_rules! sg_dma_address {
($sg:expr) => {
unsafe { (*$sg).dma_address }
};
}

macro_rules! sg_dma_len {
($sg:expr) => {
unsafe { (*$sg).dma_length }
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

macro_rules! static_branch_unlikely__11 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const18(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__5 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__12 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const14(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__19 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const23(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__6 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const10(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__13 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const15(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__20 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const19(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__21 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const22(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
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

macro_rules! static_branch_unlikely__14 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const12(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__15 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const13(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__16 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const17(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__22 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const20(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__7 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const8(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const6(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__9 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const5(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__10 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const7(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__17 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const16(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__23 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const21(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__18 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const11(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
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
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_is_before_jiffies - return true if a is before jiffies
/// @a: time (unsigned long) to compare to jiffies
///
/// Return: %true is time a is before jiffies, otherwise %false.
macro_rules! time_is_before_jiffies {
($a:expr) => {
time_after__2!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, $a)
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
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

/// upper_32_bits - return bits 32-63 of a number
/// @n: the number we're accessing
///
/// A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
/// the "right shift count >= width of type" warning when that quantity is
/// 32-bits.
macro_rules! upper_32_bits {
($n:expr) => {
(($n) >> 16i32 >> 16i32) as u32
};
}

macro_rules! usb_pipeint {
($pipe:expr) => {
usb_pipetype!($pipe) == 1
};
}

macro_rules! usb_pipetype {
($pipe:expr) => {
($pipe) >> 30i32 & 3
};
}

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

/// work_pending - Find out whether a work item is currently pending
/// @work: The work item in question
macro_rules! work_pending {
($work:expr) => {
test_bit__2!(0i32, (unsafe { core::ptr::addr_of_mut!((*$work).data) }) as *mut u64)
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut __cpu_online_mask: cpumask;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_xhci_dbg_quirks: static_call_key;
    static mut __tracepoint_xhci_dbg_quirks: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_reset_ep: static_call_key;
    static mut __tracepoint_xhci_dbg_reset_ep: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_cancel_urb: static_call_key;
    static mut __tracepoint_xhci_dbg_cancel_urb: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_ring_expansion: static_call_key;
    static mut __tracepoint_xhci_dbg_ring_expansion: tracepoint;
    static mut __SCK__tp_func_xhci_handle_event: static_call_key;
    static mut __tracepoint_xhci_handle_event: tracepoint;
    static mut __SCK__tp_func_xhci_handle_command: static_call_key;
    static mut __tracepoint_xhci_handle_command: tracepoint;
    static mut __SCK__tp_func_xhci_handle_transfer: static_call_key;
    static mut __tracepoint_xhci_handle_transfer: tracepoint;
    static mut __SCK__tp_func_xhci_queue_trb: static_call_key;
    static mut __tracepoint_xhci_queue_trb: tracepoint;
    static mut __SCK__tp_func_xhci_urb_giveback: static_call_key;
    static mut __tracepoint_xhci_urb_giveback: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_set_deq_stream: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_set_deq_stream: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_stop_ep: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_stop_ep: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_set_deq_ep: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_set_deq_ep: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_reset_ep: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_reset_ep: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_config_ep: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_config_ep: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_disable_slot: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_disable_slot: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_addr_dev: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_addr_dev: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_reset_dev: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_reset_dev: tracepoint;
    static mut __SCK__tp_func_xhci_handle_cmd_set_deq: static_call_key;
    static mut __tracepoint_xhci_handle_cmd_set_deq: tracepoint;
    static mut __SCK__tp_func_xhci_inc_enq: static_call_key;
    static mut __tracepoint_xhci_inc_enq: tracepoint;
    static mut __SCK__tp_func_xhci_inc_deq: static_call_key;
    static mut __tracepoint_xhci_inc_deq: tracepoint;
    static mut __SCK__tp_func_xhci_handle_port_status: static_call_key;
    static mut __tracepoint_xhci_handle_port_status: tracepoint;
    static mut __SCK__tp_func_xhci_ring_ep_doorbell: static_call_key;
    static mut __tracepoint_xhci_ring_ep_doorbell: tracepoint;
    static mut __SCK__tp_func_xhci_ring_host_doorbell: static_call_key;
    static mut __tracepoint_xhci_ring_host_doorbell: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn rcu_is_watching() -> bool;
    fn wait_for_completion_timeout(arg0: *mut completion, arg1: u64) -> u64;
    fn complete(arg0: *mut completion);
    fn complete_all(arg0: *mut completion);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn mod_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn cancel_delayed_work(arg0: *mut delayed_work) -> bool;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn kfree(arg0: *const ());
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn __compiletime_assert_419();
    fn is_vmalloc_addr(arg0: *const ()) -> bool;
    fn sg_pcopy_from_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *const (), arg3: u64, arg4: i64) -> u64;
    fn sg_pcopy_to_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *mut (), arg3: u64, arg4: i64) -> u64;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn dma_map_page_attrs(arg0: *mut device, arg1: *mut page, arg2: u64, arg3: u64, arg4: u32, arg5: u64) -> u64;
    fn dma_unmap_page_attrs(arg0: *mut device, arg1: u64, arg2: u64, arg3: u32, arg4: u64);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn usb_hcd_link_urb_to_ep(arg0: *mut usb_hcd, arg1: *mut urb) -> i32;
    fn usb_hcd_unlink_urb_from_ep(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_giveback_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32);
    fn usb_hcd_is_primary_hcd(arg0: *mut usb_hcd) -> i32;
    fn usb_hc_died(arg0: *mut usb_hcd);
    fn usb_hcd_poll_rh_status(arg0: *mut usb_hcd);
    fn usb_wakeup_notification(arg0: *mut usb_device, arg1: u32);
    fn usb_hcd_start_port_resume(arg0: *mut usb_bus, arg1: i32);
    fn usb_hub_clear_tt_buffer(arg0: *mut urb) -> i32;
    fn usb_hcd_resume_root_hub(arg0: *mut usb_hcd);
    fn usb_amd_quirk_pll_disable();
    fn usb_amd_quirk_pll_enable();
    fn xhci_dbg_trace(arg0: *mut xhci_hcd, arg1: Option<unsafe extern "C" fn(*mut va_format)>, arg2: *const u8, ...);
    fn xhci_get_endpoint_index(arg0: *mut usb_endpoint_descriptor) -> u32;
    fn xhci_last_valid_endpoint(arg0: u32) -> u32;
    fn xhci_ring_expansion(arg0: *mut xhci_hcd, arg1: *mut xhci_ring, arg2: u32, arg3: u32) -> i32;
    fn xhci_free_device_endpoint_resources(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: bool);
    fn xhci_dma_to_transfer_ring(arg0: *mut xhci_virt_ep, arg1: u64) -> *mut xhci_ring;
    fn xhci_alloc_command(arg0: *mut xhci_hcd, arg1: bool, arg2: u32) -> *mut xhci_command;
    fn xhci_urb_free_priv(arg0: *mut urb_priv);
    fn xhci_free_command(arg0: *mut xhci_hcd, arg1: *mut xhci_command);
    fn xhci_handshake(arg0: *mut (), arg1: u32, arg2: u32, arg3: u64) -> i32;
    fn xhci_halt(arg0: *mut xhci_hcd) -> i32;
    fn xhci_disable_interrupter(arg0: *mut xhci_hcd, arg1: *mut xhci_interrupter) -> i32;
    fn xhci_usb_endpoint_maxp(arg0: *mut usb_device, arg1: *mut usb_host_endpoint) -> i32;
    fn xhci_portsc_readl(arg0: *mut xhci_port) -> u32;
    fn xhci_set_link_state(arg0: *mut xhci_hcd, arg1: *mut xhci_port, arg2: u32);
    fn xhci_test_and_clear_bit(arg0: *mut xhci_hcd, arg1: *mut xhci_port, arg2: u32);
    fn xhci_ring_device(arg0: *mut xhci_hcd, arg1: i32);
    fn xhci_get_input_control_ctx(arg0: *mut xhci_container_ctx) -> *mut xhci_input_control_ctx;
    fn xhci_get_slot_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx) -> *mut xhci_slot_ctx;
    fn xhci_get_ep_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: u32) -> *mut xhci_ep_ctx;
    fn __traceiter_xhci_dbg_quirks(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_quirks(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_544();
    fn __traceiter_xhci_dbg_reset_ep(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_reset_ep(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_551();
    fn __traceiter_xhci_dbg_cancel_urb(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_cancel_urb(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_558();
    fn __traceiter_xhci_dbg_ring_expansion(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_ring_expansion(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_572();
    fn __traceiter_xhci_handle_event(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __SCT__tp_func_xhci_handle_event(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __compiletime_assert_586();
    fn __traceiter_xhci_handle_command(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __SCT__tp_func_xhci_handle_command(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __compiletime_assert_593();
    fn __traceiter_xhci_handle_transfer(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __SCT__tp_func_xhci_handle_transfer(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __compiletime_assert_600();
    fn __traceiter_xhci_queue_trb(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __SCT__tp_func_xhci_queue_trb(arg0: *mut (), arg1: *mut xhci_ring, arg2: *mut xhci_generic_trb, arg3: u64) -> i32;
    fn __compiletime_assert_607();
    fn __traceiter_xhci_urb_giveback(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __SCT__tp_func_xhci_urb_giveback(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __compiletime_assert_677();
    fn __traceiter_xhci_handle_cmd_set_deq_stream(arg0: *mut (), arg1: *mut xhci_stream_info, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_set_deq_stream(arg0: *mut (), arg1: *mut xhci_stream_info, arg2: u32) -> i32;
    fn __compiletime_assert_698();
    fn __traceiter_xhci_handle_cmd_stop_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_stop_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __compiletime_assert_705();
    fn __traceiter_xhci_handle_cmd_set_deq_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_set_deq_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __compiletime_assert_712();
    fn __traceiter_xhci_handle_cmd_reset_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_reset_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __compiletime_assert_719();
    fn __traceiter_xhci_handle_cmd_config_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_config_ep(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __compiletime_assert_726();
    fn __traceiter_xhci_handle_cmd_disable_slot(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_disable_slot(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_754();
    fn __traceiter_xhci_handle_cmd_addr_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_addr_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_775();
    fn __traceiter_xhci_handle_cmd_reset_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_reset_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_782();
    fn __traceiter_xhci_handle_cmd_set_deq(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_handle_cmd_set_deq(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_789();
    fn __traceiter_xhci_inc_enq(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __SCT__tp_func_xhci_inc_enq(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __compiletime_assert_838();
    fn __traceiter_xhci_inc_deq(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __SCT__tp_func_xhci_inc_deq(arg0: *mut (), arg1: *mut xhci_ring) -> i32;
    fn __compiletime_assert_845();
    fn __traceiter_xhci_handle_port_status(arg0: *mut (), arg1: *mut xhci_port, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_handle_port_status(arg0: *mut (), arg1: *mut xhci_port, arg2: u32) -> i32;
    fn __compiletime_assert_852();
    fn __traceiter_xhci_ring_ep_doorbell(arg0: *mut (), arg1: u32, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_ring_ep_doorbell(arg0: *mut (), arg1: u32, arg2: u32) -> i32;
    fn __compiletime_assert_880();
    fn __traceiter_xhci_ring_host_doorbell(arg0: *mut (), arg1: u32, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_ring_host_doorbell(arg0: *mut (), arg1: u32, arg2: u32) -> i32;
    fn __compiletime_assert_887();
    fn __compiletime_assert_918();
    fn __compiletime_assert_925();
    fn __compiletime_assert_933();
    fn __compiletime_assert_936();
    fn __compiletime_assert_937();
    fn __compiletime_assert_938();
    fn __compiletime_assert_939();
    fn __compiletime_assert_940();
    fn __compiletime_assert_941();
    fn __compiletime_assert_944();
    fn __compiletime_assert_947();
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
fn in_range64(val: u64, start: u64, len: u64) -> bool {
val.wrapping_sub(start) < len
}

fn in_range32(val: u32, start: u32, len: u32) -> bool {
val.wrapping_sub(start) < len
}

// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


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

#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__3!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__3!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits123U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_quirks, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_reset_ep, const 0 as u8, label {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_cancel_urb, const 0 as u8, label {
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
struct pollfd {
}


#[inline(always)]
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_ring_expansion, const 0 as u8, label {
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
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_event, const 0 as u8, label {
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
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_command, const 0 as u8, label {
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
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_transfer, const 0 as u8, label {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_queue_trb, const 0 as u8, label {
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
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_urb_giveback, const 0 as u8, label {
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
struct mutex_waiter {
}


#[inline(always)]
fn arch_static_branch__const10(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_set_deq_stream, const 0 as u8, label {
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn arch_static_branch__const11(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_stop_ep, const 0 as u8, label {
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
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_set_deq_ep, const 0 as u8, label {
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
struct futex_private_hash {
}


#[inline(always)]
fn arch_static_branch__const13(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_reset_ep, const 0 as u8, label {
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
fn arch_static_branch__const14(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_config_ep, const 0 as u8, label {
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
fn arch_static_branch__const15(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_disable_slot, const 0 as u8, label {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[inline(always)]
fn arch_static_branch__const16(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_addr_dev, const 0 as u8, label {
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
fn arch_static_branch__const17(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_reset_dev, const 0 as u8, label {
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
fn arch_static_branch__const18(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_cmd_set_deq, const 0 as u8, label {
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
fn arch_static_branch__const19(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_inc_enq, const 0 as u8, label {
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
fn arch_static_branch__const20(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_inc_deq, const 0 as u8, label {
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
fn arch_static_branch__const21(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_handle_port_status, const 0 as u8, label {
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
fn arch_static_branch__const22(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_ring_ep_doorbell, const 0 as u8, label {
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
fn arch_static_branch__const23(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_ring_host_doorbell, const 0 as u8, label {
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
struct hrtimer_clock_base {
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
    __slate_bits_0: __slate_bits::__SlateBits130U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[inline(always)]
fn __list_add(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*next).prev = new;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, new);
if true {
break;
}
}
}

fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, next);
if true {
break;
}
}
}

fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_20() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

fn list_is_singular(head: *const list_head) -> i32 {
(list_empty(head) == 0 && (unsafe { (*head).next }) == unsafe { (*head).prev }) as i32
}

// list_first_entry_or_null - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
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
test_bit__4!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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
struct workqueue_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits131U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
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
struct llist_node {
    next: *mut llist_node,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
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
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
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
struct load_weight {
    weight: u64,
    inv_weight: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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


fn reinit_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


fn to_delayed_work(work: *mut work_struct) -> *mut delayed_work {
container_of__5!(work, delayed_work)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn mod_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { mod_delayed_work_on(64i32, wq, dwork, delay) }
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits408U0,
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits439U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct io_uring_cmd {
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
struct ns_common {
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
struct kernfs_root {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
unsafe { __compiletime_assert_376() };
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


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_419() };
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
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
struct ep_device {
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
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mon_bus {
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
    __slate_bits_0: __slate_bits::__SlateBits620U0,
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
    __slate_bits_0: __slate_bits::__SlateBits571U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


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
fn __sg_flags(sg: *mut scatterlist) -> u32 {
((unsafe { (*sg).page_link }) & (1 | 2)) as u32
}

fn sg_chain_ptr(sg: *mut scatterlist) -> *mut scatterlist {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut scatterlist
}

fn sg_is_chain(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 1 != 0
}

fn sg_is_last(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 2 != 0
}

fn sg_next(mut sg: *mut scatterlist) -> *mut scatterlist {
if sg_is_last(sg) {
return core::ptr::null_mut::<scatterlist>();
}
let __v0: *mut scatterlist = sg;
sg = __v0.wrapping_offset(1 as isize);
if unlikely!(sg_is_chain(sg)) != 0 {
sg = sg_chain_ptr(sg);
}
sg
}

// Loop over each sg element, following the pointer to a new list if necessary
fn str_plural(num: u64) -> *const u8 {
(if num == 1 { c"".as_ptr() as *mut u8 } else { c"s".as_ptr() as *mut u8 }) as *const u8
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_sideband {
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
    __slate_bits_0: __slate_bits::__SlateBits651U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits651U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits651U2,
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
    __slate_bits_3: __slate_bits::__SlateBits651U3,
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
struct clk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reset_control {
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
struct serial_icounter_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
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
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
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
struct __SlateRecord_tty_port_h_118 {
    __slate_anon_0: __SlateRecord_tty_port_h_118_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_port_h_118_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_212 {
    lock: spinlock,
    stopped: bool,
    tco_stopped: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_218 {
    pgrp: *mut pid,
    session: *mut pid,
    lock: spinlock,
    pktstatus: u8,
    packet: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_1080 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_1169 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_1496 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_1512 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3060 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3060_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3065 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3165 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_ring_c_3978_14 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits651U0 {
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
    pub struct __SlateBits648U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits651U1 {
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
    pub struct __SlateBits651U2 {
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
    pub struct __SlateBits651U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits840U0 {
        #[bits(1)]
        pub console: u8,
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
    pub struct __SlateBits123U0 {
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
    pub struct __SlateBits131U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits408U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits439U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits620U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits571U0 {
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
    pub struct __SlateBits130U0 {
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
    pub struct __SlateBits222U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits763U0 {
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
    pub struct __SlateBits759U0 {
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
    pub struct __SlateBits753U0 {
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
    pub struct __SlateBits753U1 {
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
    pub struct __SlateBits753U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits771U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits775U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits803U0 {
        #[bits(1)]
        pub link_inactive: u32,
        #[bits(1)]
        pub connected: u32,
        #[bits(1)]
        pub lpm_incapable: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits825U0 {
        #[bits(1)]
        pub hw_lpm_support: u32,
        #[bits(1)]
        pub broken_suspend: u32,
        #[bits(1)]
        pub allow_single_roothub: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits648U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

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
fn debug_dma_mapping_error(dev: *mut device, dma_addr: u64) {
}

fn debug_dma_map_single(dev: *mut device, addr: *const (), len: u64) {
}

fn dma_mapping_error(dev: *mut device, dma_addr: u64) -> i32 {
debug_dma_mapping_error(dev, dma_addr);
if unlikely!(dma_addr == !0) != 0 {
return 12i32.wrapping_neg();
}
0
}

fn dma_map_single_attrs(dev: *mut device, ptr: *mut (), size: u64, dir: u32, attrs: u64) -> u64 {
// DMA must never operate on areas that might be remapped.
let mut __v0: i32;
let __ret_warn_on: i32 = !(!unsafe { is_vmalloc_addr(ptr as *const ()) }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: rejecting DMA map of vmalloc memory\n".as_ptr() as *const u8, dev_driver_string(dev as *const device), dev_name(dev as *const device)) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_dma_map_single_attrs_3, sym __slate_asm_string_dma_map_single_attrs_4, const 532 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(dev as *const device), dev_name(dev as *const device)) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
if __v0 != 0 {
return !0;
}
debug_dma_map_single(dev, ptr as *const (), size);
unsafe { dma_map_page_attrs(dev, virt_to_page!(ptr), offset_in_page!(ptr), size, dir, attrs) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_504: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn dma_unmap_single_attrs(dev: *mut device, addr: u64, size: u64, dir: u32, attrs: u64) {
unsafe { dma_unmap_page_attrs(dev, addr, size, dir, attrs) };
return;
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


// USB String descriptors can contain at most 126 characters.
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
fn usb_endpoint_xfer_control(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 0) as i32
}

fn usb_endpoint_xfer_isoc(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 1) as i32
}

fn usb_endpoint_maxp(epd: *const usb_endpoint_descriptor) -> i32 {
(__le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32) & 2047
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


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface {
    altsetting: *mut usb_host_interface,
    cur_altsetting: *mut usb_host_interface,
    num_altsetting: u32,
    intf_assoc: *mut usb_interface_assoc_descriptor,
    minor: i32,
    condition: u32,
    __slate_bits_0: __slate_bits::__SlateBits759U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


// this maximum is arbitrary
// USB Resume Timer: Every Host controller driver should drive the resume
// signalling on the bus for the amount of time defined by this macro.
//
// That way we will have a 'stable' behavior among all HCDs supported by Linux.
//
// Note that the USB Specification states we should drive resume for *at least*
// 20 ms, but it doesn't give an upper bound. This creates two possible
// situations which we want to avoid:
//
// (a) sometimes an msleep(20) might expire slightly before 20 ms, which causes
// us to fail USB Electrical Tests, thus failing Certification
//
// (b) Some (many) devices actually need more than 20 ms of resume signalling,
// and while we can argue that's against the USB Specification, we don't have
// control over which devices a certification laboratory will be using for
// certification. If CertLab uses a device which was tested against Windows and
// that happens to have relaxed resume signalling rules, we might fall into
// situations where we fail interoperability and electrical tests.
//
// In order to avoid both conditions, we're using a 40 ms resume timeout, which
// should cope with both LPJ calibration errors and devices not following every
// detail of the USB Specification.
// ms
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


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_bus {
    controller: *mut device,
    sysdev: *mut device,
    busnum: i32,
    bus_name: *const u8,
    uses_pio_for_control: u8,
    otg_port: u8,
    __slate_bits_0: __slate_bits::__SlateBits763U0,
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
    __slate_bits_0: __slate_bits::__SlateBits753U0,
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
    __slate_bits_1: __slate_bits::__SlateBits753U1,
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
    __slate_bits_2: __slate_bits::__SlateBits753U2,
}


// urb->transfer_flags:
//
// Note: URB_DIR_IN/OUT is automatically set in usb_submit_urb().
// report short reads as errors
// iso-only; use the first unexpired
// slot in the schedule
// urb->transfer_dma valid on submit
// Finish bulk OUT with short packet
// The following flags are used internally by usbcore and HCDs
// Transfer from device to host
// Non-scatter-gather mapping
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
    __slate_bits_0: __slate_bits::__SlateBits771U0,
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


fn usb_urb_dir_in(urb: *mut urb) -> i32 {
((unsafe { (*urb).transfer_flags }) & 512 == 512) as i32
}

fn usb_urb_dir_out(urb: *mut urb) -> i32 {
((unsafe { (*urb).transfer_flags }) & 512 == 0) as i32
}

// NOTE:  these are not the standard USB_ENDPOINT_XFER_* values!!
//
// (yet ... they're the values used by usbfs)
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
    __slate_bits_0: __slate_bits::__SlateBits775U0,
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


fn hcd_periodic_completion_in_progress(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) -> bool {
(unsafe { (*hcd).high_prio_bh.completing_ep }) == ep
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


fn lo_hi_readq(addr: *const ()) -> u64 {
let p: *const u32 = addr as *const u32;
let low: u32 = readl(p as *const ());
let high: u32 = readl(p.wrapping_offset(1 as isize) as *const ());
(low as u64).wrapping_add((high as u64) << 32i32)
}

fn lo_hi_writeq(val: u64, addr: *mut ()) {
writel(val as u32, addr);
writel((val >> 32i32) as u32, (unsafe { (addr as *const u8).wrapping_offset(4 as isize) }) as *mut ());
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_cap_regs {
    hc_capbase: u32,
    hcs_params1: u32,
    hcs_params2: u32,
    hcs_params3: u32,
    hcc_params: u32,
    db_off: u32,
    run_regs_off: u32,
    hcc_params2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port_regs {
    portsc: u32,
    portpmsc: u32,
    portli: u32,
    porthlmpc: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_op_regs {
    command: u32,
    status: u32,
    page_size: u32,
    reserved1: u32,
    reserved2: u32,
    dev_notification: u32,
    cmd_ring: u64,
    reserved3: [u32; 4],
    dcbaa_ptr: u64,
    config_reg: u32,
    reserved4: [u32; 241],
    port_regs: [xhci_port_regs; 0],
}


// USBCMD - USB command - command bitmasks
//
// start/stop HC execution - do not write unless HC is halted
// USBSTS - USB status - status bitmasks
//
// HC not running - set to 1 when run/stop bit is cleared.
// serious error, e.g. PCI parity error.  The HC will clear the run/stop bit.
// event interrupt - clear this prior to clearing any IP flags in IR set
// port change detect
// bits 5:7 reserved and zeroed
//
// save state status - '1' means xHC is saving state
// restore state status - '1' means xHC is restoring state
// true: save or restore error
// true: Controller Not Ready to accept doorbell or op reg writes after reset
// true: internal Host Controller Error - SW needs to reset and reinitialize
// stop ring immediately - abort the currently executing command
// true: command ring is running
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_intr_reg {
    iman: u32,
    imod: u32,
    erst_size: u32,
    rsvd: u32,
    erst_base: u64,
    erst_dequeue: u64,
}


// iman bitmasks
//
// bit 0 - Interrupt Pending (IP), whether there is an interrupt pending. Write-1-to-clear.
// erst_dequeue bitmasks
//
// bits 2:0 - Dequeue ERST Segment Index (DESI), is the segment number (or alias) where the
// current dequeue pointer lies. This is an optional HW hint.
// bit 3 - Event Handler Busy (EHB), whether the event ring is scheduled to be serviced by
// a work queue (or delayed service routine)?
// bits 63:4 - Event Ring Dequeue Pointer
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_run_regs {
    microframe_index: u32,
    rsvd: [u32; 7],
    ir_set: [xhci_intr_reg; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_doorbell_array {
    doorbell: [u32; 256],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_container_ctx {
    r#type: u32,
    size: i32,
    bytes: *mut u8,
    dma: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_slot_ctx {
    dev_info: u32,
    dev_info2: u32,
    tt_info: u32,
    dev_state: u32,
    reserved: [u32; 4],
}


// dev_info bitmasks
//
// Route String - 0:19
// Device speed - values defined by PORTSC Device Speed field - 20:23
// bit 24 reserved
//
// Is this LS/FS device connected through a HS hub? - bit 25
// Set if the device is a hub - bit 26
// Index of the last valid endpoint context in this device context - 27:31
// dev_info2 bitmasks
//
// Max Exit Latency (ms) - worst case time to wake up all links in dev path
// tt_info bitmasks
//
// TT Hub Slot ID - for low or full speed devices attached to a high-speed hub
// The Slot ID of the hub that isolates the high speed signaling from
// this low or full-speed device.  '0' if attached to root hub port.
// The number of the downstream facing port of the high-speed hub
// '0' if the device is not low or full speed.
// dev_state bitmasks
//
// USB device address - assigned by the HC
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_ep_ctx {
    ep_info: u32,
    ep_info2: u32,
    deq: u64,
    tx_info: u32,
    reserved: [u32; 3],
}


// ep_info bitmasks
//
// Endpoint State - bits 0:2
// 0 - disabled
// 1 - running
// 2 - halted due to halt condition - ok to manipulate endpoint ring
// 3 - stopped
// 4 - TRB error
// 5-7 - reserved
// Endpoint is set up with a Linear Stream Array (vs. Secondary Stream Array)
// hosts with LEC=1 use bits 31:24 as ESIT high bits.
// tx_info bitmasks
// bits 63:4 - TR Dequeue Pointer
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_input_control_ctx {
    drop_flags: u32,
    add_flags: u32,
    rsvd2: [u32; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_command {
    in_ctx: *mut xhci_container_ctx,
    status: u32,
    comp_param: u32,
    slot_id: i32,
    completion: *mut completion,
    command_trb: *mut xhci_trb,
    cmd_list: list_head,
    timeout_ms: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_stream_ctx {
    stream_ring: u64,
    reserved: [u32; 2],
}


// Primary stream array type, dequeue pointer is to a transfer ring
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_stream_info {
    stream_rings: *mut *mut xhci_ring,
    num_streams: u32,
    stream_ctx_array: *mut xhci_stream_ctx,
    num_stream_ctxs: u32,
    ctx_array_dma: u64,
    trb_address_map: xarray,
    free_streams_command: *mut xhci_command,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_bw_info {
    ep_interval: u32,
    mult: u32,
    num_packets: u32,
    max_packet_size: u32,
    max_esit_payload: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_virt_ep {
    vdev: *mut xhci_virt_device,
    ep_index: u32,
    ring: *mut xhci_ring,
    stream_info: *mut xhci_stream_info,
    new_ring: *mut xhci_ring,
    err_count: u32,
    ep_state: u32,
    cancelled_td_list: list_head,
    queued_deq_seg: *mut xhci_segment,
    queued_deq_ptr: *mut xhci_trb,
    skip: bool,
    bw_info: xhci_bw_info,
    bw_endpoint_list: list_head,
    stop_time: u64,
    next_uframe: i32,
    use_extended_tbc: bool,
    sideband: *mut xhci_sideband,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interval_bw {
    num_packets: u32,
    endpoints: list_head,
    overhead: [u32; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interval_bw_table {
    interval0_esit_payload: u32,
    interval_bw: [xhci_interval_bw; 16],
    bw_used: u32,
    ss_bw_in: u32,
    ss_bw_out: u32,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_xhci_msi_irq: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous", sym xhci_msi_irq, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_virt_device {
    slot_id: i32,
    udev: *mut usb_device,
    out_ctx: *mut xhci_container_ctx,
    in_ctx: *mut xhci_container_ctx,
    eps: [xhci_virt_ep; 31],
    rhub_port: *mut xhci_port,
    bw_table: *mut xhci_interval_bw_table,
    tt_info: *mut xhci_tt_bw_info,
    current_mel: u16,
    debugfs_private: *mut (),
    sideband: *mut xhci_sideband,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_root_port_bw_info {
    tts: list_head,
    num_active_tts: u32,
    bw_table: xhci_interval_bw_table,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_tt_bw_info {
    tt_list: list_head,
    slot_id: i32,
    ttport: i32,
    bw_table: xhci_interval_bw_table,
    active_eps: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_device_context_array {
    ctx_array: *mut u64,
    dma: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_transfer_event {
    buffer: u64,
    transfer_len: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_link_trb {
    segment_ptr: u64,
    intr_target: u32,
    control: u32,
}


// control bitfields
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_event_cmd {
    cmd_trb: u64,
    status: u32,
    flags: u32,
}


// Address device - disable SetAddress
// Configure Endpoint - Deconfigure
// Stop Ring - Transfer State Preserve
// Force Event
// Set Latency Tolerance Value
// Get Port Bandwidth
// Force Header
// Link TRB specific fields
// TD Size, packets remaining in this TD, bits 21:17 (5 bits, so max 31)
// xhci 1.1 uses the TD_SIZE field for TBC if Extended TBC is enabled (ETE)
// Cycle bit - indicates TRB ownership by HC or HCD
// Force next event data TRB to be evaluated before task switch.
// Used to pass OS data back after a TD completes.
// Interrupt on short packet
// Set PCIe no snoop attribute
// Chain multiple TRBs into a TD
// Interrupt on completion
// The buffer pointer contains immediate data
// TDs smaller than this might use IDT
// Block Event Interrupt
// Control transfer TRB specific fields
// Isochronous TRB specific fields
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_generic_trb {
    field: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
union xhci_trb {
    link: xhci_link_trb,
    trans_event: xhci_transfer_event,
    event_cmd: xhci_event_cmd,
    generic: xhci_generic_trb,
}


// TRB type IDs
//
// bulk, interrupt, isoc scatter/gather, and control data stage
// setup stage for control transfers
// data stage for control transfers
// status stage for control transfers
// isoc transfers
// TRB for linking ring segments
// Transfer Ring No-op (not for the command ring)
// Command TRBs
//
// Enable Slot Command
// Disable Slot Command
// Address Device Command
// Configure Endpoint Command
// Evaluate Context Command
// Reset Endpoint Command
// Stop Transfer Ring Command
// Set Transfer Ring Dequeue Pointer Command
// Reset Device Command
// Force Event Command (opt)
// Negotiate Bandwidth Command (opt)
// Set Latency Tolerance Value Command (opt)
// Get port bandwidth Command
// Force Header Command - generate a transaction or link management packet
// No-op Command - not for transfer rings
// TRB IDs 24-31 reserved
//
// Event TRBS
//
// Transfer Event
// Command Completion Event
// Port Status Change Event
// Bandwidth Request Event (opt)
// Doorbell Event (opt)
// Host Controller Event
// Device Notification Event - device sent function wake notification
// MFINDEX Wrap Event - microframe counter wrapped
// TRB IDs 40-47 reserved, 48-63 is vendor-defined
// Nec vendor-specific command completion event.
// Get NEC firmware revision.
// TRBS_PER_SEGMENT must be a multiple of 4,
// since the command ring is 64-byte aligned.
// It must also be greater than 16.
// Limits of consecutive isoc trbs that can Block Event Interrupt (BEI) if
// XHCI_AVOID_BEI quirk is in use.
#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_segment {
    trbs: *mut xhci_trb,
    next: *mut xhci_segment,
    num: u32,
    dma: u64,
    bounce_dma: u64,
    bounce_buf: *mut (),
    bounce_offs: u32,
    bounce_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_td {
    td_list: list_head,
    cancelled_td_list: list_head,
    status: i32,
    cancel_status: u32,
    urb: *mut urb,
    start_seg: *mut xhci_segment,
    start_trb: *mut xhci_trb,
    end_seg: *mut xhci_segment,
    end_trb: *mut xhci_trb,
    bounce_seg: *mut xhci_segment,
    urb_length_set: bool,
    error_mid_td: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_ring {
    first_seg: *mut xhci_segment,
    last_seg: *mut xhci_segment,
    enqueue: *mut xhci_trb,
    enq_seg: *mut xhci_segment,
    dequeue: *mut xhci_trb,
    deq_seg: *mut xhci_segment,
    td_list: list_head,
    cycle_state: u32,
    stream_id: u32,
    num_segs: u32,
    bounce_buf_len: u32,
    r#type: u32,
    old_trb_comp_code: u32,
    trb_address_map: *mut xarray,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_erst_entry {
    seg_addr: u64,
    seg_size: u32,
    rsvd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_erst {
    entries: *mut xhci_erst_entry,
    num_entries: u32,
    erst_dma_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_scratchpad {
    sp_array: *mut u64,
    sp_dma: u64,
    sp_buffers: *mut *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct urb_priv {
    num_tds: i32,
    num_tds_done: i32,
    td: [xhci_td; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct s3_save {
    command: u32,
    dev_nt: u32,
    dcbaa_ptr: u64,
    config_reg: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_bus_state {
    bus_suspended: u64,
    next_statechange: u64,
    port_c_suspend: u32,
    suspended_ports: u32,
    port_remote_wakeup: u32,
    resuming_ports: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_interrupter {
    event_ring: *mut xhci_ring,
    erst: xhci_erst,
    ir_set: *mut xhci_intr_reg,
    intr_num: u32,
    ip_autoclear: bool,
    isoc_bei_interval: u32,
    s3_iman: u32,
    s3_imod: u32,
    s3_erst_size: u32,
    s3_erst_base: u64,
    s3_erst_dequeue: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port_cap {
    psi: *mut u32,
    psi_count: u8,
    psi_uid_count: u8,
    maj_rev: u8,
    min_rev: u8,
    protocol_caps: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_port {
    port_reg: *mut xhci_port_regs,
    hw_portnum: i32,
    hcd_portnum: i32,
    rhub: *mut xhci_hub,
    port_cap: *mut xhci_port_cap,
    __slate_bits_0: __slate_bits::__SlateBits803U0,
    resume_timestamp: u64,
    rexit_active: bool,
    slot_id: i32,
    rexit_done: completion,
    u3exit_done: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_hub {
    ports: *mut *mut xhci_port,
    num_ports: u32,
    hcd: *mut usb_hcd,
    bus_state: xhci_bus_state,
    maj_rev: u8,
    min_rev: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_hcd {
    main_hcd: *mut usb_hcd,
    shared_hcd: *mut usb_hcd,
    cap_regs: *mut xhci_cap_regs,
    op_regs: *mut xhci_op_regs,
    run_regs: *mut xhci_run_regs,
    dba: *mut xhci_doorbell_array,
    hcs_params2: u32,
    hcs_params3: u32,
    hcc_params: u32,
    hcc_params2: u32,
    lock: spinlock,
    hci_version: u16,
    max_interrupters: u16,
    max_slots: u8,
    max_ports: u8,
    imod_interval: u32,
    page_size: u32,
    dma_mask_bits: u32,
    nvecs: i32,
    clk: *mut clk,
    reg_clk: *mut clk,
    reset: *mut reset_control,
    dcbaa: xhci_device_context_array,
    interrupters: *mut *mut xhci_interrupter,
    cmd_ring: *mut xhci_ring,
    cmd_ring_state: u32,
    cmd_list: list_head,
    cmd_ring_reserved_trbs: u32,
    cmd_timer: delayed_work,
    cmd_ring_stop_completion: completion,
    current_cmd: *mut xhci_command,
    scratchpad: *mut xhci_scratchpad,
    mutex: mutex,
    devs: *mut *mut xhci_virt_device,
    rh_bw: *mut xhci_root_port_bw_info,
    device_pool: *mut dma_pool,
    segment_pool: *mut dma_pool,
    small_streams_pool: *mut dma_pool,
    port_bw_pool: *mut dma_pool,
    medium_streams_pool: *mut dma_pool,
    xhc_state: u32,
    run_graceperiod: u64,
    s3: s3_save,
    quirks: u64,
    num_active_eps: u32,
    limit_active_eps: u32,
    hw_ports: *mut xhci_port,
    usb2_rhub: xhci_hub,
    usb3_rhub: xhci_hub,
    __slate_bits_0: __slate_bits::__SlateBits825U0,
    port_caps: *mut xhci_port_cap,
    num_port_caps: u32,
    comp_mode_recovery_timer: timer_list,
    port_status_u0: u32,
    test_mode: u16,
    debugfs_root: *mut dentry,
    debugfs_slots: *mut dentry,
    regset_list: list_head,
    dbc: *mut (),
    r#priv: [u64; 0],
}


fn hcd_to_xhci(hcd: *mut usb_hcd) -> *mut xhci_hcd {
let mut primary_hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
if (unsafe { usb_hcd_is_primary_hcd(hcd) }) != 0 {
primary_hcd = hcd;
} else {
primary_hcd = unsafe { (*hcd).primary_hcd };
}
(unsafe { core::ptr::addr_of_mut!((*primary_hcd).hcd_priv) as *mut u64 }) as *mut xhci_hcd
}

fn xhci_to_hcd(xhci: *mut xhci_hcd) -> *mut usb_hcd {
unsafe { (*xhci).main_hcd }
}

fn xhci_get_usb3_hcd(xhci: *mut xhci_hcd) -> *mut usb_hcd {
if !unsafe { (*xhci).shared_hcd }.is_null() {
return unsafe { (*xhci).shared_hcd };
}
if (unsafe { (*xhci).usb2_rhub.num_ports }) == 0 {
return unsafe { (*xhci).main_hcd };
}
core::ptr::null_mut::<usb_hcd>()
}

fn xhci_read_64(xhci: *const xhci_hcd, regs: *mut u64) -> u64 {
lo_hi_readq(regs as *const ())
}

fn xhci_write_64(xhci: *mut xhci_hcd, val: u64, regs: *mut u64) {
lo_hi_writeq(val, regs as *mut ());
}

fn xhci_link_chain_quirk(xhci: *mut xhci_hcd, r#type: u32) -> bool {
(unsafe { (*xhci).quirks }) & BIT_ULL!(0i32) != 0 || (r#type as u32) == 1 && (unsafe { (*xhci).quirks }) & (BIT_ULL!(9i32) | BIT_ULL!(2i32)) != 0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_queue_ctrl_tx(xhci: *mut xhci_hcd, mem_flags: u32, urb: *mut urb, slot_id: i32, ep_index: u32) -> i32 {
let mut ep_ring: *mut xhci_ring = unsafe { core::mem::zeroed() };
// 1 TRB for setup, 1 for status
let mut num_trbs: i32 = 2;
let mut ret: i32 = 0;
let mut start_cycle: i32 = 0;
let mut field: u32 = 0;
let mut urb_priv: *mut urb_priv = unsafe { core::mem::zeroed() };
ep_ring = xhci_urb_to_transfer_ring(xhci, urb);
if ep_ring.is_null() {
return 22i32.wrapping_neg();
}
// Need to copy setup packet into setup TRB, so we can't use the setup
// DMA address.
if unsafe { (*urb).setup_packet }.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(49i32) != 0 && ((unsafe { (*(*urb).dev).speed }) as u32) >= 5 {
// If next available TRB is the Link TRB in the ring segment then
// enqueue a No Op TRB, this can prevent the Setup and Data Stage
// TRB to be breaked by the Link TRB.
if last_trb_on_seg(unsafe { (*ep_ring).enq_seg }, unsafe { (*ep_ring).enqueue }.wrapping_offset(1 as isize)) {
field = (TRB_TYPE!(8i32) as u32) | unsafe { (*ep_ring).cycle_state };
queue_trb(xhci, ep_ring, false, 0u32, 0u32, TRB_INTR_TARGET!(0i32) as u32, field);
}
}
// Don't need to check if we need additional event data and normal TRBs,
// since data in control transfers will never get bigger than 16MB
// XXX: can we get a buffer that crosses 64KB boundaries?
if (unsafe { (*urb).transfer_buffer_length }) > 0 {
num_trbs = num_trbs.wrapping_add(1);
}
ret = prepare_transfer(xhci, unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }, ep_index, unsafe { (*urb).stream_id }, num_trbs as u32, urb, 0u32, mem_flags);
if ret < 0 {
return ret;
}
urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let td: *mut xhci_td = unsafe { core::ptr::addr_of_mut!((*urb_priv).td) as *mut xhci_td }.wrapping_offset(0 as isize);
// Don't give the first TRB to the hardware (by toggling the cycle bit)
// until we've finished creating all the other TRBs.  The ring's cycle
// state may change as we enqueue the other TRBs, so save it too.
let start_trb: *mut xhci_generic_trb = unsafe { core::ptr::addr_of_mut!((*(*ep_ring).enqueue).generic) };
start_cycle = (unsafe { (*ep_ring).cycle_state }) as i32;
// Queue setup TRB - see section 6.4.1.2.1
//
// FIXME better way to translate setup_packet into two u32 fields?
let setup: *mut usb_ctrlrequest = (unsafe { (*urb).setup_packet }) as *mut usb_ctrlrequest;
field = 0;
let __v2: u32 = field;
field = ((__v2 as u64) | (BIT!(6i32) | (TRB_TYPE!(2i32) as u64))) as u32;
if start_cycle == 0 {
field |= 1;
}
// xHCI 1.0/1.1 6.4.1.2.1: Transfer Type field
if ((unsafe { (*xhci).hci_version }) as i32) >= 256 || (unsafe { (*xhci).quirks }) & BIT_ULL!(21i32) != 0 {
if (unsafe { (*urb).transfer_buffer_length }) > 0 {
if ((unsafe { (*setup).bRequestType }) as i32) & 128 != 0 {
let __v6: u32 = field;
field = __v6 | (TRB_TX_TYPE!(3i32) as u32);
} else {
let __v8: u32 = field;
field = __v8 | (TRB_TX_TYPE!(2i32) as u32);
}
}
}
queue_trb(xhci, ep_ring, true, (((unsafe { (*setup).bRequestType }) as i32) | ((unsafe { (*setup).bRequest }) as i32) << 8i32 | ((unsafe { (*setup).wValue }) as i32) << 16i32) as u32, (((unsafe { (*setup).wIndex }) as i32) | ((unsafe { (*setup).wLength }) as i32) << 16i32) as u32, (TRB_LEN__2!(8i32) | TRB_INTR_TARGET!(0i32)) as u32, field); // Immediate data in pointer
// If there's data, queue data TRBs
//
// Only set interrupt on short packet for IN endpoints
if usb_urb_dir_in(urb) != 0 {
field = (BIT!(2i32) | (TRB_TYPE!(3i32) as u64)) as u32;
} else {
field = TRB_TYPE!(3i32) as u32;
}
if (unsafe { (*urb).transfer_buffer_length }) > 0 {
let mut length_field: u32 = 0;
let mut remainder: u32 = 0;
let mut addr: u64 = 0;
if xhci_urb_suitable_for_idt(urb) {
unsafe { memcpy(core::ptr::addr_of_mut!(addr) as *mut (), (*urb).transfer_buffer as *const (), (*urb).transfer_buffer_length as u64) };
loop {
core::ptr::addr_of_mut!(addr);
if true {
break;
}
}
let __v11: u32 = field;
field = ((__v11 as u64) | BIT!(6i32)) as u32;
} else {
addr = unsafe { (*urb).transfer_dma };
}
remainder = xhci_td_remainder(xhci, 0i32, (unsafe { (*urb).transfer_buffer_length }) as i32, unsafe { (*urb).transfer_buffer_length }, urb, true);
let mut __v13: u32;
let __UNIQUE_ID_x__934: u32 = remainder;
let __UNIQUE_ID_y__935: u32 = 31;
while false {
}
__v13 = if __UNIQUE_ID_x__934 < __UNIQUE_ID_y__935 { __UNIQUE_ID_x__934 } else { __UNIQUE_ID_y__935 };
length_field = TRB_LEN!(unsafe { (*urb).transfer_buffer_length }) | __v13 << 17i32 | (TRB_INTR_TARGET!(0i32) as u32);
if ((unsafe { (*setup).bRequestType }) as i32) & 128 != 0 {
let __v15: u32 = field;
field = ((__v15 as u64) | BIT!(16i32)) as u32;
}
queue_trb(xhci, ep_ring, true, lower_32_bits!(addr), upper_32_bits!(addr), length_field, field | unsafe { (*ep_ring).cycle_state });
}
// Save the DMA address of the last TRB in the TD
unsafe {
(*td).end_trb = (*ep_ring).enqueue;
}
unsafe {
(*td).end_seg = (*ep_ring).enq_seg;
}
// Queue status TRB - see Table 7 and sections 4.11.2.2 and 6.4.1.2.3
//
// If the device sent data, the status stage is an OUT transfer
if (unsafe { (*urb).transfer_buffer_length }) > 0 && ((unsafe { (*setup).bRequestType }) as i32) & 128 != 0 {
field = 0;
} else {
field = BIT!(16i32) as u32;
}
queue_trb(xhci, ep_ring, false, 0u32, 0u32, TRB_INTR_TARGET!(0i32) as u32, ((field as u64) | BIT!(5i32) | (TRB_TYPE!(4i32) as u64) | ((unsafe { (*ep_ring).cycle_state }) as u64)) as u32); // Event on completion
giveback_first_trb(xhci, slot_id, ep_index, 0u32, start_cycle, start_trb);
0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_queue_bulk_tx(xhci: *mut xhci_hcd, mem_flags: u32, urb: *mut urb, slot_id: i32, ep_index: u32) -> i32 {
let mut ring: *mut xhci_ring = unsafe { core::mem::zeroed() };
let mut urb_priv: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut sg: *mut scatterlist = core::ptr::null_mut::<scatterlist>();
let mut more_trbs_coming: bool = true;
let mut need_zero_pkt: bool = false;
let mut first_trb: bool = true;
let mut num_trbs: u32 = 0;
let mut start_cycle: u32 = 0;
let mut num_sgs: u32 = 0;
// Queue the TRBs, even if they are zero-length
let mut enqd_len: u32 = 0;
let mut block_len: u32 = 0;
let mut trb_buff_len: u32 = 0;
let mut full_len: u32 = 0;
let mut sent_len: i32 = 0;
let mut ret: i32 = 0;
let mut field: u32 = 0;
let mut length_field: u32 = 0;
let mut remainder: u32 = 0;
let mut addr: u64 = 0;
let mut send_addr: u64 = 0;
ring = xhci_urb_to_transfer_ring(xhci, urb);
if ring.is_null() {
return 22i32.wrapping_neg();
}
full_len = unsafe { (*urb).transfer_buffer_length };
// If we have scatter/gather list, we use it.
if (unsafe { (*urb).num_sgs }) != 0 && (unsafe { (*urb).transfer_flags }) & 65536 == 0 {
num_sgs = (unsafe { (*urb).num_mapped_sgs }) as u32;
sg = unsafe { (*urb).sg };
addr = sg_dma_address!(sg);
block_len = sg_dma_len!(sg);
num_trbs = count_sg_trbs_needed(urb);
} else {
num_trbs = count_trbs_needed(urb);
addr = unsafe { (*urb).transfer_dma };
block_len = full_len;
}
ret = prepare_transfer(xhci, unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }, ep_index, unsafe { (*urb).stream_id }, num_trbs, urb, 0u32, mem_flags);
if unlikely!(ret < 0) != 0 {
return ret;
}
urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
// Deal with URB_ZERO_PACKET - need one more td/trb
if (unsafe { (*urb).transfer_flags }) & 64 != 0 && (unsafe { (*urb_priv).num_tds }) > 1 {
need_zero_pkt = true;
}
let td: *mut xhci_td = unsafe { core::ptr::addr_of_mut!((*urb_priv).td) as *mut xhci_td }.wrapping_offset(0 as isize);
// Don't give the first TRB to the hardware (by toggling the cycle bit)
// until we've finished creating all the other TRBs.  The ring's cycle
// state may change as we enqueue the other TRBs, so save it too.
let start_trb: *mut xhci_generic_trb = unsafe { core::ptr::addr_of_mut!((*(*ring).enqueue).generic) };
start_cycle = unsafe { (*ring).cycle_state };
send_addr = addr;
while first_trb || enqd_len < full_len {
field = TRB_TYPE!(1i32) as u32;
// TRB buffer should not cross 64KB boundaries
trb_buff_len = TRB_BUFF_LEN_UP_TO_BOUNDARY!(addr) as u32;
trb_buff_len = min_t__4!(u32, trb_buff_len, block_len);
if enqd_len.wrapping_add(trb_buff_len) > full_len {
trb_buff_len = full_len.wrapping_sub(enqd_len);
}
// Don't change the cycle bit of the first TRB until later
if first_trb {
first_trb = false;
if start_cycle == 0 {
let __v3: u32 = field;
field = ((__v3 as u64) | BIT!(0i32)) as u32;
}
} else {
field |= unsafe { (*ring).cycle_state };
}
// Chain all the TRBs together; clear the chain bit in the last
// TRB to indicate it's the last TRB in the chain.
if enqd_len.wrapping_add(trb_buff_len) < full_len {
let __v7: u32 = field;
field = ((__v7 as u64) | BIT!(4i32)) as u32;
if trb_is_link(unsafe { (*ring).enqueue }.wrapping_offset(1 as isize)) {
if xhci_align_td(xhci, urb, enqd_len, unsafe { &mut (*core::ptr::addr_of_mut!(trb_buff_len)) }, unsafe { (*ring).enq_seg }) != 0 {
send_addr = unsafe { (*(*ring).enq_seg).bounce_dma };
// TD bounced at least, and last on this seg
unsafe {
(*td).bounce_seg = (*ring).enq_seg;
}
}
}
}
if enqd_len.wrapping_add(trb_buff_len) >= full_len {
let __v9: u32 = field;
field = ((__v9 as u64) & !BIT!(4i32)) as u32;
let __v11: u32 = field;
field = ((__v11 as u64) | BIT!(5i32)) as u32;
more_trbs_coming = false;
unsafe {
(*td).end_trb = (*ring).enqueue;
}
unsafe {
(*td).end_seg = (*ring).enq_seg;
}
if xhci_urb_suitable_for_idt(urb) {
unsafe { memcpy(core::ptr::addr_of_mut!(send_addr) as *mut (), (*urb).transfer_buffer as *const (), trb_buff_len as u64) };
loop {
core::ptr::addr_of_mut!(send_addr);
if true {
break;
}
}
let __v14: u32 = field;
field = ((__v14 as u64) | BIT!(6i32)) as u32;
}
}
// Only set interrupt on short packet for IN endpoints
if usb_urb_dir_in(urb) != 0 {
let __v16: u32 = field;
field = ((__v16 as u64) | BIT!(2i32)) as u32;
}
// Set the TRB length, TD size, and interrupter fields.
remainder = xhci_td_remainder(xhci, enqd_len as i32, trb_buff_len as i32, full_len, urb, more_trbs_coming);
let mut __v18: u32;
let __UNIQUE_ID_x__931: u32 = remainder;
let __UNIQUE_ID_y__932: u32 = 31;
while false {
}
__v18 = if __UNIQUE_ID_x__931 < __UNIQUE_ID_y__932 { __UNIQUE_ID_x__931 } else { __UNIQUE_ID_y__932 };
length_field = TRB_LEN!(trb_buff_len) | __v18 << 17i32 | (TRB_INTR_TARGET!(0i32) as u32);
queue_trb(xhci, ring, (more_trbs_coming as i32) | (need_zero_pkt as i32) != 0, lower_32_bits!(send_addr), upper_32_bits!(send_addr), length_field, field);
addr = addr.wrapping_add(trb_buff_len as u64);
sent_len = trb_buff_len as i32;
while !sg.is_null() && (sent_len as u32) >= block_len {
// New sg entry
num_sgs = num_sgs.wrapping_sub(1);
sent_len = (sent_len as u32).wrapping_sub(block_len) as i32;
sg = sg_next(sg);
if num_sgs != 0 && !sg.is_null() {
block_len = sg_dma_len!(sg);
addr = sg_dma_address!(sg);
addr = addr.wrapping_add(sent_len as u64);
}
}
block_len = block_len.wrapping_sub(sent_len as u32);
send_addr = addr;
enqd_len = enqd_len.wrapping_add(trb_buff_len);
}
if need_zero_pkt {
ret = prepare_transfer(xhci, unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }, ep_index, unsafe { (*urb).stream_id }, 1u32, urb, 1u32, mem_flags);
unsafe {
(*(core::ptr::addr_of_mut!((*urb_priv).td) as *mut xhci_td).wrapping_offset(1 as isize)).end_trb = (*ring).enqueue;
}
unsafe {
(*(core::ptr::addr_of_mut!((*urb_priv).td) as *mut xhci_td).wrapping_offset(1 as isize)).end_seg = (*ring).enq_seg;
}
field = ((((TRB_TYPE!(1i32) as u32) | unsafe { (*ring).cycle_state }) as u64) | BIT!(5i32)) as u32;
queue_trb(xhci, ring, false, 0u32, 0u32, TRB_INTR_TARGET!(0i32) as u32, field);
}
check_trb_math(urb, enqd_len as i32);
giveback_first_trb(xhci, slot_id, ep_index, unsafe { (*urb).stream_id }, start_cycle as i32, start_trb);
0
}

fn xhci_urb_to_transfer_ring(xhci: *mut xhci_hcd, urb: *mut urb) -> *mut xhci_ring {
xhci_triad_to_transfer_ring(xhci, (unsafe { (*(*urb).dev).slot_id }) as u32, unsafe { xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*urb).ep).desc)) }, unsafe { (*urb).stream_id })
}

fn xhci_urb_suitable_for_idt(urb: *mut urb) -> bool {
let __v0: bool = usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) == 0 && usb_urb_dir_out(urb) != 0;
let __v1: bool = __v0 && usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) >= 8;
if __v1 && (unsafe { (*urb).transfer_buffer_length }) <= 8 && (unsafe { (*urb).transfer_flags }) & 4 == 0 && (unsafe { (*urb).num_sgs }) == 0 {
return true;
}
false
}

fn xhci_decode_usbsts(str: *mut u8, usbsts: u32) -> *const u8 {
let mut ret: i32 = unsafe { sprintf(str, c" 0x%08x".as_ptr() as *const u8, usbsts) };
if usbsts == !0 {
return str as *const u8;
}
if usbsts & 1u32 != 0 {
let __v0: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" HCHalted".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v0);
}
if (usbsts as u64) & BIT!(2i32) != 0 {
let __v3: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" HSE".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v3);
}
if (usbsts as u64) & BIT!(3i32) != 0 {
let __v6: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" EINT".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v6);
}
if (usbsts as u64) & BIT!(4i32) != 0 {
let __v9: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" PCD".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v9);
}
if (usbsts as u64) & BIT!(8i32) != 0 {
let __v12: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" SSS".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v12);
}
if (usbsts as u64) & BIT!(9i32) != 0 {
let __v15: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" RSS".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v15);
}
if (usbsts as u64) & BIT!(10i32) != 0 {
let __v18: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" SRE".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v18);
}
if usbsts & 2048u32 != 0 {
let __v21: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" CNR".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v21);
}
if (usbsts as u64) & BIT!(12i32) != 0 {
let __v24: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" HCE".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v24);
}
str as *const u8
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ktermios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 19],
    c_ispeed: u32,
    c_ospeed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
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
struct tty_operations {
    lookup: Option<unsafe extern "C" fn(*mut tty_driver, *mut file, i32) -> *mut tty_struct>,
    install: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct)>,
    open: Option<unsafe extern "C" fn(*mut tty_struct, *mut file) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct, *mut file)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_struct)>,
    cleanup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, u64) -> i64>,
    put_char: Option<unsafe extern "C" fn(*mut tty_struct, u8) -> i32>,
    flush_chars: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write_room: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    chars_in_buffer: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i64>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    throttle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    unthrottle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    stop: Option<unsafe extern "C" fn(*mut tty_struct)>,
    start: Option<unsafe extern "C" fn(*mut tty_struct)>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    break_ctl: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    ldisc_ok: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    set_ldisc: Option<unsafe extern "C" fn(*mut tty_struct)>,
    wait_until_sent: Option<unsafe extern "C" fn(*mut tty_struct, i32)>,
    send_xchar: Option<unsafe extern "C" fn(*mut tty_struct, u8)>,
    tiocmget: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    tiocmset: Option<unsafe extern "C" fn(*mut tty_struct, u32, u32) -> i32>,
    resize: Option<unsafe extern "C" fn(*mut tty_struct, *mut winsize) -> i32>,
    get_icount: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_icounter_struct) -> i32>,
    get_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    set_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut tty_struct, *mut seq_file)>,
    proc_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_driver {
    kref: kref,
    cdevs: *mut *mut cdev,
    owner: *mut module,
    driver_name: *const u8,
    name: *const u8,
    name_base: i32,
    major: i32,
    minor_start: i32,
    num: u32,
    r#type: u32,
    subtype: u32,
    init_termios: ktermios,
    flags: u64,
    proc_entry: *mut proc_dir_entry,
    other: *mut tty_driver,
    flip_wq: *mut workqueue_struct,
    ttys: *mut *mut tty_struct,
    ports: *mut *mut tty_port,
    termios: *mut *mut ktermios,
    driver_state: *mut (),
    ops: *const tty_operations,
    tty_drivers: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ld_semaphore {
    count: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    wait_readers: u32,
    read_wait: list_head,
    write_wait: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc_ops {
    name: *mut u8,
    num: i32,
    open: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    read: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut u8, u64, *mut *mut (), u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *const u8, u64) -> i64>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    poll: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut poll_table_struct) -> u32>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    receive_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    dcd_change: Option<unsafe extern "C" fn(*mut tty_struct, bool)>,
    receive_buf2: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc {
    ops: *const tty_ldisc_ops,
    tty: *mut tty_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kfifo {
    r#in: u32,
    out: u32,
    mask: u32,
    esize: u32,
    data: *mut (),
}


// DECLARE_KFIFO_PTR - macro to declare a fifo pointer object
// @fifo: name of the declared fifo
// @type: type of the fifo elements
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_buffer {
    __slate_anon_0: __SlateRecord_tty_buffer_h_11,
    used: u32,
    size: u32,
    commit: u32,
    lookahead: u32,
    read: u32,
    flags: bool,
    __slate_pad_7: [u8; 3],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_bufhead {
    head: *mut tty_buffer,
    flip_wq: *mut workqueue_struct,
    work: work_struct,
    lock: mutex,
    priority: __SlateRecord_types_h_190,
    sentinel: tty_buffer,
    free: llist_head,
    mem_used: __SlateRecord_types_h_190,
    mem_limit: i32,
    tail: *mut tty_buffer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_operations {
    carrier_raised: Option<unsafe extern "C" fn(*mut tty_port) -> bool>,
    dtr_rts: Option<unsafe extern "C" fn(*mut tty_port, bool)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_port)>,
    activate: Option<unsafe extern "C" fn(*mut tty_port, *mut tty_struct) -> i32>,
    destruct: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_client_operations {
    receive_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port {
    buf: tty_bufhead,
    tty: *mut tty_struct,
    itty: *mut tty_struct,
    ops: *const tty_port_operations,
    client_ops: *const tty_port_client_operations,
    lock: spinlock,
    blocked_open: i32,
    count: i32,
    open_wait: wait_queue_head,
    delta_msr_wait: wait_queue_head,
    flags: u64,
    iflags: u64,
    __slate_bits_0: __slate_bits::__SlateBits840U0,
    mutex: mutex,
    buf_mutex: mutex,
    xmit_buf: *mut u8,
    xmit_fifo: __SlateRecord_tty_port_h_118,
    close_delay: u32,
    closing_wait: u32,
    drain_delay: i32,
    kref: kref,
    client_data: *mut (),
}


// tty_port::iflags bits -- use atomic bit ops
// device is initialized
// device is suspended
// device is open
// uart drivers: use the uart_port::status field and the UPSTAT_* defines
// for s/w-based flow control steering and carrier detection status
// h/w flow control enabled
// carrier detect enabled
// device exclusively opened by
// kernel
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
    kref: kref,
    index: i32,
    dev: *mut device,
    driver: *mut tty_driver,
    port: *mut tty_port,
    ops: *const tty_operations,
    ldisc: *mut tty_ldisc,
    ldisc_sem: ld_semaphore,
    atomic_write_lock: mutex,
    legacy_mutex: mutex,
    throttle_mutex: mutex,
    termios_rwsem: rw_semaphore,
    winsize_mutex: mutex,
    termios: ktermios,
    termios_locked: ktermios,
    name: [u8; 64],
    flags: u64,
    count: i32,
    receive_room: u32,
    winsize: winsize,
    flow: __SlateRecord_tty_h_212,
    ctrl: __SlateRecord_tty_h_218,
    hw_stopped: bool,
    closing: bool,
    flow_change: i32,
    link: *mut tty_struct,
    fasync: *mut fasync_struct,
    write_wait: wait_queue_head,
    read_wait: wait_queue_head,
    break_wait: wait_queue_head,
    hangup_work: work_struct,
    disc_data: *mut (),
    driver_data: *mut (),
    files_lock: spinlock,
    write_cnt: i32,
    write_buf: *mut u8,
    tty_files: list_head,
    SAK_work: work_struct,
}


fn __do_trace_xhci_dbg_quirks(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_541: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_541) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_542: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_542) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_544() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_quirks)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_543: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_543;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_quirks as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_quirks: __SlateAlign16<[u8; 16]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 113u8, 117u8, 105u8, 114u8, 107u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_quirks_545: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_quirks) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_ring.trace_xhci_dbg_quirks")]
extern "C" fn trace_xhci_dbg_quirks(vaf: *mut va_format) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_quirks.key) }) != 0 {
__do_trace_xhci_dbg_quirks(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_546: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_reset_ep(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_548: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_548) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_549: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_549) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_551() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_reset_ep)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_550: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_550;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_reset_ep as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_reset_ep: __SlateAlign16<[u8; 18]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 114u8, 101u8, 115u8, 101u8, 116u8, 95u8, 101u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_reset_ep_552: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_reset_ep) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_ring.trace_xhci_dbg_reset_ep")]
extern "C" fn trace_xhci_dbg_reset_ep(vaf: *mut va_format) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_reset_ep.key) }) != 0 {
__do_trace_xhci_dbg_reset_ep(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_553: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_cancel_urb(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_555: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_555) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_556: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_556) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_558() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_cancel_urb)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_557: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_557;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_cancel_urb as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_cancel_urb: __SlateAlign16<[u8; 20]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 99u8, 97u8, 110u8, 99u8, 101u8, 108u8, 95u8, 117u8, 114u8, 98u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_cancel_urb_559: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_cancel_urb) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_ring.trace_xhci_dbg_cancel_urb")]
extern "C" fn trace_xhci_dbg_cancel_urb(vaf: *mut va_format) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_cancel_urb.key) }) != 0 {
__do_trace_xhci_dbg_cancel_urb(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_560: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_ring_expansion(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_569: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_569) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_570: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_570) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_572() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_ring_expansion)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_571: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_571;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_ring_expansion as *const ()) };
unsafe { __v12.unwrap()(__data, vaf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_dbg_ring_expansion: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 101u8, 120u8, 112u8, 97u8, 110u8, 115u8, 105u8, 111u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_ring_expansion_573: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_ring_expansion) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci_ring.trace_xhci_dbg_ring_expansion")]
extern "C" fn trace_xhci_dbg_ring_expansion(vaf: *mut va_format) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_ring_expansion.key) }) != 0 {
__do_trace_xhci_dbg_ring_expansion(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_574: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_event(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
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
let mut __UNIQUE_ID_guard_583: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_583) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_584: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_584) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_586() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_event)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_585: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_585;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>>(__SCT__tp_func_xhci_handle_event as *const ()) };
unsafe { __v12.unwrap()(__data, ring, trb, dma) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_event: __SlateAlign16<[u8; 18]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 101u8, 118u8, 101u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_event_587: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_event) }) as *mut ();

fn trace_xhci_handle_event(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_event.key) }) != 0 {
__do_trace_xhci_handle_event(ring, trb, dma);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_588: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_command(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
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
let mut __UNIQUE_ID_guard_590: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_590) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_591: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_591) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_593() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_command)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_592: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_592;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>>(__SCT__tp_func_xhci_handle_command as *const ()) };
unsafe { __v12.unwrap()(__data, ring, trb, dma) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_command: __SlateAlign16<[u8; 20]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 111u8, 109u8, 109u8, 97u8, 110u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_command_594: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_command) }) as *mut ();

fn trace_xhci_handle_command(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_command.key) }) != 0 {
__do_trace_xhci_handle_command(ring, trb, dma);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_595: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_transfer(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
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
let mut __UNIQUE_ID_guard_597: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_597) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_598: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_598) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_600() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_transfer)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_599: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_599;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>>(__SCT__tp_func_xhci_handle_transfer as *const ()) };
unsafe { __v12.unwrap()(__data, ring, trb, dma) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_transfer: __SlateAlign16<[u8; 21]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 116u8, 114u8, 97u8, 110u8, 115u8, 102u8, 101u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_transfer_601: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_transfer) }) as *mut ();

fn trace_xhci_handle_transfer(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_transfer.key) }) != 0 {
__do_trace_xhci_handle_transfer(ring, trb, dma);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_602: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_queue_trb(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
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
let mut __UNIQUE_ID_guard_604: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_604) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_605: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_605) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_607() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_queue_trb)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_606: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_606;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring, *mut xhci_generic_trb, u64) -> i32>>(__SCT__tp_func_xhci_queue_trb as *const ()) };
unsafe { __v12.unwrap()(__data, ring, trb, dma) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_queue_trb: [u8; 15] = [120, 104, 99, 105, 95, 113, 117, 101, 117, 101, 95, 116, 114, 98, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_queue_trb_608: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_queue_trb) }) as *mut ();

fn trace_xhci_queue_trb(ring: *mut xhci_ring, trb: *mut xhci_generic_trb, dma: u64) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_queue_trb.key) }) != 0 {
__do_trace_xhci_queue_trb(ring, trb, dma);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_609: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_urb_giveback(urb: *mut urb) {
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
let mut __UNIQUE_ID_guard_674: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_674) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_675: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_675) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_urb_giveback)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_676: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_676;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>>(__SCT__tp_func_xhci_urb_giveback as *const ()) };
unsafe { __v12.unwrap()(__data, urb) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_urb_giveback: __SlateAlign16<[u8; 18]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 117u8, 114u8, 98u8, 95u8, 103u8, 105u8, 118u8, 101u8, 98u8, 97u8, 99u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_urb_giveback_678: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_urb_giveback) }) as *mut ();

fn trace_xhci_urb_giveback(urb: *mut urb) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_urb_giveback.key) }) != 0 {
__do_trace_xhci_urb_giveback(urb);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_679: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_set_deq_stream(info: *mut xhci_stream_info, stream_id: u32) {
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
let mut __UNIQUE_ID_guard_695: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_695) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_696: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_696) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_698() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq_stream)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_697: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_697;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_stream_info, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_stream_info, u32) -> i32>>(__SCT__tp_func_xhci_handle_cmd_set_deq_stream as *const ()) };
unsafe { __v12.unwrap()(__data, info, stream_id) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_set_deq_stream: __SlateAlign16<[u8; 31]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 113u8, 95u8, 115u8, 116u8, 114u8, 101u8, 97u8, 109u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_set_deq_stream_699: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_set_deq_stream) }) as *mut ();

fn trace_xhci_handle_cmd_set_deq_stream(info: *mut xhci_stream_info, stream_id: u32) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq_stream.key) }) != 0 {
__do_trace_xhci_handle_cmd_set_deq_stream(info, stream_id);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_700: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_stop_ep(ctx: *mut xhci_ep_ctx) {
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
let mut __UNIQUE_ID_guard_702: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_702) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_703: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_703) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_705() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_stop_ep)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_704: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_704;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_stop_ep as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_stop_ep: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 115u8, 116u8, 111u8, 112u8, 95u8, 101u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_stop_ep_706: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_stop_ep) }) as *mut ();

fn trace_xhci_handle_cmd_stop_ep(ctx: *mut xhci_ep_ctx) {
if static_branch_unlikely__18!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_stop_ep.key) }) != 0 {
__do_trace_xhci_handle_cmd_stop_ep(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_707: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_set_deq_ep(ctx: *mut xhci_ep_ctx) {
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
let mut __UNIQUE_ID_guard_709: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_709) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_710: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_710) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_712() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq_ep)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_711: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_711;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_set_deq_ep as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_set_deq_ep: __SlateAlign16<[u8; 27]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 113u8, 95u8, 101u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_set_deq_ep_713: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_set_deq_ep) }) as *mut ();

fn trace_xhci_handle_cmd_set_deq_ep(ctx: *mut xhci_ep_ctx) {
if static_branch_unlikely__14!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq_ep.key) }) != 0 {
__do_trace_xhci_handle_cmd_set_deq_ep(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_714: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_reset_ep(ctx: *mut xhci_ep_ctx) {
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
let mut __UNIQUE_ID_guard_716: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_716) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_717: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_717) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_719() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_reset_ep)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_718: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_718;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_reset_ep as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_reset_ep: __SlateAlign16<[u8; 25]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 114u8, 101u8, 115u8, 101u8, 116u8, 95u8, 101u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_reset_ep_720: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_reset_ep) }) as *mut ();

fn trace_xhci_handle_cmd_reset_ep(ctx: *mut xhci_ep_ctx) {
if static_branch_unlikely__15!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_reset_ep.key) }) != 0 {
__do_trace_xhci_handle_cmd_reset_ep(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_721: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_config_ep(ctx: *mut xhci_ep_ctx) {
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
let mut __UNIQUE_ID_guard_723: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_723) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_724: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_724) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_config_ep)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_725: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_725;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_config_ep as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_config_ep: __SlateAlign16<[u8; 26]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 99u8, 111u8, 110u8, 102u8, 105u8, 103u8, 95u8, 101u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_config_ep_727: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_config_ep) }) as *mut ();

fn trace_xhci_handle_cmd_config_ep(ctx: *mut xhci_ep_ctx) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_config_ep.key) }) != 0 {
__do_trace_xhci_handle_cmd_config_ep(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_728: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_disable_slot(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_751: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_751) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_752: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_752) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_754() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_disable_slot)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_753: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_753;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_disable_slot as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_disable_slot: __SlateAlign16<[u8; 29]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 100u8, 105u8, 115u8, 97u8, 98u8, 108u8, 101u8, 95u8, 115u8, 108u8, 111u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_disable_slot_755: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_disable_slot) }) as *mut ();

fn trace_xhci_handle_cmd_disable_slot(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_disable_slot.key) }) != 0 {
__do_trace_xhci_handle_cmd_disable_slot(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_756: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_addr_dev(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_772: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_772) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_773: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_773) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_775() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_addr_dev)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_774: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_774;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_addr_dev as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_addr_dev: __SlateAlign16<[u8; 25]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 97u8, 100u8, 100u8, 114u8, 95u8, 100u8, 101u8, 118u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_addr_dev_776: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_addr_dev) }) as *mut ();

fn trace_xhci_handle_cmd_addr_dev(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__17!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_addr_dev.key) }) != 0 {
__do_trace_xhci_handle_cmd_addr_dev(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_777: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_reset_dev(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_779: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_779) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_780: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_780) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_782() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_reset_dev)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_781: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_781;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_reset_dev as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_reset_dev: __SlateAlign16<[u8; 26]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 114u8, 101u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 118u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_reset_dev_783: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_reset_dev) }) as *mut ();

fn trace_xhci_handle_cmd_reset_dev(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__16!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_reset_dev.key) }) != 0 {
__do_trace_xhci_handle_cmd_reset_dev(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_784: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_cmd_set_deq(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_786: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_786) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_787: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_787) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_789() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_788: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_788;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_handle_cmd_set_deq as *const ()) };
unsafe { __v12.unwrap()(__data, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_cmd_set_deq: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 99u8, 109u8, 100u8, 95u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 113u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_cmd_set_deq_790: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_cmd_set_deq) }) as *mut ();

fn trace_xhci_handle_cmd_set_deq(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_cmd_set_deq.key) }) != 0 {
__do_trace_xhci_handle_cmd_set_deq(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_791: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_inc_enq(ring: *mut xhci_ring) {
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
let mut __UNIQUE_ID_guard_835: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_835) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_836: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_836) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_838() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_inc_enq)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_837: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_837;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>>(__SCT__tp_func_xhci_inc_enq as *const ()) };
unsafe { __v12.unwrap()(__data, ring) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_inc_enq: [u8; 13] = [120, 104, 99, 105, 95, 105, 110, 99, 95, 101, 110, 113, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_inc_enq_839: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_inc_enq) }) as *mut ();

fn trace_xhci_inc_enq(ring: *mut xhci_ring) {
if static_branch_unlikely__20!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_inc_enq.key) }) != 0 {
__do_trace_xhci_inc_enq(ring);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_840: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_inc_deq(ring: *mut xhci_ring) {
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
let mut __UNIQUE_ID_guard_842: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_842) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_843: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_843) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_845() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_inc_deq)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_844: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_844;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ring) -> i32>>(__SCT__tp_func_xhci_inc_deq as *const ()) };
unsafe { __v12.unwrap()(__data, ring) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_inc_deq: [u8; 13] = [120, 104, 99, 105, 95, 105, 110, 99, 95, 100, 101, 113, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_inc_deq_846: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_inc_deq) }) as *mut ();

fn trace_xhci_inc_deq(ring: *mut xhci_ring) {
if static_branch_unlikely__22!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_inc_deq.key) }) != 0 {
__do_trace_xhci_inc_deq(ring);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_847: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_handle_port_status(port: *mut xhci_port, portsc: u32) {
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
let mut __UNIQUE_ID_guard_849: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_849) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_850: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_850) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_852() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_handle_port_status)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_851: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_851;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_port, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_port, u32) -> i32>>(__SCT__tp_func_xhci_handle_port_status as *const ()) };
unsafe { __v12.unwrap()(__data, port, portsc) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_handle_port_status: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 97u8, 110u8, 100u8, 108u8, 101u8, 95u8, 112u8, 111u8, 114u8, 116u8, 95u8, 115u8, 116u8, 97u8, 116u8, 117u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_handle_port_status_853: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_handle_port_status) }) as *mut ();

fn trace_xhci_handle_port_status(port: *mut xhci_port, portsc: u32) {
if static_branch_unlikely__23!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_handle_port_status.key) }) != 0 {
__do_trace_xhci_handle_port_status(port, portsc);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_854: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_ring_ep_doorbell(slot: u32, doorbell: u32) {
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
let mut __UNIQUE_ID_guard_877: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_877) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_878: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_878) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_880() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_ring_ep_doorbell)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_879: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_879;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), u32, u32) -> i32>>(__SCT__tp_func_xhci_ring_ep_doorbell as *const ()) };
unsafe { __v12.unwrap()(__data, slot, doorbell) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_ring_ep_doorbell: __SlateAlign16<[u8; 22]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 101u8, 112u8, 95u8, 100u8, 111u8, 111u8, 114u8, 98u8, 101u8, 108u8, 108u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_ring_ep_doorbell_881: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_ring_ep_doorbell) }) as *mut ();

fn trace_xhci_ring_ep_doorbell(slot: u32, doorbell: u32) {
if static_branch_unlikely__21!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_ring_ep_doorbell.key) }) != 0 {
__do_trace_xhci_ring_ep_doorbell(slot, doorbell);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_882: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_ring_host_doorbell(slot: u32, doorbell: u32) {
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
let mut __UNIQUE_ID_guard_884: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_884) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_885: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_885) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_887() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_ring_host_doorbell)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_886: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_886;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), u32, u32) -> i32>>(__SCT__tp_func_xhci_ring_host_doorbell as *const ()) };
unsafe { __v12.unwrap()(__data, slot, doorbell) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_ring_host_doorbell: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 114u8, 105u8, 110u8, 103u8, 95u8, 104u8, 111u8, 115u8, 116u8, 95u8, 100u8, 111u8, 111u8, 114u8, 98u8, 101u8, 108u8, 108u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_ring_host_doorbell_888: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_ring_host_doorbell) }) as *mut ();

fn trace_xhci_ring_host_doorbell(slot: u32, doorbell: u32) {
if static_branch_unlikely__19!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_ring_host_doorbell.key) }) != 0 {
__do_trace_xhci_ring_host_doorbell(slot, doorbell);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_889: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// Ring initialization rules:
// 1. Each segment is initialized to zero, except for link TRBs.
// 2. Ring cycle state = 0.  This represents Producer Cycle State (PCS) or
//    Consumer Cycle State (CCS), depending on ring function.
// 3. Enqueue pointer = dequeue pointer = address of first TRB in the segment.
//
// Ring behavior rules:
// 1. A ring is empty if enqueue == dequeue.  This means there will always be at
//    least one free TRB in the ring.  This is useful if you want to turn that
//    into a link TRB and expand the ring.
// 2. When incrementing an enqueue or dequeue pointer, if the next TRB is a
//    link TRB, then load the pointer with the address in the link TRB.  If the
//    link TRB had its toggle bit set, you may need to update the ring cycle
//    state (see cycle bit rules).  You may have to do this multiple times
//    until you reach a non-link TRB.
// 3. A ring is full if enqueue++ (for the definition of increment above)
//    equals the dequeue pointer.
//
// Cycle bit rules:
// 1. When a consumer increments a dequeue pointer and encounters a toggle bit
//    in a link TRB, it must toggle the ring cycle state.
// 2. When a producer increments an enqueue pointer and encounters a toggle bit
//    in a link TRB, it must toggle the ring cycle state.
//
// Producer rules:
// 1. Check if ring is full before you enqueue.
// 2. Write the ring cycle state to the cycle bit in the TRB you're enqueuing.
//    Update enqueue pointer between each write (which may update the ring
//    cycle state).
// 3. Notify consumer.  If SW is producer, it rings the doorbell for command
//    and endpoint rings.  If HC is the producer for the event ring,
//    and it generates an interrupt according to interrupt modulation rules.
//
// Consumer rules:
// 1. Check if TRB belongs to you.  If the cycle bit == your ring cycle state,
//    the TRB is owned by the consumer.
// 2. Update dequeue pointer (which may update the ring cycle state) and
//    continue processing TRBs until you reach a TRB which is not owned by you.
// 3. Notify the producer.  SW is the consumer for the event ring, and it
//   updates event ring dequeue pointer.  HC is the consumer for the command and
//   endpoint rings; it generates events on the event ring for these.
/// Returns zero if the TRB isn't in this segment, otherwise it returns the DMA
/// address of the TRB.
#[unsafe(no_mangle)]
extern "C" fn xhci_trb_virt_to_dma(seg: *mut xhci_segment, trb: *mut xhci_trb) -> u64 {
let mut segment_offset: u64 = 0;
if seg.is_null() || trb.is_null() || trb < unsafe { (*seg).trbs } {
return 0;
}
// offset in TRBs
segment_offset = ((unsafe { trb.offset_from((*seg).trbs as *mut xhci_trb) }) as i64) as u64;
if segment_offset >= 256 {
return 0;
}
unsafe { (*seg).dma }.wrapping_add(segment_offset.wrapping_mul(core::mem::size_of::<xhci_trb>() as u64))
}

fn xhci_dma_to_trb(start_seg: *mut xhci_segment, dma: u64, match_seg: *mut *mut xhci_segment) -> *mut xhci_trb {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
seg = start_seg;
while !seg.is_null() {
if in_range__2!(dma, unsafe { (*seg).dma }, 4096i32) != 0 {
if !match_seg.is_null() {
unsafe {
*match_seg = seg;
}
}
return unsafe { (*seg).trbs }.wrapping_offset((dma.wrapping_sub(unsafe { (*seg).dma }) / (core::mem::size_of::<xhci_trb>() as u64)) as isize);
}
seg = if (unsafe { (*seg).next }) != start_seg { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
core::ptr::null_mut::<xhci_trb>()
}

fn trb_is_noop(trb: *mut xhci_trb) -> bool {
TRB_TYPE_NOOP_LE32!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) })
}

fn trb_is_link(trb: *mut xhci_trb) -> bool {
TRB_TYPE_LINK_LE32!(unsafe { (*trb).link.control })
}

fn last_trb_on_seg(seg: *mut xhci_segment, trb: *mut xhci_trb) -> bool {
trb == unsafe { (*seg).trbs }.wrapping_offset(255i32 as isize)
}

fn last_trb_on_ring(ring: *mut xhci_ring, seg: *mut xhci_segment, trb: *mut xhci_trb) -> bool {
last_trb_on_seg(seg, trb) && (unsafe { (*seg).next }) == unsafe { (*ring).first_seg }
}

fn link_trb_toggles_cycle(trb: *mut xhci_trb) -> bool {
((unsafe { (*trb).link.control }) as u64) & BIT!(1i32) != 0
}

fn last_td_in_urb(td: *mut xhci_td) -> bool {
let urb_priv: *mut urb_priv = (unsafe { (*(*td).urb).hcpriv }) as *mut urb_priv;
(unsafe { (*urb_priv).num_tds_done }) == unsafe { (*urb_priv).num_tds }
}

fn unhandled_event_trb(ring: *mut xhci_ring) -> bool {
((unsafe { (*(*ring).dequeue).event_cmd.flags }) as u64) & BIT!(0i32) == ((unsafe { (*ring).cycle_state }) as u64)
}

fn inc_td_cnt(urb: *mut urb) {
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let __v1: i32 = unsafe { (*urb_priv).num_tds_done };
unsafe {
(*urb_priv).num_tds_done = __v1.wrapping_add(1);
}
}

fn trb_to_noop(trb: *mut xhci_trb, noop_type: u32, unchain_links: bool) {
if trb_is_link(trb) {
if unchain_links {
let __v1: u32 = unsafe { (*trb).link.control };
let __v2: u32 = __v1 & (!BIT!(4i32) as u32);
unsafe {
(*trb).link.control = __v2;
}
}
} else {
unsafe {
*((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = 0;
}
unsafe {
*((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = 0;
}
unsafe {
*((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) = 0;
}
// Preserve only the cycle bit of this TRB
let __v3: *mut u32 = unsafe { (*trb).generic.field.as_mut_ptr() as *mut u32 }.wrapping_offset(3 as isize);
let __v4: u32 = unsafe { *__v3 };
let __v5: u32 = __v4 & (BIT!(0i32) as u32);
unsafe {
*__v3 = __v5;
}
let __v6: *mut u32 = unsafe { (*trb).generic.field.as_mut_ptr() as *mut u32 }.wrapping_offset(3 as isize);
let __v7: u32 = unsafe { *__v6 };
let __v8: u32 = __v7 | TRB_TYPE!(noop_type);
unsafe {
*__v6 = __v8;
}
}
}

fn trb_to_pos(seg: *mut xhci_segment, trb: *mut xhci_trb) -> u32 {
(unsafe { (*seg).num }.wrapping_mul(256) as i64).wrapping_add((unsafe { trb.offset_from((*seg).trbs as *mut xhci_trb) }) as i64) as u32
}

/// Updates trb to point to the next TRB in the ring, and updates seg if the next
/// TRB is in a new segment.  This does not skip over link TRBs, and it does not
/// effect the ring dequeue or enqueue pointers.
fn next_trb(seg: &mut *mut xhci_segment, trb: &mut *mut xhci_trb) {
let __v0: bool = trb_is_link(unsafe { *(trb as *mut *mut xhci_trb) }) || last_trb_on_seg(unsafe { *(seg as *mut *mut xhci_segment) }, unsafe { *(trb as *mut *mut xhci_trb) });
if __v0 {
unsafe {
*(seg as *mut *mut xhci_segment) = (*(*(seg as *mut *mut xhci_segment))).next;
}
unsafe {
*(trb as *mut *mut xhci_trb) = (*(*(seg as *mut *mut xhci_segment))).trbs;
}
} else {
let __v2: *mut xhci_trb = unsafe { *trb };
unsafe {
*trb = __v2.wrapping_offset(1 as isize);
}
}
}

/// See Cycle bit rules. SW is the consumer for the event ring only.
#[unsafe(no_mangle)]
extern "C" fn inc_deq(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
let mut link_trb_count: u32 = 0;
// event ring doesn't have link trbs, check for last trb
if ((unsafe { (*ring).r#type }) as u32) == 6 {
if !last_trb_on_seg(unsafe { (*ring).deq_seg }, unsafe { (*ring).dequeue }) {
let __v1: *mut xhci_trb = unsafe { (*ring).dequeue };
unsafe {
(*ring).dequeue = __v1.wrapping_offset(1 as isize);
}
return;
}
if last_trb_on_ring(ring, unsafe { (*ring).deq_seg }, unsafe { (*ring).dequeue }) {
unsafe {
(*ring).cycle_state = (*ring).cycle_state ^ 1;
}
}
unsafe {
(*ring).deq_seg = (*(*ring).deq_seg).next;
}
unsafe {
(*ring).dequeue = (*(*ring).deq_seg).trbs;
}
trace_xhci_inc_deq(ring);
return;
}
// All other rings have link trbs
if !trb_is_link(unsafe { (*ring).dequeue }) {
if last_trb_on_seg(unsafe { (*ring).deq_seg }, unsafe { (*ring).dequeue }) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Missing link TRB at end of segment\n".as_ptr() as *const u8) };
} else {
let __v8: *mut xhci_trb = unsafe { (*ring).dequeue };
unsafe {
(*ring).dequeue = __v8.wrapping_offset(1 as isize);
}
}
}
while trb_is_link(unsafe { (*ring).dequeue }) {
unsafe {
(*ring).deq_seg = (*(*ring).deq_seg).next;
}
unsafe {
(*ring).dequeue = (*(*ring).deq_seg).trbs;
}
trace_xhci_inc_deq(ring);
let __v10: u32 = link_trb_count;
link_trb_count = __v10.wrapping_add(1);
if __v10 > unsafe { (*ring).num_segs } {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ring is an endless link TRB loop\n".as_ptr() as *const u8) };
break;
}
}
return;
}

/// If enqueue points at a link TRB, follow links until an ordinary TRB is reached.
/// Toggle the cycle bit of passed link TRBs and optionally chain them.
fn inc_enq_past_link(xhci: *mut xhci_hcd, ring: *mut xhci_ring, chain: u32) {
let mut link_trb_count: u32 = 0;
while trb_is_link(unsafe { (*ring).enqueue }) {
// Section 6.4.4.1 of the 0.95 spec says link TRBs cannot have the chain bit
// set, but other sections talk about dealing with the chain bit set. This was
// fixed in the 0.96 specification errata, but we have to assume that all 0.95
// xHCI hardware can't handle the chain bit being cleared on a link TRB.
//
// On 0.95 and some 0.96 HCs the chain bit is set once at segment initalization
// and never changed here. On all others, modify it as requested by the caller.
if !xhci_link_chain_quirk(xhci, unsafe { (*ring).r#type }) {
let __v0: *mut xhci_trb = unsafe { (*ring).enqueue };
let __v1: u32 = unsafe { (*__v0).link.control };
let __v2: u32 = __v1 & (!BIT!(4i32) as u32);
unsafe {
(*__v0).link.control = __v2;
}
let __v3: *mut xhci_trb = unsafe { (*ring).enqueue };
unsafe {
(*__v3).link.control = (*__v3).link.control | chain;
}
}
// Give this link TRB to the hardware
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v8: *mut xhci_trb = unsafe { (*ring).enqueue };
let __v9: u32 = unsafe { (*__v8).link.control };
let __v10: u32 = __v9 ^ (BIT!(0i32) as u32);
unsafe {
(*__v8).link.control = __v10;
}
// Toggle the cycle bit after the last ring segment.
if link_trb_toggles_cycle(unsafe { (*ring).enqueue }) {
unsafe {
(*ring).cycle_state = (*ring).cycle_state ^ 1;
}
}
unsafe {
(*ring).enq_seg = (*(*ring).enq_seg).next;
}
unsafe {
(*ring).enqueue = (*(*ring).enq_seg).trbs;
}
trace_xhci_inc_enq(ring);
let __v14: u32 = link_trb_count;
link_trb_count = __v14.wrapping_add(1);
if __v14 > unsafe { (*ring).num_segs } {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Link TRB loop at enqueue\n".as_ptr() as *const u8) };
break;
}
}
}

/// See Cycle bit rules. SW is the consumer for the event ring only.
///
/// If we've just enqueued a TRB that is in the middle of a TD (meaning the
/// chain bit is set), then set the chain bit in all the following link TRBs.
/// If we've enqueued the last TRB in a TD, make sure the following link TRBs
/// have their chain bit cleared (so that each Link TRB is a separate TD).
///
/// @more_trbs_coming:	Will you enqueue more TRBs before calling
/// 			prepare_transfer()?
fn inc_enq(xhci: *mut xhci_hcd, ring: *mut xhci_ring, more_trbs_coming: bool) {
let chain: u32 = (((unsafe { *((*(*ring).enqueue).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) }) as u64) & BIT!(4i32)) as u32;
if last_trb_on_seg(unsafe { (*ring).enq_seg }, unsafe { (*ring).enqueue }) {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Tried to move enqueue past ring segment\n".as_ptr() as *const u8) };
return;
}
let __v2: *mut xhci_trb = unsafe { (*ring).enqueue };
unsafe {
(*ring).enqueue = __v2.wrapping_offset(1 as isize);
}
// If we are in the middle of a TD or the caller plans to enqueue more
// TDs as one transfer (eg. control), traverse any link TRBs right now.
// Otherwise, enqueue can stay on a link until the next prepare_ring().
// This avoids enqueue entering deq_seg and simplifies ring expansion.
if trb_is_link(unsafe { (*ring).enqueue }) && (chain != 0 || more_trbs_coming) {
inc_enq_past_link(xhci, ring, chain);
}
}

fn dma_in_range(dma: u64, start_seg: *mut xhci_segment, start_trb: *mut xhci_trb, end_seg: *mut xhci_segment, end_trb: *mut xhci_trb) -> bool {
let mut pos: u32 = 0;
let mut start: u32 = 0;
let mut end: u32 = 0;
let mut pos_seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let pos_trb: *mut xhci_trb = xhci_dma_to_trb(start_seg, dma, core::ptr::addr_of_mut!(pos_seg));
// Is the trb dma address even part of the whole ring?
if pos_trb.is_null() {
return false;
}
pos = trb_to_pos(pos_seg, pos_trb);
start = trb_to_pos(start_seg, start_trb);
end = trb_to_pos(end_seg, end_trb);
// end position is smaller than start, search range wraps around
if end < start {
return !(pos > end && pos < start);
}
pos >= start && pos <= end
}

/// If the suspect DMA address is a TRB in this TD, this function returns true
fn trb_in_td(td: *mut xhci_td, suspect_dma: u64) -> bool {
dma_in_range(suspect_dma, unsafe { (*td).start_seg }, unsafe { (*td).start_trb }, unsafe { (*td).end_seg }, unsafe { (*td).end_trb })
}

/// Return number of free normal TRBs from enqueue to dequeue pointer on ring.
/// Not counting an assumed link TRB at end of each TRBS_PER_SEGMENT sized segment.
/// Only for transfer and command rings where driver is the producer, not for
/// event rings.
#[unsafe(no_mangle)]
extern "C" fn xhci_num_trbs_free(ring: *mut xhci_ring) -> u32 {
let mut enq_seg: *mut xhci_segment = unsafe { (*ring).enq_seg };
let mut enq: *mut xhci_trb = unsafe { (*ring).enqueue };
let mut free: u32 = 0;
let mut i: i32 = 0;
// Ring might be empty even if enq != deq if enq is left on a link trb
if trb_is_link(enq) {
enq_seg = unsafe { (*enq_seg).next };
enq = unsafe { (*enq_seg).trbs };
}
// Empty ring, common case, don't walk the segments
if enq == unsafe { (*ring).dequeue } {
return unsafe { (*ring).num_segs }.wrapping_mul(255u32);
}
loop {
if (unsafe { (*ring).deq_seg }) == enq_seg && (unsafe { (*ring).dequeue }) >= enq {
return (free as i64).wrapping_add((unsafe { unsafe { (*ring).dequeue }.offset_from(enq as *mut xhci_trb) }) as i64) as u32;
}
let last_on_seg: *mut xhci_trb = unsafe { (*enq_seg).trbs }.wrapping_offset(255i32 as isize);
let __v0: u32 = free;
free = (__v0 as i64).wrapping_add((unsafe { last_on_seg.offset_from(enq as *mut xhci_trb) }) as i64) as u32;
enq_seg = unsafe { (*enq_seg).next };
enq = unsafe { (*enq_seg).trbs };
let __v3: i32 = i;
i = __v3.wrapping_add(1);
if (__v3 as u32) >= unsafe { (*ring).num_segs } {
break;
}
}
free
}

// Check to see if there's room to enqueue num_trbs on the ring and make sure
// enqueue pointer will not advance into dequeue segment. See rules above.
// return number of new segments needed to ensure this.
fn xhci_ring_expansion_needed(xhci: *mut xhci_hcd, ring: *mut xhci_ring, num_trbs: u32) -> u32 {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let mut trbs_past_seg: i32 = 0;
let enq_used: i32 = ((unsafe { unsafe { (*ring).enqueue }.offset_from((*(*ring).enq_seg).trbs as *mut xhci_trb) }) as i64) as i32;
let mut new_segs: i32 = 0;
// how many trbs will be queued past the enqueue segment?
trbs_past_seg = (enq_used as u32).wrapping_add(num_trbs).wrapping_sub(255u32) as i32;
// Consider expanding the ring already if num_trbs fills the current
// segment (i.e. trbs_past_seg == 0), not only when num_trbs goes into
// the next segment. Avoids confusing full ring with special empty ring
// case below
if trbs_past_seg < 0 {
return 0;
}
// Empty ring special case, enqueue stuck on link trb while dequeue advanced
if trb_is_link(unsafe { (*ring).enqueue }) && (unsafe { (*(*(*ring).enq_seg).next).trbs }) == unsafe { (*ring).dequeue } {
return 0;
}
new_segs = 1i32.wrapping_add(trbs_past_seg / 255i32);
seg = unsafe { (*ring).enq_seg };
while new_segs > 0 {
seg = unsafe { (*seg).next };
if seg == unsafe { (*ring).deq_seg } {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Adding %d trbs requires expanding ring by %d segments\n".as_ptr() as *const u8, num_trbs, new_segs) };
}
return new_segs as u32;
}
new_segs = new_segs.wrapping_sub(1);
}
0
}

/// Ring the host controller doorbell after placing a command on the ring
#[unsafe(no_mangle)]
extern "C" fn xhci_ring_cmd_db(xhci: *mut xhci_hcd) {
if ((unsafe { (*xhci).cmd_ring_state }) as u64) & BIT!(0i32) == 0 {
return;
}
trace_xhci_ring_host_doorbell(0u32, 0u32);
writel(0u32, unsafe { (*(*xhci).dba).doorbell.as_mut_ptr() as *mut u32 }.wrapping_offset(0 as isize) as *mut ());
// Flush PCI posted writes
readl(unsafe { (*(*xhci).dba).doorbell.as_mut_ptr() as *mut u32 }.wrapping_offset(0 as isize) as *const ());
}

fn xhci_mod_cmd_timer(xhci: *mut xhci_hcd) -> bool {
mod_delayed_work(unsafe { system_percpu_wq }, unsafe { core::ptr::addr_of_mut!((*xhci).cmd_timer) }, msecs_to_jiffies(unsafe { (*(*xhci).current_cmd).timeout_ms }))
}

fn xhci_next_queued_cmd(xhci: *mut xhci_hcd) -> *mut xhci_command {
let mut __v0: *mut xhci_command;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) };
let mut __v1: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_918() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v1;
let __v3: *mut xhci_command = if pos__ != head__ {
list_entry__25!(pos__, xhci_command)
} else {
core::ptr::null_mut::<xhci_command>()
};
__v0 = __v3;
__v0
}

/// Turn all commands on command ring with status set to "aborted" to no-op trbs.
/// If there are other commands waiting then restart the ring and kick the timer.
/// This must be called with command ring stopped and xhci->lock held.
fn xhci_handle_stopped_cmd_ring(xhci: *mut xhci_hcd, cur_cmd: *mut xhci_command) {
let mut i_cmd: *mut xhci_command = unsafe { core::mem::zeroed() };
// Turn all aborted commands in list to no-ops, then restart
i_cmd = list_first_entry__11!(unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) });
while list_entry_is_head!(i_cmd, unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) }, cmd_list) == 0 {
if (unsafe { (*i_cmd).status }) == 25 {
unsafe {
(*i_cmd).status = 24;
}
trb_to_noop(unsafe { (*i_cmd).command_trb }, 23u32, false);
// caller waiting for completion is called when command
//  completion event is received for these no-op commands
}
i_cmd = list_next_entry__9!(i_cmd, cmd_list);
}
unsafe {
(*xhci).cmd_ring_state = BIT!(0i32) as u32;
}
// ring command ring doorbell to restart the command ring
if (unsafe { (*(*xhci).cmd_ring).dequeue }) != unsafe { (*(*xhci).cmd_ring).enqueue } && ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) == 0 {
unsafe {
(*xhci).current_cmd = cur_cmd;
}
if !cur_cmd.is_null() {
xhci_mod_cmd_timer(xhci);
}
xhci_ring_cmd_db(xhci);
}
}

/// Must be called with xhci->lock held, releases and acquires lock back
fn xhci_abort_cmd_ring(xhci: *mut xhci_hcd, mut flags: u64) -> i32 {
let mut new_seg: *mut xhci_segment = unsafe { (*(*xhci).cmd_ring).deq_seg };
let mut new_deq: *mut xhci_trb = unsafe { (*(*xhci).cmd_ring).dequeue };
let mut crcr: u64 = 0;
let mut ret: i32 = 0;
reinit_completion(unsafe { core::ptr::addr_of_mut!((*xhci).cmd_ring_stop_completion) });
// The control bits like command stop, abort are located in lower
// dword of the command ring control register.
// Some controllers require all 64 bits to be written to abort the ring.
// Make sure the upper dword is valid, pointing to the next command,
// avoiding corrupting the command ring pointer in case the command ring
// is stopped by the time the upper dword is written.
next_trb(unsafe { &mut (*core::ptr::addr_of_mut!(new_seg)) }, unsafe { &mut (*core::ptr::addr_of_mut!(new_deq)) });
if trb_is_link(new_deq) {
next_trb(unsafe { &mut (*core::ptr::addr_of_mut!(new_seg)) }, unsafe { &mut (*core::ptr::addr_of_mut!(new_deq)) });
}
crcr = xhci_trb_virt_to_dma(new_seg, new_deq);
xhci_write_64(xhci, crcr | BIT_ULL!(2i32), unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) });
// Section 4.6.1.2 of xHCI 1.0 spec says software should also time the
// completion of the Command Abort operation. If CRR is not negated in 5
// seconds then driver handles it as if host died (-ENODEV).
// In the future we should distinguish between -ENODEV and -ETIMEDOUT
// and try to recover a -ETIMEDOUT with a host controller reset.
ret = unsafe { xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) as *mut (), BIT_ULL!(3i32) as u32, 0u32, 5000000u64) };
if ret < 0 {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Abort failed to stop command ring: %d\n".as_ptr() as *const u8, ret) };
unsafe { xhci_halt(xhci) };
xhci_hc_died(xhci);
return ret;
}
// Writing the CMD_RING_ABORT bit should cause a cmd completion event,
// however on some host hw the CMD_RING_RUNNING bit is correctly cleared
// but the completion event in never sent. Wait 2 secs (arbitrary
// number) to handle those cases after negation of CMD_RING_RUNNING.
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
ret = (unsafe { wait_for_completion_timeout(core::ptr::addr_of_mut!((*xhci).cmd_ring_stop_completion), msecs_to_jiffies(2000u32)) }) as i32;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
if ret == 0 {
xhci_cleanup_command_queue(xhci);
} else {
xhci_handle_stopped_cmd_ring(xhci, xhci_next_queued_cmd(xhci));
}
0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_ring_ep_doorbell(xhci: *mut xhci_hcd, slot_id: u32, ep_index: u32, stream_id: u32) {
let db_addr: *mut u32 = unsafe { (*(*xhci).dba).doorbell.as_mut_ptr() as *mut u32 }.wrapping_offset(slot_id as isize);
let ep: *mut xhci_virt_ep = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
let ep_state: u32 = unsafe { (*ep).ep_state };
// Don't ring the doorbell for this endpoint if there are pending
// cancellations because we don't want to interrupt processing.
// We don't want to restart any stream rings if there's a set dequeue
// pointer command pending because the device can choose to start any
// stream once the endpoint is on the HW schedule.
if (ep_state as u64) & (BIT!(2i32) | BIT!(0i32) | BIT!(1i32) | BIT!(8i32) | BIT!(9i32)) != 0 {
return;
}
trace_xhci_ring_ep_doorbell(slot_id, DB_VALUE!(ep_index, stream_id));
writel(DB_VALUE!(ep_index, stream_id), db_addr as *mut ());
// flush the write
readl(db_addr as *const ());
}

/// Ring the doorbell for any rings with pending URBs
#[unsafe(no_mangle)]
extern "C" fn xhci_ring_doorbell_for_active_rings(xhci: *mut xhci_hcd, slot_id: u32, ep_index: u32) {
let mut stream_id: u32 = 1;
let mut ep: *mut xhci_virt_ep = unsafe { core::mem::zeroed() };
ep = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
// A ring has pending URBs if its TD list is not empty
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) == 0 {
let __v0: bool = !unsafe { (*ep).ring }.is_null() && list_empty((unsafe { core::ptr::addr_of_mut!((*(*ep).ring).td_list) }) as *const list_head) == 0;
if __v0 {
xhci_ring_ep_doorbell(xhci, slot_id, ep_index, 0u32);
}
return;
}
while stream_id < unsafe { (*(*ep).stream_info).num_streams } {
let stream_info: *mut xhci_stream_info = unsafe { (*ep).stream_info };
if list_empty((unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*stream_info).stream_rings }.wrapping_offset(stream_id as isize))).td_list) }) as *const list_head) == 0 {
xhci_ring_ep_doorbell(xhci, slot_id, ep_index, stream_id);
}
stream_id = stream_id.wrapping_add(1);
}
}

fn xhci_get_virt_ep(xhci: *mut xhci_hcd, slot_id: u32, ep_index: u32) -> *mut xhci_virt_ep {
if slot_id == 0 || slot_id > (((unsafe { (*xhci).max_slots }) as i32) as u32) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Invalid slot_id %u\n".as_ptr() as *const u8, slot_id) };
return core::ptr::null_mut::<xhci_virt_ep>();
}
if ep_index >= 31 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Invalid endpoint index %u\n".as_ptr() as *const u8, ep_index) };
return core::ptr::null_mut::<xhci_virt_ep>();
}
if unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"No xhci virt device for slot_id %u\n".as_ptr() as *const u8, slot_id) };
return core::ptr::null_mut::<xhci_virt_ep>();
}
unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize)
}

fn xhci_virt_ep_to_ring(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, stream_id: u32) -> *mut xhci_ring {
// common case, no streams
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) == 0 {
return unsafe { (*ep).ring };
}
if unsafe { (*ep).stream_info }.is_null() {
return core::ptr::null_mut::<xhci_ring>();
}
if stream_id == 0 || stream_id >= unsafe { (*(*ep).stream_info).num_streams } {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Invalid stream_id %u request for slot_id %u ep_index %u\n".as_ptr() as *const u8, stream_id, (*(*ep).vdev).slot_id, (*ep).ep_index) };
return core::ptr::null_mut::<xhci_ring>();
}
unsafe { *unsafe { (*(*ep).stream_info).stream_rings }.wrapping_offset(stream_id as isize) }
}

/// Get the right ring for the given slot_id, ep_index and stream_id.
/// If the endpoint supports streams, boundary check the URB's stream ID.
/// If the endpoint doesn't support streams, return the singular endpoint ring.
#[unsafe(no_mangle)]
extern "C" fn xhci_triad_to_transfer_ring(xhci: *mut xhci_hcd, slot_id: u32, ep_index: u32, stream_id: u32) -> *mut xhci_ring {
let ep: *mut xhci_virt_ep = xhci_get_virt_ep(xhci, slot_id, ep_index);
if ep.is_null() {
return core::ptr::null_mut::<xhci_ring>();
}
xhci_virt_ep_to_ring(xhci, ep, stream_id)
}

/// Get the hw dequeue pointer xHC stopped on, either directly from the
/// endpoint context, or if streams are in use from the stream context.
/// The returned hw_dequeue contains the lowest four bits with cycle state
/// and possbile stream context type.
fn xhci_get_hw_deq(xhci: *mut xhci_hcd, vdev: *mut xhci_virt_device, ep_index: u32, stream_id: u32) -> u64 {
let ep: *mut xhci_virt_ep = unsafe { (*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) != 0 {
let st_ctx: *mut xhci_stream_ctx = unsafe { (*(*ep).stream_info).stream_ctx_array }.wrapping_offset(stream_id as isize);
return unsafe { (*st_ctx).stream_ring };
}
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*vdev).out_ctx, ep_index) };
unsafe { (*ep_ctx).deq }
}

/// Move the endpoint dequeue pointer to the next queued TD on ring->td_list or
/// to enqueue if no TDs are queued (empty ring)
/// All cancelled TDs on ring->td_list should be moved to ep->cancelled_td_list
/// before calling this function
fn xhci_move_deq_to_next_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, stream_id: u32) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut trb_sct: u32 = unsafe { core::mem::zeroed() };
let mut new_cycle: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(new_cycle) as *mut i32;
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut td: core::mem::MaybeUninit<*mut xhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut xhci_td = core::ptr::addr_of_mut!(td) as *mut *mut xhci_td;
let mut ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ring) as *mut *mut xhci_ring;
let mut cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut xhci_command = core::ptr::addr_of_mut!(cmd) as *mut *mut xhci_command;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
trb_sct = 0;
ret = 0;
*__slate_slot_8 = xhci_virt_ep_to_ring(xhci, ep, stream_id);
if (*__slate_slot_8).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN can't find new dequeue, invalid stream ID %u\n".as_ptr() as *const u8, stream_id);
return 19i32.wrapping_neg();
} else {
if ((*ep).ep_state as u64) & BIT!(0i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Set TR Deq already pending\n".as_ptr() as *const u8);
return 16i32.wrapping_neg();
} else {
// This function gets called from contexts where it cannot sleep
*__slate_slot_9 = xhci_alloc_command(xhci, false, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_9).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Can't alloc Set TR Deq cmd\n".as_ptr() as *const u8);
return 12i32.wrapping_neg();
} else {
// Move dequeue to the beginning of next td, or to enqueue if ring is
// empty. Avoid moving dequeue to a link trb (empty ring) as it causes
// issues on some hosts. In that case advance the enqueue to next segment
// before moving dequeue to it
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_8)).td_list) as *const list_head) != 0 {
if trb_is_link((*(*__slate_slot_8)).enqueue) {
inc_enq_past_link(xhci, *__slate_slot_8, 0u32);
}
(*ep).queued_deq_seg = (*(*__slate_slot_8)).enq_seg;
(*ep).queued_deq_ptr = (*(*__slate_slot_8)).enqueue;
*__slate_slot_5 = (*(*__slate_slot_8)).cycle_state as i32;
} else {
*__slate_slot_7 = list_first_entry__4!(core::ptr::addr_of_mut!((*(*__slate_slot_8)).td_list), xhci_td);
(*ep).queued_deq_seg = (*(*__slate_slot_7)).start_seg;
(*ep).queued_deq_ptr = (*(*__slate_slot_7)).start_trb;
*__slate_slot_5 = ((*((*(*(*__slate_slot_7)).start_trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) as u64) & BIT!(0i32)) as i32;
}
*__slate_slot_6 = xhci_trb_virt_to_dma((*ep).queued_deq_seg, (*ep).queued_deq_ptr);
if *__slate_slot_6 == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Can't find new dequeue dma of seg %p, ptr %p\n".as_ptr() as *const u8, (*ep).queued_deq_seg, (*ep).queued_deq_ptr);
ret = 22i32.wrapping_neg();
} else {
if stream_id != 0 {
trb_sct = SCT_FOR_TRB!(1i32) as u32;
}
ret = queue_command(xhci, *__slate_slot_9, lower_32_bits!(*__slate_slot_6) | trb_sct | (*__slate_slot_5 as u32), upper_32_bits!(*__slate_slot_6), STREAM_ID_FOR_TRB!(stream_id), (SLOT_ID_FOR_TRB__2!(unsafe { (*(*ep).vdev).slot_id }) as u32) | EP_INDEX_FOR_TRB!(unsafe { (*ep).ep_index }) | (TRB_TYPE!(16i32) as u32), false);
if ret >= 0 {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Set TR Deq ptr 0x%llx, cycle %u\n".as_ptr() as *const u8, *__slate_slot_6, *__slate_slot_5);
// Stop the TD queueing code from ringing the doorbell until this
// command completes. The HC won't set the dequeue pointer if the ring
// is running, and ringing the doorbell starts the ring.
let __v1: u32 = (*ep).ep_state;
let __v0: u32 = ((__v1 as u64) | BIT!(0i32)) as u32;
(*ep).ep_state = __v0;
xhci_ring_cmd_db(xhci);
return 0;
}
}
xhci_free_command(xhci, *__slate_slot_9);
(*ep).queued_deq_seg = core::ptr::null_mut::<xhci_segment>();
(*ep).queued_deq_ptr = core::ptr::null_mut::<xhci_trb>();
return ret;
}
}
}
}
}

/// flip_cycle means flip the cycle bit of all but the first and last TRB.
/// (The last TRB actually points to the ring enqueue pointer, which is not part
/// of this TD.)  This is used to remove partially enqueued isoc TDs from a ring.
fn td_to_noop(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, td: *mut xhci_td, flip_cycle: bool) {
let mut unchain_links: bool = false;
let mut seg: *mut xhci_segment = unsafe { (*td).start_seg };
let mut trb: *mut xhci_trb = unsafe { (*td).start_trb };
// link TRBs should now be unchained, but some old HCs expect otherwise
unchain_links = !xhci_link_chain_quirk(xhci, (if !unsafe { (*ep).ring }.is_null() { (unsafe { (*(*ep).ring).r#type }) as u32 } else { 4u32 }) as u32);
while true {
trb_to_noop(trb, 8u32, unchain_links);
// flip cycle if asked to
if flip_cycle && trb != unsafe { (*td).start_trb } && trb != unsafe { (*td).end_trb } {
let __v0: *mut u32 = unsafe { (*trb).generic.field.as_mut_ptr() as *mut u32 }.wrapping_offset(3 as isize);
let __v1: u32 = unsafe { *__v0 };
let __v2: u32 = __v1 ^ (BIT!(0i32) as u32);
unsafe {
*__v0 = __v2;
}
}
if trb == unsafe { (*td).end_trb } {
break;
}
next_trb(unsafe { &mut (*core::ptr::addr_of_mut!(seg)) }, unsafe { &mut (*core::ptr::addr_of_mut!(trb)) });
}
}

fn xhci_giveback_urb_in_irq(xhci: *mut xhci_hcd, cur_td: *mut xhci_td, status: i32) {
let urb: *mut urb = unsafe { (*cur_td).urb };
let urb_priv: *mut urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
let hcd: *mut usb_hcd = bus_to_hcd(unsafe { (*(*urb).dev).bus });
if usb_pipetype!(unsafe { (*urb).pipe }) == 0 {
let __v0: *mut usb_hcd = xhci_to_hcd(xhci);
let __v1: i32 = unsafe { (*__v0).self_.bandwidth_isoc_reqs };
unsafe {
(*__v0).self_.bandwidth_isoc_reqs = __v1.wrapping_sub(1);
}
if (unsafe { (*xhci_to_hcd(xhci)).self_.bandwidth_isoc_reqs }) == 0 {
if (unsafe { (*xhci).quirks }) & BIT_ULL!(3i32) != 0 {
unsafe { usb_amd_quirk_pll_enable() };
}
}
}
unsafe { xhci_urb_free_priv(urb_priv) };
unsafe { usb_hcd_unlink_urb_from_ep(hcd, urb) };
trace_xhci_urb_giveback(urb);
unsafe { usb_hcd_giveback_urb(hcd, urb, status) };
}

fn xhci_unmap_one_bounce_buffer(xhci: *mut xhci_hcd, ring: *mut xhci_ring, td: *mut xhci_td, seg: *mut xhci_segment) {
let mut len: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(len) as *mut u64;
let mut urb: *mut urb = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*xhci_to_hcd(xhci)).self_.sysdev;
urb = (*td).urb;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if usb_urb_dir_out(urb) != 0 {
dma_unmap_single!(dev, unsafe { (*seg).bounce_dma }, unsafe { (*ring).bounce_buf_len }, 1i32);
} else {
dma_unmap_single!(dev, unsafe { (*seg).bounce_dma }, unsafe { (*ring).bounce_buf_len }, 2i32);
// for in transfers we need to copy the data from bounce to sg
if (*urb).num_sgs != 0 {
*__slate_slot_0 = sg_pcopy_from_buffer((*urb).sg, (*urb).num_sgs as u32, (*seg).bounce_buf as *const (), (*seg).bounce_len as u64, (*seg).bounce_offs as i64);
if *__slate_slot_0 != ((*seg).bounce_len as u64) {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Wrong bounce buffer read length: %zu != %d\n".as_ptr() as *const u8, *__slate_slot_0, (*seg).bounce_len);
}
} else {
memcpy(((*urb).transfer_buffer as *const u8).wrapping_offset((*seg).bounce_offs as isize) as *mut (), (*seg).bounce_buf as *const (), (*seg).bounce_len as u64);
}
}
(*seg).bounce_len = 0;
(*seg).bounce_offs = 0;
}
}

fn xhci_unmap_td_bounce_buffer(xhci: *mut xhci_hcd, ring: *mut xhci_ring, td: *mut xhci_td) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
if unsafe { (*td).bounce_seg }.is_null() || ring.is_null() || unsafe { (*td).urb }.is_null() {
return;
}
// td->bounce_seg is the last one bounced, unmap them all
seg = unsafe { (*td).start_seg };
loop {
let __v1: i32 = i;
i = __v1.wrapping_add(1);
if (__v1 as u32) >= unsafe { (*ring).num_segs } {
break;
}
if (unsafe { (*seg).bounce_len }) != 0 {
xhci_unmap_one_bounce_buffer(xhci, ring, td, seg);
}
if seg == unsafe { (*td).bounce_seg } {
break;
}
seg = unsafe { (*seg).next };
}
}

fn xhci_td_cleanup(xhci: *mut xhci_hcd, td: *mut xhci_td, ep_ring: *mut xhci_ring, mut status: i32) {
let mut urb: *mut urb = core::ptr::null_mut::<urb>();
// Clean up the endpoint's TD list
urb = unsafe { (*td).urb };
// if a bounce buffer was used to align this td then unmap it
xhci_unmap_td_bounce_buffer(xhci, ep_ring, td);
// Do one last check of the actual transfer length.
// If the host controller said we transferred more data than the buffer
// length, urb->actual_length will be a very big number (since it's
// unsigned).  Play it safe and say we didn't transfer anything.
if (unsafe { (*urb).actual_length }) > unsafe { (*urb).transfer_buffer_length } {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"URB req %u and actual %u transfer length mismatch\n".as_ptr() as *const u8, (*urb).transfer_buffer_length, (*urb).actual_length) };
unsafe {
(*urb).actual_length = 0;
}
status = 0;
}
// TD might be removed from td_list if we are giving back a cancelled URB
if list_empty((unsafe { core::ptr::addr_of_mut!((*td).td_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*td).td_list) });
}
// Giving back a cancelled URB, or if a slated TD completed anyway
if list_empty((unsafe { core::ptr::addr_of_mut!((*td).cancelled_td_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*td).cancelled_td_list) });
}
inc_td_cnt(urb);
// Giveback the urb when all the tds are completed
if last_td_in_urb(td) {
let __v1: bool = (unsafe { (*urb).actual_length }) != unsafe { (*urb).transfer_buffer_length } && (unsafe { (*urb).transfer_flags }) & 1 != 0 || status != 0 && usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) == 0;
// set isoc urb status to 0 just as EHCI, UHCI, and OHCI
if usb_pipetype!(unsafe { (*urb).pipe }) == 0 {
status = 0;
}
xhci_giveback_urb_in_irq(xhci, td, status);
}
}

/// Give back previous TD and move on to the next TD.
fn xhci_dequeue_td(xhci: *mut xhci_hcd, td: *mut xhci_td, ring: *mut xhci_ring, status: u32) {
unsafe {
(*ring).dequeue = (*td).end_trb;
}
unsafe {
(*ring).deq_seg = (*td).end_seg;
}
inc_deq(xhci, ring);
xhci_td_cleanup(xhci, td, ring, status as i32);
}

/// Complete the cancelled URBs we unlinked from td_list.
fn xhci_giveback_invalidated_tds(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep) {
let mut td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut tmp_td: *mut xhci_td = unsafe { core::mem::zeroed() };
td = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) });
tmp_td = list_next_entry!(td, cancelled_td_list);
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) }, cancelled_td_list) == 0 {
let ring: *mut xhci_ring = xhci_urb_to_transfer_ring(xhci, unsafe { (*td).urb });
if ((unsafe { (*td).cancel_status }) as u32) == 4 {
xhci_td_cleanup(xhci, td, ring, unsafe { (*td).status });
}
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
return;
}
td = tmp_td;
tmp_td = list_next_entry!(tmp_td, cancelled_td_list);
}
}

fn xhci_reset_halted_ep(xhci: *mut xhci_hcd, slot_id: u32, ep_index: u32, reset_type: u32) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_1 = xhci_alloc_command(xhci, false, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_1).is_null() {
ret = 12i32.wrapping_neg();
} else {
ret = xhci_queue_reset_ep(xhci, *__slate_slot_1, slot_id as i32, ep_index, reset_type);
}
if ret != 0 {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR queuing reset endpoint for slot %d ep_index %d, %d\n".as_ptr() as *const u8, slot_id, ep_index, ret);
}
return ret;
}
}

fn xhci_handle_halted_endpoint(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, td: *mut xhci_td, reset_type: u32) -> i32 {
let rhub_port: *mut xhci_port = unsafe { (*(*ep).vdev).rhub_port };
let slot_id: u32 = (unsafe { (*(*ep).vdev).slot_id }) as u32;
let mut err: i32 = 0;
// Avoid resetting endpoint if link is inactive or device disonnected.
// Can cause host hang.
// Device will be reset to recover an inactive link, so don't do anything
if ((unsafe { (*rhub_port).__slate_bits_0.__get_link_inactive() }) as i32) != 0 || ((unsafe { (*rhub_port).__slate_bits_0.__get_connected() }) as i32) == 0 {
let __v1: u32 = unsafe { (*ep).ep_state };
let __v2: u32 = ((__v1 as u64) | BIT!(9i32)) as u32;
unsafe {
(*ep).ep_state = __v2;
}
return 19i32.wrapping_neg();
}
// add td to cancelled list and let reset ep handler take care of it
if (reset_type as u32) == 0 {
let __v4: u32 = unsafe { (*ep).ep_state };
let __v5: u32 = ((__v4 as u64) | BIT!(6i32)) as u32;
unsafe {
(*ep).ep_state = __v5;
}
let __v6: bool = !td.is_null() && list_empty((unsafe { core::ptr::addr_of_mut!((*td).cancelled_td_list) }) as *const list_head) != 0;
if __v6 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*td).cancelled_td_list) }, unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) });
unsafe {
(*td).cancel_status = 1;
}
}
}
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(1i32) != 0 {
return 0;
}
err = xhci_reset_halted_ep(xhci, slot_id, unsafe { (*ep).ep_index }, reset_type);
if err != 0 {
return err;
}
let __v8: u32 = unsafe { (*ep).ep_state };
let __v9: u32 = ((__v8 as u64) | BIT!(1i32)) as u32;
unsafe {
(*ep).ep_state = __v9;
}
xhci_ring_cmd_db(xhci);
0
}

// Fix up the ep ring first, so HW stops executing cancelled TDs.
// We have the xHCI lock, so nothing can modify this list until we drop it.
// We're also in the event handler, so we can't get re-interrupted if another
// Stop Endpoint command completes.
//
// only call this when ring is not in a running state
fn xhci_invalidate_cancelled_tds(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep) -> i32 {
let mut td: *mut xhci_td = core::ptr::null_mut::<xhci_td>();
let mut tmp_td: *mut xhci_td = core::ptr::null_mut::<xhci_td>();
let mut cached_td: *mut xhci_td = core::ptr::null_mut::<xhci_td>();
let mut hw_deq: u64 = 0;
let mut err: i32 = 0;
// This is not going to work if the hardware is changing its dequeue
// pointers as we look at them. Completion handler will call us later.
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(0i32) != 0 {
return 0;
}
td = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) });
tmp_td = list_next_entry!(td, cancelled_td_list);
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) }, cancelled_td_list) == 0 {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Removing canceled TD starting at 0x%llx (dma) in stream %u URB %p".as_ptr() as *const u8, xhci_trb_virt_to_dma((*td).start_seg, (*td).start_trb), (*(*td).urb).stream_id, (*td).urb) };
list_del_init(unsafe { core::ptr::addr_of_mut!((*td).td_list) });
let ring: *mut xhci_ring = xhci_urb_to_transfer_ring(xhci, unsafe { (*td).urb });
if ring.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Cancelled URB %p has invalid stream ID %u.\n".as_ptr() as *const u8, (*td).urb, (*(*td).urb).stream_id) };
} else {
// device disconnected or link error, ep will be dropped
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(9i32) != 0 {
unsafe {
(*td).cancel_status = 4;
}
} else {
// If a ring stopped on the TD we need to cancel then we have to
// move the xHC endpoint ring dequeue pointer past this TD.
// Rings halted due to STALL may show hw_deq is past the stalled
// TD, but still require a set TR Deq command to flush xHC cache.
hw_deq = xhci_get_hw_deq(xhci, unsafe { (*ep).vdev }, unsafe { (*ep).ep_index }, unsafe { (*(*td).urb).stream_id });
let __v5: u64 = hw_deq;
hw_deq = __v5 & GENMASK_ULL__11!(63i32, 4i32);
let __v7: bool = ((unsafe { (*td).cancel_status }) as u32) == 1 || trb_in_td(td, hw_deq);
if __v7 {
'__slate_break_8: {
match (unsafe { (*td).cancel_status }) as u32 {
2 | 4 => {
break '__slate_break_8; // TD is already no-op
// set TR deq command already queued
}
0 | 1 | 3 => {
if !cached_td.is_null() {
if (unsafe { (*(*cached_td).urb).stream_id }) != unsafe { (*(*td).urb).stream_id } {
// Multiple streams case, defer move dq
unsafe {
(*td).cancel_status = 3;
}
break '__slate_break_8;
}
// Should never happen, but clear the TD if it does
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Found multiple active URBs %p and %p in stream %u?\n".as_ptr() as *const u8, (*td).urb, (*cached_td).urb, (*(*td).urb).stream_id) };
td_to_noop(xhci, ep, cached_td, false);
unsafe {
(*cached_td).cancel_status = 4;
}
}
// TD is cached, clear it
td_to_noop(xhci, ep, td, false);
unsafe {
(*td).cancel_status = 2;
}
cached_td = td;
}
_ => {
}
}
}
} else {
td_to_noop(xhci, ep, td, false);
unsafe {
(*td).cancel_status = 4;
}
}
}
}
td = tmp_td;
tmp_td = list_next_entry!(tmp_td, cancelled_td_list);
}
// If there's no need to move the dequeue pointer then we're done
if cached_td.is_null() {
return 0;
}
err = xhci_move_deq_to_next_td(xhci, ep, unsafe { (*(*cached_td).urb).stream_id });
if err != 0 {
// Failed to move past cached td, just set cached TDs to no-op
td = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) });
tmp_td = list_next_entry!(td, cancelled_td_list);
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) }, cancelled_td_list) == 0 {
// Deferred TDs need to have the deq pointer set after the above command
// completes, so if that failed we just give up on all of them (and
// complain loudly since this could cause issues due to caching).
if !(((unsafe { (*td).cancel_status }) as u32) != 2 && ((unsafe { (*td).cancel_status }) as u32) != 3) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Failed to clear cancelled cached URB %p, mark clear anyway\n".as_ptr() as *const u8, (*td).urb) };
td_to_noop(xhci, ep, td, false);
unsafe {
(*td).cancel_status = 4;
}
}
td = tmp_td;
tmp_td = list_next_entry!(tmp_td, cancelled_td_list);
}
}
0
}

/// Erase queued TDs from transfer ring(s) and give back those the xHC didn't
/// stop on. If necessary, queue commands to move the xHC off cancelled TDs it
/// stopped on. Those will be given back later when the commands complete.
///
/// Call under xhci->lock on a stopped endpoint.
#[unsafe(no_mangle)]
extern "C" fn xhci_process_cancelled_tds(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep) {
xhci_invalidate_cancelled_tds(xhci, ep);
xhci_giveback_invalidated_tds(xhci, ep);
}

/// Returns the TD the endpoint ring halted on.
/// Only call for non-running rings without streams.
fn find_halted_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep) -> *mut xhci_td {
let mut hw_deq: u64 = 0;
if list_empty((unsafe { core::ptr::addr_of_mut!((*(*ep).ring).td_list) }) as *const list_head) == 0 {
// Not streams compatible
hw_deq = xhci_get_hw_deq(xhci, unsafe { (*ep).vdev }, unsafe { (*ep).ep_index }, 0u32);
let __v0: u64 = hw_deq;
hw_deq = __v0 & GENMASK_ULL__4!(63i32, 4i32);
let td: *mut xhci_td = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*(*ep).ring).td_list) }, xhci_td);
if trb_in_td(td, hw_deq) {
return td;
}
}
core::ptr::null_mut::<xhci_td>()
}

/// When we get a command completion for a Stop Endpoint Command, we need to
/// unlink any cancelled TDs from the ring.  There are two ways to do that:
///
///  1. If the HW was in the middle of processing the TD that needs to be
///     cancelled, then we must move the ring's dequeue pointer past the last TRB
///     in the TD with a Set Dequeue Pointer Command.
///  2. Otherwise, we turn all the TRBs in the TD into No-op TRBs (with the chain
///     bit cleared) so that the HW will skip over them.
fn xhci_handle_cmd_stop_ep(xhci: *mut xhci_hcd, slot_id: i32, trb: *mut xhci_trb, comp_code: u32) {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
let mut reset_type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(reset_type) as *mut u32;
let mut td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
td = core::ptr::null_mut::<xhci_td>();
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if unlikely__3!(TRB_TO_SUSPEND_PORT!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) })) != 0 {
if unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }.is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Stop endpoint command completion for disabled slot %u\n".as_ptr() as *const u8, slot_id);
}
return;
} else {
*__slate_slot_18 = TRB_TO_EP_INDEX!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
*__slate_slot_17 = xhci_get_virt_ep(xhci, slot_id as u32, *__slate_slot_18);
if (*__slate_slot_17).is_null() {
return;
} else {
'__join_2: {
*__slate_slot_16 = xhci_get_ep_ctx(xhci, (*(*(*__slate_slot_17)).vdev).out_ctx, *__slate_slot_18);
trace_xhci_handle_cmd_stop_ep(*__slate_slot_16);
if comp_code == 19 {
// If stop endpoint command raced with a halting endpoint we need to
// reset the host side endpoint first.
// If the TD we halted on isn't cancelled the TD should be given back
// with a proper error code, and the ring dequeue moved past the TD.
// If streams case we can't find hw_deq, or the TD we halted on so do a
// soft reset.
//
// Proper error code is unknown here, it would be -EPIPE if device side
// of enadpoit halted (aka STALL), and -EPROTO if not (transaction error)
// We use -EPROTO, if device is stalled it should return a stall error on
// next transfer, which then will return -EPIPE, and device side stall is
// noted and cleared by class driver.
let __t0: u32 = GET_EP_CTX_STATE!(*__slate_slot_16);
if __t0 == 2 {
// If the halt happened before Stop Endpoint failed, its transfer event
// should have already been handled and Reset Endpoint should be pending.
if ((*(*__slate_slot_17)).ep_state as u64) & BIT!(1i32) == 0 {
if ((*(*__slate_slot_17)).ep_state as u64) & BIT!(4i32) != 0 {
*__slate_slot_14 = 1;
} else {
*__slate_slot_14 = 0;
td = find_halted_td(xhci, *__slate_slot_17);
if !td.is_null() {
(*td).status = 71i32.wrapping_neg();
}
}
// reset ep, reset handler cleans up cancelled tds
*__slate_slot_12 = xhci_handle_halted_endpoint(xhci, *__slate_slot_17, td, *__slate_slot_14);
if *__slate_slot_12 != 0 {
break '__join_2;
}
}
let __v11: *mut xhci_virt_ep = *__slate_slot_17;
let __v10: u32 = (*__v11).ep_state;
let __v9: u32 = ((__v10 as u64) & !BIT!(2i32)) as u32;
(*__v11).ep_state = __v9;
// Reset EP handler will clean up cancelled TDs
return;
} else {
if __t0 == 3 {
if ((*(*__slate_slot_17)).ep_state as u64) & BIT!(1i32) != 0 {
break '__join_2;
} else {
// Per xHCI 4.6.9, Stop Endpoint command on a Stopped
// EP is a Context State Error, and EP stays Stopped.
//
// But maybe it failed on Halted, and somebody ran Reset
// Endpoint later. EP state is now Stopped and EP_HALTED
// still set because Reset EP handler will run after us.
// On some HCs EP state remains Stopped for some tens of
// us to a few ms or more after a doorbell ring, and any
// new Stop Endpoint fails without aborting the restart.
// This handler may run quickly enough to still see this
// Stopped state, but it will soon change to Running.
//
// Assume this bug on unexpected Stop Endpoint failures.
// Keep retrying until the EP starts and stops again or
// up to a timeout (a defective HC may never start, or a
// driver bug may cause stopping an already stopped EP).
if time_is_before_jiffies!(unsafe { (*(*__slate_slot_17)).stop_time }.wrapping_add(msecs_to_jiffies(100u32))) {
break '__join_2;
}
}
} else {
if __t0 != 1 {
break '__join_2;
}
}
}
// Race, HW handled stop ep cmd before ep was running
*__slate_slot_13 = xhci_alloc_command(xhci, false, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_13).is_null() {
let __v8: *mut xhci_virt_ep = *__slate_slot_17;
let __v7: u32 = (*__v8).ep_state;
let __v6: u32 = ((__v7 as u64) & !BIT!(2i32)) as u32;
(*__v8).ep_state = __v6;
return;
} else {
xhci_queue_stop_endpoint(xhci, *__slate_slot_13, slot_id, *__slate_slot_18, 0i32);
xhci_ring_cmd_db(xhci);
return;
}
}
}
// link is inactive or disconnected, ep is not running and shouldn't be restarted
if ((*(*(*(*__slate_slot_17)).vdev).rhub_port).__slate_bits_0.__get_link_inactive() as i32) != 0 || ((*(*(*(*__slate_slot_17)).vdev).rhub_port).__slate_bits_0.__get_connected() as i32) == 0 {
let __v5: *mut xhci_virt_ep = *__slate_slot_17;
let __v4: u32 = (*__v5).ep_state;
let __v3: u32 = ((__v4 as u64) | BIT!(9i32)) as u32;
(*__v5).ep_state = __v3;
}
// will queue a set TR deq if stopped on a cancelled, uncleared TD
xhci_invalidate_cancelled_tds(xhci, *__slate_slot_17);
let __v2: *mut xhci_virt_ep = *__slate_slot_17;
let __v1: u32 = (*__v2).ep_state;
let __v0: u32 = ((__v1 as u64) & !BIT!(2i32)) as u32;
(*__v2).ep_state = __v0;
// Otherwise ring the doorbell(s) to restart queued transfers
xhci_giveback_invalidated_tds(xhci, *__slate_slot_17);
xhci_ring_doorbell_for_active_rings(xhci, slot_id as u32, *__slate_slot_18);
}
}
}
}

fn xhci_kill_ring_urbs(xhci: *mut xhci_hcd, ring: *mut xhci_ring) {
let mut cur_td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut tmp: *mut xhci_td = unsafe { core::mem::zeroed() };
cur_td = list_first_entry__13!(unsafe { core::ptr::addr_of_mut!((*ring).td_list) });
tmp = list_next_entry__10!(cur_td, td_list);
while list_entry_is_head!(cur_td, unsafe { core::ptr::addr_of_mut!((*ring).td_list) }, td_list) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*cur_td).td_list) });
if list_empty((unsafe { core::ptr::addr_of_mut!((*cur_td).cancelled_td_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*cur_td).cancelled_td_list) });
}
xhci_unmap_td_bounce_buffer(xhci, ring, cur_td);
inc_td_cnt(unsafe { (*cur_td).urb });
if last_td_in_urb(cur_td) {
xhci_giveback_urb_in_irq(xhci, cur_td, 108i32.wrapping_neg());
}
cur_td = tmp;
tmp = list_next_entry__10!(tmp, td_list);
}
}

fn xhci_kill_endpoint_urbs(xhci: *mut xhci_hcd, slot_id: i32, ep_index: i32) {
let mut cur_td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut tmp: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut ep: *mut xhci_virt_ep = unsafe { core::mem::zeroed() };
let mut ring: *mut xhci_ring = unsafe { core::mem::zeroed() };
ep = xhci_get_virt_ep(xhci, slot_id as u32, ep_index as u32);
if ep.is_null() {
return;
}
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) != 0 || ((unsafe { (*ep).ep_state }) as u64) & BIT!(5i32) != 0 {
let mut stream_id: i32 = 1;
while (stream_id as u32) < unsafe { (*(*ep).stream_info).num_streams } {
ring = unsafe { *unsafe { (*(*ep).stream_info).stream_rings }.wrapping_offset(stream_id as isize) };
if !ring.is_null() {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Killing URBs for slot ID %u, ep index %u, stream %u".as_ptr() as *const u8, slot_id, ep_index, stream_id) };
xhci_kill_ring_urbs(xhci, ring);
}
stream_id = stream_id.wrapping_add(1);
}
} else {
ring = unsafe { (*ep).ring };
if ring.is_null() {
return;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Killing URBs for slot ID %u, ep index %u".as_ptr() as *const u8, slot_id, ep_index) };
xhci_kill_ring_urbs(xhci, ring);
}
cur_td = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) });
tmp = list_next_entry!(cur_td, cancelled_td_list);
while list_entry_is_head!(cur_td, unsafe { core::ptr::addr_of_mut!((*ep).cancelled_td_list) }, cancelled_td_list) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*cur_td).cancelled_td_list) });
inc_td_cnt(unsafe { (*cur_td).urb });
if last_td_in_urb(cur_td) {
xhci_giveback_urb_in_irq(xhci, cur_td, 108i32.wrapping_neg());
}
cur_td = tmp;
tmp = list_next_entry!(tmp, cancelled_td_list);
}
}

/// host controller died, register read returns 0xffffffff
/// Complete pending commands, mark them ABORTED.
/// URBs need to be given back as usb core might be waiting with device locks
/// held for the URBs to finish during device disconnect, blocking host remove.
///
/// Call with xhci->lock held.
/// lock is relased and re-acquired while giving back urb.
#[unsafe(no_mangle)]
extern "C" fn xhci_hc_died(xhci: *mut xhci_hcd) {
let mut notify: bool = false;
// return any pending urbs, remove may be waiting for them
let mut i: i32 = 0;
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
return;
}
notify = ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(2i32) == 0;
if notify {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI host controller not responding, assume dead\n".as_ptr() as *const u8) };
}
let __v2: u32 = unsafe { (*xhci).xhc_state };
let __v3: u32 = ((__v2 as u64) | BIT!(0i32)) as u32;
unsafe {
(*xhci).xhc_state = __v3;
}
xhci_cleanup_command_queue(xhci);
while i <= ((unsafe { (*xhci).max_slots }) as i32) {
if !unsafe { *unsafe { (*xhci).devs }.wrapping_offset(i as isize) }.is_null() {
for j in 0i32..31i32 {
xhci_kill_endpoint_urbs(xhci, i, j);
}
}
i = i.wrapping_add(1);
}
// inform usb core hc died if PCI remove isn't already handling it
if notify {
unsafe { usb_hc_died(xhci_to_hcd(xhci)) };
}
}

/// When we get a completion for a Set Transfer Ring Dequeue Pointer command,
/// we need to clear the set deq pending flag in the endpoint ring state, so that
/// the TD queueing code can ring the doorbell again.  We also need to ring the
/// endpoint doorbell to restart the ring, but only if there aren't more
/// cancellations pending.
fn xhci_handle_cmd_set_deq(xhci: *mut xhci_hcd, slot_id: i32, trb: *mut xhci_trb, cmd_comp_code: u32) {
let mut slot_state: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(slot_state) as *mut u32;
let mut ep_state: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(ep_state) as *mut u32;
let mut deq: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(deq) as *mut u64;
let mut tmp_td: core::mem::MaybeUninit<*mut xhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut xhci_td = core::ptr::addr_of_mut!(tmp_td) as *mut *mut xhci_td;
let mut td: core::mem::MaybeUninit<*mut xhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut xhci_td = core::ptr::addr_of_mut!(td) as *mut *mut xhci_td;
let mut stream_ctx: core::mem::MaybeUninit<*mut xhci_stream_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut xhci_stream_ctx = core::ptr::addr_of_mut!(stream_ctx) as *mut *mut xhci_stream_ctx;
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
let mut ep_ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ep_ring) as *mut *mut xhci_ring;
let mut stream_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(stream_id) as *mut u32;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
*__slate_slot_16 = TRB_TO_EP_INDEX!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
*__slate_slot_15 = TRB_TO_STREAM_ID!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) });
*__slate_slot_13 = xhci_get_virt_ep(xhci, slot_id as u32, *__slate_slot_16);
if (*__slate_slot_13).is_null() {
return;
} else {
*__slate_slot_14 = xhci_virt_ep_to_ring(xhci, *__slate_slot_13, *__slate_slot_15);
if (*__slate_slot_14).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Set TR deq ptr command for freed stream ID %u\n".as_ptr() as *const u8, *__slate_slot_15);
// XXX: Harmless???
} else {
*__slate_slot_12 = xhci_get_ep_ctx(xhci, (*(*(*__slate_slot_13)).vdev).out_ctx, *__slate_slot_16);
*__slate_slot_11 = xhci_get_slot_ctx(xhci, (*(*(*__slate_slot_13)).vdev).out_ctx);
trace_xhci_handle_cmd_set_deq(*__slate_slot_11);
trace_xhci_handle_cmd_set_deq_ep(*__slate_slot_12);
if ((*(*__slate_slot_13)).ep_state as u64) & BIT!(4i32) != 0 {
*__slate_slot_10 = unsafe { (*(*(*__slate_slot_13)).stream_info).stream_ctx_array }.wrapping_offset(*__slate_slot_15 as isize);
trace_xhci_handle_cmd_set_deq_stream((*(*__slate_slot_13)).stream_info, *__slate_slot_15);
}
if cmd_comp_code != 1 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __t0: u32 = cmd_comp_code;
if __t0 == 5 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Set TR Deq Ptr cmd invalid because of stream ID configuration\n".as_ptr() as *const u8);
} else {
if __t0 == 19 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Set TR Deq Ptr cmd failed due to incorrect slot or ep state.\n".as_ptr() as *const u8);
*__slate_slot_6 = GET_EP_CTX_STATE!(*__slate_slot_12);
*__slate_slot_5 = (*(*__slate_slot_11)).dev_state;
*__slate_slot_5 = GET_SLOT_STATE!(*__slate_slot_5);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Slot state = %u, EP state = %u".as_ptr() as *const u8, *__slate_slot_5, *__slate_slot_6);
} else {
if __t0 == 11 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Set TR Deq Ptr cmd failed because slot %u was not enabled.\n".as_ptr() as *const u8, slot_id);
} else {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Set TR Deq Ptr cmd with unknown completion code of %u.\n".as_ptr() as *const u8, cmd_comp_code);
}
}
}
// OK what do we do now?  The endpoint state is hosed, and we
// should never get to this point if the synchronization between
// queueing, and endpoint state are correct.  This might happen
// if the device gets disconnected after we've finished
// cancelling URBs, which might not be an error...
} else {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// 4.6.10 deq ptr is written to the stream ctx for streams
if ((*(*__slate_slot_13)).ep_state as u64) & BIT!(4i32) != 0 {
*__slate_slot_7 = (*(*__slate_slot_10)).stream_ring & GENMASK_ULL__14!(63i32, 4i32);
// Cadence xHCI controllers store some endpoint state
// information within Rsvd0 fields of Stream Endpoint
// context. This field is not cleared during Set TR
// Dequeue Pointer command which causes XDMA to skip
// over transfer ring and leads to data loss on stream
// pipe.
// To fix this issue driver must clear Rsvd0 field.
if (*xhci).quirks & BIT_ULL!(48i32) != 0 {
*((*(*__slate_slot_10)).reserved.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = 0;
*((*(*__slate_slot_10)).reserved.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = 0;
}
} else {
*__slate_slot_7 = (*(*__slate_slot_12)).deq & GENMASK_ULL__10!(63i32, 4i32);
}
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_cancel_urb), c"Successful Set TR Deq Ptr cmd, deq = @%08llx".as_ptr() as *const u8, *__slate_slot_7);
if xhci_trb_virt_to_dma((*(*__slate_slot_13)).queued_deq_seg, (*(*__slate_slot_13)).queued_deq_ptr) == *__slate_slot_7 {
// Update the ring's dequeue segment and dequeue pointer
// to reflect the new position.
(*(*__slate_slot_14)).deq_seg = (*(*__slate_slot_13)).queued_deq_seg;
(*(*__slate_slot_14)).dequeue = (*(*__slate_slot_13)).queued_deq_ptr;
} else {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Mismatch between completed Set TR Deq Ptr command & xHCI internal state.\n".as_ptr() as *const u8);
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ep deq seg = %p, deq ptr = %p\n".as_ptr() as *const u8, (*(*__slate_slot_13)).queued_deq_seg, (*(*__slate_slot_13)).queued_deq_ptr);
}
}
// HW cached TDs cleared from cache, give them back
*__slate_slot_9 = list_first_entry__3!(core::ptr::addr_of_mut!((*(*__slate_slot_13)).cancelled_td_list));
let __v4: *mut xhci_td = list_next_entry!(*__slate_slot_9, cancelled_td_list);
*__slate_slot_8 = __v4;
while list_entry_is_head!(*__slate_slot_9, core::ptr::addr_of_mut!((*(*__slate_slot_13)).cancelled_td_list), cancelled_td_list) == 0 {
*__slate_slot_14 = xhci_urb_to_transfer_ring(xhci, (*(*__slate_slot_9)).urb);
if ((*(*__slate_slot_9)).cancel_status as u32) == 2 {
(*(*__slate_slot_9)).cancel_status = 4;
xhci_td_cleanup(xhci, *__slate_slot_9, *__slate_slot_14, (*(*__slate_slot_9)).status);
}
*__slate_slot_9 = *__slate_slot_8;
let __v3: *mut xhci_td = list_next_entry!(*__slate_slot_8, cancelled_td_list);
*__slate_slot_8 = __v3;
}
}
let __v2: *mut xhci_virt_ep = *__slate_slot_13;
let __v1: u32 = (*__v2).ep_state;
let __v0: u32 = ((__v1 as u64) & !BIT!(0i32)) as u32;
(*__v2).ep_state = __v0;
(*(*__slate_slot_13)).queued_deq_seg = core::ptr::null_mut::<xhci_segment>();
(*(*__slate_slot_13)).queued_deq_ptr = core::ptr::null_mut::<xhci_trb>();
// Check for deferred or newly cancelled TDs
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_13)).cancelled_td_list) as *const list_head) == 0 {
xhci_invalidate_cancelled_tds(xhci, *__slate_slot_13);
// Try to restart the endpoint if all is done
xhci_ring_doorbell_for_active_rings(xhci, slot_id as u32, *__slate_slot_16);
// Start giving back any TDs invalidated above
xhci_giveback_invalidated_tds(xhci, *__slate_slot_13);
} else {
// Restart any rings with pending URBs
xhci_ring_doorbell_for_active_rings(xhci, slot_id as u32, *__slate_slot_16);
}
}
}
}

fn xhci_handle_cmd_reset_ep(xhci: *mut xhci_hcd, slot_id: i32, trb: *mut xhci_trb, cmd_comp_code: u32) {
let ep_index: u32 = TRB_TO_EP_INDEX!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
let ep: *mut xhci_virt_ep = xhci_get_virt_ep(xhci, slot_id as u32, ep_index);
if ep.is_null() {
return;
}
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*(*ep).vdev).out_ctx, ep_index) };
trace_xhci_handle_cmd_reset_ep(ep_ctx);
// This command will only fail if the endpoint wasn't halted,
// but we don't care.
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_reset_ep), c"Ignoring reset ep completion code of %u".as_ptr() as *const u8, cmd_comp_code) };
// Cleanup cancelled TDs as ep is stopped. May queue a Set TR Deq cmd
xhci_invalidate_cancelled_tds(xhci, ep);
// Clear our internal halted state
let __v1: u32 = unsafe { (*ep).ep_state };
let __v2: u32 = ((__v1 as u64) & !BIT!(1i32)) as u32;
unsafe {
(*ep).ep_state = __v2;
}
xhci_giveback_invalidated_tds(xhci, ep);
// if this was a soft reset, then restart
if ((unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) }) as u64) & BIT!(9i32) != 0 {
xhci_ring_doorbell_for_active_rings(xhci, slot_id as u32, ep_index);
}
}

fn xhci_handle_cmd_enable_slot(slot_id: i32, command: *mut xhci_command, cmd_comp_code: u32) {
if cmd_comp_code == 1 {
unsafe {
(*command).slot_id = slot_id;
}
} else {
unsafe {
(*command).slot_id = 0;
}
}
}

fn xhci_handle_cmd_disable_slot(xhci: *mut xhci_hcd, slot_id: i32, cmd_comp_code: u32) {
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
if virt_dev.is_null() {
return;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*virt_dev).out_ctx) };
trace_xhci_handle_cmd_disable_slot(slot_ctx);
if (unsafe { (*xhci).quirks }) & BIT_ULL!(5i32) != 0 {
unsafe { xhci_free_device_endpoint_resources(xhci, virt_dev, true) };
}
// Delete default control endpoint resources
if cmd_comp_code == 1 {
unsafe {
*unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset(slot_id as isize) = 0;
}
unsafe {
*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) = core::ptr::null_mut::<xhci_virt_device>();
}
}
}

fn xhci_handle_cmd_config_ep(xhci: *mut xhci_hcd, slot_id: i32) {
let mut ep_index: u32 = 0;
let mut add_flags: u32 = 0;
// Configure endpoint commands can come from the USB core configuration
// or alt setting changes, or when streams were being configured.
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
if virt_dev.is_null() {
return;
}
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*virt_dev).in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Could not get input context, bad type.\n".as_ptr() as *const u8) };
return;
}
add_flags = unsafe { (*ctrl_ctx).add_flags };
// Input ctx add_flags are the endpoint index plus one
ep_index = unsafe { xhci_last_valid_endpoint(add_flags) }.wrapping_sub(1);
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*virt_dev).out_ctx, ep_index) };
trace_xhci_handle_cmd_config_ep(ep_ctx);
return;
}

fn xhci_handle_cmd_addr_dev(xhci: *mut xhci_hcd, slot_id: i32) {
let vdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
if vdev.is_null() {
return;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*vdev).out_ctx) };
trace_xhci_handle_cmd_addr_dev(slot_ctx);
}

fn xhci_handle_cmd_reset_dev(xhci: *mut xhci_hcd, slot_id: i32) {
let vdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
if vdev.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Reset device command completion for disabled slot %u\n".as_ptr() as *const u8, slot_id) };
return;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*vdev).out_ctx) };
trace_xhci_handle_cmd_reset_dev(slot_ctx);
}

fn xhci_handle_cmd_nec_get_fw(xhci: *mut xhci_hcd, event: *mut xhci_event_cmd) {
if (unsafe { (*xhci).quirks }) & BIT_ULL!(2i32) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN NEC_GET_FW command on non-NEC host\n".as_ptr() as *const u8) };
return;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"NEC firmware version %2x.%02x".as_ptr() as *const u8, NEC_FW_MAJOR!(unsafe { (*event).status }), NEC_FW_MINOR!(unsafe { (*event).status })) };
}

fn xhci_complete_del_and_free_cmd(cmd: *mut xhci_command, comp_code: u32, comp_param: u32) {
list_del(unsafe { core::ptr::addr_of_mut!((*cmd).cmd_list) });
if !unsafe { (*cmd).completion }.is_null() {
unsafe {
(*cmd).status = comp_code;
}
unsafe {
(*cmd).comp_param = comp_param;
}
unsafe { complete((*cmd).completion) };
} else {
unsafe { kfree(cmd as *const ()) };
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_cleanup_command_queue(xhci: *mut xhci_hcd) {
let mut cur_cmd: *mut xhci_command = unsafe { core::mem::zeroed() };
let mut tmp_cmd: *mut xhci_command = unsafe { core::mem::zeroed() };
unsafe {
(*xhci).current_cmd = core::ptr::null_mut::<xhci_command>();
}
cur_cmd = list_first_entry__11!(unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) });
tmp_cmd = list_next_entry__9!(cur_cmd, cmd_list);
while list_entry_is_head!(cur_cmd, unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) }, cmd_list) == 0 {
xhci_complete_del_and_free_cmd(cur_cmd, 25u32, 0u32);
cur_cmd = tmp_cmd;
tmp_cmd = list_next_entry__9!(tmp_cmd, cmd_list);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_ring.xhci_handle_command_timeout")]
extern "C" fn xhci_handle_command_timeout(work: *mut work_struct) {
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
// If timeout work is pending, or current_cmd is NULL, it means we
// raced with command completion. Command is handled so just return.
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut usbsts: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(usbsts) as *mut u32;
let mut cmd_field3: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(cmd_field3) as *mut u32;
let mut hw_ring_state: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(hw_ring_state) as *mut u64;
let mut str: core::mem::MaybeUninit<__SlateAlign16<[u8; 500]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut [u8; 500] = core::ptr::addr_of_mut!(str) as *mut [u8; 500];
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
unsafe {
{
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 500usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_10 = container_of__52!(to_delayed_work(work), xhci_hcd);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_10)).lock), *__slate_slot_9);
if unsafe { (*(*__slate_slot_10)).current_cmd }.is_null() {
__v4 = true;
} else {
__v4 = delayed_work_pending!(core::ptr::addr_of_mut!((*(*__slate_slot_10)).cmd_timer)) != 0;
}
}
if __v4 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_10)).lock), *__slate_slot_9);
return;
} else {
*__slate_slot_6 = *((*(*(*(*__slate_slot_10)).current_cmd).command_trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize);
*__slate_slot_5 = readl(core::ptr::addr_of_mut!((*(*(*__slate_slot_10)).op_regs).status) as *const ());
// Bail out and tear down xhci if a stop endpoint command failed
if TRB_FIELD_TO_TYPE!(*__slate_slot_6) == 15 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_10)).self_.controller as *const device, c"xHCI host not responding to stop endpoint command\n".as_ptr() as *const u8);
*__slate_slot_3 = xhci_get_virt_ep(*__slate_slot_10, TRB_TO_SLOT_ID!(*__slate_slot_6), TRB_TO_EP_INDEX!(*__slate_slot_6));
if !(*__slate_slot_3).is_null() {
let __v2: *mut xhci_virt_ep = *__slate_slot_3;
let __v1: u32 = (*__v2).ep_state;
let __v0: u32 = ((__v1 as u64) & !BIT!(2i32)) as u32;
(*__v2).ep_state = __v0;
}
xhci_halt(*__slate_slot_10);
xhci_hc_died(*__slate_slot_10);
} else {
// mark this command to be cancelled
(*(*(*__slate_slot_10)).current_cmd).status = 25;
// Make sure command ring is running before aborting it
*__slate_slot_7 = xhci_read_64(*__slate_slot_10 as *const xhci_hcd, core::ptr::addr_of_mut!((*(*(*__slate_slot_10)).op_regs).cmd_ring));
if *__slate_slot_7 == !0 {
xhci_hc_died(*__slate_slot_10);
} else {
if ((*(*__slate_slot_10)).cmd_ring_state as u64) & BIT!(0i32) != 0 && *__slate_slot_7 & BIT_ULL!(3i32) != 0 {
// Prevent new doorbell, and start command abort
(*(*__slate_slot_10)).cmd_ring_state = BIT!(1i32) as u32;
xhci_abort_cmd_ring(*__slate_slot_10, *__slate_slot_9);
} else {
// host removed. Bail out
if ((*(*__slate_slot_10)).xhc_state as u64) & BIT!(2i32) != 0 {
xhci_cleanup_command_queue(*__slate_slot_10);
} else {
// command timeout on stopped ring, ring can't be aborted
xhci_handle_stopped_cmd_ring(*__slate_slot_10, (*(*__slate_slot_10)).current_cmd);
}
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_10)).lock), *__slate_slot_9);
return;
}
}
}

fn handle_cmd_completion(xhci: *mut xhci_hcd, event: *mut xhci_event_cmd) {
let mut cmd_type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(cmd_type) as *mut u32;
let mut cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut xhci_command = core::ptr::addr_of_mut!(cmd) as *mut *mut xhci_command;
let mut cmd_trb: core::mem::MaybeUninit<*mut xhci_trb> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut xhci_trb = core::ptr::addr_of_mut!(cmd_trb) as *mut *mut xhci_trb;
let mut cmd_comp_code: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(cmd_comp_code) as *mut u32;
let mut cmd_dequeue_dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u64 = core::ptr::addr_of_mut!(cmd_dequeue_dma) as *mut u64;
let mut cmd_dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(cmd_dma) as *mut u64;
let mut status: u32 = unsafe { core::mem::zeroed() };
let mut slot_id: u32 = unsafe { core::mem::zeroed() };
unsafe {
slot_id = TRB_TO_SLOT_ID!(unsafe { (*event).flags });
status = (*event).status;
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if slot_id > (((*xhci).max_slots as i32) as u32) {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Invalid slot_id %u\n".as_ptr() as *const u8, slot_id);
return;
} else {
*__slate_slot_14 = (*event).cmd_trb;
*__slate_slot_11 = (*(*xhci).cmd_ring).dequeue;
trace_xhci_handle_command((*xhci).cmd_ring, core::ptr::addr_of_mut!((*(*__slate_slot_11)).generic), *__slate_slot_14);
*__slate_slot_12 = GET_COMP_CODE!(unsafe { (*event).status });
// If CMD ring stopped we own the trbs between enqueue and dequeue
if *__slate_slot_12 == 24 {
complete_all(core::ptr::addr_of_mut!((*xhci).cmd_ring_stop_completion));
return;
} else {
*__slate_slot_13 = xhci_trb_virt_to_dma((*(*xhci).cmd_ring).deq_seg, *__slate_slot_11);
// Check whether the completion event is for our internal kept
// command.
if *__slate_slot_13 == 0 || *__slate_slot_14 != *__slate_slot_13 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR mismatched command completion event\n".as_ptr() as *const u8);
return;
} else {
*__slate_slot_10 = list_first_entry__12!(core::ptr::addr_of_mut!((*xhci).cmd_list), xhci_command);
cancel_delayed_work(core::ptr::addr_of_mut!((*xhci).cmd_timer));
if (*(*__slate_slot_10)).command_trb != (*(*xhci).cmd_ring).dequeue {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Command completion event does not match command\n".as_ptr() as *const u8);
return;
} else {
'__join_0: {
// Host aborted the command ring, check if the current command was
// supposed to be aborted, otherwise continue normally.
// The command ring is stopped now, but the xHC will issue a Command
// Ring Stopped event which will cause us to restart it.
if *__slate_slot_12 == 25 {
(*xhci).cmd_ring_state = BIT!(2i32) as u32;
if (*(*__slate_slot_10)).status == 25 {
if (*xhci).current_cmd == *__slate_slot_10 {
(*xhci).current_cmd = core::ptr::null_mut::<xhci_command>();
break '__join_0;
} else {
break '__join_0;
}
}
}
*__slate_slot_9 = TRB_FIELD_TO_TYPE!(unsafe { *((*(*__slate_slot_11)).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
let __t0: u32 = *__slate_slot_9;
if __t0 == 9 {
xhci_handle_cmd_enable_slot(slot_id as i32, *__slate_slot_10, *__slate_slot_12);
} else {
if __t0 == 10 {
xhci_handle_cmd_disable_slot(xhci, slot_id as i32, *__slate_slot_12);
} else {
if __t0 == 12 {
if unsafe { (*(*__slate_slot_10)).completion }.is_null() {
xhci_handle_cmd_config_ep(xhci, slot_id as i32);
}
} else {
if __t0 != 13 {
if __t0 == 11 {
xhci_handle_cmd_addr_dev(xhci, slot_id as i32);
} else {
if __t0 == 15 {
let __ret_warn_on: i32 = (slot_id != TRB_TO_SLOT_ID!(unsafe { *((*(*__slate_slot_11)).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) })) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("919: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 919b - ., 3; .popsection\n# {0}", const 919 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_handle_cmd_completion_22, sym __slate_asm_string_handle_cmd_completion_23, const 1889 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("920: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 920b - ., 4; .popsection\n# {0}", const 920 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if unsafe { (*(*__slate_slot_10)).completion }.is_null() {
xhci_handle_cmd_stop_ep(xhci, slot_id as i32, *__slate_slot_11, *__slate_slot_12);
}
} else {
if __t0 == 16 {
let __ret_warn_on: i32 = (slot_id != TRB_TO_SLOT_ID!(unsafe { *((*(*__slate_slot_11)).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) })) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("921: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 921b - ., 3; .popsection\n# {0}", const 921 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_handle_cmd_completion_20, sym __slate_asm_string_handle_cmd_completion_21, const 1896 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("922: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 922b - ., 4; .popsection\n# {0}", const 922 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
xhci_handle_cmd_set_deq(xhci, slot_id as i32, *__slate_slot_11, *__slate_slot_12);
} else {
if __t0 == 23 {
if (*(*__slate_slot_10)).status == 24 {
*__slate_slot_12 = 24;
}
// Is this an aborted command turned to NO-OP?
} else {
if __t0 == 14 {
let __ret_warn_on: i32 = (slot_id != TRB_TO_SLOT_ID!(unsafe { *((*(*__slate_slot_11)).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) })) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("923: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 923b - ., 3; .popsection\n# {0}", const 923 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_handle_cmd_completion_18, sym __slate_asm_string_handle_cmd_completion_19, const 1906 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("924: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 924b - ., 4; .popsection\n# {0}", const 924 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
xhci_handle_cmd_reset_ep(xhci, slot_id as i32, *__slate_slot_11, *__slate_slot_12);
} else {
if __t0 == 17 {
slot_id = TRB_TO_SLOT_ID!(unsafe { *((*(*__slate_slot_11)).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
// SLOT_ID field in reset device cmd completion event TRB is 0.
// Use the SLOT_ID from the command TRB instead (xhci 4.6.11)
xhci_handle_cmd_reset_dev(xhci, slot_id as i32);
} else {
if __t0 == 49 {
xhci_handle_cmd_nec_get_fw(xhci, event);
} else {
if __t0 != 21 {
while false {
}
_dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"INFO unknown command type %d\n".as_ptr() as *const u8, *__slate_slot_9); // Skip over unknown commands on the event ring
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
}
// restart timer if this wasn't the last command
if list_is_singular(core::ptr::addr_of_mut!((*xhci).cmd_list) as *const list_head) == 0 {
(*xhci).current_cmd = list_first_entry__12!(core::ptr::addr_of_mut!((*(*__slate_slot_10)).cmd_list), xhci_command);
xhci_mod_cmd_timer(xhci);
} else {
if (*xhci).current_cmd == *__slate_slot_10 {
(*xhci).current_cmd = core::ptr::null_mut::<xhci_command>();
}
}
}
xhci_complete_del_and_free_cmd(*__slate_slot_10, *__slate_slot_12, COMP_PARAM!(status));
inc_deq(xhci, (*xhci).cmd_ring);
}
}
}
}
}
}

fn handle_vendor_event(xhci: *mut xhci_hcd, event: *mut xhci_trb, trb_type: u32) {
if trb_type == 48 && (unsafe { (*xhci).quirks }) & BIT_ULL!(2i32) != 0 {
handle_cmd_completion(xhci, unsafe { core::ptr::addr_of_mut!((*event).event_cmd) });
}
}

fn handle_device_notification(xhci: *mut xhci_hcd, event: *mut xhci_trb) {
let slot_id: u32 = TRB_TO_SLOT_ID!(unsafe { *((*event).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
if unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Device Notification event for unused slot %u\n".as_ptr() as *const u8, slot_id) };
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Device Wake Notification event for slot ID %u\n".as_ptr() as *const u8, slot_id) };
}
let udev: *mut usb_device = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).udev };
if !udev.is_null() && !unsafe { (*udev).parent }.is_null() {
unsafe { usb_wakeup_notification((*udev).parent, (*udev).portnum as u32) };
}
}

/// Quirk hanlder for errata seen on Cavium ThunderX2 processor XHCI
/// Controller.
/// As per ThunderX2errata-129 USB 2 device may come up as USB 1
/// If a connection to a USB 1 device is followed by another connection
/// to a USB 2 device.
///
/// Reset the PHY after the USB device is disconnected if device speed
/// is less than HCD_USB3.
/// Retry the reset sequence max of 4 times checking the PLL lock status.
fn xhci_cavium_reset_phy_quirk(xhci: *mut xhci_hcd) {
let hcd: *mut usb_hcd = xhci_to_hcd(xhci);
let mut pll_lock_check: u32 = 0;
let mut retry_count: u32 = 4;
loop {
// Assert PHY reset
writel(111u32, (unsafe { ((*hcd).regs as *const u8).wrapping_offset(4168 as isize) }) as *mut ());
udelay(10u64);
// De-assert the PHY reset
writel(127u32, (unsafe { ((*hcd).regs as *const u8).wrapping_offset(4168 as isize) }) as *mut ());
udelay(200u64);
pll_lock_check = readl(((unsafe { ((*hcd).regs as *const u8).wrapping_offset(4208 as isize) }) as *mut ()) as *const ());
let __v1: bool = if pll_lock_check & 1 == 0 {
let __v3: u32 = retry_count.wrapping_sub(1);
retry_count = __v3;
__v3 != 0
} else {
false
};
if !__v1 {
break;
}
}
}

fn handle_port_status(xhci: *mut xhci_hcd, event: *mut xhci_trb) {
// We might get interrupts after shared_hcd is removed
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut pls: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(pls) as *mut u32;
let mut portsc: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(portsc) as *mut u32;
let mut port_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(port_id) as *mut u32;
let mut cmd_reg: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(cmd_reg) as *mut u32;
let mut hcd_portnum: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(hcd_portnum) as *mut u32;
let mut bogus_port_status: bool = unsafe { core::mem::zeroed() };
let mut hcd: core::mem::MaybeUninit<*mut usb_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut usb_hcd = core::ptr::addr_of_mut!(hcd) as *mut *mut usb_hcd;
let mut port: core::mem::MaybeUninit<*mut xhci_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut xhci_port = core::ptr::addr_of_mut!(port) as *mut *mut xhci_port;
let mut bus_state: core::mem::MaybeUninit<*mut xhci_bus_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut xhci_bus_state = core::ptr::addr_of_mut!(bus_state) as *mut *mut xhci_bus_state;
let mut vdev: *mut xhci_virt_device = unsafe { core::mem::zeroed() };
unsafe {
vdev = core::ptr::null_mut::<xhci_virt_device>();
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
bogus_port_status = false;
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Port status change events always have a successful completion code
if GET_COMP_CODE!(unsafe { *((*event).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) }) != 1 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: xHC returned failed port status event\n".as_ptr() as *const u8);
}
*__slate_slot_6 = GET_PORT_ID!(unsafe { *((*event).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) });
if *__slate_slot_6 <= 0 || *__slate_slot_6 > (((*xhci).max_ports as i32) as u32) {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Port change event with invalid port ID %d\n".as_ptr() as *const u8, *__slate_slot_6);
return;
} else {
'__join_2: {
*__slate_slot_11 = unsafe { (*xhci).hw_ports }.wrapping_offset((*__slate_slot_6).wrapping_sub(1) as isize);
if (*__slate_slot_11).is_null() || unsafe { (*(*__slate_slot_11)).rhub }.is_null() || (*(*__slate_slot_11)).hcd_portnum == ((1i32.wrapping_neg() as u8) as i32) {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Port change event, no port for port ID %u\n".as_ptr() as *const u8, *__slate_slot_6);
bogus_port_status = true;
} else {
if (*(*__slate_slot_11)).slot_id != 0 {
vdev = *unsafe { (*xhci).devs }.wrapping_offset((*(*__slate_slot_11)).slot_id as isize);
}
if (*(*__slate_slot_11)).rhub == core::ptr::addr_of_mut!((*xhci).usb3_rhub) {
__v3 = xhci_get_usb3_hcd(xhci).is_null();
} else {
__v3 = false;
}
if __v3 {
bogus_port_status = true;
} else {
*__slate_slot_10 = (*(*(*__slate_slot_11)).rhub).hcd;
*__slate_slot_12 = core::ptr::addr_of_mut!((*(*(*__slate_slot_11)).rhub).bus_state);
*__slate_slot_8 = (*(*__slate_slot_11)).hcd_portnum as u32;
*__slate_slot_5 = xhci_portsc_readl(*__slate_slot_11);
*__slate_slot_4 = *__slate_slot_5 & 480u32;
trace_xhci_handle_port_status(*__slate_slot_11, *__slate_slot_5);
if (*(*__slate_slot_10)).state == 4 {
usb_hcd_resume_root_hub(*__slate_slot_10);
}
// Tag broken links to avoid retries while hub driver sorts it out.
// Link status is not relible while port is in reset.
if *__slate_slot_5 & 16u32 == 0 {
let __t0 = (*__slate_slot_4 == 192u32) as u32;
(*(*__slate_slot_11)).__slate_bits_0.__set_link_inactive(__t0);
let __t1 = (*__slate_slot_5 & 1u32 != 0) as u32;
(*(*__slate_slot_11)).__slate_bits_0.__set_connected(__t1);
}
if *__slate_slot_5 & 4194304u32 != 0 && *__slate_slot_5 & 480u32 == 480u32 {
*__slate_slot_7 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
if *__slate_slot_7 & 1u32 == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHC is not running.\n".as_ptr() as *const u8);
break '__join_2;
} else {
if DEV_SUPERSPEED_ANY!(*__slate_slot_5) {
// Set a flag to say the port signaled remote wakeup,
// so we can tell the difference between the end of
// device and host initiated resume.
let __v2: *mut xhci_bus_state = *__slate_slot_12;
(*__v2).port_remote_wakeup = (*__v2).port_remote_wakeup | ((1i32 << *__slate_slot_8) as u32);
xhci_test_and_clear_bit(xhci, *__slate_slot_11, 4194304u32);
usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*(*__slate_slot_10)).self_), *__slate_slot_8 as i32);
xhci_set_link_state(xhci, *__slate_slot_11, 0u32);
// Need to wait until the next link state change
// indicates the device is actually in U0.
bogus_port_status = true;
break '__join_2;
} else {
if test_bit__3!(*__slate_slot_8, core::ptr::addr_of_mut!((*(*__slate_slot_12)).resuming_ports)) == 0 {
(*(*__slate_slot_11)).resume_timestamp = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(40u32));
set_bit(*__slate_slot_8 as i64, core::ptr::addr_of_mut!((*(*__slate_slot_12)).resuming_ports) as *mut u64);
// Do the rest in GetPortStatus after resume time delay.
// Avoid polling roothub status before that so that a
// usb device auto-resume latency around ~40ms.
set_bit(2i64, core::ptr::addr_of_mut!((*(*__slate_slot_10)).flags) as *mut u64);
mod_timer(core::ptr::addr_of_mut!((*(*__slate_slot_10)).rh_timer), (*(*__slate_slot_11)).resume_timestamp);
usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*(*__slate_slot_10)).self_), *__slate_slot_8 as i32);
bogus_port_status = true;
}
}
}
}
if *__slate_slot_5 & 4194304u32 != 0 && DEV_SUPERSPEED_ANY!(*__slate_slot_5) && (*__slate_slot_5 & 480u32 == 0u32 || *__slate_slot_5 & 480u32 == 32u32 || *__slate_slot_5 & 480u32 == 64u32) {
complete(core::ptr::addr_of_mut!((*(*__slate_slot_11)).u3exit_done));
// We've just brought the device into U0/1/2 through either the
// Resume state after a device remote wakeup, or through the
// U3Exit state after a host-initiated resume.  If it's a device
// initiated remote wake, don't pass up the link state change,
// so the roothub behavior is consistent with external
// USB 3.0 hub behavior.
if !vdev.is_null() {
xhci_ring_device(xhci, (*(*__slate_slot_11)).slot_id);
}
if (*(*__slate_slot_12)).port_remote_wakeup & ((1i32 << *__slate_slot_8) as u32) != 0 {
xhci_test_and_clear_bit(xhci, *__slate_slot_11, 4194304u32);
usb_wakeup_notification((*(*__slate_slot_10)).self_.root_hub, (*__slate_slot_8).wrapping_add(1));
bogus_port_status = true;
break '__join_2;
}
}
// Check to see if xhci-hub.c is waiting on RExit to U0 transition (or
// RExit to a disconnect state).  If so, let the driver know it's
// out of the RExit state.
if (*(*__slate_slot_10)).speed < 64 && (*(*__slate_slot_11)).rexit_active {
complete(core::ptr::addr_of_mut!((*(*__slate_slot_11)).rexit_done));
(*(*__slate_slot_11)).rexit_active = false;
bogus_port_status = true;
} else {
if (*(*__slate_slot_10)).speed < 64 {
xhci_test_and_clear_bit(xhci, *__slate_slot_11, 4194304u32);
if (*xhci).quirks & BIT_ULL!(34i32) != 0 && *__slate_slot_5 & 131072u32 != 0 && *__slate_slot_5 & 1u32 == 0 {
xhci_cavium_reset_phy_quirk(xhci);
}
}
}
}
}
}
if bogus_port_status {
return;
} else {
// Don't make the USB core poll the roothub if we got a bad port status
// change event.  Besides, at that point we can't tell which roothub
// (USB 2.0 or USB 3.0) to kick.
// xHCI port-status-change events occur when the "or" of all the
// status-change bits in the portsc register changes from 0 to 1.
// New status changes won't cause an event if any other change
// bits are still set.  When an event occurs, switch over to
// polling to avoid losing status changes.
set_bit(2i64, core::ptr::addr_of_mut!((*(*__slate_slot_10)).flags) as *mut u64);
spin_unlock(core::ptr::addr_of_mut!((*xhci).lock));
// Pass this up to the core
usb_hcd_poll_rh_status(*__slate_slot_10);
spin_lock(core::ptr::addr_of_mut!((*xhci).lock));
}
}
}
}

fn xhci_clear_hub_tt_buffer(xhci: *mut xhci_hcd, td: *mut xhci_td, ep: *mut xhci_virt_ep) {
// As part of low/full-speed endpoint-halt processing
// we must clear the TT buffer (USB 2.0 specification 11.17.5).
let __v0: bool = !unsafe { (*(*(*td).urb).dev).tt }.is_null() && !usb_pipeint!(unsafe { (*(*td).urb).pipe }) && (unsafe { (*(*(*(*td).urb).dev).tt).hub }) != unsafe { (*xhci_to_hcd(xhci)).self_.root_hub };
if __v0 && ((unsafe { (*ep).ep_state }) as u64) & BIT!(8i32) == 0 {
let __v2: u32 = unsafe { (*ep).ep_state };
let __v3: u32 = ((__v2 as u64) | BIT!(8i32)) as u32;
unsafe {
(*ep).ep_state = __v3;
}
unsafe {
(*(*(*td).urb).ep).hcpriv = (*(*td).urb).dev as *mut ();
}
if (unsafe { usb_hub_clear_tt_buffer((*td).urb) }) != 0 {
let __v5: u32 = unsafe { (*ep).ep_state };
let __v6: u32 = ((__v5 as u64) & !BIT!(8i32)) as u32;
unsafe {
(*ep).ep_state = __v6;
}
}
}
}

/// Check if xhci internal endpoint state has gone to a "halt" state due to an
/// error or stall, including default control pipe protocol stall.
/// The internal halt needs to be cleared with a reset endpoint command.
///
/// External device side is also halted in functional stall cases. Class driver
/// will clear the device halt with a CLEAR_FEATURE(ENDPOINT_HALT) request later.
fn xhci_halted_host_endpoint(xhci: *mut xhci_hcd, ep_ctx: *mut xhci_ep_ctx, comp_code: u32) -> bool {
let ep_type: i32 = CTX_TO_EP_TYPE!(unsafe { (*ep_ctx).ep_info2 }) as i32;
// on xHCI this always halts, including protocol stall
// The 0.95 spec says a babbling control endpoint is not halted.
// The 0.96 spec says it is. Some HW claims to be 0.95
// compliant, but it halts the control endpoint anyway.
// Check endpoint context if endpoint is halted.
// these errors halt all non-isochronous endpoints
match comp_code {
6 => {
return true; // on xHCI this always halts, including protocol stall
}
3 => {
if ((unsafe { (*xhci).hci_version }) as i32) <= 149 && ep_type == 4 {
return GET_EP_CTX_STATE!(ep_ctx) == 2;
}
// The 0.95 spec says a babbling control endpoint is not halted.
// The 0.96 spec says it is. Some HW claims to be 0.95
// compliant, but it halts the control endpoint anyway.
// Check endpoint context if endpoint is halted.
return ep_type != 5 && ep_type != 1; // these errors halt all non-isochronous endpoints
}
4 | 36 => {
return ep_type != 5 && ep_type != 1; // these errors halt all non-isochronous endpoints
}
_ => {
}
}
false
}

#[unsafe(no_mangle)]
extern "C" fn xhci_is_vendor_info_code(xhci: *mut xhci_hcd, trb_comp_code: u32) -> i32 {
if trb_comp_code >= 224 && trb_comp_code <= 255 {
// Vendor defined "informational" completion code,
// treat as not-an-error.
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Vendor defined info completion code %u\n".as_ptr() as *const u8, trb_comp_code) };
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Treating code as success.\n".as_ptr() as *const u8) };
}
return 1;
}
0
}

fn finish_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, ep_ring: *mut xhci_ring, td: *mut xhci_td, trb_comp_code: u32) {
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*(*ep).vdev).out_ctx, (*ep).ep_index) };
match trb_comp_code {
26..=28 => {
return;
// The "Stop Endpoint" completion will take care of any
// stopped TDs. A stopped TD may be restarted, so don't update
// the ring dequeue pointer or take this TD off any lists yet.
}
_ => {
}
}
if xhci_halted_host_endpoint(xhci, ep_ctx, trb_comp_code) {
// xhci internal endpoint state will go to a "halt" state for
// any stall, including default control pipe protocol stall.
// To clear the host side halt we need to issue a reset endpoint
// command, followed by a set dequeue command to move past the
// TD.
// Class drivers clear the device side halt from a functional
// stall later. Hub TT buffer should only be cleared for FS/LS
// devices behind HS hubs for functional stalls.
if !((unsafe { (*ep).ep_index }) == 0 && trb_comp_code == 6) {
xhci_clear_hub_tt_buffer(xhci, td, ep);
}
xhci_handle_halted_endpoint(xhci, ep, td, 0u32);
return; // xhci_handle_halted_endpoint marked td cancelled
}
xhci_dequeue_td(xhci, td, ep_ring, (unsafe { (*td).status }) as u32);
}

/// sum trb lengths from the first trb up to stop_trb, _excluding_ stop_trb
fn sum_trb_lengths(td: *mut xhci_td, stop_trb: *mut xhci_trb) -> u32 {
let mut sum: u32 = 0;
let mut trb: *mut xhci_trb = unsafe { (*td).start_trb };
let mut seg: *mut xhci_segment = unsafe { (*td).start_seg };
while trb != stop_trb {
let __v0: bool = !trb_is_noop(trb) && !trb_is_link(trb);
if __v0 {
let __v1: u32 = sum;
sum = __v1.wrapping_add(TRB_LEN!(unsafe { *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) }));
}
next_trb(unsafe { &mut (*core::ptr::addr_of_mut!(seg)) }, unsafe { &mut (*core::ptr::addr_of_mut!(trb)) });
}
sum
}

/// Process control tds, update urb status and actual_length.
fn process_ctrl_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, ep_ring: *mut xhci_ring, td: *mut xhci_td, ep_trb: *mut xhci_trb, event: *mut xhci_transfer_event) {
let mut trb_type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(trb_type) as *mut u32;
let mut requested: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(requested) as *mut u32;
let mut remaining: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(remaining) as *mut u32;
let mut trb_comp_code: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(trb_comp_code) as *mut u32;
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
unsafe {
'__join_0: {
'__join_5: {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = TRB_FIELD_TO_TYPE!(unsafe { *((*ep_trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) });
*__slate_slot_4 = xhci_get_ep_ctx(xhci, (*(*ep).vdev).out_ctx, (*ep).ep_index);
*__slate_slot_3 = GET_COMP_CODE!(unsafe { (*event).transfer_len });
*__slate_slot_1 = (*(*td).urb).transfer_buffer_length;
*__slate_slot_2 = EVENT_TRB_LEN!(unsafe { (*event).transfer_len });
let __t1: u32 = *__slate_slot_3;
if __t1 == 1 {
if *__slate_slot_0 != 4 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: Success on ctrl %s TRB without IOC set?\n".as_ptr() as *const u8, if *__slate_slot_0 == 3 { c"data".as_ptr() as *mut u8 } else { c"setup".as_ptr() as *mut u8 });
(*td).status = 108i32.wrapping_neg();
break '__join_5;
} else {
(*td).status = 0;
break '__join_5;
}
} else {
if __t1 == 13 {
(*td).status = 0;
break '__join_5;
} else {
if __t1 == 28 {
if *__slate_slot_0 == 3 || *__slate_slot_0 == 1 {
(*(*td).urb).actual_length = *__slate_slot_2;
break '__join_0;
} else {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: Stopped Short Packet on ctrl setup or status TRB\n".as_ptr() as *const u8);
break '__join_0;
}
} else {
if __t1 == 26 {
let __t0: u32 = *__slate_slot_0;
if __t0 == 2 {
(*(*td).urb).actual_length = 0;
break '__join_0;
} else {
if __t0 != 3 {
if __t0 != 1 {
if __t0 == 4 {
(*(*td).urb).actual_length = *__slate_slot_1;
break '__join_0;
} else {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: unexpected TRB Type %d\n".as_ptr() as *const u8, *__slate_slot_0);
break '__join_0;
}
}
}
}
(*(*td).urb).actual_length = (*__slate_slot_1).wrapping_sub(*__slate_slot_2);
break '__join_0;
} else {
if __t1 == 27 {
break '__join_0;
} else {
if __t1 != 6 {
if !xhci_halted_host_endpoint(xhci, *__slate_slot_4, *__slate_slot_3) {
break '__join_5;
}
}
}
}
}
}
}
if *__slate_slot_0 == 3 || *__slate_slot_0 == 1 {
(*(*td).urb).actual_length = (*__slate_slot_1).wrapping_sub(*__slate_slot_2);
} else {
if !(*td).urb_length_set {
(*(*td).urb).actual_length = 0;
}
}
// Did we transfer part of the data (middle) phase?
break '__join_0;
}
// stopped at setup stage, no data transferred
if *__slate_slot_0 != 2 {
// if on data stage then update the actual_length of the URB and flag it
// as set, so it won't be overwritten in the event for the last TRB.
if *__slate_slot_0 == 3 || *__slate_slot_0 == 1 {
(*td).urb_length_set = true;
(*(*td).urb).actual_length = (*__slate_slot_1).wrapping_sub(*__slate_slot_2);
return;
} else {
// at status stage
if !(*td).urb_length_set {
(*(*td).urb).actual_length = *__slate_slot_1;
}
}
}
}
finish_td(xhci, ep, ep_ring, td, *__slate_slot_3);
}
}

/// Process isochronous tds, update urb packet status and actual_length.
fn process_isoc_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, ep_ring: *mut xhci_ring, td: *mut xhci_td, ep_trb: *mut xhci_trb, event: *mut xhci_transfer_event) {
let mut ep_trb_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(ep_trb_len) as *mut u32;
let mut requested: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(requested) as *mut u32;
let mut remaining: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(remaining) as *mut u32;
let mut sum_trbs_for_length: bool = unsafe { core::mem::zeroed() };
let mut trb_comp_code: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(trb_comp_code) as *mut u32;
let mut frame: core::mem::MaybeUninit<*mut usb_iso_packet_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut usb_iso_packet_descriptor = core::ptr::addr_of_mut!(frame) as *mut *mut usb_iso_packet_descriptor;
let mut idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(idx) as *mut i32;
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
unsafe {
'__join_7: {
'__join_18: {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
sum_trbs_for_length = false;
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_7 = GET_COMP_CODE!(unsafe { (*event).transfer_len });
*__slate_slot_10 = (*(*td).urb).hcpriv as *mut urb_priv;
*__slate_slot_9 = (*(*__slate_slot_10)).num_tds_done;
*__slate_slot_8 = (core::ptr::addr_of_mut!((*(*td).urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(*__slate_slot_9 as isize);
*__slate_slot_4 = (*(*__slate_slot_8)).length;
*__slate_slot_5 = EVENT_TRB_LEN!(unsafe { (*event).transfer_len });
*__slate_slot_3 = TRB_LEN!(unsafe { *((*ep_trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) });
// handle completion code
let __t0: u32 = *__slate_slot_7;
if __t0 == 1 {
if (*td).error_mid_td {
break '__join_7;
} else {
// Don't overwrite status if TD had an error, see xHCI 4.9.1
if *__slate_slot_5 != 0 {
sum_trbs_for_length = true;
}
(*(*__slate_slot_8)).status = 0;
break '__join_7;
}
} else {
if __t0 == 13 {
(*(*__slate_slot_8)).status = 0;
sum_trbs_for_length = true;
break '__join_7;
} else {
if __t0 == 18 {
(*(*__slate_slot_8)).status = 70i32.wrapping_neg();
break '__join_7;
} else {
if __t0 == 3 {
sum_trbs_for_length = true;
break '__join_18;
} else {
if __t0 == 31 {
break '__join_18;
} else {
if __t0 == 23 {
(*(*__slate_slot_8)).status = 18i32.wrapping_neg();
sum_trbs_for_length = true;
if ep_trb != (*td).end_trb {
(*td).error_mid_td = true;
break '__join_7;
} else {
break '__join_7;
}
} else {
if __t0 != 22 {
if __t0 != 6 {
if __t0 == 4 {
(*(*__slate_slot_8)).status = 71i32.wrapping_neg();
sum_trbs_for_length = true;
if ep_trb != (*td).end_trb {
(*td).error_mid_td = true;
break '__join_7;
} else {
break '__join_7;
}
} else {
if __t0 == 26 {
sum_trbs_for_length = true;
break '__join_7;
} else {
if __t0 == 28 {
(*(*__slate_slot_8)).status = 0;
// field normally containing residue now contains transferred
*__slate_slot_4 = *__slate_slot_5;
break '__join_7;
} else {
if __t0 == 27 {
sum_trbs_for_length = true; // exclude stopped trb with invalid length from length sum
*__slate_slot_3 = 0;
*__slate_slot_5 = 0;
break '__join_7;
} else {
sum_trbs_for_length = true;
(*(*__slate_slot_8)).status = 1i32.wrapping_neg();
break '__join_7;
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
}
(*(*__slate_slot_8)).status = 71i32.wrapping_neg();
break '__join_7;
}
(*(*__slate_slot_8)).status = 75i32.wrapping_neg();
if ep_trb != (*td).end_trb {
(*td).error_mid_td = true;
}
}
if !(*td).urb_length_set {
if sum_trbs_for_length {
(*(*__slate_slot_8)).actual_length = sum_trb_lengths(td, ep_trb).wrapping_add(*__slate_slot_3).wrapping_sub(*__slate_slot_5);
} else {
(*(*__slate_slot_8)).actual_length = *__slate_slot_4;
}
let __v2: *mut urb = (*td).urb;
let __v1: u32 = (*__v2).actual_length;
(*__v2).actual_length = __v1.wrapping_add((*(*__slate_slot_8)).actual_length);
}
if (*td).error_mid_td && ep_trb != (*td).end_trb {
(*td).urb_length_set = true;
return;
} else {
// Don't give back TD yet if we encountered an error mid TD
finish_td(xhci, ep, ep_ring, td, *__slate_slot_7);
}
}
}

/// Process bulk and interrupt tds, update urb status and actual_length.
fn process_bulk_intr_td(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, ep_ring: *mut xhci_ring, td: *mut xhci_td, ep_trb: *mut xhci_trb, event: *mut xhci_transfer_event) {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut ep_trb_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(ep_trb_len) as *mut u32;
let mut requested: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(requested) as *mut u32;
let mut remaining: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(remaining) as *mut u32;
let mut trb_comp_code: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(trb_comp_code) as *mut u32;
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
unsafe {
'__join_3: {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = xhci_get_slot_ctx(xhci, (*(*ep).vdev).out_ctx);
*__slate_slot_7 = GET_COMP_CODE!(unsafe { (*event).transfer_len });
*__slate_slot_6 = EVENT_TRB_LEN!(unsafe { (*event).transfer_len });
*__slate_slot_4 = TRB_LEN!(unsafe { *((*ep_trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) });
*__slate_slot_5 = (*(*td).urb).transfer_buffer_length;
let __t0: u32 = *__slate_slot_7;
if __t0 == 1 {
(*ep).err_count = 0;
// handle success with untransferred data as short packet
if ep_trb != (*td).end_trb || *__slate_slot_6 != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Successful completion on short TX\n".as_ptr() as *const u8);
}
(*td).status = 0;
} else {
if __t0 == 13 {
(*ep).err_count = 0;
(*td).status = 0;
} else {
if __t0 == 28 {
(*(*td).urb).actual_length = *__slate_slot_6;
break '__join_3;
} else {
if __t0 == 27 {
(*(*td).urb).actual_length = sum_trb_lengths(td, ep_trb);
// stopped on ep trb with invalid length, exclude it
break '__join_3;
} else {
if __t0 == 4 {
if (*xhci).quirks & BIT_ULL!(40i32) != 0 {
__v3 = true;
} else {
let __v1: u32 = (*ep).err_count;
(*ep).err_count = __v1.wrapping_add(1);
__v3 = __v1 > 3;
}
if !(__v3 || (*(*__slate_slot_8)).tt_info & 255 != 0) {
(*td).status = 0;
xhci_handle_halted_endpoint(xhci, ep, td, 1u32);
return;
}
}
}
}
}
}
if ep_trb == (*td).end_trb {
(*(*td).urb).actual_length = (*__slate_slot_5).wrapping_sub(*__slate_slot_6);
} else {
(*(*td).urb).actual_length = sum_trb_lengths(td, ep_trb).wrapping_add(*__slate_slot_4).wrapping_sub(*__slate_slot_6);
}
}
if *__slate_slot_6 > *__slate_slot_5 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"bad transfer trb length %d in event trb\n".as_ptr() as *const u8, *__slate_slot_6);
(*(*td).urb).actual_length = 0;
}
finish_td(xhci, ep, ep_ring, td, *__slate_slot_7);
}
// do nothing
}

/// Transfer events which don't point to a transfer TRB, see xhci 4.17.4
fn handle_transferless_tx_event(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, trb_comp_code: u32) -> i32 {
match trb_comp_code {
4 | 6 | 10 | 34 => {
let __v1: u32 = unsafe { (*ep).err_count };
unsafe {
(*ep).err_count = __v1.wrapping_add(1);
}
if __v1 > 3 {
xhci_handle_halted_endpoint(xhci, ep, core::ptr::null_mut::<xhci_td>(), 0u32);
} else {
xhci_handle_halted_endpoint(xhci, ep, core::ptr::null_mut::<xhci_td>(), 1u32);
}
}
14 | 15 | 27 => {
}
_ => {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Transfer event %u for unknown stream ring slot %u ep %u\n".as_ptr() as *const u8, trb_comp_code, (*(*ep).vdev).slot_id, (*ep).ep_index) };
return 19i32.wrapping_neg();
}
}
0
}

fn xhci_spurious_success_tx_event(xhci: *mut xhci_hcd, ring: *mut xhci_ring) -> bool {
match unsafe { (*ring).old_trb_comp_code } {
13 => {
return (unsafe { (*xhci).quirks }) & BIT_ULL!(4i32) != 0;
}
3 | 4 | 31 => {
return (unsafe { (*xhci).quirks }) & BIT_ULL!(49i32) != 0 && ((unsafe { (*ring).r#type }) as u32) == 1;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn find_td_by_dma(ep_ring: *mut xhci_ring, dma: u64) -> *mut xhci_td {
let mut td: *mut xhci_td = unsafe { core::mem::zeroed() };
if dma != 0 {
td = list_first_entry__13!(unsafe { core::ptr::addr_of_mut!((*ep_ring).td_list) });
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*ep_ring).td_list) }, td_list) == 0 {
if trb_in_td(td, dma) {
return td;
}
td = list_next_entry__10!(td, td_list);
}
}
core::ptr::null_mut::<xhci_td>()
}

// If this function returns an error condition, it means it got a Transfer
// event with a corrupted Slot ID, Endpoint ID, or TRB DMA address.
// At this point, the host controller is probably hosed and should be reset.
fn handle_tx_event(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter, event: *mut xhci_transfer_event) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Don't print wanings if ring is empty due to a stopped endpoint generating an
// extra completion event if the device was suspended. Or, a event for the last TRB
// of a short TD we already got a short event for. The short TD is already removed
// from the TD list.
let mut __v1: bool = unsafe { core::mem::zeroed() };
// We don't know how many TDs were missed when ep_trb_dma is zero (as permitted by
// xHCI 1.0) or bogus. Bail out leaving ep->skip set, next event will sort it out.
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut pos__: *mut list_head = unsafe { core::mem::zeroed() };
let mut head__: *mut list_head = unsafe { core::mem::zeroed() };
let mut ring_xrun_event: bool = unsafe { core::mem::zeroed() };
let mut trb_comp_code: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(trb_comp_code) as *mut u32;
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut ep_trb: core::mem::MaybeUninit<*mut xhci_trb> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut xhci_trb = core::ptr::addr_of_mut!(ep_trb) as *mut *mut xhci_trb;
let mut ep_trb_dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(ep_trb_dma) as *mut u64;
let mut td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut ep_index: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(ep_index) as *mut i32;
let mut slot_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(slot_id) as *mut u32;
let mut ep_ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ep_ring) as *mut *mut xhci_ring;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
unsafe {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
td = core::ptr::null_mut::<xhci_td>();
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
status = 115i32.wrapping_neg();
let __slate_auto_bytes_6 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ring_xrun_event = false;
*__slate_slot_17 = TRB_TO_SLOT_ID!(unsafe { (*event).flags });
*__slate_slot_16 = TRB_TO_EP_ID!(unsafe { (*event).flags }).wrapping_sub(1) as i32;
*__slate_slot_10 = GET_COMP_CODE!(unsafe { (*event).transfer_len });
*__slate_slot_14 = (*event).buffer;
*__slate_slot_19 = xhci_get_virt_ep(xhci, *__slate_slot_17, *__slate_slot_16 as u32);
if (*__slate_slot_19).is_null() {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR Invalid Transfer event\n".as_ptr() as *const u8);
} else {
*__slate_slot_18 = xhci_dma_to_transfer_ring(*__slate_slot_19, *__slate_slot_14);
*__slate_slot_11 = xhci_get_ep_ctx(xhci, (*(*(*__slate_slot_19)).vdev).out_ctx, *__slate_slot_16 as u32);
if GET_EP_CTX_STATE!(*__slate_slot_11) == 0 {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR Transfer event for disabled endpoint slot %u ep %u\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
} else {
if (*__slate_slot_18).is_null() {
return handle_transferless_tx_event(xhci, *__slate_slot_19, *__slate_slot_10);
} else {
// find the transfer trb this events points to
*__slate_slot_13 = xhci_dma_to_trb((*(*__slate_slot_18)).deq_seg, *__slate_slot_14, core::ptr::null_mut::<*mut xhci_segment>());
// Look for common error cases
let __t0: u32 = *__slate_slot_10;
if __t0 == 1 {
if EVENT_TRB_LEN!(unsafe { (*event).transfer_len }) != 0 {
*__slate_slot_10 = 13;
}
} else {
if __t0 != 13 {
if __t0 != 26 {
if __t0 != 27 {
if __t0 != 28 {
if __t0 == 6 {
status = 32i32.wrapping_neg();
} else {
if __t0 == 36 {
status = 71i32.wrapping_neg();
} else {
if __t0 == 4 {
status = 71i32.wrapping_neg();
} else {
if __t0 == 3 {
status = 75i32.wrapping_neg();
} else {
if __t0 == 5 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: TRB error for slot %u ep %u on endpoint\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
status = 84i32.wrapping_neg();
} else {
if __t0 == 2 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: HC couldn't access mem fast enough for slot %u ep %u\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
status = 63i32.wrapping_neg();
} else {
if __t0 == 18 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: bandwidth overrun event for slot %u ep %u on endpoint\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
} else {
if __t0 == 31 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: buffer overrun event for slot %u ep %u on endpoint".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
} else {
if __t0 == 14 {
// When the Isoch ring is empty, the xHC will generate
// a Ring Overrun Event for IN Isoch endpoint or Ring
// Underrun Event for OUT Isoch endpoint.
ring_xrun_event = true;
} else {
if __t0 == 15 {
ring_xrun_event = true;
} else {
if __t0 == 23 {
(*(*__slate_slot_19)).skip = true;
// When encounter missed service error, one or more isoc tds
// may be missed by xHC.
// Set skip flag of the ep_ring; Complete the missed tds as
// short transfer when process the ep_ring next time.
} else {
if __t0 == 20 {
(*(*__slate_slot_19)).skip = true;
return 0;
} else {
if __t0 == 22 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: detect an incompatible device for slot %u ep %u".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16); // needs disable slot command to recover
status = 71i32.wrapping_neg();
} else {
if xhci_is_vendor_info_code(xhci, *__slate_slot_10) != 0 {
status = 0;
} else {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR Unknown event condition %u for slot %u ep %u , HC probably busted\n".as_ptr() as *const u8, *__slate_slot_10, *__slate_slot_17, *__slate_slot_16);
if !(*(*__slate_slot_19)).skip {
return 0;
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
}
}
}
}
}
}
}
}
}
head__ = core::ptr::addr_of_mut!((*(*__slate_slot_18)).td_list);
while false {
}
let __v5: *mut list_head = core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head);
pos__ = __v5;
if pos__ != head__ {
__v4 = list_entry__3!(pos__, xhci_td);
} else {
__v4 = core::ptr::null_mut::<xhci_td>();
}
// xhci 4.10.2 states isoc endpoints should continue
// processing the next TD if there was an error mid TD.
// So host like NEC don't generate an event for the last
// isoc TRB even if the IOC flag is set.
// xhci 4.9.1 states that if there are errors in mult-TRB
// TDs xHC should generate an error for that TRB, and if xHC
// proceeds to the next TD it should genete an event for
// any TRB with IOC flag on the way. Other host follow this.
//
// We wait for the final IOC event, but if we get an event
// anywhere outside this TD, just give it back already.
td = __v4;
if !td.is_null() && (*td).error_mid_td {
__v3 = !trb_in_td(td, *__slate_slot_14);
} else {
__v3 = false;
}
if __v3 {
xhci_dequeue_td(xhci, td, *__slate_slot_18, (*td).status as u32);
}
if *__slate_slot_10 == 23 {
__v2 = find_td_by_dma(*__slate_slot_18, *__slate_slot_14).is_null();
} else {
__v2 = false;
}
if __v2 {
return 0;
} else {
'__join_6: {
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_18)).td_list) as *const list_head) != 0 {
if *__slate_slot_10 != 26 && *__slate_slot_10 != 27 && !ring_xrun_event {
__v1 = !xhci_spurious_success_tx_event(xhci, *__slate_slot_18);
} else {
__v1 = false;
}
if __v1 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Event TRB for slot %u ep %u with no TDs queued\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
}
(*(*__slate_slot_19)).skip = false;
} else {
'__join_15: {
'__join_27: {
loop {
td = list_first_entry__4!(core::ptr::addr_of_mut!((*(*__slate_slot_18)).td_list), xhci_td);
// Is this TRB not part of the currently executing TD?
if !trb_in_td(td, *__slate_slot_14) {
if (*(*__slate_slot_19)).skip {
__v0 = usb_endpoint_xfer_isoc(core::ptr::addr_of_mut!((*(*(*td).urb).ep).desc) as *const usb_endpoint_descriptor) != 0;
} else {
__v0 = false;
}
if __v0 {
// this event is unlikely to match any TD, don't skip them all
if *__slate_slot_10 == 27 {
return 0;
} else {
// TD was missed, skip it. Core already initialized frame->status
// to -EXDEV and frame->actual_length to 0, nothing more to do.
xhci_dequeue_td(xhci, td, *__slate_slot_18, 0u32);
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_18)).td_list) as *const list_head) == 0 {
if ring_xrun_event {
// If we are here, we are on xHCI 1.0 host with no
// idea how many TDs were missed or where the xrun
// occurred. New TDs may have been added after the
// xrun, so skip only one TD to be safe.
return 0;
}
} else {
break '__join_27;
}
}
} else {
break;
}
} else {
if (*(*__slate_slot_19)).skip {
(*(*__slate_slot_19)).skip = false;
}
// If ep->skip is set, it means there are missed tds on the
// endpoint ring need to take care of.
// Process them as short transfer until reach the td pointed by
// the event.
}
if !(*(*__slate_slot_19)).skip {
break '__join_15;
}
}
// TD was queued after xrun, maybe xrun was on a link, don't panic yet
if ring_xrun_event {
return 0;
} else {
// Skip the Force Stopped Event. The 'ep_trb' of FSE is not in the current
// TD pointed by 'ep_ring->dequeue' because that the hardware dequeue
// pointer still at the previous TRB of the current TD. The previous TRB
// maybe a Link TD or the last TRB of the previous TD. The command
// completion handle will take care the rest.
if *__slate_slot_10 == 26 || *__slate_slot_10 == 27 {
return 0;
} else {
// Some hosts give a spurious success event after a short
// transfer or error on last TRB. Ignore it.
if xhci_spurious_success_tx_event(xhci, *__slate_slot_18) {
(*(*__slate_slot_18)).old_trb_comp_code = 0;
return 0;
} else {
// HC is busted, give up!
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Event dma %pad for ep %d status %d not part of TD at %016llx - %016llx\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_14), *__slate_slot_16, *__slate_slot_10, xhci_trb_virt_to_dma((*td).start_seg, (*td).start_trb), xhci_trb_virt_to_dma((*td).end_seg, (*td).end_trb));
return 108i32.wrapping_neg();
}
}
}
}
(*(*__slate_slot_19)).skip = false;
td = core::ptr::null_mut::<xhci_td>();
break '__join_6;
}
(*(*__slate_slot_18)).old_trb_comp_code = *__slate_slot_10;
// Get out if a TD was queued at enqueue after the xrun occurred
if ring_xrun_event {
return 0;
} else {
trace_xhci_handle_transfer(*__slate_slot_18, *__slate_slot_13 as *mut xhci_generic_trb, *__slate_slot_14);
// No-op TRB could trigger interrupts in a case where a URB was killed
// and a STALL_ERROR happens right after the endpoint ring stopped.
// Reset the halted endpoint. Otherwise, the endpoint remains stalled
// indefinitely.
if !trb_is_noop(*__slate_slot_13) {
(*td).status = status;
// update the urb's actual_length and give back to the core
if usb_endpoint_xfer_control(core::ptr::addr_of_mut!((*(*(*td).urb).ep).desc) as *const usb_endpoint_descriptor) != 0 {
process_ctrl_td(xhci, *__slate_slot_19, *__slate_slot_18, td, *__slate_slot_13, event);
} else {
if usb_endpoint_xfer_isoc(core::ptr::addr_of_mut!((*(*(*td).urb).ep).desc) as *const usb_endpoint_descriptor) != 0 {
process_isoc_td(xhci, *__slate_slot_19, *__slate_slot_18, td, *__slate_slot_13, event);
} else {
process_bulk_intr_td(xhci, *__slate_slot_19, *__slate_slot_18, td, *__slate_slot_13, event);
}
}
return 0;
}
}
}
}
if xhci_halted_host_endpoint(xhci, *__slate_slot_11, *__slate_slot_10) {
xhci_handle_halted_endpoint(xhci, *__slate_slot_19, td, 0u32);
}
return 0;
}
}
}
}
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"@%016llx %08x %08x %08x %08x\n".as_ptr() as *const u8, xhci_trb_virt_to_dma((*(*ir).event_ring).deq_seg, (*(*ir).event_ring).dequeue), lower_32_bits!(unsafe { (*event).buffer }), upper_32_bits!(unsafe { (*event).buffer }), (*event).transfer_len, (*event).flags);
return 19i32.wrapping_neg();
}
// Skip codes that require special handling depending on
// transfer type
// Completion codes for endpoint stopped state
// Completion codes for endpoint halted state
// Completion codes for endpoint error state
// completion codes not indicating endpoint state change
}

/// This function handles one OS-owned event on the event ring. It may drop
/// xhci->lock between event processing (e.g. to pass up port status changes).
fn xhci_handle_event_trb(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter, event: *mut xhci_trb) -> i32 {
let mut trb_type: u32 = 0;
trace_xhci_handle_event(unsafe { (*ir).event_ring }, unsafe { core::ptr::addr_of_mut!((*event).generic) }, xhci_trb_virt_to_dma(unsafe { (*(*ir).event_ring).deq_seg }, unsafe { (*(*ir).event_ring).dequeue }));
// Barrier between reading the TRB_CYCLE (valid) flag before, and any
// speculative reads of the event's flags/data below.
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("lfence", options(nostack, att_syntax));
}
if true {
break;
}
}
trb_type = TRB_FIELD_TO_TYPE!(unsafe { (*event).event_cmd.flags });
// FIXME: Handle more event types.
match trb_type {
33 => {
handle_cmd_completion(xhci, unsafe { core::ptr::addr_of_mut!((*event).event_cmd) });
}
34 => {
handle_port_status(xhci, event);
}
32 => {
handle_tx_event(xhci, ir, unsafe { core::ptr::addr_of_mut!((*event).trans_event) });
}
38 => {
handle_device_notification(xhci, event);
}
_ => {
if trb_type >= 48 {
handle_vendor_event(xhci, event, trb_type);
} else {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR unknown event type %d\n".as_ptr() as *const u8, trb_type) };
}
}
}
// Any of the above functions may drop and re-acquire the lock, so check
// to make sure a watchdog timer didn't mark the host as non-responsive.
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
return 19i32.wrapping_neg();
}
0
}

/// Update Event Ring Dequeue Pointer:
/// - When all events have finished
/// - To avoid "Event Ring Full Error" condition
#[unsafe(no_mangle)]
extern "C" fn xhci_update_erst_dequeue(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter, clear_ehb: bool) {
let mut temp_64: u64 = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
let deq: u64 = xhci_trb_virt_to_dma(unsafe { (*(*ir).event_ring).deq_seg }, unsafe { (*(*ir).event_ring).dequeue });
if deq == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN something wrong with SW event ring dequeue ptr\n".as_ptr() as *const u8) };
}
// Per 4.9.4, Software writes to the ERDP register shall always advance
// the Event Ring Dequeue Pointer value.
if temp_64 & GENMASK_ULL__3!(63i32, 4i32) == deq & GENMASK_ULL__16!(63i32, 4i32) && !clear_ehb {
return;
}
// Update HC event ring dequeue pointer
temp_64 = ((unsafe { (*(*(*ir).event_ring).deq_seg).num }) & 7) as u64;
let __v1: u64 = temp_64;
temp_64 = __v1 | deq & GENMASK_ULL__8!(63i32, 4i32);
// Clear the event handler busy flag (RW1C)
if clear_ehb {
let __v3: u64 = temp_64;
temp_64 = __v3 | BIT_ULL!(3i32);
}
xhci_write_64(xhci, temp_64, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
}

/// Clear the interrupt pending bit for a specific interrupter.
fn xhci_clear_interrupt_pending(ir: *mut xhci_interrupter) {
if !unsafe { (*ir).ip_autoclear } {
let mut iman: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
let __v0: u32 = iman;
iman = ((__v0 as u64) | BIT!(0i32)) as u32;
writel(iman, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *mut ());
// Read operation to guarantee the write has been flushed from posted buffers
readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
}
}

/// Handle all OS-owned events on an interrupter event ring. It may drop
/// and reaquire xhci->lock between event processing.
fn xhci_handle_events(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter, skip_events: bool) -> i32 {
let mut event_loop: i32 = 0;
let mut err: i32 = 0;
let mut temp: u64 = 0;
xhci_clear_interrupt_pending(ir);
// Event ring hasn't been allocated yet.
if unsafe { (*ir).event_ring }.is_null() || unsafe { (*(*ir).event_ring).dequeue }.is_null() {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR interrupter event ring not ready\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 || ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(1i32) != 0 {
// Clear the event handler busy flag (RW1C)
temp = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
xhci_write_64(xhci, temp | BIT_ULL!(3i32), unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
return 19i32.wrapping_neg();
}
// Process all OS owned event TRBs on this event ring
while unhandled_event_trb(unsafe { (*ir).event_ring }) {
if !skip_events {
err = xhci_handle_event_trb(xhci, ir, unsafe { (*(*ir).event_ring).dequeue });
}
// If half a segment of events have been handled in one go then
// update ERDP, and force isoc trbs to interrupt more often
let __v1: i32 = event_loop;
event_loop = __v1.wrapping_add(1);
if __v1 > 256 / 2 {
xhci_update_erst_dequeue(xhci, ir, false);
if (unsafe { (*ir).isoc_bei_interval }) > 8 {
unsafe {
(*ir).isoc_bei_interval = (*ir).isoc_bei_interval / 2;
}
}
event_loop = 0;
}
// Update SW event ring dequeue pointer
inc_deq(xhci, unsafe { (*ir).event_ring });
if err != 0 {
break;
}
}
xhci_update_erst_dequeue(xhci, ir, true);
0
}

/// Move the event ring dequeue pointer to skip events kept in the secondary
/// event ring.  This is used to ensure that pending events in the ring are
/// acknowledged, so the xHCI HCD can properly enter suspend/resume.  The
/// secondary ring is typically maintained by an external component.
#[unsafe(no_mangle)]
extern "C" fn xhci_skip_sec_intr_events(xhci: *mut xhci_hcd, ring: *mut xhci_ring, ir: *mut xhci_interrupter) {
let mut erdp_reg: u64 = 0;
let mut deq: u64 = 0;
// disable irq, ack pending interrupt and ack all pending events
unsafe { xhci_disable_interrupter(xhci, ir) };
// last acked event trb is in erdp reg
erdp_reg = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
deq = erdp_reg & GENMASK_ULL__19!(63i32, 4i32);
if deq == 0 {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"event ring handling not required\n".as_ptr() as *const u8) };
return;
}
let current_trb: *mut xhci_trb = unsafe { (*(*ir).event_ring).dequeue };
// read cycle state of the last acked trb to find out CCS
unsafe {
(*ring).cycle_state = (((*current_trb).event_cmd.flags as u64) & BIT!(0i32)) as u32;
}
xhci_handle_events(xhci, ir, true);
}

/// xHCI spec says we can get an interrupt, and if the HC has an error condition,
/// we might get bad data out of the event ring.  Section 4.10.2.7 has a list of
/// indicators of an event TRB error, but we check the status *first* to be safe.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_ring.xhci_irq")]
extern "C" fn xhci_irq(hcd: *mut usb_hcd) -> u32 {
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut ret: u32 = unsafe { core::mem::zeroed() };
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
xhci = hcd_to_xhci(hcd);
ret = 1;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
spin_lock(core::ptr::addr_of_mut!((*xhci).lock));
// Check if the xHC generated the interrupt, or the irq is shared
*__slate_slot_0 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *const ());
if *__slate_slot_0 == !0 {
xhci_hc_died(xhci);
} else {
if (*__slate_slot_0 as u64) & BIT!(3i32) == 0 {
ret = 0;
} else {
if (*__slate_slot_0 as u64) & BIT!(12i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARNING: Host Controller Error\n".as_ptr() as *const u8);
xhci_halt(xhci);
} else {
if (*__slate_slot_0 as u64) & BIT!(2i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARNING: Host System Error\n".as_ptr() as *const u8);
xhci_halt(xhci);
} else {
// Clear the op reg interrupt status first,
// so we can receive interrupts from other MSI-X interrupters.
// USBSTS bits are write 1 to clear.
writel(BIT!(3i32) as u32, core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut ());
// This is the handler of the primary interrupter
xhci_handle_events(xhci, *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize), false);
}
}
}
}
spin_unlock(core::ptr::addr_of_mut!((*xhci).lock));
return ret;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci_ring.xhci_msi_irq")]
extern "C" fn xhci_msi_irq(irq: i32, hcd: *mut ()) -> u32 {
xhci_irq(hcd as *mut usb_hcd)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_msi_irq_926: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, *mut ()>(Some(xhci_msi_irq)) };

// 		Endpoint Ring Operations
/// Generic function for queueing a TRB on a ring.
/// The caller must have checked to make sure there's room on the ring.
///
/// @more_trbs_coming:	Will you enqueue more TRBs before calling
/// 			prepare_transfer()?
fn queue_trb(xhci: *mut xhci_hcd, ring: *mut xhci_ring, more_trbs_coming: bool, field1: u32, field2: u32, field3: u32, field4: u32) {
let trb: *mut xhci_generic_trb = unsafe { core::ptr::addr_of_mut!((*(*ring).enqueue).generic) };
unsafe {
*((*trb).field.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = field1;
}
unsafe {
*((*trb).field.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = field2;
}
unsafe {
*((*trb).field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) = field3;
}
// make sure TRB is fully written before giving it to the controller
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
*((*trb).field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize) = field4;
}
trace_xhci_queue_trb(ring, trb, xhci_trb_virt_to_dma(unsafe { (*ring).enq_seg }, unsafe { (*ring).enqueue }));
inc_enq(xhci, ring, more_trbs_coming);
}

/// Does various checks on the endpoint ring, and makes it ready to queue num_trbs.
/// expand ring if it start to be full.
fn prepare_ring(xhci: *mut xhci_hcd, ep_ring: *mut xhci_ring, ep_state: u32, num_trbs: u32, mem_flags: u32) -> i32 {
let mut new_segs: u32 = 0;
// Make sure the endpoint has been added to xHC schedule
match ep_state {
0 => {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN urb submitted to disabled ep\n".as_ptr() as *const u8) };
// USB core changed config/interfaces without notifying us,
// or hardware is reporting the wrong state.
return 2i32.wrapping_neg();
}
4 => {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN waiting for error on ep to be cleared\n".as_ptr() as *const u8) };
// FIXME event handling code for error needs to clear it
//
// XXX not sure if this should be -ENOENT or not
return 22i32.wrapping_neg();
}
2 => {
}
1 | 3 => {
}
_ => {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR unknown endpoint state for ep\n".as_ptr() as *const u8) };
// FIXME issue Configure Endpoint command to try to get the HC
// back into a known state.
return 22i32.wrapping_neg();
}
}
if ep_ring != unsafe { (*xhci).cmd_ring } {
new_segs = xhci_ring_expansion_needed(xhci, ep_ring, num_trbs);
} else {
if xhci_num_trbs_free(ep_ring) <= num_trbs {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Do not support expand command ring\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
}
if new_segs != 0 {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_ring_expansion), c"ERROR no room on ep ring, try ring expansion".as_ptr() as *const u8) };
if (unsafe { xhci_ring_expansion(xhci, ep_ring, new_segs, mem_flags) }) != 0 {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ring expansion failed\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
}
// Ensure that new TRBs won't overwrite a link
if trb_is_link(unsafe { (*ep_ring).enqueue }) {
inc_enq_past_link(xhci, ep_ring, 0u32);
}
if last_trb_on_seg(unsafe { (*ep_ring).enq_seg }, unsafe { (*ep_ring).enqueue }) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Missing link TRB at end of ring segment\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
0
}

fn prepare_transfer(xhci: *mut xhci_hcd, xdev: *mut xhci_virt_device, ep_index: u32, stream_id: u32, num_trbs: u32, urb: *mut urb, td_index: u32, mem_flags: u32) -> i32 {
let mut ret: i32 = 0;
let mut urb_priv: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut td: *mut xhci_td = unsafe { core::mem::zeroed() };
let mut ep_ring: *mut xhci_ring = unsafe { core::mem::zeroed() };
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*xdev).out_ctx, ep_index) };
ep_ring = xhci_triad_to_transfer_ring(xhci, (unsafe { (*xdev).slot_id }) as u32, ep_index, stream_id);
if ep_ring.is_null() {
return 22i32.wrapping_neg();
}
ret = prepare_ring(xhci, ep_ring, GET_EP_CTX_STATE!(ep_ctx), num_trbs, mem_flags);
if ret != 0 {
return ret;
}
urb_priv = (unsafe { (*urb).hcpriv }) as *mut urb_priv;
td = unsafe { core::ptr::addr_of_mut!((*urb_priv).td) as *mut xhci_td }.wrapping_offset(td_index as isize);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*td).td_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*td).cancelled_td_list) });
if td_index == 0 {
ret = unsafe { usb_hcd_link_urb_to_ep(bus_to_hcd((*(*urb).dev).bus), urb) };
if unlikely__2!(ret) != 0 {
return ret;
}
}
unsafe {
(*td).urb = urb;
}
// Add this TD to the tail of the endpoint ring's TD list
list_add_tail(unsafe { core::ptr::addr_of_mut!((*td).td_list) }, unsafe { core::ptr::addr_of_mut!((*ep_ring).td_list) });
unsafe {
(*td).start_seg = (*ep_ring).enq_seg;
}
unsafe {
(*td).start_trb = (*ep_ring).enqueue;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn count_trbs(addr: u64, len: u64) -> u32 {
let mut num_trbs: u32 = (len.wrapping_add(addr & 65535u64).wrapping_add(65536u64).wrapping_sub(1) / 65536u64) as u32;
if num_trbs == 0 {
num_trbs = num_trbs.wrapping_add(1);
}
num_trbs
}

fn count_trbs_needed(urb: *mut urb) -> u32 {
count_trbs(unsafe { (*urb).transfer_dma }, (unsafe { (*urb).transfer_buffer_length }) as u64)
}

fn count_sg_trbs_needed(urb: *mut urb) -> u32 {
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
let mut len: u32 = 0;
let mut full_len: u32 = unsafe { (*urb).transfer_buffer_length };
let mut num_trbs: u32 = 0;
sg = unsafe { (*urb).sg };
while i < ((unsafe { (*urb).num_mapped_sgs }) as u32) {
len = sg_dma_len!(sg);
let __v1: u32 = count_trbs(sg_dma_address!(sg), len as u64);
num_trbs = num_trbs.wrapping_add(__v1);
len = min_t__4!(u32, len, full_len);
full_len = full_len.wrapping_sub(len);
if full_len == 0 {
break;
}
i = i.wrapping_add(1);
sg = sg_next(sg);
}
num_trbs
}

fn count_isoc_trbs_needed(urb: *mut urb, i: i32) -> u32 {
let addr: u64 = unsafe { (*urb).transfer_dma }.wrapping_add((unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).offset }) as u64);
let len: u64 = (unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).length }) as u64;
count_trbs(addr, len)
}

fn check_trb_math(urb: *mut urb, running_total: i32) {
if unlikely!((running_total as u32) != unsafe { (*urb).transfer_buffer_length }) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*urb).dev).dev) as *const device, c"%s - ep %#x - Miscalculated tx length, queued %#x (%d), asked for %#x (%d)\n".as_ptr() as *const u8, c"check_trb_math".as_ptr() as *mut u8, (*(*urb).ep).desc.bEndpointAddress as i32, running_total, running_total, (*urb).transfer_buffer_length, (*urb).transfer_buffer_length) };
}
}

fn giveback_first_trb(xhci: *mut xhci_hcd, slot_id: i32, ep_index: u32, stream_id: u32, start_cycle: i32, start_trb: *mut xhci_generic_trb) {
// Pass all the TRBs to the hardware at once and make sure this write
// isn't reordered.
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
if start_cycle != 0 {
let __v2: *mut u32 = unsafe { (*start_trb).field.as_mut_ptr() as *mut u32 }.wrapping_offset(3 as isize);
unsafe {
*__v2 = *__v2 | (start_cycle as u32);
}
} else {
let __v5: *mut u32 = unsafe { (*start_trb).field.as_mut_ptr() as *mut u32 }.wrapping_offset(3 as isize);
let __v6: u32 = unsafe { *__v5 };
let __v7: u32 = __v6 & (!BIT!(0i32) as u32);
unsafe {
*__v5 = __v7;
}
}
xhci_ring_ep_doorbell(xhci, slot_id as u32, ep_index, stream_id);
}

fn check_interval(urb: *mut urb, ep_ctx: *mut xhci_ep_ctx) {
let xhci_interval: i32 = EP_INTERVAL_TO_UFRAMES!(unsafe { (*ep_ctx).ep_info });
let mut ep_interval: i32 = unsafe { (*urb).interval };
// Convert to microframes
if ((unsafe { (*(*urb).dev).speed }) as u32) == 1 || ((unsafe { (*(*urb).dev).speed }) as u32) == 2 {
ep_interval = ep_interval.wrapping_mul(8);
}
// FIXME change this to a warning and a suggestion to use the new API
// to set the polling interval (once the API is added).
if xhci_interval != ep_interval {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*urb).dev).dev) as *const device, c"Driver uses different interval (%d microframe%s) than xHCI (%d microframe%s)\n".as_ptr() as *const u8, ep_interval, str_plural(ep_interval as u64), xhci_interval, str_plural(xhci_interval as u64)) };
}
unsafe {
(*urb).interval = xhci_interval;
}
// Convert back to frames for LS/FS devices
if ((unsafe { (*(*urb).dev).speed }) as u32) == 1 || ((unsafe { (*(*urb).dev).speed }) as u32) == 2 {
unsafe {
(*urb).interval = (*urb).interval / 8;
}
}
}
}

/// xHCI uses normal TRBs for both bulk and interrupt.  When the interrupt
/// endpoint is to be serviced, the xHC will consume (at most) one TD.  A TD
/// (comprised of sg list entries) can take several service intervals to
/// transmit.
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_intr_tx(xhci: *mut xhci_hcd, mem_flags: u32, urb: *mut urb, slot_id: i32, ep_index: u32) -> i32 {
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).out_ctx, ep_index) };
check_interval(urb, ep_ctx);
xhci_queue_bulk_tx(xhci, mem_flags, urb, slot_id, ep_index)
}

/// For xHCI 1.0 host controllers, TD size is the number of max packet sized
/// packets remaining in the TD (*not* including this TRB).
///
/// Total TD packet count = total_packet_count =
///     DIV_ROUND_UP(TD size in bytes / wMaxPacketSize)
///
/// Packets transferred up to and including this TRB = packets_transferred =
///     rounddown(total bytes transferred including this TRB / wMaxPacketSize)
///
/// TD size = total_packet_count - packets_transferred
///
/// For xHCI 0.96 and older, TD size field should be the remaining bytes
/// including this TRB, right shifted by 10
///
/// For all hosts it must fit in bits 21:17, so it can't be bigger than 31.
/// This is taken care of in the TRB_TD_SIZE() macro
///
/// The last TRB in a TD must have the TD size set to zero.
fn xhci_td_remainder(xhci: *mut xhci_hcd, transferred: i32, mut trb_buff_len: i32, td_total_len: u32, urb: *mut urb, more_trbs_coming: bool) -> u32 {
let mut maxp: u32 = 0;
let mut total_packet_count: u32 = 0;
// MTK xHCI 0.96 contains some features from 1.0
if ((unsafe { (*xhci).hci_version }) as i32) < 256 && (unsafe { (*xhci).quirks }) & BIT_ULL!(21i32) == 0 {
return td_total_len.wrapping_sub(transferred as u32) >> 10i32;
}
// One TRB with a zero-length data packet.
if !more_trbs_coming || transferred == 0 && trb_buff_len == 0 || (trb_buff_len as u32) == td_total_len {
return 0;
}
// for MTK xHCI 0.96, TD size include this TRB, but not in 1.x
if (unsafe { (*xhci).quirks }) & BIT_ULL!(21i32) != 0 && ((unsafe { (*xhci).hci_version }) as i32) < 256 {
trb_buff_len = 0;
}
maxp = (unsafe { xhci_usb_endpoint_maxp((*urb).dev, (*urb).ep) }) as u32;
total_packet_count = td_total_len.wrapping_add(maxp).wrapping_sub(1) / maxp;
// Queueing functions don't count the current TRB into transferred
total_packet_count.wrapping_sub((transferred.wrapping_add(trb_buff_len) as u32) / maxp)
}

fn xhci_align_td(xhci: *mut xhci_hcd, urb: *mut urb, enqd_len: u32, trb_buff_len: &mut u32, seg: *mut xhci_segment) -> i32 {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut unalign: u32 = 0;
let max_pkt: u32 = (unsafe { xhci_usb_endpoint_maxp((*urb).dev, (*urb).ep) }) as u32;
let mut new_buff_len: u32 = 0;
let mut len: u64 = 0;
unalign = enqd_len.wrapping_add(unsafe { *(trb_buff_len as *mut u32) }) % max_pkt;
// we got lucky, last normal TRB data on segment is packet aligned
if unalign == 0 {
return 0;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Unaligned %d bytes, buff len %d\n".as_ptr() as *const u8, unalign, *(trb_buff_len as *mut u32)) };
}
// is the last nornal TRB alignable by splitting it
if (unsafe { *(trb_buff_len as *mut u32) }) > unalign {
let __v1: u32 = unsafe { *trb_buff_len };
unsafe {
*trb_buff_len = __v1.wrapping_sub(unalign);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"split align, new buff len %d\n".as_ptr() as *const u8, *(trb_buff_len as *mut u32)) };
}
return 0;
}
// We want enqd_len + trb_buff_len to sum up to a number aligned to
// number which is divisible by the endpoint's wMaxPacketSize. IOW:
// (size of currently enqueued TRBs + remainder) % wMaxPacketSize == 0.
new_buff_len = max_pkt.wrapping_sub(enqd_len % max_pkt);
if new_buff_len > unsafe { (*urb).transfer_buffer_length }.wrapping_sub(enqd_len) {
new_buff_len = unsafe { (*urb).transfer_buffer_length }.wrapping_sub(enqd_len);
}
// create a max max_pkt sized bounce buffer pointed to by last trb
if usb_urb_dir_out(urb) != 0 {
if (unsafe { (*urb).num_sgs }) != 0 {
len = unsafe { sg_pcopy_to_buffer((*urb).sg, (*urb).num_sgs as u32, (*seg).bounce_buf, new_buff_len as u64, enqd_len as i64) };
if len != (new_buff_len as u64) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Wrong bounce buffer write length: %zu != %d\n".as_ptr() as *const u8, len, new_buff_len) };
}
} else {
unsafe { memcpy((*seg).bounce_buf, (((*urb).transfer_buffer as *const u8).wrapping_offset(enqd_len as isize) as *mut ()) as *const (), new_buff_len as u64) };
}
unsafe {
(*seg).bounce_dma = dma_map_single!(dev, unsafe { (*seg).bounce_buf }, max_pkt, 1i32);
}
} else {
unsafe {
(*seg).bounce_dma = dma_map_single!(dev, unsafe { (*seg).bounce_buf }, max_pkt, 2i32);
}
}
if dma_mapping_error(dev, unsafe { (*seg).bounce_dma }) != 0 {
// try without aligning. Some host controllers survive
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Failed mapping bounce buffer, not aligning\n".as_ptr() as *const u8) };
return 0;
}
unsafe {
*(trb_buff_len as *mut u32) = new_buff_len;
}
unsafe {
(*seg).bounce_len = new_buff_len;
}
unsafe {
(*seg).bounce_offs = enqd_len;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Bounce align, new buff len %d\n".as_ptr() as *const u8, *(trb_buff_len as *mut u32)) };
}
1
}

// This is very similar to what ehci-q.c qtd_fill() does
// Caller must have locked xhci->lock
/// The transfer burst count field of the isochronous TRB defines the number of
/// bursts that are required to move all packets in this TD.  Only SuperSpeed
/// devices can burst up to bMaxBurst number of packets per service interval.
/// This field is zero based, meaning a value of zero in the field means one
/// burst.  Basically, for everything but SuperSpeed devices, this field will be
/// zero.  Only xHCI 1.0 host controllers support this field.
fn xhci_get_burst_count(xhci: *mut xhci_hcd, urb: *mut urb, total_packet_count: u32) -> u32 {
let mut max_burst: u32 = 0;
if ((unsafe { (*xhci).hci_version }) as i32) < 256 || ((unsafe { (*(*urb).dev).speed }) as u32) < 5 {
return 0;
}
max_burst = (unsafe { (*(*urb).ep).ss_ep_comp.bMaxBurst }) as u32;
(total_packet_count.wrapping_add(max_burst.wrapping_add(1)).wrapping_sub(1) / max_burst.wrapping_add(1)).wrapping_sub(1)
}

/// Returns the number of packets in the last "burst" of packets.  This field is
/// valid for all speeds of devices.  USB 2.0 devices can only do one "burst", so
/// the last burst packet count is equal to the total number of packets in the
/// TD.  SuperSpeed endpoints can have up to 3 bursts.  All but the last burst
/// must contain (bMaxBurst + 1) number of packets, but the last burst can
/// contain 1 to (bMaxBurst + 1) packets.
fn xhci_get_last_burst_packet_count(xhci: *mut xhci_hcd, urb: *mut urb, total_packet_count: u32) -> u32 {
let mut max_burst: u32 = 0;
let mut residue: u32 = 0;
if ((unsafe { (*xhci).hci_version }) as i32) < 256 {
return 0;
}
if ((unsafe { (*(*urb).dev).speed }) as u32) >= 5 {
// bMaxBurst is zero based: 0 means 1 packet per burst
max_burst = (unsafe { (*(*urb).ep).ss_ep_comp.bMaxBurst }) as u32;
residue = total_packet_count % max_burst.wrapping_add(1);
// If residue is zero, the last burst contains (max_burst + 1)
// number of packets, but the TLBPC field is zero-based.
if residue == 0 {
return max_burst;
}
return residue.wrapping_sub(1);
}
if total_packet_count == 0 {
return 0;
}
total_packet_count.wrapping_sub(1)
}

// Returns the Isochronous Scheduling Threshold in Microframes. 1 Frame is 8 Microframes.
fn xhci_ist_microframes(xhci: *mut xhci_hcd) -> i32 {
let mut ist: i32 = 0;
let mut __v0: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_937() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_938() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) == 0 {
unsafe { __compiletime_assert_939() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) >> ((core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_940() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_941() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*xhci).hcs_params2 }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2))) >> ((core::mem::size_of::<__SlateRecord_xhci_ring_c_3978_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)).trailing_zeros() as i32);
__v0 = __v2;
ist = __v0 as i32;
if ((unsafe { (*xhci).hcs_params2 }) as u64) & BIT!(3i32) != 0 {
ist = ist.wrapping_mul(8);
}
ist
}

fn xhci_isoc_td_uses_frame_id(xhci: *mut xhci_hcd, urb: *mut urb, ep: *mut xhci_virt_ep, i: i32) -> bool {
if (unsafe { (*urb).transfer_flags }) & 2 != 0 {
return false;
}
if ((unsafe { (*xhci).hcc_params }) as u64) & BIT!(11i32) != 0 {
return true;
}
// set frame id for first TD of first URB in stream
if (unsafe { (*ep).next_uframe }) == 1i32.wrapping_neg() && i == 0 {
return true;
}
false
}

/// Check if frame is in the valid frame window, including start and end.
/// If start > end then assume window wrapped around at a limit the frame
/// value won't exceed.
fn xhci_frame_in_range(frame: u32, start: u32, end: u32) -> bool {
// frame window end wrapped around
if start > end {
return frame >= start || frame <= end;
}
frame >= start && frame <= end
}

/// Set the urb->start_frame of the URB.
///
/// Returns microframe index of first TD
fn xhci_get_isoc_start_frame(xhci: *mut xhci_hcd, urb: *mut urb, ep: *mut xhci_virt_ep) -> i32 {
let mut curr_frame: u32 = 0;
let mut start_uframe: u32 = 0;
let mut urb_start: u32 = 0;
let mut urb_end: u32 = 0;
let mut win_start: u32 = 0;
let mut win_end: u32 = 0;
let mut frame_unit: bool = false;
let mut uinterval: i32 = 0;
let mut mfindex: u32 = 0;
let mut ist: i32 = 0;
// check if urb uses frame units instead of microframes
frame_unit = ((unsafe { (*(*urb).dev).speed }) as u32) == 2 || ((unsafe { (*(*urb).dev).speed }) as u32) == 1;
uinterval = unsafe { (*urb).interval };
if frame_unit {
uinterval = uinterval.wrapping_mul(8);
}
// get current microframe index and isoc scheduling threshold
mfindex = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).run_regs).microframe_index) }) as *const ());
ist = xhci_ist_microframes(xhci);
// calculate valid frame window, in frame units, see xhci 4.11.2.5
curr_frame = MFINDEX_TO_FRAME!(mfindex);
win_start = curr_frame.wrapping_add(DIV_ROUND_UP_POW2!(ist, 8i32) as u32).wrapping_add(1) % 2048;
win_end = curr_frame.wrapping_add(895) % 2048;
// Is this the first URB starting the whole isoc data flow?
if (unsafe { (*ep).next_uframe }) < 0 {
// align first URB to next interval boundary, or at last to full frame
start_uframe = mfindex.wrapping_add(ist as u32).wrapping_add(10);
start_uframe = roundup!(start_uframe, 8i32);
start_uframe = roundup!(start_uframe, uinterval) % 16384u32;
} else {
// URB is mid stream and expected to handle the next frame
start_uframe = (unsafe { (*ep).next_uframe }) as u32;
urb_start = start_uframe / 8;
urb_end = start_uframe.wrapping_add(unsafe { (*urb).number_of_packets }.wrapping_mul(uinterval) as u32) / 8;
urb_end %= 2048;
if !xhci_frame_in_range(urb_start, win_start, win_end) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ill-timed isoc URB %p for start frame %d, range %d-%d\n".as_ptr() as *const u8, urb, urb_start, win_start, win_end) };
}
}
if !xhci_frame_in_range(urb_end, win_start, win_end) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ill-timed isoc URB %p for end frame %d, range %d-%d\n".as_ptr() as *const u8, urb, urb_start, win_start, win_end) };
}
}
}
// set urb->start_frame
unsafe {
(*urb).start_frame = (if frame_unit { start_uframe / 8 } else { start_uframe }) as i32;
}
start_uframe as i32
}

/// Check if we should generate event interrupt for a TD in an isoc URB
fn trb_block_event_intr(xhci: *mut xhci_hcd, num_tds: i32, i: i32, ir: *mut xhci_interrupter) -> bool {
if ((unsafe { (*xhci).hci_version }) as i32) < 256 {
return false;
}
// always generate an event interrupt for the last TD
if i == num_tds.wrapping_sub(1) {
return false;
}
// If AVOID_BEI is set the host handles full event rings poorly,
// generate an event at least every 8th TD to clear the event ring
if i != 0 && (unsafe { (*ir).isoc_bei_interval }) != 0 && (unsafe { (*xhci).quirks }) & BIT_ULL!(15i32) != 0 {
return (i as u32) % unsafe { (*ir).isoc_bei_interval } != 0;
}
true
}

// This is for isoc transfer
fn xhci_queue_isoc_tx(xhci: *mut xhci_hcd, mem_flags: u32, urb: *mut urb, slot_id: i32, ep_index: u32) -> i32 {
let mut __UNIQUE_ID_y__943: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__942: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__946: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__945: u32 = unsafe { core::mem::zeroed() };
let mut remainder: u32 = unsafe { core::mem::zeroed() };
let mut sia_frame_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut u32 = core::ptr::addr_of_mut!(sia_frame_id) as *mut u32;
let mut last_burst_pkt_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u32 = core::ptr::addr_of_mut!(last_burst_pkt_count) as *mut u32;
let mut burst_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut u32 = core::ptr::addr_of_mut!(burst_count) as *mut u32;
let mut max_pkt: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut u32 = core::ptr::addr_of_mut!(max_pkt) as *mut u32;
let mut total_pkt_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut u32 = core::ptr::addr_of_mut!(total_pkt_count) as *mut u32;
let mut start_uframe: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut i32 = core::ptr::addr_of_mut!(start_uframe) as *mut i32;
let mut uinterval: i32 = unsafe { core::mem::zeroed() };
let mut xep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(xep) as *mut *mut xhci_virt_ep;
let mut more_trbs_coming: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut bool = core::ptr::addr_of_mut!(more_trbs_coming) as *mut bool;
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_51: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut start_addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_54: *mut u64 = core::ptr::addr_of_mut!(start_addr) as *mut u64;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_55: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut td_remain_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_56: *mut i32 = core::ptr::addr_of_mut!(td_remain_len) as *mut i32;
let mut td_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_57: *mut i32 = core::ptr::addr_of_mut!(td_len) as *mut i32;
let mut trb_buff_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_58: *mut i32 = core::ptr::addr_of_mut!(trb_buff_len) as *mut i32;
let mut running_total: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_59: *mut i32 = core::ptr::addr_of_mut!(running_total) as *mut i32;
let mut length_field: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut u32 = core::ptr::addr_of_mut!(length_field) as *mut u32;
let mut field: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_61: *mut u32 = core::ptr::addr_of_mut!(field) as *mut u32;
let mut start_cycle: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_62: *mut i32 = core::ptr::addr_of_mut!(start_cycle) as *mut i32;
let mut first_trb: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_63: *mut bool = core::ptr::addr_of_mut!(first_trb) as *mut bool;
let mut start_trb: core::mem::MaybeUninit<*mut xhci_generic_trb> = core::mem::MaybeUninit::uninit();
let __slate_slot_64: *mut *mut xhci_generic_trb = core::ptr::addr_of_mut!(start_trb) as *mut *mut xhci_generic_trb;
let mut trbs_per_td: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_65: *mut i32 = core::ptr::addr_of_mut!(trbs_per_td) as *mut i32;
let mut num_tds: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_66: *mut i32 = core::ptr::addr_of_mut!(num_tds) as *mut i32;
let mut td: core::mem::MaybeUninit<*mut xhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_67: *mut *mut xhci_td = core::ptr::addr_of_mut!(td) as *mut *mut xhci_td;
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_68: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut ep_ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_69: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ep_ring) as *mut *mut xhci_ring;
let mut ir: core::mem::MaybeUninit<*mut xhci_interrupter> = core::mem::MaybeUninit::uninit();
let __slate_slot_70: *mut *mut xhci_interrupter = core::ptr::addr_of_mut!(ir) as *mut *mut xhci_interrupter;
unsafe {
let __slate_auto_bytes_4 = __slate_slot_70 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_69 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_68 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_67 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_66 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_65 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_64 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_63 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_12 = __slate_slot_62 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_61 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_14 = __slate_slot_60 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_15 = __slate_slot_59 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_16 = __slate_slot_58 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_17 = __slate_slot_57 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_18 = __slate_slot_56 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_18.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_19 = __slate_slot_55 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_19.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_20 = __slate_slot_54 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_20.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_21 = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_21.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_22 = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_22.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_23 = __slate_slot_51 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_23.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_24 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_24.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_25 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_25.add(0usize), 0u8, 8usize);
uinterval = (*urb).interval;
let __slate_auto_bytes_26 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_26.add(0usize), 0u8, 4usize);
*__slate_slot_49 = ((*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize);
*__slate_slot_69 = (*((*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring;
*__slate_slot_70 = *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize);
*__slate_slot_66 = (*urb).number_of_packets;
if *__slate_slot_66 < 1 {
return 22i32.wrapping_neg();
} else {
*__slate_slot_54 = (*urb).transfer_dma;
*__slate_slot_64 = core::ptr::addr_of_mut!((*(*(*__slate_slot_69)).enqueue).generic);
*__slate_slot_62 = (*(*__slate_slot_69)).cycle_state as i32;
*__slate_slot_68 = (*urb).hcpriv as *mut urb_priv;
if ((*(*urb).dev).speed as u32) == 2 || ((*(*urb).dev).speed as u32) == 1 {
uinterval = unsafe { (*urb).interval }.wrapping_mul(8);
}
*__slate_slot_47 = xhci_get_isoc_start_frame(xhci, urb, *__slate_slot_49);
// Queue the TRBs for each TD, even if they are zero-length
*__slate_slot_52 = 0;
'__join_3: {
'__join_39: {
'__join_11: {
while *__slate_slot_52 < *__slate_slot_66 {
let __slate_auto_bytes = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_63 = true;
*__slate_slot_59 = 0;
*__slate_slot_53 = (*__slate_slot_54).wrapping_add((*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(*__slate_slot_52 as isize)).offset as u64);
*__slate_slot_57 = (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(*__slate_slot_52 as isize)).length as i32;
*__slate_slot_56 = *__slate_slot_57;
*__slate_slot_45 = xhci_usb_endpoint_maxp((*urb).dev, (*urb).ep) as u32;
*__slate_slot_46 = (*__slate_slot_57 as u32).wrapping_add(*__slate_slot_45).wrapping_sub(1) / *__slate_slot_45;
// A zero-length transfer still involves at least one packet.
if *__slate_slot_46 == 0 {
let __v41: u32 = *__slate_slot_46;
*__slate_slot_46 = __v41.wrapping_add(1);
}
*__slate_slot_44 = xhci_get_burst_count(xhci, urb, *__slate_slot_46);
*__slate_slot_43 = xhci_get_last_burst_packet_count(xhci, urb, *__slate_slot_46);
*__slate_slot_65 = count_isoc_trbs_needed(urb, *__slate_slot_52) as i32;
*__slate_slot_55 = prepare_transfer(xhci, *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize), ep_index, (*urb).stream_id, *__slate_slot_65 as u32, urb, *__slate_slot_52 as u32, mem_flags);
if *__slate_slot_55 < 0 {
break '__join_39;
} else {
*__slate_slot_67 = (core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_52 as isize);
// Choose SIA or frame ID based scheduling for this TD
if xhci_isoc_td_uses_frame_id(xhci, urb, *__slate_slot_49, *__slate_slot_52) {
*__slate_slot_42 = ((*__slate_slot_47).wrapping_add((*__slate_slot_52).wrapping_mul(uinterval)) / 8) as u32;
*__slate_slot_42 = TRB_FRAME_ID!(*__slate_slot_42 % 2048);
} else {
*__slate_slot_42 = BIT!(31i32) as u32;
}
// Set isoc specific data for the first TRB in a TD.
// Prevent HW from getting the TRBs by keeping the cycle state
// inverted in the first TDs isoc TRB.
*__slate_slot_61 = (TRB_TYPE!(5i32) as u32) | TRB_TLBPC!(*__slate_slot_43) | *__slate_slot_42 | if *__slate_slot_52 != 0 { (*(*__slate_slot_69)).cycle_state } else { ((*__slate_slot_62 == 0) as i32) as u32 };
// xhci 1.1 with ETE uses TD_Size field for TBC, old is Rsvdz
if !(*(*__slate_slot_49)).use_extended_tbc {
let __v39: u32 = *__slate_slot_61;
let __v38: u32 = __v39 | TRB_TBC!(*__slate_slot_44);
*__slate_slot_61 = __v38;
}
// fill the rest of the TRB fields, and remaining normal TRBs
*__slate_slot_51 = 0;
while *__slate_slot_51 < *__slate_slot_65 {
remainder = 0;
// only first TRB is isoc, overwrite otherwise
if !(*__slate_slot_63) {
*__slate_slot_61 = (TRB_TYPE!(1i32) as u32) | (*(*__slate_slot_69)).cycle_state;
}
// Only set interrupt on short packet for IN EPs
if usb_urb_dir_in(urb) != 0 {
let __v36: u32 = *__slate_slot_61;
let __v35: u32 = ((__v36 as u64) | BIT!(2i32)) as u32;
*__slate_slot_61 = __v35;
}
// Set the chain bit for all except the last TRB
if *__slate_slot_51 < (*__slate_slot_65).wrapping_sub(1) {
*__slate_slot_50 = true;
let __v30: u32 = *__slate_slot_61;
let __v29: u32 = ((__v30 as u64) | BIT!(4i32)) as u32;
*__slate_slot_61 = __v29;
} else {
*__slate_slot_50 = false;
(*(*__slate_slot_67)).end_trb = (*(*__slate_slot_69)).enqueue;
(*(*__slate_slot_67)).end_seg = (*(*__slate_slot_69)).enq_seg;
let __v34: u32 = *__slate_slot_61;
let __v33: u32 = ((__v34 as u64) | BIT!(5i32)) as u32;
*__slate_slot_61 = __v33;
if trb_block_event_intr(xhci, *__slate_slot_66, *__slate_slot_52, *__slate_slot_70) {
let __v32: u32 = *__slate_slot_61;
let __v31: u32 = ((__v32 as u64) | BIT!(9i32)) as u32;
*__slate_slot_61 = __v31;
}
}
// Calculate TRB length
*__slate_slot_58 = TRB_BUFF_LEN_UP_TO_BOUNDARY!(*__slate_slot_53) as i32;
if *__slate_slot_58 > *__slate_slot_56 {
*__slate_slot_58 = *__slate_slot_56;
}
// Set the TRB length, TD size, & interrupter fields.
remainder = xhci_td_remainder(xhci, *__slate_slot_59, *__slate_slot_58, *__slate_slot_57 as u32, urb, *__slate_slot_50);
*__slate_slot_60 = (TRB_LEN__2!(*__slate_slot_58) | TRB_INTR_TARGET!(0i32)) as u32;
// xhci 1.1 with ETE uses TD Size field for TBC
if *__slate_slot_63 && (*(*__slate_slot_49)).use_extended_tbc {
__UNIQUE_ID_x__942 = *__slate_slot_44;
__UNIQUE_ID_y__943 = 31;
while false {
}
*__slate_slot_60 |= (if __UNIQUE_ID_x__942 < __UNIQUE_ID_y__943 { __UNIQUE_ID_x__942 } else { __UNIQUE_ID_y__943 }) << 17i32;
} else {
__UNIQUE_ID_x__945 = remainder;
__UNIQUE_ID_y__946 = 31;
while false {
}
*__slate_slot_60 |= (if __UNIQUE_ID_x__945 < __UNIQUE_ID_y__946 { __UNIQUE_ID_x__945 } else { __UNIQUE_ID_y__946 }) << 17i32;
}
*__slate_slot_63 = false;
queue_trb(xhci, *__slate_slot_69, *__slate_slot_50, lower_32_bits!(*__slate_slot_53), upper_32_bits!(*__slate_slot_53), *__slate_slot_60, *__slate_slot_61);
let __v16: i32 = *__slate_slot_59;
*__slate_slot_59 = __v16.wrapping_add(*__slate_slot_58);
let __v14: u64 = *__slate_slot_53;
*__slate_slot_53 = __v14.wrapping_add(*__slate_slot_58 as u64);
let __v12: i32 = *__slate_slot_56;
*__slate_slot_56 = __v12.wrapping_sub(*__slate_slot_58);
let __v10: i32 = *__slate_slot_51;
*__slate_slot_51 = __v10.wrapping_add(1);
}
// Check TD length
if *__slate_slot_59 != *__slate_slot_57 {
break '__join_11;
} else {
let __v8: i32 = *__slate_slot_52;
*__slate_slot_52 = __v8.wrapping_add(1);
}
}
}
(*(*__slate_slot_49)).next_uframe = (*__slate_slot_47).wrapping_add((*__slate_slot_66).wrapping_mul(uinterval)) % 16384i32;
if (*xhci_to_hcd(xhci)).self_.bandwidth_isoc_reqs == 0 {
if (*xhci).quirks & BIT_ULL!(3i32) != 0 {
usb_amd_quirk_pll_disable();
}
}
let __v6: *mut usb_hcd = xhci_to_hcd(xhci);
let __v5: i32 = (*__v6).self_.bandwidth_isoc_reqs;
(*__v6).self_.bandwidth_isoc_reqs = __v5.wrapping_add(1);
giveback_first_trb(xhci, slot_id, ep_index, (*urb).stream_id, *__slate_slot_62, *__slate_slot_64);
return 0;
}
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ISOC TD length unmatch\n".as_ptr() as *const u8);
*__slate_slot_55 = 22i32.wrapping_neg();
break '__join_3;
}
if *__slate_slot_52 == 0 {
return *__slate_slot_55;
}
}
let __v3: i32 = *__slate_slot_52;
*__slate_slot_52 = __v3.wrapping_sub(1);
while *__slate_slot_52 >= 0 {
list_del_init(core::ptr::addr_of_mut!((*(core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_52 as isize)).td_list));
let __v1: i32 = *__slate_slot_52;
*__slate_slot_52 = __v1.wrapping_sub(1);
}
// Clean up a partially enqueued isoc transfer.
// Use the first TD as a temporary variable to turn the TDs we've queued
// into No-ops with a software-owned cycle bit. That way the hardware
// won't accidentally start executing bogus TDs when we partially
// overwrite them.  td->start_trb and td->start_seg are already set.
(*(core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(0 as isize)).end_trb = (*(*__slate_slot_69)).enqueue;
// Every TRB except the first & last will have its cycle bit flipped.
td_to_noop(xhci, *__slate_slot_49, (core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(0 as isize), true);
// Reset the ring enqueue back to the first TRB and its cycle bit.
(*(*__slate_slot_69)).enqueue = (*(core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(0 as isize)).start_trb;
(*(*__slate_slot_69)).enq_seg = (*(core::ptr::addr_of_mut!((*(*__slate_slot_68)).td) as *mut xhci_td).wrapping_offset(0 as isize)).start_seg;
(*(*__slate_slot_69)).cycle_state = *__slate_slot_62 as u32;
usb_hcd_unlink_urb_from_ep(bus_to_hcd((*(*urb).dev).bus), urb);
return *__slate_slot_55;
}
}
}

/// Check transfer ring to guarantee there is enough room for the urb.
/// Update ISO URB start_frame and interval.
/// Update interval as xhci_queue_intr_tx does. Use xhci frame_index to
/// update urb->start_frame if URB_ISO_ASAP is set in transfer_flags or
/// Contiguous Frame ID is not supported by HC.
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_isoc_tx_prepare(xhci: *mut xhci_hcd, mem_flags: u32, urb: *mut urb, slot_id: i32, ep_index: u32) -> i32 {
let mut ep_ring: *mut xhci_ring = unsafe { core::mem::zeroed() };
let mut num_tds: i32 = 0;
let mut num_trbs: i32 = 0;
let mut ret: i32 = 0;
let xdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
let xep: *mut xhci_virt_ep = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
ep_ring = unsafe { (*((*xdev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring };
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*xdev).out_ctx, ep_index) };
num_tds = unsafe { (*urb).number_of_packets };
for i in 0..num_tds {
let __v0: u32 = count_isoc_trbs_needed(urb, i);
num_trbs = (num_trbs as u32).wrapping_add(__v0) as i32;
}
// Check the ring to guarantee there is enough room for the whole urb.
// Do not insert any td of the urb to the ring if the check failed.
ret = prepare_ring(xhci, ep_ring, GET_EP_CTX_STATE!(ep_ctx), num_trbs as u32, mem_flags);
if ret != 0 {
return ret;
}
// Check interval value. This should be done before we start to
// calculate the start frame value.
check_interval(urb, ep_ctx);
// Schedule the URB discontiguously if all previous URBs have completed.
// XXX core can't tell if completions are pending but not running yet.
let __v6: bool = list_empty((unsafe { core::ptr::addr_of_mut!((*ep_ring).td_list) }) as *const list_head) != 0 && !hcd_periodic_completion_in_progress(xhci_to_hcd(xhci), unsafe { (*urb).ep });
if __v6 {
if GET_EP_CTX_STATE!(ep_ctx) == 1 {
}
unsafe {
(*xep).next_uframe = 1i32.wrapping_neg();
}
}
xhci_queue_isoc_tx(xhci, mem_flags, urb, slot_id, ep_index)
}

// 		Command Ring Operations
/// Generic function for queueing a command TRB on the command ring.
/// Check to make sure there's room on the command ring for one command TRB.
/// Also check that there's room reserved for commands that must not fail.
/// If this is a command that must not fail, meaning command_must_succeed = TRUE,
/// then only check for the number of reserved spots.
/// Don't decrement xhci->cmd_ring_reserved_trbs after we've queued the TRB
/// because the command event handler may want to resubmit a failed command.
fn queue_command(xhci: *mut xhci_hcd, cmd: *mut xhci_command, field1: u32, field2: u32, field3: u32, field4: u32, command_must_succeed: bool) -> i32 {
let mut reserved_trbs: i32 = (unsafe { (*xhci).cmd_ring_reserved_trbs }) as i32;
let hcd: *mut usb_hcd = xhci_to_hcd(xhci);
let mut ret: i32 = 0;
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 || ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(1i32) != 0 {
return 108i32.wrapping_neg();
}
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Can't queue command, xHC not accessible\n".as_ptr() as *const u8) };
return 108i32.wrapping_neg();
}
if !command_must_succeed {
reserved_trbs = reserved_trbs.wrapping_add(1);
}
ret = prepare_ring(xhci, unsafe { (*xhci).cmd_ring }, 1u32, reserved_trbs as u32, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if ret < 0 {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERR: No room for command on command ring\n".as_ptr() as *const u8) };
if command_must_succeed {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERR: Reserved TRB counting for unfailable commands failed.\n".as_ptr() as *const u8) };
}
return ret;
}
unsafe {
(*cmd).command_trb = (*(*xhci).cmd_ring).enqueue;
}
// if there are no other commands queued we start the timeout timer
if list_empty((unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) }) as *const list_head) != 0 {
unsafe {
(*xhci).current_cmd = cmd;
}
xhci_mod_cmd_timer(xhci);
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*cmd).cmd_list) }, unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) });
queue_trb(xhci, unsafe { (*xhci).cmd_ring }, false, field1, field2, field3, field4 | unsafe { (*(*xhci).cmd_ring).cycle_state });
0
}

/// Queue a slot enable or disable request on the command ring
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_slot_control(xhci: *mut xhci_hcd, cmd: *mut xhci_command, trb_type: u32, slot_id: u32) -> i32 {
queue_command(xhci, cmd, 0u32, 0u32, 0u32, TRB_TYPE!(trb_type) | SLOT_ID_FOR_TRB!(slot_id), false)
}

/// Queue an address device command TRB
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_address_device(xhci: *mut xhci_hcd, cmd: *mut xhci_command, in_ctx_ptr: u64, slot_id: u32, setup: u32) -> i32 {
queue_command(xhci, cmd, lower_32_bits!(in_ctx_ptr), upper_32_bits!(in_ctx_ptr), 0u32, ((((TRB_TYPE!(11i32) as u32) | SLOT_ID_FOR_TRB!(slot_id)) as u64) | if (setup as u32) == 0 { BIT!(9i32) } else { 0 }) as u32, false)
}

#[unsafe(no_mangle)]
extern "C" fn xhci_queue_vendor_command(xhci: *mut xhci_hcd, cmd: *mut xhci_command, field1: u32, field2: u32, field3: u32, field4: u32) -> i32 {
queue_command(xhci, cmd, field1, field2, field3, field4, false)
}

/// Queue a reset device command TRB
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_reset_device(xhci: *mut xhci_hcd, cmd: *mut xhci_command, slot_id: u32) -> i32 {
queue_command(xhci, cmd, 0u32, 0u32, 0u32, (TRB_TYPE!(17i32) as u32) | SLOT_ID_FOR_TRB!(slot_id), false)
}

/// Queue a configure endpoint command TRB
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_configure_endpoint(xhci: *mut xhci_hcd, cmd: *mut xhci_command, in_ctx_ptr: u64, slot_id: u32, command_must_succeed: bool) -> i32 {
queue_command(xhci, cmd, lower_32_bits!(in_ctx_ptr), upper_32_bits!(in_ctx_ptr), 0u32, (TRB_TYPE!(12i32) as u32) | SLOT_ID_FOR_TRB!(slot_id), command_must_succeed)
}

/// Queue a get root hub port bandwidth command TRB
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_get_port_bw(xhci: *mut xhci_hcd, cmd: *mut xhci_command, in_ctx_ptr: u64, dev_speed: u8, command_must_succeed: bool) -> i32 {
queue_command(xhci, cmd, lower_32_bits!(in_ctx_ptr), upper_32_bits!(in_ctx_ptr), 0u32, (TRB_TYPE!(21i32) | DEV_SPEED_FOR_TRB!(dev_speed)) as u32, command_must_succeed)
}

/// Queue an evaluate context command TRB
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_evaluate_context(xhci: *mut xhci_hcd, cmd: *mut xhci_command, in_ctx_ptr: u64, slot_id: u32, command_must_succeed: bool) -> i32 {
queue_command(xhci, cmd, lower_32_bits!(in_ctx_ptr), upper_32_bits!(in_ctx_ptr), 0u32, (TRB_TYPE!(13i32) as u32) | SLOT_ID_FOR_TRB!(slot_id), command_must_succeed)
}

/// Suspend is set to indicate "Stop Endpoint Command" is being issued to stop
/// activity on an endpoint that is about to be suspended.
#[unsafe(no_mangle)]
extern "C" fn xhci_queue_stop_endpoint(xhci: *mut xhci_hcd, cmd: *mut xhci_command, slot_id: i32, ep_index: u32, suspend: i32) -> i32 {
let trb_slot_id: u32 = SLOT_ID_FOR_TRB__2!(slot_id) as u32;
let trb_ep_index: u32 = EP_INDEX_FOR_TRB!(ep_index);
let r#type: u32 = TRB_TYPE!(15i32) as u32;
let trb_suspend: u32 = SUSPEND_PORT_FOR_TRB!(suspend) as u32;
queue_command(xhci, cmd, 0u32, 0u32, 0u32, trb_slot_id | trb_ep_index | r#type | trb_suspend, false)
}

#[unsafe(no_mangle)]
extern "C" fn xhci_queue_reset_ep(xhci: *mut xhci_hcd, cmd: *mut xhci_command, slot_id: i32, ep_index: u32, reset_type: u32) -> i32 {
let trb_slot_id: u32 = SLOT_ID_FOR_TRB__2!(slot_id) as u32;
let trb_ep_index: u32 = EP_INDEX_FOR_TRB!(ep_index);
let mut r#type: u32 = TRB_TYPE!(14i32) as u32;
if (reset_type as u32) == 1 {
let __v0: u32 = r#type;
r#type = ((__v0 as u64) | BIT!(9i32)) as u32;
}
queue_command(xhci, cmd, 0u32, 0u32, 0u32, trb_slot_id | trb_ep_index | r#type, false)
}

static __slate_asm_string_dma_map_single_attrs_3: [u8; 44] = *b"%s %s: rejecting DMA map of vmalloc memory\n\0";

static __slate_asm_string_dma_map_single_attrs_4: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_handle_cmd_completion_18: [u8; 1] = *b"\0";

static __slate_asm_string_handle_cmd_completion_19: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-ring.c\0";

static __slate_asm_string_handle_cmd_completion_20: [u8; 1] = *b"\0";

static __slate_asm_string_handle_cmd_completion_21: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-ring.c\0";

static __slate_asm_string_handle_cmd_completion_22: [u8; 1] = *b"\0";

static __slate_asm_string_handle_cmd_completion_23: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci-ring.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
