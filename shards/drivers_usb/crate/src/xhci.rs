
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
//! writeable
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
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
//! Generic info of form tag = "info"
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! poll_timeout_us - Periodically poll and perform an operation until
//!                   a condition is met or a timeout occurs
//!
//! @op: Operation
//! @cond: Break condition
//! @sleep_us: Maximum time to sleep between operations in us (0 tight-loops).
//!            Please read usleep_range() function description for details and
//!            limitations.
//! @timeout_us: Timeout in us, 0 means never timeout
//! @sleep_before_op: if it is true, sleep @sleep_us before operation.
//!
//! When available, you'll probably want to use one of the specialized
//! macros defined below rather than this macro directly.
//!
//! Returns: 0 on success and -ETIMEDOUT upon a timeout. Must not
//! be called from atomic context if sleep_us or timeout_us are used.
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
//! FIXME defined twice, from xhci.h
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! HC should halt within 16 ms, but use 32 ms as some hosts take longer
//! PORTSC - Port Status and Control Register - port_status_base bitmasks
//!
//! true: device connected
//! hc_capbase - bitmasks
//!
//! bits 7:0 - Capability Registers Length
//! Max number of Devices Slots. xHCI specification section 5.3.3
//! Valid values are in the range of 1 to 255.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// umin - return minimum of two non-negative values
//   Signed types are zero extended to match a larger unsigned type.
// @x: first value
// @y: second value
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// Overflow Flag
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Timer expired
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Cannot send after transport endpoint shutdown
// Connection timed out
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// page_folio - Converts from page to folio.
// @p: The page.
//
// Every page is part of a folio.  This function cannot be called on a
// NULL pointer.
//
// Context: No reference, nor lock is required on @page.  If the caller
// does not hold a reference, this call may race with a folio split, so
// it should re-check the folio still contains this page after gaining
// a reference on the folio.
// Return: The folio which contains this page.
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
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// guarantee 'op' and 'cond' are evaluated after timeout expired
// poll_timeout_us_atomic - Periodically poll and perform an operation until
//                          a condition is met or a timeout occurs
//
// @op: Operation
// @cond: Break condition
// @delay_us: Time to udelay between operations in us (0 tight-loops).
//            Please read udelay() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
// @delay_before_op: if it is true, delay @delay_us before operation.
//
// This macro does not rely on timekeeping.  Hence it is safe to call even when
// timekeeping is suspended, at the expense of an underestimation of wall clock
// time, which is rather minimal with a non-zero delay_us.
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout.
// guarantee 'op' and 'cond' are evaluated after timeout expired
// read_poll_timeout - Periodically poll an address until a condition is
//                     met or a timeout occurs
// @op: accessor function (takes @args as its arguments)
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @sleep_us: Maximum time to sleep between reads in us (0 tight-loops). Please
//            read usleep_range() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
// @sleep_before_read: if it is true, sleep @sleep_us before read.
// @args: arguments for @op poll
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @args is stored in @val. Must not
// be called from atomic context if sleep_us or timeout_us are used.
// read_poll_timeout_atomic - Periodically poll an address until a condition is
//                            met or a timeout occurs
// @op: accessor function (takes @args as its arguments)
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @delay_us: Time to udelay between reads in us (0 tight-loops). Please
//            read udelay() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
// @delay_before_read: if it is true, delay @delay_us before read.
// @args: arguments for @op poll
//
// This macro does not rely on timekeeping.  Hence it is safe to call even when
// timekeeping is suspended, at the expense of an underestimation of wall clock
// time, which is rather minimal with a non-zero @delay_us.
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @args is stored in @val.
// readx_poll_timeout - Periodically poll an address until a condition is met or a timeout occurs
// @op: accessor function (takes @addr as its only argument)
// @addr: Address to poll
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @sleep_us: Maximum time to sleep between reads in us (0 tight-loops). Please
//            read usleep_range() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @addr is stored in @val. Must not
// be called from atomic context if sleep_us or timeout_us are used.
// readx_poll_timeout_atomic - Periodically poll an address until a condition is met or a timeout occurs
// @op: accessor function (takes @addr as its only argument)
// @addr: Address to poll
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @delay_us: Time to udelay between reads in us (0 tight-loops). Please
//            read udelay() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @addr is stored in @val.
// HCCPARAMS offset from PCI base address
// HCCPARAMS contains the first extended capability pointer
// Extended capability register fields
// USB 2.0 xHCI 1.0 hardware LMP capability - section 7.2.2.1.3.2
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
// use 512 microseconds as USB2 LPM L1 default timeout.
// Set default HIRD/BESL value to 4 (350/400us) for USB2 L1 LPM resume latency.
// Safe to use with mixed HIRD and BESL systems (host and device) and is used
// by other operating systems.
//
// XHCI 1.0 errata 8/14/12 Table 13 notes:
// "Software should choose xHC BESL/BESLD field values that do not violate a
// device's resume latency requirements,
// e.g. not program values > '4' if BLC = '1' and a HIRD device is attached,
// or not program values < '4' if BLC = '0' and a BESL device is attached.
// bits 15:8 - Rsvd
//
// bits 31:16 - Host Controller Interface Version Number
// HCSPARAMS1 - hcs_params1 - bitmasks
//
// bits 7:0 - Number of Device Slots
// bits 18:8 - Number of Interrupters, max values is 1024
// bits 23:19 - Rsvd
//
// bits 31:24 - Max Ports, max values is 255
// bits 15:8 - Rsvd
//
// bits 31:16 - U2 Device Exit Latency, Max U2 to U0 latency for the roothub ports
// HCCPARAMS1 - hcc_params - bitmasks
//
// bit 0 - 64-bit Addressing Capability
// DBOFF - db_off - bitmasks
//
// bits 1:0 - Rsvd
//
// bits 31:2 - Doorbell Array Offset
// RTSOFF - run_regs_off - bitmasks
//
// bits 4:0 - Rsvd
//
// bits 31:5 - Runtime Register Space Offse
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

macro_rules! DIV_ROUND_DOWN_ULL {
($ll:expr, $d:expr) => {
{
let mut __v: u64;
let mut __v2: u64 = $ll;
do_div__2!(__v2, $d);
__v = __v2;
__v
}
};
}

macro_rules! DIV_ROUND_UP_ULL {
($ll:expr, $d:expr) => {
DIV_ROUND_DOWN_ULL!((($ll) as u64).wrapping_add(($d) as u64).wrapping_sub(1), $d)
};
}

macro_rules! DMA_BIT_MASK__2 {
($n:expr) => {
GENMASK_ULL__17!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! DMA_BIT_MASK__3 {
($n:expr) => {
GENMASK_ULL__15!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! DMA_BIT_MASK__4 {
($n:expr) => {
GENMASK_ULL__18!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! DMA_BIT_MASK__5 {
($n:expr) => {
GENMASK_ULL__13!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! EP_IS_ADDED {
($ctrl_ctx:expr, $i:expr) => {
(unsafe { (*$ctrl_ctx).add_flags }) & ((1i32 << $i.wrapping_add(1)) as u32)
};
}

macro_rules! EP_IS_DROPPED {
($ctrl_ctx:expr, $i:expr) => {
(unsafe { (*$ctrl_ctx).drop_flags }) & ((1i32 << $i.wrapping_add(1)) as u32)
};
}

macro_rules! GENMASK_INPUT_CHECK__5 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_5543>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__6 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_647>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__10 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_513>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__11 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_5537>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__13 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_482>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__14 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_5467>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__17 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_496>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__18 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_5547>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__19 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_499>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__24 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_5534>() as i32
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
macro_rules! GENMASK_TYPE__5 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__10!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
(GENMASK_INPUT_CHECK__18!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
(GENMASK_INPUT_CHECK__5!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__9 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__14!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__10 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__24!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__11 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__13!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__13 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__19!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__14 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__17!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__19 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__11!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__24 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__6!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__13!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__5 {
($h:expr, $l:expr) => {
GENMASK_TYPE__24!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__7 {
($h:expr, $l:expr) => {
GENMASK_TYPE__14!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__13 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__15 {
($h:expr, $l:expr) => {
GENMASK_TYPE__19!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__17 {
($h:expr, $l:expr) => {
GENMASK_TYPE__8!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__18 {
($h:expr, $l:expr) => {
GENMASK_TYPE__10!(u64, $h, $l)
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__5!(u64, $h, $l)
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__9!(u64, $h, $l)
};
}

macro_rules! GENMASK__6 {
($h:expr, $l:expr) => {
GENMASK_TYPE__11!(u64, $h, $l)
};
}

macro_rules! GET_EP_CTX_STATE {
($ctx:expr) => {
(unsafe { (*$ctx).ep_info }) & 7
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

macro_rules! LAST_CTX {
($p:expr) => {
($p) << 27i32
};
}

macro_rules! MAX_PACKET_DECODED {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

macro_rules! MAX_PACKET {
($p:expr) => {
(($p) & 65535) << 16i32
};
}

macro_rules! PASS_BUCKET_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PASS_KMALLOC_PARAMS {
($_size:expr) => {
PASS_BUCKET_PARAMS!($_size)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PORT_BESLD {
($p:expr) => {
(($p) & 15) << 10i32
};
}

/// USB2 Protocol PORTHLPMC
macro_rules! PORT_HIRDM {
($p:expr) => {
($p) & 3
};
}

macro_rules! PORT_HIRD {
($p:expr) => {
(($p) & 15) << 4i32
};
}

macro_rules! PORT_L1DS {
($p:expr) => {
(($p) & 255) << 8i32
};
}

macro_rules! PORT_L1_TIMEOUT {
($p:expr) => {
(($p) & 255) << 2i32
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! TRB_TYPE {
($p:expr) => {
($p) << 10i32
};
}

macro_rules! TT_THINK_TIME {
($p:expr) => {
(($p) & 3) << 16i32
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

macro_rules! USB_GET_BESL_BASELINE {
($p:expr) => {
(($p) & 3840u32) >> 8i32
};
}

macro_rules! USB_GET_BESL_DEEP {
($p:expr) => {
(($p) & 61440u32) >> 12i32
};
}

/// Maximum number of ports under a hub device
macro_rules! XHCI_MAX_PORTS {
($p:expr) => {
(($p) & 255) << 24i32
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
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

macro_rules! __const_hweight16 {
($w:expr) => {
__const_hweight8!($w).wrapping_add(__const_hweight8!(($w) >> 8i32))
};
}

macro_rules! __const_hweight32 {
($w:expr) => {
__const_hweight16!($w).wrapping_add(__const_hweight16!(($w) >> 16i32))
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
(((($w) as u64) & 1u64 != 0) as i32).wrapping_add(((($w) as u64) & 2u64 != 0) as i32).wrapping_add(((($w) as u64) & 4u64 != 0) as i32).wrapping_add(((($w) as u64) & 8u64 != 0) as i32).wrapping_add(((($w) as u64) & 16u64 != 0) as i32).wrapping_add(((($w) as u64) & 32u64 != 0) as i32).wrapping_add(((($w) as u64) & 64u64 != 0) as i32).wrapping_add(((($w) as u64) & 128u64 != 0) as i32) as u32
};
}

/// __flex_counter() - Get pointer to counter member for the given
///                    flexible array, if it was annotated with __counted_by()
/// @FAM: Pointer to flexible array member of an addressable struct instance
///
/// For example, with:
///
/// 	struct foo {
/// 		int counter;
/// 		short array[] __counted_by(counter);
/// 	} *p;
///
/// __flex_counter(p->array) will resolve to &p->counter.
///
/// Note that Clang may not allow this to be assigned to a separate
/// variable; it must be used directly.
///
/// If p->array is unannotated, this returns (void *)NULL.
macro_rules! __flex_counter__2 {
($__obj_ptr:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__obj_ptr).num_tds) }
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

macro_rules! __le16_to_cpu {
($x:expr) => {
($x) as u16
};
}

macro_rules! __must_be_array__24 {
() => {
core::mem::size_of::<__SlateRecord_xhci_c_1642_4>() as i32
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

macro_rules! __or {
() => {
____or!()
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

/// __set_flex_counter() - Set the counter associated with the given flexible
///                        array member that has been annoated by __counted_by().
/// @FAM: Instance of flexible array member within a given struct.
/// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
///
/// This is a no-op if no annotation exists. Count needs to be checked with
/// overflows_flex_counter_type() before using this function.
macro_rules! __set_flex_counter {
($COUNT:expr, $__obj_ptr:expr) => {
let mut __v: i32;
let __v2: i32 = ($COUNT) as i32;
unsafe {
*__flex_counter__2!($__obj_ptr) = __v2;
}
__v = __v2;
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__5 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
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
macro_rules! container_of__46 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__62!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__48 {
($ptr:expr) => {
{
let mut __v: *mut xhci_virt_ep;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__17!() as isize)) }) as *mut ()) as *mut xhci_virt_ep;
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
let mut __v: *mut xhci_hcd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__9!() as isize)) }) as *mut ()) as *mut xhci_hcd;
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
macro_rules! container_of__65 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__45!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! dma_map_single__2 {
($d:expr, $a:expr, $s:expr, $r:expr) => {
dma_map_single_attrs($d, $a, ($s) as u64, $r, 0u64)
};
}

macro_rules! dma_unmap_single__2 {
($d:expr, $a:expr, $s:expr, $r:expr) => {
dma_unmap_single_attrs($d, $a, ($s) as u64, $r, 0u64);
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
macro_rules! do_div__2 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __v2: u32 = ($base) as u32;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
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
size_mul($count, (core::mem::size_of::<xhci_td>() as u64).wrapping_add(__must_be_array__24!() as u64))
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32($w)
};
__v
}
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
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<u32>() as u64) <= 4 { __ilog2_u32($n) } else { __ilog2_u64(($n) as u64) } }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
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

/// Add a ktime_t variable and a scalar nanosecond value.
/// res = kt + nsval:
macro_rules! ktime_add_ns {
($kt:expr, $nsval:expr) => {
(($kt) as u64).wrapping_add($nsval)
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
macro_rules! list_entry__18 {
($ptr:expr) => {
container_of__48!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__41 {
($ptr:expr, $type:ty) => {
container_of__46!($ptr, $type)
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
macro_rules! list_first_entry__27 {
($ptr:expr) => {
list_entry__18!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__23 {
($pos:expr, $member:ident) => {
list_entry__18!(unsafe { (*$pos).$member.next })
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

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
if true {
break;
}
}
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__17 {
() => {
core::mem::offset_of!(xhci_virt_ep, bw_endpoint_list) as u64
};
}

macro_rules! offsetof__9 {
() => {
632u64
};
}

macro_rules! offsetof__45 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, ep0) as u64
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, driver) as u64
};
}

macro_rules! offsetof__62 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, bw_endpoint_list) as u64
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

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) == 1 { 1u64 } else { 1u64 << ilog2__2!($n.wrapping_sub(1)).wrapping_add(1) } } else { __roundup_pow_of_two(($n) as u64) }
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
size_add(8u64, flex_array_size__5!($count))
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
struct_size!($count)
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

macro_rules! test_bit__6 {
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

macro_rules! timer_container_of__2 {
($callback_timer:expr) => {
container_of__57!($callback_timer)
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

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! to_usb_driver {
($d:expr) => {
{
let mut __v: *mut usb_driver;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!(usb_driver) as isize)) }) as *mut ()) as *mut usb_driver;
__v
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

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
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
    static mut __SCK__might_resched: static_call_key;
    static mut __cpu_online_mask: cpumask;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_ullong: kernel_param_ops;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_xhci_dbg_address: static_call_key;
    static mut __tracepoint_xhci_dbg_address: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_context_change: static_call_key;
    static mut __tracepoint_xhci_dbg_context_change: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_quirks: static_call_key;
    static mut __tracepoint_xhci_dbg_quirks: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_cancel_urb: static_call_key;
    static mut __tracepoint_xhci_dbg_cancel_urb: tracepoint;
    static mut __SCK__tp_func_xhci_dbg_init: static_call_key;
    static mut __tracepoint_xhci_dbg_init: tracepoint;
    static mut __SCK__tp_func_xhci_address_ctx: static_call_key;
    static mut __tracepoint_xhci_address_ctx: tracepoint;
    static mut __SCK__tp_func_xhci_setup_device: static_call_key;
    static mut __tracepoint_xhci_setup_device: tracepoint;
    static mut __SCK__tp_func_xhci_urb_enqueue: static_call_key;
    static mut __tracepoint_xhci_urb_enqueue: tracepoint;
    static mut __SCK__tp_func_xhci_urb_dequeue: static_call_key;
    static mut __tracepoint_xhci_urb_dequeue: tracepoint;
    static mut __SCK__tp_func_xhci_add_endpoint: static_call_key;
    static mut __tracepoint_xhci_add_endpoint: tracepoint;
    static mut __SCK__tp_func_xhci_alloc_dev: static_call_key;
    static mut __tracepoint_xhci_alloc_dev: tracepoint;
    static mut __SCK__tp_func_xhci_free_dev: static_call_key;
    static mut __tracepoint_xhci_free_dev: tracepoint;
    static mut __SCK__tp_func_xhci_discover_or_reset_device: static_call_key;
    static mut __tracepoint_xhci_discover_or_reset_device: tracepoint;
    static mut __SCK__tp_func_xhci_setup_device_slot: static_call_key;
    static mut __tracepoint_xhci_setup_device_slot: tracepoint;
    static mut __SCK__tp_func_xhci_configure_endpoint: static_call_key;
    static mut __tracepoint_xhci_configure_endpoint: tracepoint;
    static mut __SCK__tp_func_xhci_address_ctrl_ctx: static_call_key;
    static mut __tracepoint_xhci_address_ctrl_ctx: tracepoint;
    static mut __SCK__tp_func_xhci_configure_endpoint_ctrl_ctx: static_call_key;
    static mut __tracepoint_xhci_configure_endpoint_ctrl_ctx: tracepoint;
    static mut __SCK__tp_func_xhci_portsc_writel: static_call_key;
    static mut __tracepoint_xhci_portsc_writel: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strstr(arg0: *const u8, arg1: *const u8) -> *mut u8;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn schedule_timeout_uninterruptible(arg0: i64) -> i64;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn iomem_get_mapping() -> *mut address_space;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn rcu_is_watching() -> bool;
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion(arg0: *mut completion);
    fn ktime_get() -> i64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn add_timer(arg0: *mut timer_list);
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn __compiletime_assert_179();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_cache_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32, arg3: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_large_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn is_vmalloc_addr(arg0: *const ()) -> bool;
    fn sg_pcopy_from_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *const (), arg3: u64, arg4: i64) -> u64;
    fn sg_pcopy_to_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *mut (), arg3: u64, arg4: i64) -> u64;
    fn dma_pool_create_node(arg0: *const u8, arg1: *mut device, arg2: u64, arg3: u64, arg4: u64, arg5: i32) -> *mut dma_pool;
    fn dma_map_page_attrs(arg0: *mut device, arg1: *mut page, arg2: u64, arg3: u64, arg4: u32, arg5: u64) -> u64;
    fn dma_unmap_page_attrs(arg0: *mut device, arg1: u64, arg2: u64, arg3: u32, arg4: u64);
    fn dma_set_mask(arg0: *mut device, arg1: u64) -> i32;
    fn dma_set_coherent_mask(arg0: *mut device, arg1: u64) -> i32;
    fn iommu_get_domain_for_dev(arg0: *mut device) -> *mut iommu_domain;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn dmi_get_system_info(arg0: i32) -> *const u8;
    fn usb_disabled() -> i32;
    fn usb_endpoint_is_hs_isoc_double(arg0: *mut usb_device, arg1: *const usb_host_endpoint) -> bool;
    fn usb_hcd_check_unlink_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32) -> i32;
    fn usb_hcd_unlink_urb_from_ep(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_giveback_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32);
    fn usb_hcd_map_urb_for_dma(arg0: *mut usb_hcd, arg1: *mut urb, arg2: u32) -> i32;
    fn usb_hcd_unmap_urb_for_dma(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_is_primary_hcd(arg0: *mut usb_hcd) -> i32;
    fn usb_hcd_poll_rh_status(arg0: *mut usb_hcd);
    fn usb_root_hub_lost_power(arg0: *mut usb_device);
    fn usb_hcd_resume_root_hub(arg0: *mut usb_hcd);
    fn usb_amd_dev_put();
    fn usb_asmedia_modifyflowcontrol(arg0: *mut pci_dev);
    fn usb_disable_xhci_ports(arg0: *mut pci_dev);
    fn xhci_get_slot_state(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx) -> *mut u8;
    fn xhci_dbg_trace(arg0: *mut xhci_hcd, arg1: Option<unsafe extern "C" fn(*mut va_format)>, arg2: *const u8, ...);
    fn xhci_mem_cleanup(arg0: *mut xhci_hcd);
    fn xhci_mem_init(arg0: *mut xhci_hcd, arg1: u32) -> i32;
    fn xhci_free_virt_device(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: i32);
    fn xhci_free_virt_devices_depth_first(arg0: *mut xhci_hcd, arg1: i32);
    fn xhci_alloc_virt_device(arg0: *mut xhci_hcd, arg1: i32, arg2: *mut usb_device, arg3: u32) -> i32;
    fn xhci_setup_addressable_virt_dev(arg0: *mut xhci_hcd, arg1: *mut usb_device) -> i32;
    fn xhci_copy_ep0_dequeue_into_input_ctx(arg0: *mut xhci_hcd, arg1: *mut usb_device);
    fn xhci_endpoint_zero(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: *mut usb_host_endpoint);
    fn xhci_clear_endpoint_bw_info(arg0: *mut xhci_bw_info);
    fn xhci_rh_bw_cleanup(arg0: *mut xhci_hcd);
    fn xhci_update_bw_info(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: *mut xhci_input_control_ctx, arg3: *mut xhci_virt_device);
    fn xhci_endpoint_copy(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: *mut xhci_container_ctx, arg3: u32);
    fn xhci_slot_copy(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: *mut xhci_container_ctx);
    fn xhci_endpoint_init(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: *mut usb_device, arg3: *mut usb_host_endpoint, arg4: u32) -> i32;
    fn xhci_ring_free(arg0: *mut xhci_hcd, arg1: *mut xhci_ring);
    fn xhci_initialize_ring_info(arg0: *mut xhci_ring);
    fn xhci_ring_init(arg0: *mut xhci_hcd, arg1: *mut xhci_ring);
    fn xhci_free_endpoint_ring(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: u32);
    fn xhci_alloc_stream_info(arg0: *mut xhci_hcd, arg1: u32, arg2: u32, arg3: u32, arg4: u32) -> *mut xhci_stream_info;
    fn xhci_free_stream_info(arg0: *mut xhci_hcd, arg1: *mut xhci_stream_info);
    fn xhci_setup_streams_ep_input_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_ep_ctx, arg2: *mut xhci_stream_info);
    fn xhci_setup_no_streams_ep_input_ctx(arg0: *mut xhci_ep_ctx, arg1: *mut xhci_virt_ep);
    fn xhci_alloc_command(arg0: *mut xhci_hcd, arg1: bool, arg2: u32) -> *mut xhci_command;
    fn xhci_alloc_command_with_ctx(arg0: *mut xhci_hcd, arg1: bool, arg2: u32) -> *mut xhci_command;
    fn xhci_urb_free_priv(arg0: *mut urb_priv);
    fn xhci_free_command(arg0: *mut xhci_hcd, arg1: *mut xhci_command);
    fn xhci_irq(arg0: *mut usb_hcd) -> u32;
    fn xhci_alloc_tt_info(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: *mut usb_device, arg3: *mut usb_tt, arg4: u32) -> i32;
    fn xhci_trb_virt_to_dma(arg0: *mut xhci_segment, arg1: *mut xhci_trb) -> u64;
    fn xhci_is_vendor_info_code(arg0: *mut xhci_hcd, arg1: u32) -> i32;
    fn xhci_ring_cmd_db(arg0: *mut xhci_hcd);
    fn xhci_queue_slot_control(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u32, arg3: u32) -> i32;
    fn xhci_queue_address_device(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u64, arg3: u32, arg4: u32) -> i32;
    fn xhci_queue_vendor_command(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u32, arg3: u32, arg4: u32, arg5: u32) -> i32;
    fn xhci_queue_stop_endpoint(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: i32, arg3: u32, arg4: i32) -> i32;
    fn xhci_queue_ctrl_tx(arg0: *mut xhci_hcd, arg1: u32, arg2: *mut urb, arg3: i32, arg4: u32) -> i32;
    fn xhci_queue_bulk_tx(arg0: *mut xhci_hcd, arg1: u32, arg2: *mut urb, arg3: i32, arg4: u32) -> i32;
    fn xhci_queue_intr_tx(arg0: *mut xhci_hcd, arg1: u32, arg2: *mut urb, arg3: i32, arg4: u32) -> i32;
    fn xhci_queue_isoc_tx_prepare(arg0: *mut xhci_hcd, arg1: u32, arg2: *mut urb, arg3: i32, arg4: u32) -> i32;
    fn xhci_queue_configure_endpoint(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u64, arg3: u32, arg4: bool) -> i32;
    fn xhci_queue_get_port_bw(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u64, arg3: u8, arg4: bool) -> i32;
    fn xhci_queue_evaluate_context(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u64, arg3: u32, arg4: bool) -> i32;
    fn xhci_queue_reset_device(arg0: *mut xhci_hcd, arg1: *mut xhci_command, arg2: u32) -> i32;
    fn xhci_handle_command_timeout(arg0: *mut work_struct);
    fn xhci_ring_doorbell_for_active_rings(arg0: *mut xhci_hcd, arg1: u32, arg2: u32);
    fn xhci_cleanup_command_queue(arg0: *mut xhci_hcd);
    fn xhci_process_cancelled_tds(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_ep);
    fn xhci_add_interrupter(arg0: *mut xhci_hcd, arg1: u32);
    fn xhci_hub_control(arg0: *mut usb_hcd, arg1: u16, arg2: u16, arg3: u16, arg4: *mut u8, arg5: u16) -> i32;
    fn xhci_hub_status_data(arg0: *mut usb_hcd, arg1: *mut u8) -> i32;
    fn xhci_get_rhub(arg0: *mut usb_hcd) -> *mut xhci_hub;
    fn xhci_port_is_tunneled(arg0: *mut xhci_hcd, arg1: *mut xhci_port) -> u32;
    fn xhci_hc_died(arg0: *mut xhci_hcd);
    fn xhci_bus_suspend(arg0: *mut usb_hcd) -> i32;
    fn xhci_bus_resume(arg0: *mut usb_hcd) -> i32;
    fn xhci_get_resuming_ports(arg0: *mut usb_hcd) -> u64;
    fn xhci_port_state_to_neutral(arg0: u32) -> u32;
    fn xhci_get_input_control_ctx(arg0: *mut xhci_container_ctx) -> *mut xhci_input_control_ctx;
    fn xhci_get_slot_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx) -> *mut xhci_slot_ctx;
    fn xhci_get_ep_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: u32) -> *mut xhci_ep_ctx;
    fn xhci_triad_to_transfer_ring(arg0: *mut xhci_hcd, arg1: u32, arg2: u32, arg3: u32) -> *mut xhci_ring;
    fn __traceiter_xhci_dbg_address(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_address(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_539();
    fn __traceiter_xhci_dbg_context_change(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_context_change(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_546();
    fn __traceiter_xhci_dbg_quirks(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_quirks(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_553();
    fn __traceiter_xhci_dbg_cancel_urb(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_cancel_urb(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_567();
    fn __traceiter_xhci_dbg_init(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __SCT__tp_func_xhci_dbg_init(arg0: *mut (), arg1: *mut va_format) -> i32;
    fn __compiletime_assert_574();
    fn __traceiter_xhci_address_ctx(arg0: *mut (), arg1: *mut xhci_hcd, arg2: *mut xhci_container_ctx) -> i32;
    fn __SCT__tp_func_xhci_address_ctx(arg0: *mut (), arg1: *mut xhci_hcd, arg2: *mut xhci_container_ctx) -> i32;
    fn __compiletime_assert_588();
    fn __traceiter_xhci_setup_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __SCT__tp_func_xhci_setup_device(arg0: *mut (), arg1: *mut xhci_virt_device) -> i32;
    fn __compiletime_assert_658();
    fn __traceiter_xhci_urb_enqueue(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __SCT__tp_func_xhci_urb_enqueue(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __compiletime_assert_679();
    fn __traceiter_xhci_urb_dequeue(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __SCT__tp_func_xhci_urb_dequeue(arg0: *mut (), arg1: *mut urb) -> i32;
    fn __compiletime_assert_693();
    fn __traceiter_xhci_add_endpoint(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __SCT__tp_func_xhci_add_endpoint(arg0: *mut (), arg1: *mut xhci_ep_ctx) -> i32;
    fn __compiletime_assert_742();
    fn __traceiter_xhci_alloc_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_alloc_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_749();
    fn __traceiter_xhci_free_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_free_dev(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_756();
    fn __traceiter_xhci_discover_or_reset_device(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_discover_or_reset_device(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_770();
    fn __traceiter_xhci_setup_device_slot(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_setup_device_slot(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_777();
    fn __traceiter_xhci_configure_endpoint(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __SCT__tp_func_xhci_configure_endpoint(arg0: *mut (), arg1: *mut xhci_slot_ctx) -> i32;
    fn __compiletime_assert_805();
    fn __traceiter_xhci_address_ctrl_ctx(arg0: *mut (), arg1: *mut xhci_input_control_ctx) -> i32;
    fn __SCT__tp_func_xhci_address_ctrl_ctx(arg0: *mut (), arg1: *mut xhci_input_control_ctx) -> i32;
    fn __compiletime_assert_812();
    fn __traceiter_xhci_configure_endpoint_ctrl_ctx(arg0: *mut (), arg1: *mut xhci_input_control_ctx) -> i32;
    fn __SCT__tp_func_xhci_configure_endpoint_ctrl_ctx(arg0: *mut (), arg1: *mut xhci_input_control_ctx) -> i32;
    fn __compiletime_assert_819();
    fn __traceiter_xhci_portsc_writel(arg0: *mut (), arg1: *mut xhci_port, arg2: u32) -> i32;
    fn __SCT__tp_func_xhci_portsc_writel(arg0: *mut (), arg1: *mut xhci_port, arg2: u32) -> i32;
    fn __compiletime_assert_882();
    fn xhci_debugfs_init(arg0: *mut xhci_hcd);
    fn xhci_debugfs_exit(arg0: *mut xhci_hcd);
    fn xhci_debugfs_create_root();
    fn xhci_debugfs_remove_root();
    fn xhci_debugfs_create_slot(arg0: *mut xhci_hcd, arg1: i32);
    fn xhci_debugfs_remove_slot(arg0: *mut xhci_hcd, arg1: i32);
    fn xhci_debugfs_create_endpoint(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: i32);
    fn xhci_debugfs_remove_endpoint(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: i32);
    fn xhci_debugfs_create_stream_files(arg0: *mut xhci_hcd, arg1: *mut xhci_virt_device, arg2: i32);
    fn __compiletime_assert_935();
    fn __compiletime_assert_947();
    fn __compiletime_assert_948();
    fn __compiletime_assert_949();
    fn __compiletime_assert_950();
    fn __compiletime_assert_951();
    fn __compiletime_assert_952();
    fn __compiletime_assert_953();
    fn __compiletime_assert_954();
    fn __compiletime_assert_955();
    fn __compiletime_assert_956();
    fn __compiletime_assert_959();
    fn __compiletime_assert_960();
    fn __compiletime_assert_961();
    fn __compiletime_assert_962();
    fn __compiletime_assert_963();
    fn __compiletime_assert_966();
    fn __compiletime_assert_968();
    fn __compiletime_assert_969();
    fn __compiletime_assert_970();
    fn __compiletime_assert_971();
    fn __compiletime_assert_972();
    fn __compiletime_assert_973();
    fn __compiletime_assert_974();
    fn __compiletime_assert_975();
    fn __compiletime_assert_976();
    fn __compiletime_assert_977();
    fn __compiletime_assert_978();
    fn __compiletime_assert_979();
    fn __compiletime_assert_980();
    fn __compiletime_assert_981();
    fn __compiletime_assert_982();
    fn __compiletime_assert_985();
    fn __compiletime_assert_986();
    fn __compiletime_assert_987();
    fn __compiletime_assert_988();
    fn __compiletime_assert_989();
    fn __compiletime_assert_990();
    fn __compiletime_assert_993();
    fn __compiletime_assert_994();
    fn __compiletime_assert_995();
    fn __compiletime_assert_996();
    fn __compiletime_assert_997();
    fn __compiletime_assert_998();
    fn __compiletime_assert_1001();
    fn __compiletime_assert_1002();
    fn __compiletime_assert_1003();
    fn __compiletime_assert_1004();
    fn __compiletime_assert_1005();
    fn __compiletime_assert_1006();
    fn __compiletime_assert_1007();
    fn __compiletime_assert_1008();
    fn __compiletime_assert_1009();
    fn __compiletime_assert_1010();
    fn __compiletime_assert_1011();
    fn __compiletime_assert_1019();
    fn __compiletime_assert_1020();
    fn __compiletime_assert_1021();
    fn __compiletime_assert_1022();
    fn __compiletime_assert_1023();
    fn __compiletime_assert_1024();
    fn __compiletime_assert_1025();
    fn __compiletime_assert_1026();
    fn __compiletime_assert_1027();
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
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
}


#[inline(always)]
fn __arch_hweight32(mut w: u32) -> u32 {
let mut res: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight32\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntl %edi, %eax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u32, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
res
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn fls_long(l: u64) -> u32 {
if ((const { core::mem::size_of::<u64>() }) as u64) == 4 {
return fls(l as u32) as u32;
}
fls64(l) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn __roundup_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n.wrapping_sub(1))
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_address, const 0 as u8, label {
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
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_context_change, const 0 as u8, label {
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
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
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
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
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
struct futex_private_hash {
}


#[inline(always)]
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_dbg_init, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_address_ctx, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_setup_device, const 0 as u8, label {
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
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_urb_enqueue, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_urb_dequeue, const 0 as u8, label {
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
fn arch_static_branch__const10(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_add_endpoint, const 0 as u8, label {
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
fn arch_static_branch__const11(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_alloc_dev, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_free_dev, const 0 as u8, label {
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
fn arch_static_branch__const13(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_discover_or_reset_device, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_setup_device_slot, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_configure_endpoint, const 0 as u8, label {
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
fn arch_static_branch__const16(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_address_ctrl_ctx, const 0 as u8, label {
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
struct hrtimer_clock_base {
}


#[inline(always)]
fn arch_static_branch__const17(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_configure_endpoint_ctrl_ctx, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_xhci_portsc_writel, const 0 as u8, label {
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
struct seccomp_filter {
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
    __slate_bits_0: __slate_bits::__SlateBits133U0,
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
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
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


#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
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

#[inline(always)]
fn raw_atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
raw_atomic_fetch_add_unless(v, a, u) != u
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
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_unless(v, a, u)
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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
struct workqueue_struct {
}


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
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

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
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

#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn cpu_relax() {
native_pause();
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
    __slate_bits_0: __slate_bits::__SlateBits134U0,
    pkru: u32,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits228U0,
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


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct irq_domain {
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
struct user_namespace {
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
struct ns_common {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits291U0,
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
    __slate_bits_0: __slate_bits::__SlateBits73U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits73U1,
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
    __slate_bits_2: __slate_bits::__SlateBits73U2,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
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
struct dev_pm_qos {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

fn ktime_compare(cmp1: i64, cmp2: i64) -> i32 {
if cmp1 < cmp2 {
return 1i32.wrapping_neg();
}
if cmp1 > cmp2 {
return 1;
}
0
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


fn ktime_add_us(kt: i64, usec: u64) -> i64 {
ktime_add_ns!(kt, usec.wrapping_mul(1000)) as i64
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
struct kmem_cache {
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
struct mnt_idmap {
}


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_179() };
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
struct assoc_array_ptr {
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
struct key_type {
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


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits416U0,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
    __slate_bits_0: __slate_bits::__SlateBits465U0,
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
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
    __slate_bits_0: __slate_bits::__SlateBits467U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct posix_acl {
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits492U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits492U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits492U2,
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
    __slate_bits_3: __slate_bits::__SlateBits492U3,
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
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
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
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
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
struct hotplug_slot {
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
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
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
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
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
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_pool {
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
    __slate_bits_0: __slate_bits::__SlateBits294U0,
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
struct iova_bitmap {
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
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_viommu {
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
    __slate_bits_0: __slate_bits::__SlateBits620U0,
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
struct ep_device {
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
#[inline(always)]
fn _kmalloc_node_noprof(size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_node_noprof(size, flags, node) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_node_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, node, size) };
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(size), flags, node) }
}

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
struct mon_bus {
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
struct gen_pool {
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


// module_param - typesafe helper for a module/cmdline parameter
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// @name becomes the module parameter, or (prefixed by KBUILD_MODNAME and a
// ".") the kernel commandline parameter.  Note that - is changed to _, so
// the user can use "foo-bar=1" even for variable "foo_bar".
//
// @perm is 0 if the variable is not to appear in sysfs, or 0444
// for world-readable, 0644 for root-writable, etc.  Note that if it
// is writable, you may need to use kernel_param_lock() around
// accesses (esp. charp, which can be kfreed when it changes).
//
// The @type is simply pasted to refer to a param_ops_##type and a
// param_check_##type: for convenience many standard types are provided but
// you can create your own by defining those variables.
//
// Standard types are:
// 	byte, hexint, short, ushort, int, uint, long, ulong
// 	charp: a character pointer
// 	bool: a bool, values 0/1, y/n, Y/N.
// 	invbool: the above, only sense-reversed (N = true).
// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
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
    __slate_bits_0: __slate_bits::__SlateBits489U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_to_node(dev: *mut device) -> i32 {
unsafe { (*dev).numa_node }
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
    __slate_bits_0: __slate_bits::__SlateBits733U0,
    kobj: kobject,
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
    __slate_bits_0: __slate_bits::__SlateBits728U0,
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
    __slate_bits_1: __slate_bits::__SlateBits728U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits728U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits728U3,
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
    __slate_bits_4: __slate_bits::__SlateBits728U4,
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
    __slate_bits_0: __slate_bits::__SlateBits446U0,
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
struct serial_icounter_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
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


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
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
fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
}


fn compound_order(page: *const page) -> u32 {
let folio: *const folio = page as *const folio;
if test_bit__6!(6i32, unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags.f) }) == 0 {
return 0;
}
folio_large_order(folio)
}

fn page_size(page: *const page) -> u64 {
_AC!() << 12i32 << compound_order(page)
}

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
struct __SlateRecord_xhci_c_38 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_38_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_38_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_38_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_38_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_38_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_42_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_482 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_496 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_499 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_513 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_647 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_1642_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3511_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_3641_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_4612_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5464_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5467 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5472_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_12 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5478_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_12 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5479_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_12 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5483_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5484_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5485_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5534 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5537 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5543 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_c_5547 {
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
    pub struct __SlateBits291U0 {
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
    pub struct __SlateBits73U0 {
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
    pub struct __SlateBits73U1 {
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
    pub struct __SlateBits294U0 {
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
    pub struct __SlateBits492U0 {
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
    pub struct __SlateBits489U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U1 {
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
    pub struct __SlateBits492U2 {
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
    pub struct __SlateBits492U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits769U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits705U0 {
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
    pub struct __SlateBits920U0 {
        #[bits(1)]
        pub console: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits73U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits126U0 {
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
    pub struct __SlateBits134U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits416U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits620U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits465U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits467U0 {
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
    pub struct __SlateBits133U0 {
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
    pub struct __SlateBits228U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits837U0 {
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
    pub struct __SlateBits833U0 {
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
    pub struct __SlateBits826U0 {
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
    pub struct __SlateBits826U1 {
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
    pub struct __SlateBits826U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits846U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits850U0 {
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
    pub struct __SlateBits883U0 {
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
    pub struct __SlateBits859U0 {
        #[bits(1)]
        pub hw_lpm_support: u32,
        #[bits(1)]
        pub broken_suspend: u32,
        #[bits(1)]
        pub allow_single_roothub: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits446U0 {
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
    pub struct __SlateBits733U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits728U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits728U1 {
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
    pub struct __SlateBits728U2 {
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
    pub struct __SlateBits728U3 {
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
    pub struct __SlateBits728U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
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
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_505: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn dma_unmap_single_attrs(dev: *mut device, addr: u64, size: u64, dir: u32, attrs: u64) {
unsafe { dma_unmap_page_attrs(dev, addr, size, dir, attrs) };
return;
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
    __slate_bits_0: __slate_bits::__SlateBits769U0,
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
    __slate_bits_0: __slate_bits::__SlateBits705U0,
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
struct iommu_attach_handle {
    domain: *mut iommu_domain,
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

fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
}

fn str_enable_disable(v: bool) -> *const u8 {
(if v { c"enable".as_ptr() as *mut u8 } else { c"disable".as_ptr() as *mut u8 }) as *const u8
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
fn usb_endpoint_num(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bEndpointAddress }) as i32) & 15
}

fn usb_endpoint_type(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bmAttributes }) as i32) & 3
}

fn usb_endpoint_dir_in(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bEndpointAddress }) as i32) & 128 == 128) as i32
}

fn usb_endpoint_dir_out(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bEndpointAddress }) as i32) & 128 == 0) as i32
}

fn usb_endpoint_xfer_bulk(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 2) as i32
}

fn usb_endpoint_xfer_control(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 0) as i32
}

fn usb_endpoint_xfer_int(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 3) as i32
}

fn usb_endpoint_xfer_isoc(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bmAttributes }) as i32) & 3 == 1) as i32
}

fn usb_endpoint_is_bulk_out(epd: *const usb_endpoint_descriptor) -> i32 {
let __v0: bool = usb_endpoint_xfer_bulk(epd) != 0 && usb_endpoint_dir_out(epd) != 0;
__v0 as i32
}

fn usb_endpoint_maxp(epd: *const usb_endpoint_descriptor) -> i32 {
(__le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32) & 2047
}

fn usb_endpoint_interrupt_type(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bmAttributes }) as i32) & 48
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


fn usb_ss_max_streams(comp: *const usb_ss_ep_comp_descriptor) -> i32 {
let mut max_streams: i32 = 0;
if comp.is_null() {
return 0;
}
max_streams = ((unsafe { (*comp).bmAttributes }) as i32) & 31;
if max_streams == 0 {
return 0;
}
max_streams = 1i32 << max_streams;
max_streams
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


// A U1 timeout of 0x0 means the parent hub will reject any transitions to U1.
// 0xff means the parent hub will accept transitions to U1, but will not
// initiate a transition.
//
// A U1 timeout of 0x1 to 0x7F also causes the hub to initiate a transition to
// U1 after that many microseconds.  Timeouts of 0x80 to 0xFE are reserved
// values.
//
// A U2 timeout of 0x0 means the parent hub will reject any transitions to U2.
// 0xff means the parent hub will accept transitions to U2, but will not
// initiate a transition.
//
// A U2 timeout of 0x1 to 0xFE also causes the hub to initiate a transition to
// U2 after N*256 microseconds.  Therefore a U2 timeout value of 0x1 means a U2
// idle timer of 256 microseconds, 0x2 means 512 microseconds, 0xFE means
// 65.024ms.
// The Set System Exit Latency control transfer provides one byte each for
// U1 SEL and U1 PEL, so the max exit latency is 0xFF.  U2 SEL and U2 PEL each
// are two bytes long.
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_get_noresume(dev: *mut device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*dev).power.usage_count) });
}

fn pm_runtime_put_noidle(dev: *mut device) {
atomic_add_unless(unsafe { core::ptr::addr_of_mut!((*dev).power.usage_count) }, 1i32.wrapping_neg(), 0i32);
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
    __slate_bits_0: __slate_bits::__SlateBits833U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_bus {
    controller: *mut device,
    sysdev: *mut device,
    busnum: i32,
    bus_name: *const u8,
    uses_pio_for_control: u8,
    otg_port: u8,
    __slate_bits_0: __slate_bits::__SlateBits837U0,
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
    __slate_bits_0: __slate_bits::__SlateBits826U0,
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
    __slate_bits_1: __slate_bits::__SlateBits826U1,
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
    __slate_bits_2: __slate_bits::__SlateBits826U2,
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
    __slate_bits_0: __slate_bits::__SlateBits827U0,
}


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
    __slate_bits_0: __slate_bits::__SlateBits846U0,
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
    __slate_bits_0: __slate_bits::__SlateBits850U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_sideband_event {
    r#type: u32,
    evt_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_sideband {
    xhci: *mut xhci_hcd,
    vdev: *mut xhci_virt_device,
    eps: [*mut xhci_virt_ep; 31],
    ir: *mut xhci_interrupter,
    r#type: u32,
    mutex: mutex,
    intf: *mut usb_interface,
    notify_client: Option<unsafe extern "C" fn(*mut usb_interface, *mut xhci_sideband_event) -> i32>,
}


fn xhci_sideband_notify_ep_ring_free(sb: *mut xhci_sideband, ep_index: u32) {
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_xhci_portsc_writel: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_portsc_readl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_run: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_stop: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_shutdown: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_suspend: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_resume: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_get_endpoint_index: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_drop_endpoint: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_add_endpoint: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_stop_endpoint_sync: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_check_bandwidth: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_reset_bandwidth: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_update_hub_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_gen_setup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_xhci_init_driver: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_xhci_hcd__1028_5737_xhci_hcd_init6:\t\t\t\n.long\t{16} - .\t\n.previous\t\t\t\t\t\n", sym xhci_portsc_writel, sym xhci_portsc_readl, sym xhci_run, sym xhci_stop, sym xhci_shutdown, sym xhci_suspend, sym xhci_resume, sym xhci_get_endpoint_index, sym xhci_drop_endpoint, sym xhci_add_endpoint, sym xhci_stop_endpoint_sync, sym xhci_check_bandwidth, sym xhci_reset_bandwidth, sym xhci_update_hub_device, sym xhci_gen_setup, sym xhci_init_driver, sym xhci_hcd_init, options(att_syntax));

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

// Max Number of Ports. xHCI specification section 5.3.3
// Valid values are in the range of 1 to 255.
// Max number of Interrupter Register Sets. xHCI specification section 5.3.3
// Valid values are in the range of 1 to 1024.
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
// Reset HC - resets internal HC state machine and all registers (except
// PCI config regs).  HC does NOT drive a USB reset on the downstream ports.
// The xHCI driver must reinitialize the xHC after setting this bit.
// Event Interrupt Enable - a '1' allows interrupts from the host controller
// Host System Error Interrupt Enable - get out-of-band signal for HC errors
// host controller save/restore state.
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
// DNCTRL - Device Notification Control Register - dev_notification bitmasks
// Generate a device notification event when the HC sees a transaction with a
// notification type that matches a bit set in this bit field.
// Most of the device notification types should only be used for debug.
// SW does need to pay attention to function wake notifications.
// CRCR - Command Ring Control Register - cmd_ring bitmasks
//
// bit 0 - Cycle bit indicates the ownership of the command ring
// bits 63:6 - Command Ring pointer
// bits 15:0 - HCD page shift bit
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
// bit 1 - Interrupt Enable (IE), whether the interrupter is capable of generating an interrupt
// imod bitmasks
//
// bits 15:0 - Interrupt Moderation Interval, the minimum interval between interrupts
// (in 250ns intervals). The interval between interrupts will be longer if there are no
// events on the event ring. Default is 4000 (1 ms).
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


// "Block" sizes in bytes the hardware uses for different device speeds.
// The logic in this part of the hardware limits the number of bits the hardware
// can use, so must represent bandwidth in a less precise manner to mimic what
// the scheduler hardware computes.
// The TTs need to claim roughly twice as much bandwidth (94 bytes per
// microframe ~= 24Mbps) of the HS bus as the devices can actually use because
// of overhead associated with split transfers crossing microframe boundaries.
// 31 blocks is pure protocol overhead.
// Bandwidth limits in blocks
// Percentage of bus bandwidth reserved for non-periodic transfers
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


// Transfer event flags bitfield, also for select command completion events
// Endpoint ID 1 - 31
// Transfer event TRB length bit mask
fn xhci_trb_comp_code_string(status: u8) -> *const u8 {
match status as i32 {
0 => {
return c"Invalid".as_ptr() as *const u8;
}
1 => {
return c"Success".as_ptr() as *const u8;
}
2 => {
return c"Data Buffer Error".as_ptr() as *const u8;
}
3 => {
return c"Babble Detected".as_ptr() as *const u8;
}
4 => {
return c"USB Transaction Error".as_ptr() as *const u8;
}
5 => {
return c"TRB Error".as_ptr() as *const u8;
}
6 => {
return c"Stall Error".as_ptr() as *const u8;
}
7 => {
return c"Resource Error".as_ptr() as *const u8;
}
8 => {
return c"Bandwidth Error".as_ptr() as *const u8;
}
9 => {
return c"No Slots Available Error".as_ptr() as *const u8;
}
10 => {
return c"Invalid Stream Type Error".as_ptr() as *const u8;
}
11 => {
return c"Slot Not Enabled Error".as_ptr() as *const u8;
}
12 => {
return c"Endpoint Not Enabled Error".as_ptr() as *const u8;
}
13 => {
return c"Short Packet".as_ptr() as *const u8;
}
14 => {
return c"Ring Underrun".as_ptr() as *const u8;
}
15 => {
return c"Ring Overrun".as_ptr() as *const u8;
}
16 => {
return c"VF Event Ring Full Error".as_ptr() as *const u8;
}
17 => {
return c"Parameter Error".as_ptr() as *const u8;
}
18 => {
return c"Bandwidth Overrun Error".as_ptr() as *const u8;
}
19 => {
return c"Context State Error".as_ptr() as *const u8;
}
20 => {
return c"No Ping Response Error".as_ptr() as *const u8;
}
21 => {
return c"Event Ring Full Error".as_ptr() as *const u8;
}
22 => {
return c"Incompatible Device Error".as_ptr() as *const u8;
}
23 => {
return c"Missed Service Error".as_ptr() as *const u8;
}
24 => {
return c"Command Ring Stopped".as_ptr() as *const u8;
}
25 => {
return c"Command Aborted".as_ptr() as *const u8;
}
26 => {
return c"Stopped".as_ptr() as *const u8;
}
27 => {
return c"Stopped - Length Invalid".as_ptr() as *const u8;
}
28 => {
return c"Stopped - Short Packet".as_ptr() as *const u8;
}
29 => {
return c"Max Exit Latency Too Large Error".as_ptr() as *const u8;
}
31 => {
return c"Isoch Buffer Overrun".as_ptr() as *const u8;
}
32 => {
return c"Event Lost Error".as_ptr() as *const u8;
}
33 => {
return c"Undefined Error".as_ptr() as *const u8;
}
34 => {
return c"Invalid Stream ID Error".as_ptr() as *const u8;
}
35 => {
return c"Secondary Bandwidth Error".as_ptr() as *const u8;
}
36 => {
return c"Split Transaction Error".as_ptr() as *const u8;
}
_ => {
return c"Unknown!!".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_link_trb {
    segment_ptr: u64,
    intr_target: u32,
    control: u32,
}


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
// Set TR Dequeue Pointer command TRB fields, 6.4.3.9
// Link TRB specific fields
// Normal TRB fields
//
// transfer_len bitmasks - bits 0:16
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
// Isochronous TRB specific fields
// TRB cache size for xHC with TRB cache
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
// Nec vendor-specific command completion event.
// Get NEC firmware revision.
// TRBS_PER_SEGMENT must be a multiple of 4,
// since the command ring is 64-byte aligned.
// It must also be greater than 16.
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


// xHCI command default timeout value in milliseconds.
// USB 3.2 spec, section 9.2.6.1
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
    __slate_bits_0: __slate_bits::__SlateBits883U0,
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
    __slate_bits_0: __slate_bits::__SlateBits859U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_driver_overrides {
    extra_priv_size: u64,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    add_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    drop_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    check_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    reset_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    update_hub_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_tt, u32) -> i32>,
    hub_control: Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>,
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

fn xhci_hcd_is_usb3(hcd: *mut usb_hcd) -> bool {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
hcd == xhci_get_usb3_hcd(xhci)
}

fn xhci_has_one_roothub(xhci: *mut xhci_hcd) -> bool {
((unsafe { (*xhci).__slate_bits_0.__get_allow_single_roothub() }) as i32) != 0 && ((unsafe { (*xhci).usb2_rhub.num_ports }) == 0 || (unsafe { (*xhci).usb3_rhub.num_ports }) == 0)
}

fn xhci_read_64(xhci: *const xhci_hcd, regs: *mut u64) -> u64 {
lo_hi_readq(regs as *const ())
}

fn xhci_write_64(xhci: *mut xhci_hcd, val: u64, regs: *mut u64) {
lo_hi_writeq(val, regs as *mut ());
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_gen_setup")]
extern "C" fn xhci_gen_setup(hcd: *mut usb_hcd, get_quirks: Option<unsafe extern "C" fn(*mut device, *mut xhci_hcd)>) -> i32 {
let mut __slate_compound_70: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_72: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
// TODO: Check with DWC3 clients for sysdev according to
// quirks
let dev: *mut device = unsafe { (*hcd).self_.sysdev };
let mut retval: i32 = 0;
let mut hcs_params1: u32 = 0;
let mut hc_capbase: u32 = 0;
// Accept arbitrarily long scatter-gather lists
unsafe {
(*hcd).self_.sg_tablesize = 4294967295u32;
}
// support to build packet from discontinuous buffers
unsafe {
(*hcd).self_.__slate_bits_0.__set_no_sg_constraint(1u32);
}
// XHCI controllers don't stop the ep queue on short packets :|
unsafe {
(*hcd).self_.__slate_bits_0.__set_no_stop_on_short(1u32);
}
xhci = hcd_to_xhci(hcd);
if (unsafe { usb_hcd_is_primary_hcd(hcd) }) == 0 {
xhci_hcd_init_usb3_data(xhci, hcd);
return 0;
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*xhci).mutex) }, c"&xhci->mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5456) });
if true {
break;
}
}
unsafe {
(*xhci).main_hcd = hcd;
}
unsafe {
(*xhci).cap_regs = (*hcd).regs as *mut xhci_cap_regs;
}
hc_capbase = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).hc_capbase) }) as *const ());
if hc_capbase == !0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host controller not accessible, removed?\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
let mut __v2: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5464_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_968() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v4: u64;
loop {
if false {
unsafe { __compiletime_assert_969() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5464_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) == 0 {
unsafe { __compiletime_assert_970() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5464_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5464_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_971() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5464_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5464_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5464_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5464_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_972() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = ((hc_capbase as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5464_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5464_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32);
__v2 = __v4;
unsafe {
(*xhci).op_regs = (((*hcd).regs as *const u8).wrapping_offset(__v2 as isize) as *mut ()) as *mut xhci_op_regs;
}
unsafe {
(*xhci).run_regs = (((*hcd).regs as *const u8).wrapping_offset(((readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).run_regs_off) as *const ()) as u64) & GENMASK__4!(31i32, 5i32)) as isize) as *mut ()) as *mut xhci_run_regs;
}
// Cache read-only capability registers
hcs_params1 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).hcs_params1) }) as *const ());
unsafe {
(*xhci).hcs_params2 = readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).hcs_params2) as *const ());
}
unsafe {
(*xhci).hcs_params3 = readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).hcs_params3) as *const ());
}
let mut __v9: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5472_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_973() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v11: u64;
loop {
if false {
unsafe { __compiletime_assert_974() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5472_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) == 0 {
unsafe { __compiletime_assert_975() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5472_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5472_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_976() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5472_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5472_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5472_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5472_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_977() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = ((hc_capbase as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5472_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5472_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32);
__v9 = __v11;
unsafe {
(*xhci).hci_version = __v9 as u16;
}
unsafe {
(*xhci).hcc_params = readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).hcc_params) as *const ());
}
if ((unsafe { (*xhci).hci_version }) as i32) > 256 {
unsafe {
(*xhci).hcc_params2 = readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).hcc_params2) as *const ());
}
}
unsafe {
(*xhci).dma_mask_bits = 64;
}
let mut __v16: u64;
let mut __UNIQUE_ID_x__983: u64 = 0;
let mut __v17: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5478_6>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_978() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v19: u64;
loop {
if false {
unsafe { __compiletime_assert_979() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5478_8>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) == 0 {
unsafe { __compiletime_assert_980() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5478_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5478_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_981() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5478_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5478_12>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5478_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5478_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_982() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v19 = ((hcs_params1 as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5478>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5478_2>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32);
__v17 = __v19;
__UNIQUE_ID_x__983 = __v17;
let __UNIQUE_ID_y__984: i32 = 255;
loop {
if __types_ok__4!(__UNIQUE_ID_x__983, __UNIQUE_ID_y__984) == 0 {
unsafe { __compiletime_assert_985() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v16 = if __UNIQUE_ID_x__983 < (__UNIQUE_ID_y__984 as u64) { __UNIQUE_ID_x__983 } else { __UNIQUE_ID_y__984 as u64 };
unsafe {
(*xhci).max_slots = __v16 as u8;
}
let mut __v25: u64;
let mut __UNIQUE_ID_x__991: u64 = 0;
let mut __v26: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5479_6>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_986() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v28: u64;
loop {
if false {
unsafe { __compiletime_assert_987() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5479_8>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) == 0 {
unsafe { __compiletime_assert_988() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5479_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5479_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_989() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5479_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5479_12>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5479_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5479_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_990() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v28 = ((hcs_params1 as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5479>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5479_2>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 24i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32);
__v26 = __v28;
__UNIQUE_ID_x__991 = __v26;
let __UNIQUE_ID_y__992: i32 = 127;
loop {
if __types_ok__4!(__UNIQUE_ID_x__991, __UNIQUE_ID_y__992) == 0 {
unsafe { __compiletime_assert_993() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v25 = if __UNIQUE_ID_x__991 < (__UNIQUE_ID_y__992 as u64) { __UNIQUE_ID_x__991 } else { __UNIQUE_ID_y__992 as u64 };
unsafe {
(*xhci).max_ports = __v25 as u8;
}
// xhci-plat or xhci-pci might have set max_interrupters already
if (unsafe { (*xhci).max_interrupters }) == 0 {
let mut __v34: u64;
let mut __UNIQUE_ID_x__999: u64 = 0;
let mut __v35: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5483_6>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_994() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v37: u64;
loop {
if false {
unsafe { __compiletime_assert_995() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5483_8>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) == 0 {
unsafe { __compiletime_assert_996() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5483_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5483_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_997() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5483_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5483_12>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5483_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5483_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_998() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v37 = ((hcs_params1 as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5483>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5483_2>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32);
__v35 = __v37;
__UNIQUE_ID_x__999 = __v35;
let __UNIQUE_ID_y__1000: i32 = 128;
loop {
if __types_ok__4!(__UNIQUE_ID_x__999, __UNIQUE_ID_y__1000) == 0 {
unsafe { __compiletime_assert_1001() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v34 = if __UNIQUE_ID_x__999 < (__UNIQUE_ID_y__1000 as u64) { __UNIQUE_ID_x__999 } else { __UNIQUE_ID_y__1000 as u64 };
unsafe {
(*xhci).max_interrupters = __v34 as u16;
}
} else {
let mut __v43: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5484_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_1002() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v45: u64;
loop {
if false {
unsafe { __compiletime_assert_1003() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5484_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) == 0 {
unsafe { __compiletime_assert_1004() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5484_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5484_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_1005() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5484_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5484_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5484_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5484_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_1006() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v45 = ((hcs_params1 as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5484_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5484_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32);
__v43 = __v45;
if (((unsafe { (*xhci).max_interrupters }) as i32) as u64) > __v43 {
let mut __v50: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_5485_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_1007() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v52: u64;
loop {
if false {
unsafe { __compiletime_assert_1008() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_5485_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) == 0 {
unsafe { __compiletime_assert_1009() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_5485_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5485_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_1010() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_5485_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5485_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_5485_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_5485_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_1011() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v52 = ((hcs_params1 as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_5485_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_5485_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(18)).trailing_zeros() as i32);
__v50 = __v52;
unsafe {
(*xhci).max_interrupters = __v50 as u16;
}
}
}
unsafe {
(*xhci).quirks = (*xhci).quirks | quirks;
}
if get_quirks != None {
unsafe { get_quirks.unwrap()(dev, xhci) };
}
// In xhci controllers which follow xhci 1.0 spec gives a spurious
// success event after a short transfer. This quirk will ignore such
// spurious event.
if ((unsafe { (*xhci).hci_version }) as i32) > 150 {
let __v61: u64 = unsafe { (*xhci).quirks };
let __v62: u64 = __v61 | BIT_ULL!(4i32);
unsafe {
(*xhci).quirks = __v62;
}
}
if ((unsafe { (*xhci).hci_version }) as i32) == 149 && (unsafe { link_quirk }) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"QUIRK: Not clearing Link TRB chain bits".as_ptr() as *const u8) };
}
let __v64: u64 = unsafe { (*xhci).quirks };
let __v65: u64 = __v64 | BIT_ULL!(0i32);
unsafe {
(*xhci).quirks = __v65;
}
}
// Make sure the HC is halted.
retval = xhci_halt(xhci);
if retval != 0 {
return retval;
}
xhci_zero_64b_regs(xhci);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Resetting HCD\n".as_ptr() as *const u8) };
}
// Reset the internal HC memory state and registers.
retval = xhci_reset(xhci, 10000000u64);
if retval != 0 {
return retval;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Reset complete\n".as_ptr() as *const u8) };
}
// On some xHCI controllers (e.g. R-Car SoCs), the AC64 bit (bit 0)
// of HCCPARAMS1 is set to 1. However, the xHCs don't support 64-bit
// address memory pointers actually. So, this driver clears the AC64
// bit of xhci->hcc_params to call dma_set_coherent_mask(dev,
// DMA_BIT_MASK(32)) in this xhci_gen_setup().
if (unsafe { (*xhci).quirks }) & BIT_ULL!(23i32) != 0 {
let __v67: u32 = unsafe { (*xhci).hcc_params };
let __v68: u32 = ((__v67 as u64) & !BIT!(0i32)) as u32;
unsafe {
(*xhci).hcc_params = __v68;
}
}
// Set dma_mask and coherent_dma_mask to 64-bits if xHC supports
// 64-bit addressing, unless a controller-specific quirk callback
// limits the usable address width.
let __v69: bool = ((unsafe { (*xhci).hcc_params }) as u64) & BIT!(0i32) != 0 && (unsafe { dma_set_mask(dev, DMA_BIT_MASK__4!(unsafe { (*xhci).dma_mask_bits })) }) == 0;
if __v69 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Enabling %u-bit DMA addresses.\n".as_ptr() as *const u8, (*xhci).dma_mask_bits) };
}
unsafe { dma_set_coherent_mask(dev, DMA_BIT_MASK__3!(unsafe { (*xhci).dma_mask_bits })) };
} else {
// This is to avoid error in cases where a 32-bit USB
// controller is used on a 64-bit capable system.
retval = unsafe { dma_set_mask(dev, DMA_BIT_MASK__2!(32i32)) };
if retval != 0 {
return retval;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Enabling 32-bit DMA addresses.\n".as_ptr() as *const u8) };
}
unsafe { dma_set_coherent_mask(dev, DMA_BIT_MASK__5!(32i32)) };
}
'__slate_break_71: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
unsafe {
*core::ptr::addr_of_mut!((*xhci).lock) = *{
__slate_compound_70 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_70)
};
}
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*xhci).cmd_list) });
'__slate_break_75: loop {
'__slate_break_74: loop {
'__slate_break_73: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*xhci).cmd_timer)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*xhci).cmd_timer)).work)).data = *{
__slate_compound_72 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_72)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*xhci).cmd_timer)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*xhci).cmd_timer)).work)).func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut work_struct)>>(xhci_handle_command_timeout as *const ()) };
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*xhci).cmd_timer)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
init_completion(unsafe { core::ptr::addr_of_mut!((*xhci).cmd_ring_stop_completion) });
xhci_hcd_page_size(xhci);
// Allocate xHCI data structures
retval = unsafe { xhci_mem_init(xhci, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if retval != 0 {
return retval;
}
// Initialize HCD and host controller data structures
xhci_init(hcd);
if xhci_hcd_is_usb3(hcd) {
xhci_hcd_init_usb3_data(xhci, hcd);
} else {
xhci_hcd_init_usb2_data(xhci, hcd);
}
while false {
}
unsafe { _dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"hcc params 0x%08x hci version 0x%x quirks 0x%016llx\n".as_ptr() as *const u8, (*xhci).hcc_params, (*xhci).hci_version as i32, (*xhci).quirks) };
0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_set_interrupter_moderation(ir: *mut xhci_interrupter, mut imod_interval: u32) -> i32 {
let mut imod: u32 = 0;
if ir.is_null() || unsafe { (*ir).ir_set }.is_null() {
return 22i32.wrapping_neg();
}
// IMODI value in IMOD register is in 250ns increments
let mut __v0: u64;
let __UNIQUE_ID_x__933: u64 = ((imod_interval / 250).wrapping_add(0) as u64).wrapping_add(0).wrapping_add(0);
let __UNIQUE_ID_y__934: u64 = 65535u64;
loop {
if __types_ok__3!(__UNIQUE_ID_x__933, __UNIQUE_ID_y__934) == 0 {
unsafe { __compiletime_assert_935() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__933 < __UNIQUE_ID_y__934 { __UNIQUE_ID_x__933 } else { __UNIQUE_ID_y__934 };
imod_interval = __v0 as u32;
imod = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).imod) }) as *const ());
let __v3: u32 = imod & 4294901760u32;
imod = __v3;
imod |= imod_interval;
writel(imod, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).imod) }) as *mut ());
0
}

fn xhci_urb_to_transfer_ring(xhci: *mut xhci_hcd, urb: *mut urb) -> *mut xhci_ring {
unsafe { xhci_triad_to_transfer_ring(xhci, (*(*urb).dev).slot_id as u32, xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*urb).ep).desc)), (*urb).stream_id) }
}

fn xhci_urb_suitable_for_idt(urb: *mut urb) -> bool {
let __v0: bool = usb_endpoint_xfer_isoc((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) == 0 && usb_urb_dir_out(urb) != 0;
let __v1: bool = __v0 && usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) >= 8;
if __v1 && (unsafe { (*urb).transfer_buffer_length }) <= 8 && (unsafe { (*urb).transfer_flags }) & 4 == 0 && (unsafe { (*urb).num_sgs }) == 0 {
return true;
}
false
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
    __slate_bits_0: __slate_bits::__SlateBits920U0,
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


fn xhci_create_dbc_dev(xhci: *mut xhci_hcd) -> i32 {
0
}

fn xhci_remove_dbc_dev(xhci: *mut xhci_hcd) {
}

fn xhci_dbc_init() -> i32 {
0
}

fn xhci_dbc_exit() {
}

fn xhci_dbc_suspend(xhci: *mut xhci_hcd) -> i32 {
0
}

fn xhci_dbc_resume(xhci: *mut xhci_hcd) -> i32 {
0
}

fn __do_trace_xhci_dbg_address(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_536: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_536) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_537: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_537) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_539() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_address)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_538: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_538;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_address as *const ()) };
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
static __trace_check_xhci_dbg_address: __SlateAlign16<[u8; 17]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 97u8, 100u8, 100u8, 114u8, 101u8, 115u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_address_540: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_address) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci.trace_xhci_dbg_address")]
extern "C" fn trace_xhci_dbg_address(vaf: *mut va_format) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_address.key) }) != 0 {
__do_trace_xhci_dbg_address(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_541: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_context_change(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_543: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_543) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_544: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_544) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_546() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_context_change)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_545: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_545;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_context_change as *const ()) };
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
static __trace_check_xhci_dbg_context_change: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 98u8, 103u8, 95u8, 99u8, 111u8, 110u8, 116u8, 101u8, 120u8, 116u8, 95u8, 99u8, 104u8, 97u8, 110u8, 103u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_context_change_547: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_context_change) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci.trace_xhci_dbg_context_change")]
extern "C" fn trace_xhci_dbg_context_change(vaf: *mut va_format) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_context_change.key) }) != 0 {
__do_trace_xhci_dbg_context_change(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_548: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

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
let mut __UNIQUE_ID_guard_550: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_550) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_551: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_551) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_553() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_quirks)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_552: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_552;
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
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_quirks_554: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_quirks) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci.trace_xhci_dbg_quirks")]
extern "C" fn trace_xhci_dbg_quirks(vaf: *mut va_format) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_quirks.key) }) != 0 {
__do_trace_xhci_dbg_quirks(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_555: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

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
let mut __UNIQUE_ID_guard_564: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_564) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_565: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_565) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_567() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_cancel_urb)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_566: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_566;
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
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_cancel_urb_568: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_cancel_urb) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci.trace_xhci_dbg_cancel_urb")]
extern "C" fn trace_xhci_dbg_cancel_urb(vaf: *mut va_format) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_cancel_urb.key) }) != 0 {
__do_trace_xhci_dbg_cancel_urb(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_569: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_dbg_init(vaf: *mut va_format) {
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
let mut __UNIQUE_ID_guard_571: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_571) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_572: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_572) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_574() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_init)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_573: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_573;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut va_format) -> i32>>(__SCT__tp_func_xhci_dbg_init as *const ()) };
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
static __trace_check_xhci_dbg_init: [u8; 14] = [120, 104, 99, 105, 95, 100, 98, 103, 95, 105, 110, 105, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_dbg_init_575: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_dbg_init) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.xhci.trace_xhci_dbg_init")]
extern "C" fn trace_xhci_dbg_init(vaf: *mut va_format) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_dbg_init.key) }) != 0 {
__do_trace_xhci_dbg_init(vaf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_576: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_address_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx) {
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
let mut __UNIQUE_ID_guard_585: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_585) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_586: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_586) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_588() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_address_ctx)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_587: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_587;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_hcd, *mut xhci_container_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_hcd, *mut xhci_container_ctx) -> i32>>(__SCT__tp_func_xhci_address_ctx as *const ()) };
unsafe { __v12.unwrap()(__data, xhci, ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_address_ctx: __SlateAlign16<[u8; 17]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 97u8, 100u8, 100u8, 114u8, 101u8, 115u8, 115u8, 95u8, 99u8, 116u8, 120u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_address_ctx_589: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_address_ctx) }) as *mut ();

fn trace_xhci_address_ctx(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_address_ctx.key) }) != 0 {
__do_trace_xhci_address_ctx(xhci, ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_590: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_setup_device(vdev: *mut xhci_virt_device) {
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
let mut __UNIQUE_ID_guard_655: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_655) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_656: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_656) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_658() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_setup_device)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_657: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_657;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_virt_device) -> i32>>(__SCT__tp_func_xhci_setup_device as *const ()) };
unsafe { __v12.unwrap()(__data, vdev) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_setup_device: __SlateAlign16<[u8; 18]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_setup_device_659: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_setup_device) }) as *mut ();

fn trace_xhci_setup_device(vdev: *mut xhci_virt_device) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_setup_device.key) }) != 0 {
__do_trace_xhci_setup_device(vdev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_660: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_urb_enqueue(urb: *mut urb) {
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
let mut __UNIQUE_ID_guard_676: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_676) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_677: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_677) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_urb_enqueue)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_678: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_678;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>>(__SCT__tp_func_xhci_urb_enqueue as *const ()) };
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
static __trace_check_xhci_urb_enqueue: __SlateAlign16<[u8; 17]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 117u8, 114u8, 98u8, 95u8, 101u8, 110u8, 113u8, 117u8, 101u8, 117u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_urb_enqueue_680: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_urb_enqueue) }) as *mut ();

fn trace_xhci_urb_enqueue(urb: *mut urb) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_urb_enqueue.key) }) != 0 {
__do_trace_xhci_urb_enqueue(urb);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_681: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_urb_dequeue(urb: *mut urb) {
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
let mut __UNIQUE_ID_guard_690: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_690) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_691: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_691) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_693() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_urb_dequeue)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_692: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_692;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut urb) -> i32>>(__SCT__tp_func_xhci_urb_dequeue as *const ()) };
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
static __trace_check_xhci_urb_dequeue: __SlateAlign16<[u8; 17]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 117u8, 114u8, 98u8, 95u8, 100u8, 101u8, 113u8, 117u8, 101u8, 117u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_urb_dequeue_694: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_urb_dequeue) }) as *mut ();

fn trace_xhci_urb_dequeue(urb: *mut urb) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_urb_dequeue.key) }) != 0 {
__do_trace_xhci_urb_dequeue(urb);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_695: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_add_endpoint(ctx: *mut xhci_ep_ctx) {
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
let mut __UNIQUE_ID_guard_739: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_739) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_740: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_740) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_742() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_add_endpoint)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_741: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_741;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_ep_ctx) -> i32>>(__SCT__tp_func_xhci_add_endpoint as *const ()) };
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
static __trace_check_xhci_add_endpoint: __SlateAlign16<[u8; 18]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 97u8, 100u8, 100u8, 95u8, 101u8, 110u8, 100u8, 112u8, 111u8, 105u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_add_endpoint_743: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_add_endpoint) }) as *mut ();

fn trace_xhci_add_endpoint(ctx: *mut xhci_ep_ctx) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_add_endpoint.key) }) != 0 {
__do_trace_xhci_add_endpoint(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_744: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_alloc_dev(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_746: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_746) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_747: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_747) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_749() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_dev)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_748: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_748;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_alloc_dev as *const ()) };
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
static __trace_check_xhci_alloc_dev: [u8; 15] = [120, 104, 99, 105, 95, 97, 108, 108, 111, 99, 95, 100, 101, 118, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_alloc_dev_750: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_alloc_dev) }) as *mut ();

fn trace_xhci_alloc_dev(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__18!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_alloc_dev.key) }) != 0 {
__do_trace_xhci_alloc_dev(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_751: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_free_dev(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_753: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_753) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_754: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_754) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_756() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_free_dev)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_755: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_755;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_free_dev as *const ()) };
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
static __trace_check_xhci_free_dev: [u8; 14] = [120, 104, 99, 105, 95, 102, 114, 101, 101, 95, 100, 101, 118, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_free_dev_757: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_free_dev) }) as *mut ();

fn trace_xhci_free_dev(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__14!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_free_dev.key) }) != 0 {
__do_trace_xhci_free_dev(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_758: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_discover_or_reset_device(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_767: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_767) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_768: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_768) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_770() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_discover_or_reset_device)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_769: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_769;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_discover_or_reset_device as *const ()) };
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
static __trace_check_xhci_discover_or_reset_device: __SlateAlign16<[u8; 30]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 100u8, 105u8, 115u8, 99u8, 111u8, 118u8, 101u8, 114u8, 95u8, 111u8, 114u8, 95u8, 114u8, 101u8, 115u8, 101u8, 116u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_discover_or_reset_device_771: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_discover_or_reset_device) }) as *mut ();

fn trace_xhci_discover_or_reset_device(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__15!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_discover_or_reset_device.key) }) != 0 {
__do_trace_xhci_discover_or_reset_device(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_772: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_setup_device_slot(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_774: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_774) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_775: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_775) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_777() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_setup_device_slot)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_776: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_776;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_setup_device_slot as *const ()) };
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
static __trace_check_xhci_setup_device_slot: __SlateAlign16<[u8; 23]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 95u8, 100u8, 101u8, 118u8, 105u8, 99u8, 101u8, 95u8, 115u8, 108u8, 111u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_setup_device_slot_778: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_setup_device_slot) }) as *mut ();

fn trace_xhci_setup_device_slot(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_setup_device_slot.key) }) != 0 {
__do_trace_xhci_setup_device_slot(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_779: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_configure_endpoint(ctx: *mut xhci_slot_ctx) {
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
let mut __UNIQUE_ID_guard_802: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_802) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_803: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_803) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_805() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_configure_endpoint)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_804: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_804;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_slot_ctx) -> i32>>(__SCT__tp_func_xhci_configure_endpoint as *const ()) };
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
static __trace_check_xhci_configure_endpoint: __SlateAlign16<[u8; 24]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 99u8, 111u8, 110u8, 102u8, 105u8, 103u8, 117u8, 114u8, 101u8, 95u8, 101u8, 110u8, 100u8, 112u8, 111u8, 105u8, 110u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_configure_endpoint_806: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_configure_endpoint) }) as *mut ();

fn trace_xhci_configure_endpoint(ctx: *mut xhci_slot_ctx) {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_configure_endpoint.key) }) != 0 {
__do_trace_xhci_configure_endpoint(ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_807: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_address_ctrl_ctx(ctrl_ctx: *mut xhci_input_control_ctx) {
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
let mut __UNIQUE_ID_guard_809: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_809) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_810: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_810) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_812() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_address_ctrl_ctx)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_811: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_811;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_input_control_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_input_control_ctx) -> i32>>(__SCT__tp_func_xhci_address_ctrl_ctx as *const ()) };
unsafe { __v12.unwrap()(__data, ctrl_ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_address_ctrl_ctx: __SlateAlign16<[u8; 22]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 97u8, 100u8, 100u8, 114u8, 101u8, 115u8, 115u8, 95u8, 99u8, 116u8, 114u8, 108u8, 95u8, 99u8, 116u8, 120u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_address_ctrl_ctx_813: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_address_ctrl_ctx) }) as *mut ();

fn trace_xhci_address_ctrl_ctx(ctrl_ctx: *mut xhci_input_control_ctx) {
if static_branch_unlikely__17!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_address_ctrl_ctx.key) }) != 0 {
__do_trace_xhci_address_ctrl_ctx(ctrl_ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_814: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_configure_endpoint_ctrl_ctx(ctrl_ctx: *mut xhci_input_control_ctx) {
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
let mut __UNIQUE_ID_guard_816: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_816) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_817: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_817) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_819() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_configure_endpoint_ctrl_ctx)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_818: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_818;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_input_control_ctx) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_input_control_ctx) -> i32>>(__SCT__tp_func_xhci_configure_endpoint_ctrl_ctx as *const ()) };
unsafe { __v12.unwrap()(__data, ctrl_ctx) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_xhci_configure_endpoint_ctrl_ctx: __SlateAlign16<[u8; 33]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 99u8, 111u8, 110u8, 102u8, 105u8, 103u8, 117u8, 114u8, 101u8, 95u8, 101u8, 110u8, 100u8, 112u8, 111u8, 105u8, 110u8, 116u8, 95u8, 99u8, 116u8, 114u8, 108u8, 95u8, 99u8, 116u8, 120u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_configure_endpoint_ctrl_ctx_820: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_configure_endpoint_ctrl_ctx) }) as *mut ();

fn trace_xhci_configure_endpoint_ctrl_ctx(ctrl_ctx: *mut xhci_input_control_ctx) {
if static_branch_unlikely__16!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_configure_endpoint_ctrl_ctx.key) }) != 0 {
__do_trace_xhci_configure_endpoint_ctrl_ctx(ctrl_ctx);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_821: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_xhci_portsc_writel(port: *mut xhci_port, portsc: u32) {
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
let mut __UNIQUE_ID_guard_879: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_879) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_880: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_880) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_882() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_xhci_portsc_writel)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_881: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_881;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut xhci_port, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut xhci_port, u32) -> i32>>(__SCT__tp_func_xhci_portsc_writel as *const ()) };
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
static __trace_check_xhci_portsc_writel: __SlateAlign16<[u8; 19]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 112u8, 111u8, 114u8, 116u8, 115u8, 99u8, 95u8, 119u8, 114u8, 105u8, 116u8, 101u8, 108u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_xhci_portsc_writel_883: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_xhci_portsc_writel) }) as *mut ();

fn trace_xhci_portsc_writel(port: *mut xhci_port, portsc: u32) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_xhci_portsc_writel.key) }) != 0 {
__do_trace_xhci_portsc_writel(port, portsc);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_884: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Some 0.95 hardware can't handle the chain bit on a Link TRB being cleared
static mut link_quirk: i32 = unsafe { core::mem::zeroed() };

fn __check_link_quirk() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(link_quirk) }
}

static __param_str_link_quirk: __SlateAlign16<[u8; 20]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 108u8, 105u8, 110u8, 107u8, 95u8, 113u8, 117u8, 105u8, 114u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_link_quirk: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_link_quirk.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_xhci_c_38>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_38_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_38_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_38_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_38_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_38_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(link_quirk) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_927: [u8; 33] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 108, 105, 110, 107, 95, 113, 117, 105, 114, 107, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_928: [u8; 65] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 108, 105, 110, 107, 95, 113, 117, 105, 114, 107, 58, 68, 111, 110, 39, 116, 32, 99, 108, 101, 97, 114, 32, 116, 104, 101, 32, 99, 104, 97, 105, 110, 32, 98, 105, 116, 32, 111, 110, 32, 97, 32, 108, 105, 110, 107, 32, 84, 82, 66, 0];

static mut quirks: u64 = unsafe { core::mem::zeroed() };

fn __check_quirks() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(quirks) }
}

static __param_str_quirks: __SlateAlign16<[u8; 16]> = __SlateAlign16([120u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 113u8, 117u8, 105u8, 114u8, 107u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_quirks: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_quirks.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ullong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_xhci_c_42>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_42_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_42_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_42_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_42_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_xhci_c_42_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(quirks) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_929: [u8; 32] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 113, 117, 105, 114, 107, 115, 58, 117, 108, 108, 111, 110, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_930: [u8; 67] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 113, 117, 105, 114, 107, 115, 58, 66, 105, 116, 32, 102, 108, 97, 103, 115, 32, 102, 111, 114, 32, 113, 117, 105, 114, 107, 115, 32, 116, 111, 32, 98, 101, 32, 101, 110, 97, 98, 108, 101, 100, 32, 97, 115, 32, 100, 101, 102, 97, 117, 108, 116, 0];

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_portsc_writel")]
extern "C" fn xhci_portsc_writel(port: *mut xhci_port, val: u32) {
trace_xhci_portsc_writel(port, val);
writel(val, (unsafe { core::ptr::addr_of_mut!((*(*port).port_reg).portsc) }) as *mut ());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_portsc_writel_931: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_port, u32)>, *mut ()>(Some(xhci_portsc_writel)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_portsc_readl")]
extern "C" fn xhci_portsc_readl(port: *mut xhci_port) -> u32 {
readl((unsafe { core::ptr::addr_of_mut!((*(*port).port_reg).portsc) }) as *const ())
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_portsc_readl_932: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_port) -> u32>, *mut ()>(Some(xhci_portsc_readl)) };

fn td_on_ring(td: *mut xhci_td, ring: *mut xhci_ring) -> bool {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
if td.is_null() || unsafe { (*td).start_seg }.is_null() {
return false;
}
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
if seg == unsafe { (*td).start_seg } {
return true;
}
seg = if (unsafe { (*seg).next }) != unsafe { (*ring).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
false
}

/// xhci_handshake - spin reading hc until handshake completes or fails
/// @ptr: address of hc register to be read
/// @mask: bits to look at in result of read
/// @done: value of those bits when handshake succeeds
/// @usec: timeout in microseconds
///
/// Returns negative errno, or zero on success
///
/// Success happens when the "mask" bits have the specified value (hardware
/// handshake done).  There are two failure modes:  "usec" have passed (major
/// hardware flakeout), or the register reads as all-ones (hardware removed).
#[unsafe(no_mangle)]
extern "C" fn xhci_handshake(ptr: *mut (), mask: u32, done: u32, __timeout_us: u64) -> i32 {
let mut result: u32 = 0;
let mut ret: i32 = 0;
let mut __v0: i32;
let mut __left_ns: i64 = __timeout_us.wrapping_mul(1000) as i64;
let __delay_us: u64 = 1;
let __delay_ns: u64 = __delay_us.wrapping_mul(1000);
let mut ___ret: i32 = 0;
if false && __delay_us != 0 {
udelay(__delay_us);
if __timeout_us != 0 {
__left_ns = (__left_ns as u64).wrapping_sub(__delay_ns) as i64;
}
}
loop {
let __expired: bool = __timeout_us != 0 && __left_ns < 0;
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
result = readl(ptr as *const ());
if result & mask == done || result == !0 {
___ret = 0;
break;
}
if __expired {
___ret = 110i32.wrapping_neg();
break;
}
if __delay_us != 0 {
udelay(__delay_us);
if __timeout_us != 0 {
__left_ns = (__left_ns as u64).wrapping_sub(__delay_ns) as i64;
}
}
cpu_relax();
if __timeout_us != 0 {
__left_ns = __left_ns.wrapping_sub(1);
}
}
__v0 = ___ret;
ret = __v0;
if result == !0 {
return 19i32.wrapping_neg();
}
// card removed
ret
}

/// Disable interrupts and begin the xHCI halting process.
#[unsafe(no_mangle)]
extern "C" fn xhci_quiesce(xhci: *mut xhci_hcd) {
let mut cmd: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *const ());
let __v1: u32 = cmd & 4294966258u32;
cmd = __v1;
writel(cmd, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
}

/// Force HC into halt state.
///
/// Disable any IRQs and clear the run/stop bit.
/// HC will complete any current and actively pipelined transactions, and
/// should halt within 16 ms of the run/stop bit being cleared.
/// Read HC Halted bit in the status register to see when the HC is finished.
#[unsafe(no_mangle)]
extern "C" fn xhci_halt(xhci: *mut xhci_hcd) -> i32 {
let mut ret: i32 = 0;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"// Halt the HC".as_ptr() as *const u8) };
xhci_quiesce(xhci);
ret = xhci_handshake((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut (), 1u32, 1u32, 32000u64);
if ret != 0 {
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host halt failed, %d\n".as_ptr() as *const u8, ret) };
}
return ret;
}
let __v2: u32 = unsafe { (*xhci).xhc_state };
let __v3: u32 = ((__v2 as u64) | BIT!(1i32)) as u32;
unsafe {
(*xhci).xhc_state = __v3;
}
unsafe {
(*xhci).cmd_ring_state = BIT!(2i32) as u32;
}
ret
}

/// Set the run bit and wait for the host to be running.
#[unsafe(no_mangle)]
extern "C" fn xhci_start(xhci: *mut xhci_hcd) -> i32 {
let mut temp: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *const ());
let mut ret: i32 = 0;
temp |= 1u32;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"// Turn on HC, cmd = 0x%x.".as_ptr() as *const u8, temp) };
writel(temp, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
// Wait for the HCHalted Status bit to be 0 to indicate the host is
// running.
ret = xhci_handshake((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut (), 1u32, 0u32, 32000u64);
if ret == 110i32.wrapping_neg() {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host took too long to start, waited %u microseconds.\n".as_ptr() as *const u8, 32000i32) };
}
if ret == 0 {
// clear state flags. Including dying, halted or removing
unsafe {
(*xhci).xhc_state = 0;
}
unsafe {
(*xhci).run_graceperiod = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(500u32));
}
}
ret
}

/// Reset a halted HC.
///
/// This resets pipelines, timers, counters, state machines, etc.
/// Transactions will be terminated immediately, and operational registers
/// will be set to their defaults.
#[unsafe(no_mangle)]
extern "C" fn xhci_reset(xhci: *mut xhci_hcd, timeout_us: u64) -> i32 {
let mut command: u32 = 0;
let state: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *const ());
let mut ret: i32 = 0;
if state == !0 {
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host not accessible, reset failed.\n".as_ptr() as *const u8) };
}
return 19i32.wrapping_neg();
}
if state & 1u32 == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host controller not halted, aborting reset.\n".as_ptr() as *const u8) };
return 0;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"// Reset the HC".as_ptr() as *const u8) };
command = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *const ());
let __v2: u32 = command;
command = ((__v2 as u64) | BIT!(1i32)) as u32;
writel(command, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
// Existing Intel xHCI controllers require a delay of 1 mS,
// after setting the CMD_RESET bit, and before accessing any
// HC registers. This allows the HC to complete the
// reset operation and be ready for HC register access.
// Without this delay, the subsequent HC register access,
// may result in a system hang very rarely.
if (unsafe { (*xhci).quirks }) & BIT_ULL!(12i32) != 0 {
udelay(1000u64);
}
ret = xhci_handshake((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut (), BIT!(1i32) as u32, 0u32, timeout_us);
if ret != 0 {
return ret;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(28i32) != 0 {
unsafe { usb_asmedia_modifyflowcontrol(to_pci_dev!(unsafe { (*xhci_to_hcd(xhci)).self_.controller })) };
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Wait for controller to be ready for doorbell rings".as_ptr() as *const u8) };
// xHCI cannot write to any doorbells or operational registers other
// than status until the "Controller Not Ready" flag is cleared.
ret = xhci_handshake((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut (), 2048u32, 0u32, timeout_us);
unsafe {
(*xhci).usb2_rhub.bus_state.port_c_suspend = 0;
}
unsafe {
(*xhci).usb2_rhub.bus_state.suspended_ports = 0;
}
unsafe {
(*xhci).usb2_rhub.bus_state.resuming_ports = 0;
}
unsafe {
(*xhci).usb3_rhub.bus_state.port_c_suspend = 0;
}
unsafe {
(*xhci).usb3_rhub.bus_state.suspended_ports = 0;
}
unsafe {
(*xhci).usb3_rhub.bus_state.resuming_ports = 0;
}
ret
}

fn xhci_zero_64b_regs(xhci: *mut xhci_hcd) {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.sysdev };
let mut err: i32 = 0;
let mut i: i32 = 0;
let mut val: u64 = 0;
// Some Renesas controllers get into a weird state if they are
// reset while programmed with 64bit addresses (they will preserve
// the top half of the address in internal, non visible
// registers). You end up with half the address coming from the
// kernel, and the other half coming from the firmware. Also,
// changing the programming leads to extra accesses even if the
// controller is supposed to be halted. The controller ends up with
// a fatal fault, and is then ripe for being properly reset.
//
// Special care is taken to only apply this if the device is behind
// an iommu. Doing anything when there is no iommu is definitely
// unsafe...
let domain: *mut iommu_domain = unsafe { iommu_get_domain_for_dev(dev) };
if (unsafe { (*xhci).quirks }) & BIT_ULL!(32i32) == 0 || domain.is_null() || (unsafe { (*domain).r#type }) == 4u32 {
return;
}
while false {
}
unsafe { _dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Zeroing 64bit base registers, expecting fault\n".as_ptr() as *const u8) };
// Clear HSEIE so that faults do not get signaled
val = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *const ()) as u64;
let __v2: u64 = val & 18446744073709551607u64;
val = __v2;
writel(val as u32, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
// Clear HSE (aka FATAL)
val = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *const ()) as u64;
let __v3: u64 = val;
val = __v3 | BIT!(2i32);
writel(val as u32, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut ());
// Now zero the registers, and brace for impact
val = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr) });
if upper_32_bits!(val) != 0 {
xhci_write_64(xhci, 0u64, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr) });
}
val = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) });
if upper_32_bits!(val) != 0 {
xhci_write_64(xhci, 0u64, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) });
}
while i < ((unsafe { (*xhci).max_interrupters }) as i32) {
let mut ir: *mut xhci_intr_reg = unsafe { core::mem::zeroed() };
ir = unsafe { (*(*xhci).run_regs).ir_set.as_mut_ptr() as *mut xhci_intr_reg }.wrapping_offset(i as isize);
val = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*ir).erst_base) });
if upper_32_bits!(val) != 0 {
xhci_write_64(xhci, 0u64, unsafe { core::ptr::addr_of_mut!((*ir).erst_base) });
}
val = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*ir).erst_dequeue) });
if upper_32_bits!(val) != 0 {
xhci_write_64(xhci, 0u64, unsafe { core::ptr::addr_of_mut!((*ir).erst_dequeue) });
}
i = i.wrapping_add(1);
}
// Wait for the fault to appear. It will be cleared on reset
err = xhci_handshake((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut (), BIT!(2i32) as u32, BIT!(2i32) as u32, 32000u64);
if err == 0 {
while false {
}
unsafe { _dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Fault detected\n".as_ptr() as *const u8) };
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_enable_interrupter(ir: *mut xhci_interrupter) -> i32 {
let mut iman: u32 = 0;
if ir.is_null() || unsafe { (*ir).ir_set }.is_null() {
return 22i32.wrapping_neg();
}
iman = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
let __v0: u32 = iman;
iman = ((__v0 as u64) & !BIT!(0i32)) as u32;
let __v2: u32 = iman;
iman = ((__v2 as u64) | BIT!(1i32)) as u32;
writel(iman, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *mut ());
// Read operation to guarantee the write has been flushed from posted buffers
readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_disable_interrupter(xhci: *mut xhci_hcd, ir: *mut xhci_interrupter) -> i32 {
let mut iman: u32 = 0;
if ir.is_null() || unsafe { (*ir).ir_set }.is_null() {
return 22i32.wrapping_neg();
}
iman = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
let __v0: u32 = iman;
iman = ((__v0 as u64) & !BIT!(0i32)) as u32;
let __v2: u32 = iman;
iman = ((__v2 as u64) & !BIT!(1i32)) as u32;
writel(iman, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *mut ());
iman = readl((unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *const ());
if (iman as u64) & BIT!(0i32) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Interrupt pending\n".as_ptr() as *const u8, c"xhci_disable_interrupter".as_ptr() as *mut u8) };
}
}
0
}

// interrupt moderation interval imod_interval in nanoseconds
#[unsafe(link_section = ".text.slate_distinct.xhci.compliance_mode_recovery")]
extern "C" fn compliance_mode_recovery(t: *mut timer_list) {
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
let mut temp: u32 = 0;
let mut i: i32 = 0;
xhci = timer_container_of__2!(t);
let rhub: *mut xhci_hub = unsafe { core::ptr::addr_of_mut!((*xhci).usb3_rhub) };
let hcd: *mut usb_hcd = unsafe { (*rhub).hcd };
if hcd.is_null() {
return;
}
while (i as u32) < unsafe { (*rhub).num_ports } {
temp = xhci_portsc_readl(unsafe { *unsafe { (*rhub).ports }.wrapping_offset(i as isize) });
if temp & 480u32 == 320 {
// Compliance Mode Detected. Letting USB Core
// handle the Warm Reset
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Compliance mode detected->port %d".as_ptr() as *const u8, i.wrapping_add(1)) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Attempting compliance mode recovery".as_ptr() as *const u8) };
if (unsafe { (*hcd).state }) == 4 {
unsafe { usb_hcd_resume_root_hub(hcd) };
}
unsafe { usb_hcd_poll_rh_status(hcd) };
}
i = i.wrapping_add(1);
}
if (unsafe { (*xhci).port_status_u0 }) != ((1i32 << unsafe { (*rhub).num_ports }).wrapping_sub(1) as u32) {
unsafe { mod_timer(core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(2000u32))) };
}
}

/// Quirk to work around issue generated by the SN65LVPE502CP USB3.0 re-driver
/// that causes ports behind that hardware to enter compliance mode sometimes.
/// The quirk creates a timer that polls every 2 seconds the link state of
/// each host controller's port and recovers it by issuing a Warm reset
/// if Compliance mode is detected, otherwise the port will become "dead" (no
/// device connections or disconnections will be detected anymore). Becasue no
/// status event is generated when entering compliance mode (per xhci spec),
/// this quirk is needed on systems that have the failing hardware installed.
fn compliance_mode_recovery_timer_init(xhci: *mut xhci_hcd) {
unsafe {
(*xhci).port_status_u0 = 0;
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer) }, Some(compliance_mode_recovery), 0i32);
unsafe {
(*xhci).comp_mode_recovery_timer.expires = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(2000u32));
}
unsafe { add_timer(core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer)) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Compliance mode recovery timer initialized".as_ptr() as *const u8) };
}

/// This function identifies the systems that have installed the SN65LVPE502CP
/// USB3.0 re-driver and that need the Compliance Mode Quirk.
/// Systems:
/// Vendor: Hewlett-Packard -> System Models: Z420, Z620 and Z820
fn xhci_compliance_mode_recovery_timer_quirk_check() -> bool {
let dmi_product_name: *const u8 = unsafe { dmi_get_system_info(7i32) };
let dmi_sys_vendor: *const u8 = unsafe { dmi_get_system_info(6i32) };
if dmi_product_name.is_null() || dmi_sys_vendor.is_null() {
return false;
}
if unsafe { strstr(dmi_sys_vendor, c"Hewlett-Packard".as_ptr() as *const u8) }.is_null() {
return false;
}
let __v0: bool = !unsafe { strstr(dmi_product_name, c"Z420".as_ptr() as *const u8) }.is_null() || !unsafe { strstr(dmi_product_name, c"Z620".as_ptr() as *const u8) }.is_null();
let __v1: bool = __v0 || !unsafe { strstr(dmi_product_name, c"Z820".as_ptr() as *const u8) }.is_null();
let __v2: bool = __v1 || !unsafe { strstr(dmi_product_name, c"Z1 Workstation".as_ptr() as *const u8) }.is_null();
if __v2 {
return true;
}
false
}

fn xhci_all_ports_seen_u0(xhci: *mut xhci_hcd) -> i32 {
((unsafe { (*xhci).port_status_u0 }) == ((1i32 << unsafe { (*xhci).usb3_rhub.num_ports }).wrapping_sub(1) as u32)) as i32
}

fn xhci_hcd_page_size(xhci: *mut xhci_hcd) {
let mut page_size: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).page_size) }) as *const ()) & 65535;
if !is_power_of_2(page_size as u64) {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Invalid page size register = 0x%x\n".as_ptr() as *const u8, page_size) };
// Fallback to 4K page size, since that's common
page_size = 1;
}
unsafe {
(*xhci).page_size = page_size << 12i32;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"HCD page size set to %iK".as_ptr() as *const u8, (*xhci).page_size >> 10i32) };
}

fn xhci_enable_max_dev_slots(xhci: *mut xhci_hcd) {
let mut config_reg: u32 = 0;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"xHC can handle at most %d device slots".as_ptr() as *const u8, (*xhci).max_slots as i32) };
config_reg = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).config_reg) }) as *const ());
let __v0: u32 = config_reg;
config_reg = ((__v0 as u64) & !GENMASK__6!(7i32, 0i32)) as u32;
config_reg |= ((unsafe { (*xhci).max_slots }) as i32) as u32;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Setting Max device slots reg = 0x%x".as_ptr() as *const u8, config_reg) };
writel(config_reg, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).config_reg) }) as *mut ());
}

fn xhci_set_cmd_ring_deq(xhci: *mut xhci_hcd) {
let mut deq_dma: u64 = unsafe { xhci_trb_virt_to_dma((*(*xhci).cmd_ring).deq_seg, (*(*xhci).cmd_ring).dequeue) };
let mut crcr: u64 = 0;
let __v0: u64 = deq_dma;
deq_dma = __v0 & GENMASK_ULL__7!(63i32, 6i32);
crcr = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) });
let __v2: u64 = crcr;
crcr = __v2 & !GENMASK_ULL__2!(63i32, 6i32);
crcr |= deq_dma;
let __v6: u64 = crcr;
crcr = __v6 & !BIT_ULL!(0i32);
crcr |= (unsafe { (*(*xhci).cmd_ring).cycle_state }) as u64;
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Setting command ring address to 0x%llx".as_ptr() as *const u8, crcr) };
xhci_write_64(xhci, crcr, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).cmd_ring) });
}

fn xhci_set_doorbell_ptr(xhci: *mut xhci_hcd) {
let offset: u32 = ((readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).db_off) }) as *const ()) as u64) & GENMASK__3!(31i32, 2i32)) as u32;
unsafe {
(*xhci).dba = ((((*xhci).cap_regs as *mut ()) as *const u8).wrapping_offset(offset as isize) as *mut ()) as *mut xhci_doorbell_array;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Doorbell array is located at offset 0x%x from cap regs base addr".as_ptr() as *const u8, offset) };
}

/// Enable USB 3.0 device notifications for function remote wake, which is necessary
/// for allowing USB 3.0 devices to do remote wakeup from U3 (device suspend).
fn xhci_set_dev_notifications(xhci: *mut xhci_hcd) {
let mut dev_notf: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dev_notification) }) as *const ());
let __v1: u32 = dev_notf & 4294901760u32;
dev_notf = __v1;
let __v2: u32 = dev_notf;
dev_notf = ((__v2 as u64) | BIT!(1i32)) as u32;
writel(dev_notf, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dev_notification) }) as *mut ());
}

/// Setup basic xHCI registers
fn xhci_init(hcd: *mut usb_hcd) {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Starting %s".as_ptr() as *const u8, c"xhci_init".as_ptr() as *mut u8) };
// Set the Number of Device Slots Enabled to the maximum supported value
xhci_enable_max_dev_slots(xhci);
// Initialize the Command ring
unsafe { xhci_ring_init(xhci, (*xhci).cmd_ring) };
// Reserve one command ring TRB for disabling LPM.
// Since the USB core grabs the shared usb_bus bandwidth mutex before
// disabling LPM, we only need to reserve one TRB for all devices.
unsafe {
(*xhci).cmd_ring_reserved_trbs = 1;
}
// Set the address in the Command Ring Control register
xhci_set_cmd_ring_deq(xhci);
// Set Device Context Base Address Array pointer
xhci_write_64(xhci, unsafe { (*xhci).dcbaa.dma }, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr) });
// Set Doorbell array pointer
xhci_set_doorbell_ptr(xhci);
// Set USB 3.0 device notifications for function remote wake
xhci_set_dev_notifications(xhci);
// Initialize the Primary interrupter
unsafe { xhci_ring_init(xhci, (*(*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize))).event_ring) };
unsafe { xhci_add_interrupter(xhci, 0u32) };
unsafe {
(*(*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize))).isoc_bei_interval = 32;
}
// Initializing Compliance Mode Recovery Data If Needed
if xhci_compliance_mode_recovery_timer_quirk_check() {
let __v1: u64 = unsafe { (*xhci).quirks };
let __v2: u64 = __v1 | BIT_ULL!(14i32);
unsafe {
(*xhci).quirks = __v2;
}
compliance_mode_recovery_timer_init(xhci);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Finished %s".as_ptr() as *const u8, c"xhci_init".as_ptr() as *mut u8) };
}

fn xhci_run_finished(xhci: *mut xhci_hcd) -> i32 {
let ir: *mut xhci_interrupter = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize) };
let mut flags: u64 = 0;
let mut temp: u32 = 0;
// Enable interrupts before starting the host (xhci 4.2 and 5.5.2).
// Protect the short window before host is running with a lock
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Enable interrupts".as_ptr() as *const u8) };
temp = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *const ());
temp |= 4u32;
writel(temp, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Enable primary interrupter".as_ptr() as *const u8) };
xhci_enable_interrupter(ir);
if xhci_start(xhci) != 0 {
xhci_halt(xhci);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
return 19i32.wrapping_neg();
}
unsafe {
(*xhci).cmd_ring_state = BIT!(0i32) as u32;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(2i32) != 0 {
unsafe { xhci_ring_cmd_db(xhci) };
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
0
}

/// Start the HC after it was halted.
///
/// This function is called by the USB core when the HC driver is added.
/// Its opposite is xhci_stop().
///
/// xhci_init() must be called once before this function can be called.
/// Reset the HC, enable device slot contexts, program DCBAAP, and
/// set command ring pointer and event ring pointer.
///
/// Setup MSI-X vectors and enable interrupts.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_run")]
extern "C" fn xhci_run(hcd: *mut usb_hcd) -> i32 {
let mut temp_64: u64 = 0;
let mut ret: i32 = 0;
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let ir: *mut xhci_interrupter = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize) };
// Start the xHCI host controller running only after the USB 2.0 roothub
// is setup.
unsafe {
(*hcd).__slate_bits_0.__set_uses_new_polling(1u32);
}
if ((unsafe { (*hcd).__slate_bits_0.__get_msi_enabled() }) as i32) != 0 {
unsafe {
(*ir).ip_autoclear = true;
}
}
if (unsafe { usb_hcd_is_primary_hcd(hcd) }) == 0 {
return xhci_run_finished(xhci);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"xhci_run".as_ptr() as *const u8) };
temp_64 = xhci_read_64(xhci as *const xhci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
let __v0: u64 = temp_64;
temp_64 = __v0 & GENMASK_ULL__5!(63i32, 4i32);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"ERST deq = 64'h%0lx".as_ptr() as *const u8, temp_64) };
xhci_set_interrupter_moderation(ir, unsafe { (*xhci).imod_interval });
if (unsafe { (*xhci).quirks }) & BIT_ULL!(2i32) != 0 {
let command: *mut xhci_command = unsafe { xhci_alloc_command(xhci, false, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if command.is_null() {
return 12i32.wrapping_neg();
}
ret = unsafe { xhci_queue_vendor_command(xhci, command, 0u32, 0u32, 0u32, TRB_TYPE!(49i32) as u32) };
if ret != 0 {
unsafe { xhci_free_command(xhci, command) };
}
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"Finished %s for main hcd".as_ptr() as *const u8, c"xhci_run".as_ptr() as *mut u8) };
xhci_create_dbc_dev(xhci);
unsafe { xhci_debugfs_init(xhci) };
if xhci_has_one_roothub(xhci) {
return xhci_run_finished(xhci);
}
set_bit(8i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_run_936: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>, *mut ()>(Some(xhci_run)) };

/// Stop xHCI driver.
///
/// This function is called by the USB core when the HC driver is removed.
/// Its opposite is xhci_run().
///
/// Disable device contexts, disable IRQs, and quiesce the HC.
/// Reset the HC, finish any completed transactions, and cleanup memory.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_stop")]
extern "C" fn xhci_stop(hcd: *mut usb_hcd) {
let mut temp: u32 = 0;
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let ir: *mut xhci_interrupter = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*xhci).mutex)) };
// Only halt host and free memory after both hcds are removed
if (unsafe { usb_hcd_is_primary_hcd(hcd) }) == 0 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*xhci).mutex)) };
return;
}
xhci_remove_dbc_dev(xhci);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
let __v1: u32 = unsafe { (*xhci).xhc_state };
let __v2: u32 = ((__v1 as u64) | BIT!(1i32)) as u32;
unsafe {
(*xhci).xhc_state = __v2;
}
unsafe {
(*xhci).cmd_ring_state = BIT!(2i32) as u32;
}
xhci_halt(xhci);
xhci_reset(xhci, 250000u64);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
// Deleting Compliance Mode Recovery Timer
let __v3: bool = (unsafe { (*xhci).quirks }) & BIT_ULL!(14i32) != 0 && xhci_all_ports_seen_u0(xhci) == 0;
if __v3 {
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer)) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"%s: compliance mode recovery timer deleted".as_ptr() as *const u8, c"xhci_stop".as_ptr() as *mut u8) };
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(3i32) != 0 {
unsafe { usb_amd_dev_put() };
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"// Disabling event ring interrupts".as_ptr() as *const u8) };
temp = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *const ());
writel((((temp & 4294959104u32) as u64) | BIT!(3i32)) as u32, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *mut ());
xhci_disable_interrupter(xhci, ir);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"cleaning up memory".as_ptr() as *const u8) };
unsafe { xhci_mem_cleanup(xhci) };
unsafe { xhci_debugfs_exit(xhci) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"xhci_stop completed - status = %x".as_ptr() as *const u8, readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *const ())) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*xhci).mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_stop_937: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd)>, *mut ()>(Some(xhci_stop)) };

/// Shutdown HC (not bus-specific)
///
/// This is called when the machine is rebooting or halting.  We assume that the
/// machine will be powered off, and the HC's internal state will be reset.
/// Don't bother to free memory.
///
/// This will only ever be called with the main usb_hcd (the USB3 roothub).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_shutdown")]
extern "C" fn xhci_shutdown(hcd: *mut usb_hcd) {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if (unsafe { (*xhci).quirks }) & BIT_ULL!(13i32) != 0 {
unsafe { usb_disable_xhci_ports(to_pci_dev!(unsafe { (*hcd).self_.sysdev })) };
}
// Don't poll the roothubs after shutdown.
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: stopping usb%d port polling.\n".as_ptr() as *const u8, c"xhci_shutdown".as_ptr() as *mut u8, (*hcd).self_.busnum) };
}
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*hcd).rh_timer)) };
if !unsafe { (*xhci).shared_hcd }.is_null() {
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*(*xhci).shared_hcd).flags) }) as *mut u64);
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*(*xhci).shared_hcd).rh_timer)) };
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
xhci_halt(xhci);
// Workaround for spurious wakeps at shutdown with HSW, and for boot
// firmware delay in ADL-P PCH if port are left in U3 at shutdown
if (unsafe { (*xhci).quirks }) & BIT_ULL!(18i32) != 0 || (unsafe { (*xhci).quirks }) & BIT_ULL!(44i32) != 0 {
xhci_reset(xhci, 250000u64);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*xhci).lock) });
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_init), c"xhci_shutdown completed - status = %x".as_ptr() as *const u8, readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *const ())) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_shutdown_938: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd)>, *mut ()>(Some(xhci_shutdown)) };

fn xhci_save_registers(xhci: *mut xhci_hcd) {
let mut ir: *mut xhci_interrupter = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
unsafe {
(*xhci).s3.command = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
}
unsafe {
(*xhci).s3.dev_nt = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).dev_notification) as *const ());
}
unsafe {
(*xhci).s3.dcbaa_ptr = xhci_read_64(xhci as *const xhci_hcd, core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr));
}
unsafe {
(*xhci).s3.config_reg = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).config_reg) as *const ());
}
// save both primary and all secondary interrupters
//
// fixme, shold we lock  to prevent race with remove secondary interrupter?
while i < (((unsafe { (*xhci).max_interrupters }) as i32) as u32) {
ir = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) };
if !ir.is_null() {
unsafe {
(*ir).s3_erst_size = readl(core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) as *const ());
}
unsafe {
(*ir).s3_erst_base = xhci_read_64(xhci as *const xhci_hcd, core::ptr::addr_of_mut!((*(*ir).ir_set).erst_base));
}
unsafe {
(*ir).s3_erst_dequeue = xhci_read_64(xhci as *const xhci_hcd, core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue));
}
unsafe {
(*ir).s3_iman = readl(core::ptr::addr_of_mut!((*(*ir).ir_set).iman) as *const ());
}
unsafe {
(*ir).s3_imod = readl(core::ptr::addr_of_mut!((*(*ir).ir_set).imod) as *const ());
}
}
i = i.wrapping_add(1);
}
}

fn xhci_restore_registers(xhci: *mut xhci_hcd) {
let mut ir: *mut xhci_interrupter = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
writel(unsafe { (*xhci).s3.command }, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).command) }) as *mut ());
writel(unsafe { (*xhci).s3.dev_nt }, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dev_notification) }) as *mut ());
xhci_write_64(xhci, unsafe { (*xhci).s3.dcbaa_ptr }, unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr) });
writel(unsafe { (*xhci).s3.config_reg }, (unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).config_reg) }) as *mut ());
// FIXME should we lock to protect against freeing of interrupters
while i < (((unsafe { (*xhci).max_interrupters }) as i32) as u32) {
ir = unsafe { *unsafe { (*xhci).interrupters }.wrapping_offset(i as isize) };
if !ir.is_null() {
writel(unsafe { (*ir).s3_erst_size }, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_size) }) as *mut ());
xhci_write_64(xhci, unsafe { (*ir).s3_erst_base }, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_base) });
xhci_write_64(xhci, unsafe { (*ir).s3_erst_dequeue }, unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).erst_dequeue) });
writel(unsafe { (*ir).s3_iman }, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).iman) }) as *mut ());
writel(unsafe { (*ir).s3_imod }, (unsafe { core::ptr::addr_of_mut!((*(*ir).ir_set).imod) }) as *mut ());
}
i = i.wrapping_add(1);
}
}

/// The whole command ring must be cleared to zero when we suspend the host.
///
/// The host doesn't save the command ring pointer in the suspend well, so we
/// need to re-program it on resume.  Unfortunately, the pointer must be 64-byte
/// aligned, because of the reserved bits in the command ring dequeue pointer
/// register.  Therefore, we can't just set the dequeue pointer back in the
/// middle of the ring (TRBs are 16-byte aligned).
fn xhci_clear_command_ring(xhci: *mut xhci_hcd) {
let mut seg: *mut xhci_segment = unsafe { core::mem::zeroed() };
let ring: *mut xhci_ring = unsafe { (*xhci).cmd_ring };
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
// erase all TRBs before the link
unsafe { memset((*seg).trbs as *mut (), 0i32, (core::mem::size_of::<xhci_trb>() as u64).wrapping_mul(255u64)) };
// clear link cycle bit
let __v0: *mut xhci_trb = unsafe { (*seg).trbs }.wrapping_offset(255i32 as isize);
let __v1: u32 = unsafe { (*__v0).link.control };
let __v2: u32 = __v1 & (!BIT!(0i32) as u32);
unsafe {
(*__v0).link.control = __v2;
}
seg = if (unsafe { (*seg).next }) != unsafe { (*ring).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
unsafe { xhci_initialize_ring_info(ring) };
// Reset the hardware dequeue pointer.
// Yes, this will need to be re-written after resume, but we're paranoid
// and want to make sure the hardware doesn't access bogus memory
// because, say, the BIOS or an SMI started the host without changing
// the command ring pointers.
xhci_set_cmd_ring_deq(xhci);
}

// Disable port wake bits if do_wakeup is not set.
//
// Also clear a possible internal port wake state left hanging for ports that
// detected termination but never successfully enumerated (trained to 0U).
// Internal wake causes immediate xHCI wake after suspend. PORT_CSC write done
// at enumeration clears this wake, force one here as well for unconnected ports
fn xhci_disable_hub_port_wake(xhci: *mut xhci_hcd, rhub: *mut xhci_hub, do_wakeup: bool) {
let mut flags: u64 = 0;
let mut t1: u32 = 0;
let mut t2: u32 = 0;
let mut portsc: u32 = 0;
let mut i: i32 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
while (i as u32) < unsafe { (*rhub).num_ports } {
portsc = xhci_portsc_readl(unsafe { *unsafe { (*rhub).ports }.wrapping_offset(i as isize) });
t1 = unsafe { xhci_port_state_to_neutral(portsc) };
t2 = t1;
// clear wake bits if do_wake is not set
if !do_wakeup {
let __v6: u32 = t2 & 4060086271u32;
t2 = __v6;
}
// Don't touch csc bit if connected or connect change is set
if portsc & 131073u32 == 0 {
t2 |= 131072u32;
}
if t1 != t2 {
xhci_portsc_writel(unsafe { *unsafe { (*rhub).ports }.wrapping_offset(i as isize) }, t2);
}
i = i.wrapping_add(1);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
}

fn xhci_pending_portevent(xhci: *mut xhci_hcd) -> bool {
let mut ports: *mut *mut xhci_port = unsafe { core::mem::zeroed() };
let mut port_index: i32 = 0;
let status: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *const ());
let mut portsc: u32 = 0;
if (status as u64) & BIT!(3i32) != 0 {
return true;
}
// Checking STS_EINT is not enough as there is a lag between a change
// bit being set and the Port Status Change Event that it generated
// being written to the Event Ring. See note in xhci 1.1 section 4.19.2.
port_index = (unsafe { (*xhci).usb2_rhub.num_ports }) as i32;
ports = unsafe { (*xhci).usb2_rhub.ports };
loop {
let __v0: i32 = port_index;
port_index = __v0.wrapping_sub(1);
if __v0 == 0 {
break;
}
portsc = xhci_portsc_readl(unsafe { *ports.wrapping_offset(port_index as isize) });
if portsc & 16646144u32 != 0 || portsc & 480u32 == 480u32 {
return true;
}
}
port_index = (unsafe { (*xhci).usb3_rhub.num_ports }) as i32;
ports = unsafe { (*xhci).usb3_rhub.ports };
loop {
let __v3: i32 = port_index;
port_index = __v3.wrapping_sub(1);
if __v3 == 0 {
break;
}
portsc = xhci_portsc_readl(unsafe { *ports.wrapping_offset(port_index as isize) });
if portsc & 33423360u32 != 0 || portsc & 480u32 == 480u32 {
return true;
}
}
false
}

/// Stop HC (not bus-specific)
///
/// This is called when the machine transition into S3/S4 mode.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_suspend")]
extern "C" fn xhci_suspend(xhci: *mut xhci_hcd, do_wakeup: bool) -> i32 {
// Deleting Compliance Mode Recovery Timer because the xHCI Host
// is about to be suspended.
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut usbsts: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(usbsts) as *mut u32;
let mut command: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(command) as *mut u32;
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut delay: u32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
delay = 64000u32;
hcd = xhci_to_hcd(xhci);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*hcd).state == 0 {
return 0;
} else {
if (*hcd).state != 4 || !unsafe { (*xhci).shared_hcd }.is_null() && (*(*xhci).shared_hcd).state != 4 {
return 22i32.wrapping_neg();
} else {
// Clear root port wake on bits if wakeup not allowed.
xhci_disable_hub_port_wake(xhci, core::ptr::addr_of_mut!((*xhci).usb3_rhub), do_wakeup);
xhci_disable_hub_port_wake(xhci, core::ptr::addr_of_mut!((*xhci).usb2_rhub), do_wakeup);
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
return 0;
} else {
xhci_dbc_suspend(xhci);
// Don't poll the roothubs on bus suspend.
clear_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
timer_delete_sync(core::ptr::addr_of_mut!((*hcd).rh_timer));
if !unsafe { (*xhci).shared_hcd }.is_null() {
clear_bit(2i64, core::ptr::addr_of_mut!((*(*xhci).shared_hcd).flags) as *mut u64);
timer_delete_sync(core::ptr::addr_of_mut!((*(*xhci).shared_hcd).rh_timer));
}
if (*xhci).quirks & BIT_ULL!(30i32) != 0 {
usleep_range(1000u64, 1500u64);
}
spin_lock_irq(core::ptr::addr_of_mut!((*xhci).lock));
clear_bit(0i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
if !unsafe { (*xhci).shared_hcd }.is_null() {
clear_bit(0i64, core::ptr::addr_of_mut!((*(*xhci).shared_hcd).flags) as *mut u64);
}
// step 1: stop endpoint
//
// skipped assuming that port suspend has done
// Check if command ring is empty
if list_empty(core::ptr::addr_of_mut!((*xhci).cmd_list) as *const list_head) == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Suspending and stopping xHC with pending command!\n".as_ptr() as *const u8);
}
'__join_0: {
// step 2: clear Run/Stop bit
*__slate_slot_8 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
let __v5: u32 = *__slate_slot_8 & 4294967294u32;
*__slate_slot_8 = __v5;
writel(*__slate_slot_8, core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *mut ());
// Some chips from Fresco Logic need an extraordinary delay
let __v4: u32 = delay;
delay = __v4.wrapping_mul((if (*xhci).quirks & BIT_ULL!(17i32) != 0 { 10i32 } else { 1i32 }) as u32);
*__slate_slot_11 = xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut (), 1u32, 1u32, delay as u64);
if *__slate_slot_11 != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Clearing Run/Stop bit failed %d\n".as_ptr() as *const u8, *__slate_slot_11);
} else {
xhci_clear_command_ring(xhci);
// step 3: save registers
xhci_save_registers(xhci);
// step 4: set CSS flag
*__slate_slot_8 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
let __v2: u32 = *__slate_slot_8;
let __v1: u32 = ((__v2 as u64) | BIT!(8i32)) as u32;
*__slate_slot_8 = __v1;
writel(*__slate_slot_8, core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *mut ());
*__slate_slot_11 = xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut (), BIT!(8i32) as u32, 0u32, 20000u64);
*__slate_slot_7 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *const ());
(*xhci).__slate_bits_0.__set_broken_suspend(0u32);
if *__slate_slot_11 != 0 {
// AMD SNPS xHC 3.0 occasionally does not clear the
// SSS bit of USBSTS and when driver tries to poll
// to see if the xHC clears BIT(8) which never happens
// and driver assumes that controller is not responding
// and times out. To workaround this, its good to check
// if SRE and HCE bits are not set (as per xhci
// Section 5.4.2) and bypass the timeout.
if (*xhci).quirks & BIT_ULL!(35i32) == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Controller Save State failed %d\n".as_ptr() as *const u8, *__slate_slot_11);
break '__join_0;
} else {
if (*__slate_slot_7 as u64) & (BIT!(10i32) | BIT!(12i32)) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Controller Save State failed, USBSTS 0x%08x\n".as_ptr() as *const u8, *__slate_slot_7);
break '__join_0;
} else {
(*xhci).__slate_bits_0.__set_broken_suspend(1u32);
}
}
} else {
if (*__slate_slot_7 as u64) & BIT!(10i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Suspend Save Error (SRE), USBSTS 0x%08x\n".as_ptr() as *const u8, *__slate_slot_7);
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
if (*xhci).quirks & BIT_ULL!(14i32) != 0 {
__v0 = xhci_all_ports_seen_u0(xhci) == 0;
} else {
__v0 = false;
}
if __v0 {
timer_delete_sync(core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer));
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"%s: compliance mode recovery timer deleted".as_ptr() as *const u8, c"xhci_suspend".as_ptr() as *mut u8);
}
return 0;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
return 110i32.wrapping_neg();
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_suspend_939: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_hcd, bool) -> i32>, *mut ()>(Some(xhci_suspend)) };

/// start xHC (not bus-specific)
///
/// This is called when the machine transition from S3/S4 mode.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_resume")]
extern "C" fn xhci_resume(xhci: *mut xhci_hcd, power_lost: bool, is_auto_resume: bool) -> i32 {
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut reset_registers: bool = unsafe { core::mem::zeroed() };
let mut suspended_usb3_devs: bool = unsafe { core::mem::zeroed() };
let mut pending_portevent: bool = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut seg: core::mem::MaybeUninit<*mut xhci_segment> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut xhci_segment = core::ptr::addr_of_mut!(seg) as *mut *mut xhci_segment;
let mut hcd: *mut usb_hcd = unsafe { core::mem::zeroed() };
let mut temp: u32 = unsafe { core::mem::zeroed() };
let mut command: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(command) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
temp = 0;
hcd = xhci_to_hcd(xhci);
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
retval = 0;
pending_portevent = false;
suspended_usb3_devs = false;
reset_registers = false;
if (*hcd).state == 0 {
return 0;
} else {
// Wait a bit if either of the roothubs need to settle from the
// transition into bus suspend.
if time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*xhci).usb2_rhub.bus_state.next_statechange }) {
__v8 = true;
} else {
__v8 = time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*xhci).usb3_rhub.bus_state.next_statechange });
}
if __v8 {
msleep(100u32);
}
set_bit(0i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
if !unsafe { (*xhci).shared_hcd }.is_null() {
set_bit(0i64, core::ptr::addr_of_mut!((*(*xhci).shared_hcd).flags) as *mut u64);
}
spin_lock_irq(core::ptr::addr_of_mut!((*xhci).lock));
if power_lost || ((*xhci).__slate_bits_0.__get_broken_suspend() as i32) != 0 || (*xhci).quirks & BIT_ULL!(7i32) != 0 {
reset_registers = true;
} else {
// Some controllers might lose power during suspend, so wait
// for controller not ready bit to clear, just as in xHC init.
retval = xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut (), 2048u32, 0u32, 10000000u64);
if retval != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Controller not ready at resume %d\n".as_ptr() as *const u8, retval);
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
return retval;
} else {
// step 1: restore register
xhci_restore_registers(xhci);
// step 2: initialize command ring buffer
xhci_set_cmd_ring_deq(xhci);
// step 3: restore state and start state
//
// step 3: set CRS flag
*__slate_slot_16 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
let __v7: u32 = *__slate_slot_16;
let __v6: u32 = ((__v7 as u64) | BIT!(9i32)) as u32;
*__slate_slot_16 = __v6;
writel(*__slate_slot_16, core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *mut ());
// Some controllers take up to 55+ ms to complete the controller
// restore so setting the timeout to 100ms. Xhci specification
// doesn't mention any timeout value.
if xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut (), BIT!(9i32) as u32, 0u32, 100000u64) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: xHC restore state timeout\n".as_ptr() as *const u8);
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
return 110i32.wrapping_neg();
} else {
// re-initialize the HC on Restore Error, or Host Controller Error
temp = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *const ());
if (temp as u64) & (BIT!(10i32) | BIT!(12i32)) != 0 && ((*xhci).xhc_state as u64) & BIT!(2i32) == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHC error in resume, USBSTS 0x%x, Reinit\n".as_ptr() as *const u8, temp);
reset_registers = true;
}
}
}
}
if reset_registers {
if (*xhci).quirks & BIT_ULL!(14i32) != 0 {
__v5 = xhci_all_ports_seen_u0(xhci) == 0;
} else {
__v5 = false;
}
if __v5 {
timer_delete_sync(core::ptr::addr_of_mut!((*xhci).comp_mode_recovery_timer));
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Compliance Mode Recovery Timer deleted!".as_ptr() as *const u8);
}
// Let the USB core know _both_ roothubs lost power.
usb_root_hub_lost_power((*(*xhci).main_hcd).self_.root_hub);
if !unsafe { (*xhci).shared_hcd }.is_null() {
usb_root_hub_lost_power((*(*xhci).shared_hcd).self_.root_hub);
}
xhci_halt(xhci);
xhci_zero_64b_regs(xhci);
if ((*xhci).xhc_state as u64) & BIT!(2i32) != 0 {
retval = 19i32.wrapping_neg();
} else {
retval = xhci_reset(xhci, 10000000u64);
}
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
if retval != 0 {
return retval;
} else {
cancel_delayed_work_sync(core::ptr::addr_of_mut!((*xhci).cmd_timer));
// Delete all remaining commands
xhci_cleanup_command_queue(xhci);
// Clear data which is re-initilized during runtime
*__slate_slot_13 = (*(*(*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize))).event_ring).first_seg;
while !(*__slate_slot_13).is_null() {
memset((*(*__slate_slot_13)).trbs as *mut (), 0i32, (core::mem::size_of::<xhci_trb>() as u64).wrapping_mul(256));
*__slate_slot_13 = if (*(*__slate_slot_13)).next != (*(*(*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize))).event_ring).first_seg { (*(*__slate_slot_13)).next } else { core::ptr::null_mut::<xhci_segment>() };
}
i = (*xhci).max_slots as i32;
while i > 0 {
xhci_free_virt_devices_depth_first(xhci, i);
i = i.wrapping_sub(1);
}
xhci_rh_bw_cleanup(xhci);
(*xhci).cmd_ring_reserved_trbs = 0;
*__slate_slot_13 = (*(*xhci).cmd_ring).first_seg;
while !(*__slate_slot_13).is_null() {
memset((*(*__slate_slot_13)).trbs as *mut (), 0i32, (core::mem::size_of::<xhci_trb>() as u64).wrapping_mul(256));
*__slate_slot_13 = if (*(*__slate_slot_13)).next != (*(*xhci).cmd_ring).first_seg { (*(*__slate_slot_13)).next } else { core::ptr::null_mut::<xhci_segment>() };
}
xhci_debugfs_exit(xhci);
xhci_init(hcd);
// USB core calls the PCI reinit and start functions twice:
// first with the primary HCD, and then with the secondary HCD.
// If we don't do the same, the host will never be started.
retval = xhci_run(hcd);
if retval == 0 && !unsafe { (*xhci).shared_hcd }.is_null() {
retval = xhci_run((*xhci).shared_hcd);
}
if retval != 0 {
return retval;
} else {
// Resume roothubs unconditionally as PORTSC change bits are not
// immediately visible after xHC reset
(*hcd).state = 4;
if !unsafe { (*xhci).shared_hcd }.is_null() {
(*(*xhci).shared_hcd).state = 4;
usb_hcd_resume_root_hub((*xhci).shared_hcd);
}
usb_hcd_resume_root_hub(hcd);
}
}
} else {
// step 4: set Run/Stop bit
*__slate_slot_16 = readl(core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *const ());
*__slate_slot_16 |= 1u32;
writel(*__slate_slot_16, core::ptr::addr_of_mut!((*(*xhci).op_regs).command) as *mut ());
xhci_handshake(core::ptr::addr_of_mut!((*(*xhci).op_regs).status) as *mut (), 1u32, 0u32, 250000u64);
// step 5: walk topology and initialize portsc,
// portpmsc and portli
//
// this is done in bus_resume
// step 6: restart each of the previously
// Running endpoints by ringing their doorbells
spin_unlock_irq(core::ptr::addr_of_mut!((*xhci).lock));
xhci_dbc_resume(xhci);
// Resume roothubs only if there are pending events.
// USB 3 devices resend U3 LFPS wake after a 100ms delay if
// the first wake signalling failed, give it that chance if
// there are suspended USB 3 devices.
if (*xhci).usb3_rhub.bus_state.suspended_ports != 0 || (*xhci).usb3_rhub.bus_state.bus_suspended != 0 {
suspended_usb3_devs = true;
}
pending_portevent = xhci_pending_portevent(xhci);
if suspended_usb3_devs && !pending_portevent && is_auto_resume {
msleep(120u32);
pending_portevent = xhci_pending_portevent(xhci);
}
if pending_portevent {
if !unsafe { (*xhci).shared_hcd }.is_null() {
usb_hcd_resume_root_hub((*xhci).shared_hcd);
}
usb_hcd_resume_root_hub(hcd);
}
// If system is subject to the Quirk, Compliance Mode Timer needs to
// be re-initialized Always after a system resume. Ports are subject
// to suffer the Compliance Mode issue again. It doesn't matter if
// ports have entered previously to U0 before system's suspension.
if (*xhci).quirks & BIT_ULL!(14i32) != 0 {
compliance_mode_recovery_timer_init(xhci);
}
}
if (*xhci).quirks & BIT_ULL!(28i32) != 0 {
usb_asmedia_modifyflowcontrol(to_pci_dev!(unsafe { (*hcd).self_.controller }));
}
// Re-enable port polling.
if !unsafe { (*xhci).shared_hcd }.is_null() {
set_bit(2i64, core::ptr::addr_of_mut!((*(*xhci).shared_hcd).flags) as *mut u64);
usb_hcd_poll_rh_status((*xhci).shared_hcd);
}
set_bit(2i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
usb_hcd_poll_rh_status(hcd);
return retval;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_resume_940: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_hcd, bool, bool) -> i32>, *mut ()>(Some(xhci_resume)) };

fn xhci_map_temp_buffer(hcd: *mut usb_hcd, urb: *mut urb) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut buf_len: u32 = 0;
let dir: u32 = (if usb_urb_dir_in(urb) != 0 { 2i32 } else { 1i32 }) as u32;
buf_len = unsafe { (*urb).transfer_buffer_length };
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(buf_len as u64, 2336u32, dev_to_node(unsafe { (*hcd).self_.sysdev }), unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(buf_len as u64, 2336u32, dev_to_node(unsafe { (*hcd).self_.sysdev }), unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let temp: *mut () = __v0;
if temp.is_null() {
return 12i32.wrapping_neg();
}
if usb_urb_dir_out(urb) != 0 {
unsafe { sg_pcopy_to_buffer((*urb).sg, (*urb).num_sgs as u32, temp, buf_len as u64, 0i64) };
}
unsafe {
(*urb).transfer_buffer = temp;
}
unsafe {
(*urb).transfer_dma = dma_map_single__2!(unsafe { (*hcd).self_.sysdev }, unsafe { (*urb).transfer_buffer }, unsafe { (*urb).transfer_buffer_length }, dir);
}
if dma_mapping_error(unsafe { (*hcd).self_.sysdev }, unsafe { (*urb).transfer_dma }) != 0 {
ret = 11i32.wrapping_neg();
unsafe { kfree(temp as *const ()) };
} else {
unsafe {
(*urb).transfer_flags = (*urb).transfer_flags | 65536;
}
}
ret
}

fn xhci_urb_temp_buffer_required(hcd: *mut usb_hcd, urb: *mut urb) -> bool {
let mut ret: bool = false;
let mut i: u32 = 0;
let mut len: u32 = 0;
let mut trb_size: u32 = 0;
let mut max_pkt: u32 = 0;
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut tail_sg: *mut scatterlist = unsafe { core::mem::zeroed() };
tail_sg = unsafe { (*urb).sg };
max_pkt = xhci_usb_endpoint_maxp(unsafe { (*urb).dev }, unsafe { (*urb).ep }) as u32;
if (unsafe { (*urb).num_sgs }) == 0 {
return ret;
}
if ((unsafe { (*(*urb).dev).speed }) as u32) >= 5 {
trb_size = 16;
} else {
trb_size = 8;
}
if (unsafe { (*urb).transfer_buffer_length }) != 0 && (unsafe { (*urb).transfer_flags }) & 4 == 0 {
i = 0;
sg = unsafe { (*urb).sg };
while i < ((unsafe { (*urb).num_sgs }) as u32) {
len = len.wrapping_add(unsafe { (*sg).length });
if i > trb_size.wrapping_sub(2) {
len = len.wrapping_sub(unsafe { (*tail_sg).length });
if len < max_pkt {
ret = true;
break;
}
tail_sg = sg_next(tail_sg);
}
i = i.wrapping_add(1);
sg = sg_next(sg);
}
}
ret
}

fn xhci_unmap_temp_buf(hcd: *mut usb_hcd, urb: *mut urb) {
let mut len: u32 = 0;
let mut buf_len: u32 = 0;
let dir: u32 = (if usb_urb_dir_in(urb) != 0 { 2i32 } else { 1i32 }) as u32;
buf_len = unsafe { (*urb).transfer_buffer_length };
if IS_ENABLED__2!() != 0 && (unsafe { (*urb).transfer_flags }) & 65536 != 0 {
dma_unmap_single__2!(unsafe { (*hcd).self_.sysdev }, unsafe { (*urb).transfer_dma }, unsafe { (*urb).transfer_buffer_length }, dir);
}
if usb_urb_dir_in(urb) != 0 {
len = (unsafe { sg_pcopy_from_buffer((*urb).sg, (*urb).num_sgs as u32, (*urb).transfer_buffer as *const (), buf_len as u64, 0i64) }) as u32;
if len != buf_len {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(hcd_to_xhci(hcd))).self_.controller as *const device, c"Copy from tmp buf to urb sg list failed\n".as_ptr() as *const u8) };
}
unsafe {
(*urb).actual_length = len;
}
}
}
let __v2: u32 = (unsafe { (*urb).transfer_flags }) & 4294901759u32;
unsafe {
(*urb).transfer_flags = __v2;
}
unsafe { kfree((*urb).transfer_buffer as *const ()) };
unsafe {
(*urb).transfer_buffer = core::ptr::null_mut::<()>();
}
}

/// Bypass the DMA mapping if URB is suitable for Immediate Transfer (IDT),
/// we'll copy the actual data into the TRB address register. This is limited to
/// transfers up to 8 bytes on output endpoints of any kind with wMaxPacketSize
/// >= 8 bytes. If suitable for IDT only one Transfer TRB per TD is allowed.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_map_urb_for_dma")]
extern "C" fn xhci_map_urb_for_dma(hcd: *mut usb_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if xhci_urb_suitable_for_idt(urb) {
return 0;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(39i32) != 0 {
if xhci_urb_temp_buffer_required(hcd, urb) {
return xhci_map_temp_buffer(hcd, urb);
}
}
unsafe { usb_hcd_map_urb_for_dma(hcd, urb, mem_flags) }
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_unmap_urb_for_dma")]
extern "C" fn xhci_unmap_urb_for_dma(hcd: *mut usb_hcd, urb: *mut urb) {
let mut unmap_temp_buf: bool = false;
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if (unsafe { (*urb).num_sgs }) != 0 && (unsafe { (*urb).transfer_flags }) & 65536 != 0 {
unmap_temp_buf = true;
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(39i32) != 0 && unmap_temp_buf {
xhci_unmap_temp_buf(hcd, urb);
} else {
unsafe { usb_hcd_unmap_urb_for_dma(hcd, urb) };
}
}

/// xhci_get_endpoint_index - Used for passing endpoint bitmasks between the core and
/// HCDs.  Find the index for an endpoint given its descriptor.  Use the return
/// value to right shift 1 for the bitmask.
/// @desc: USB endpoint descriptor to determine index for
///
/// Index  = (epnum * 2) + direction - 1,
/// where direction = 0 for OUT, 1 for IN.
/// For control endpoints, the IN index is used (OUT index is unused), so
/// index = (epnum * 2) + direction - 1 = (epnum * 2) + 1 - 1 = (epnum * 2)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_get_endpoint_index")]
extern "C" fn xhci_get_endpoint_index(desc: *mut usb_endpoint_descriptor) -> u32 {
let index: u32 = if usb_endpoint_xfer_control(desc as *const usb_endpoint_descriptor) != 0 {
usb_endpoint_num(desc as *const usb_endpoint_descriptor).wrapping_mul(2) as u32
} else {
(usb_endpoint_num(desc as *const usb_endpoint_descriptor).wrapping_mul(2) as u32).wrapping_add((if usb_endpoint_dir_in(desc as *const usb_endpoint_descriptor) != 0 { 1i32 } else { 0i32 }) as u32).wrapping_sub(1)
};
index
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_get_endpoint_index_941: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_endpoint_descriptor) -> u32>, *mut ()>(Some(xhci_get_endpoint_index)) };

/// The reverse operation to xhci_get_endpoint_index. Calculate the USB endpoint
/// address from the XHCI endpoint index.
fn xhci_get_endpoint_address(ep_index: u32) -> u32 {
let number: u32 = ep_index.wrapping_add(2).wrapping_sub(1) / 2;
let direction: u32 = (if ep_index % 2 != 0 { 0i32 } else { 128i32 }) as u32;
direction | number
}

/// Find the flag for this endpoint (for use in the control context).  Use the
/// endpoint index to create a bitmask.  The slot context is bit 0, endpoint 0 is
/// bit 1, etc.
fn xhci_get_endpoint_flag(desc: *mut usb_endpoint_descriptor) -> u32 {
(1i32 << xhci_get_endpoint_index(desc).wrapping_add(1)) as u32
}

/// Compute the last valid endpoint context index.  Basically, this is the
/// endpoint index plus one.  For slot contexts with more than valid endpoint,
/// we find the most significant bit set in the added contexts flags.
/// e.g. ep 1 IN (with epnum 0x81) => added_ctxs = 0b1000
/// fls(0b1000) = 4, but the endpoint context index is 3, so subtract one.
#[unsafe(no_mangle)]
extern "C" fn xhci_last_valid_endpoint(added_ctxs: u32) -> u32 {
fls(added_ctxs).wrapping_sub(1) as u32
}

/// Returns 1 if the arguments are OK;
/// returns 0 this is a root hub; returns -EINVAL for NULL pointers.
fn xhci_check_args(hcd: *mut usb_hcd, udev: *mut usb_device, ep: *mut usb_host_endpoint, check_ep: i32, check_virt_dev: bool, func: *const u8) -> i32 {
if hcd.is_null() || check_ep != 0 && ep.is_null() || udev.is_null() {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017xHCI %s called with invalid args\n".as_ptr() as *const u8, func) };
}
__v0 = 0;
return 22i32.wrapping_neg();
}
if unsafe { (*udev).parent }.is_null() {
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017xHCI %s called for root hub\n".as_ptr() as *const u8, func) };
}
__v1 = 0;
return 0;
}
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if check_virt_dev {
if (unsafe { (*udev).slot_id }) == 0 || unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) }.is_null() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI %s called with unaddressed device\n".as_ptr() as *const u8, func) };
}
return 22i32.wrapping_neg();
}
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
if (unsafe { (*virt_dev).udev }) != udev {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI %s called with udev and virt_dev does not match\n".as_ptr() as *const u8, func) };
}
return 22i32.wrapping_neg();
}
}
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(1i32) != 0 {
return 19i32.wrapping_neg();
}
1
}

/// Full speed devices may have a max packet size greater than 8 bytes, but the
/// USB core doesn't know that until it reads the first 8 bytes of the
/// descriptor.  If the usb_device's max packet size changes after that point,
/// we need to issue an evaluate context command and wait on it.
fn xhci_check_ep0_maxpacket(xhci: *mut xhci_hcd, vdev: *mut xhci_virt_device) -> i32 {
let mut ep_ctx: *mut xhci_ep_ctx = unsafe { core::mem::zeroed() };
let mut command: *mut xhci_command = unsafe { core::mem::zeroed() };
let mut max_packet_size: i32 = 0;
let mut hw_max_packet_size: i32 = 0;
let mut ret: i32 = 0;
ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*vdev).out_ctx, 0u32) };
hw_max_packet_size = MAX_PACKET_DECODED!(unsafe { (*ep_ctx).ep_info2 }) as i32;
max_packet_size = usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*(*vdev).udev).ep0.desc) }) as *const usb_endpoint_descriptor);
if hw_max_packet_size == max_packet_size {
return 0;
}
match max_packet_size {
8 | 9 | 16 | 32 | 64 => {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Max Packet Size for ep 0 changed.".as_ptr() as *const u8) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Max packet size in usb_device = %d".as_ptr() as *const u8, max_packet_size) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Max packet size in xHCI HW = %d".as_ptr() as *const u8, hw_max_packet_size) };
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Issuing evaluate context command.".as_ptr() as *const u8) };
command = unsafe { xhci_alloc_command(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if command.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*command).in_ctx = (*vdev).in_ctx;
}
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*command).in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_check_ep0_maxpacket".as_ptr() as *mut u8) };
ret = 12i32.wrapping_neg();
} else {
// Set up the modified control endpoint 0
unsafe { xhci_endpoint_copy(xhci, (*vdev).in_ctx, (*vdev).out_ctx, 0u32) };
ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*command).in_ctx, 0u32) };
let __v4: u32 = (unsafe { (*ep_ctx).ep_info }) & 4294967288u32;
unsafe {
(*ep_ctx).ep_info = __v4;
}
// must clear
let __v7: u32 = (unsafe { (*ep_ctx).ep_info2 }) & 65535u32;
unsafe {
(*ep_ctx).ep_info2 = __v7;
}
let __v9: u32 = unsafe { (*ep_ctx).ep_info2 };
let __v10: u32 = __v9 | (MAX_PACKET!(max_packet_size) as u32);
unsafe {
(*ep_ctx).ep_info2 = __v10;
}
unsafe {
(*ctrl_ctx).add_flags = BIT!(1i32) as u32;
}
unsafe {
(*ctrl_ctx).drop_flags = 0;
}
ret = xhci_configure_endpoint(xhci, unsafe { (*vdev).udev }, command, true, false);
// Clean up the input context for later use by bandwidth functions
unsafe {
(*ctrl_ctx).add_flags = BIT!(0i32) as u32;
}
}
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe { kfree((*command).completion as *const ()) };
unsafe { kfree(command as *const ()) };
ret
}

/// non-error returns are a promise to giveback() the urb later
/// we drop ownership so next owner (or urb unlink) can get it
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_urb_enqueue")]
extern "C" fn xhci_urb_enqueue(hcd: *mut usb_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut __slate_compound_24: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_ptr: *mut urb_priv = unsafe { core::mem::zeroed() };
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __count: u64 = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut num_tds: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(num_tds) as *mut i32;
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut ep_state: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u32 = core::ptr::addr_of_mut!(ep_state) as *mut *mut u32;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
let mut slot_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(slot_id) as *mut u32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
{
xhci = hcd_to_xhci(hcd);
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
ret = 0;
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_12 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*urb).ep).desc));
if usb_endpoint_xfer_isoc(core::ptr::addr_of_mut!((*(*urb).ep).desc) as *const usb_endpoint_descriptor) != 0 {
*__slate_slot_9 = (*urb).number_of_packets;
} else {
if usb_endpoint_is_bulk_out(core::ptr::addr_of_mut!((*(*urb).ep).desc) as *const usb_endpoint_descriptor) != 0 && (*urb).transfer_buffer_length > 0 && (*urb).transfer_flags & 64 != 0 {
__v8 = (*urb).transfer_buffer_length % (usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*urb).ep).desc) as *const usb_endpoint_descriptor) as u32) == 0;
} else {
__v8 = false;
}
if __v8 {
*__slate_slot_9 = 2;
} else {
*__slate_slot_9 = 1;
}
}
}
__count = *__slate_slot_9 as u64;
__obj_size = struct_size_t!(__count);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, mem_flags, *{
__slate_compound_24 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_24)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, mem_flags, *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
}
__obj_ptr = *__slate_slot_1 as *mut urb_priv;
if !__obj_ptr.is_null() {
__set_flex_counter!(__count, __obj_ptr);
}
*__slate_slot_10 = __obj_ptr;
if (*__slate_slot_10).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_0: {
(*(*__slate_slot_10)).num_tds = *__slate_slot_9;
(*(*__slate_slot_10)).num_tds_done = 0;
(*urb).hcpriv = *__slate_slot_10 as *mut ();
trace_xhci_urb_enqueue(urb);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_15);
ret = xhci_check_args(hcd, (*urb).dev, (*urb).ep, 1i32, true, c"xhci_urb_enqueue".as_ptr() as *const u8);
if ret <= 0 {
ret = if ret != 0 { ret } else { 22i32.wrapping_neg() };
} else {
*__slate_slot_13 = (*(*urb).dev).slot_id as u32;
if HCD_HW_ACCESSIBLE!(hcd) == 0 {
ret = 108i32.wrapping_neg();
} else {
if ((*(*(*unsafe { (*xhci).devs }.wrapping_offset(*__slate_slot_13 as isize))).rhub_port).__slate_bits_0.__get_link_inactive() as i32) != 0 {
ret = 19i32.wrapping_neg();
} else {
if ((*xhci).xhc_state as u64) & BIT!(0i32) != 0 {
ret = 108i32.wrapping_neg();
} else {
*__slate_slot_11 = core::ptr::addr_of_mut!((*((*(*unsafe { (*xhci).devs }.wrapping_offset(*__slate_slot_13 as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_12 as isize)).ep_state);
if (*(*__slate_slot_11) as u64) & (BIT!(3i32) | BIT!(5i32)) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: Can't enqueue URB, ep in streams transition state %x\n".as_ptr() as *const u8, *(*__slate_slot_11));
ret = 22i32.wrapping_neg();
} else {
if (*(*__slate_slot_11) as u64) & BIT!(7i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Can't enqueue URB while manually clearing toggle\n".as_ptr() as *const u8);
ret = 22i32.wrapping_neg();
} else {
let __t0: i32 = usb_endpoint_type(core::ptr::addr_of_mut!((*(*urb).ep).desc) as *const usb_endpoint_descriptor);
if __t0 == 0 {
ret = xhci_queue_ctrl_tx(xhci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), urb, *__slate_slot_13 as i32, *__slate_slot_12);
} else {
if __t0 == 2 {
ret = xhci_queue_bulk_tx(xhci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), urb, *__slate_slot_13 as i32, *__slate_slot_12);
} else {
if __t0 == 3 {
ret = xhci_queue_intr_tx(xhci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), urb, *__slate_slot_13 as i32, *__slate_slot_12);
} else {
if __t0 == 1 {
ret = xhci_queue_isoc_tx_prepare(xhci, (BIT!(5i32) as u32) | (BIT!(11i32) as u32), urb, *__slate_slot_13 as i32, *__slate_slot_12);
}
}
}
}
if ret == 0 {
break '__join_0;
}
}
}
}
}
}
}
xhci_urb_free_priv(*__slate_slot_10);
(*urb).hcpriv = core::ptr::null_mut::<()>();
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_15);
return ret;
}
}
}

/// Remove the URB's TD from the endpoint ring.  This may cause the HC to stop
/// USB transfers, potentially stopping in the middle of a TRB buffer.  The HC
/// should pick up where it left off in the TD, unless a Set Transfer Ring
/// Dequeue Pointer is issued.
///
/// The TRBs that make up the buffers for the canceled URB will be "removed" from
/// the ring.  Since the ring is a contiguous structure, they can't be physically
/// removed.  Instead, there are two options:
///
///  1) If the HC is in the middle of processing the URB to be canceled, we
///     simply move the ring's dequeue pointer past those TRBs using the Set
///     Transfer Ring Dequeue Pointer command.  This will be the common case,
///     when drivers timeout on the last submitted URB and attempt to cancel.
///
///  2) If the HC is in the middle of a different TD, we turn the TRBs into a
///     series of 1-TRB transfer no-op TDs.  (No-ops shouldn't be chained.)  The
///     HC will need to invalidate the any TRBs it has cached after the stop
///     endpoint command, as noted in the xHCI 0.95 errata.
///
///  3) The TD may have completed by the time the Stop Endpoint Command
///     completes, so software needs to handle that case too.
///
/// This function should protect against the TD enqueueing code ringing the
/// doorbell while this code is waiting for a Stop Endpoint command to complete.
/// It also needs to account for multiple cancellations on happening at the same
/// time for the same endpoint.
///
/// Note that this function can be called in any context, or so says
/// usb_hcd_unlink_urb()
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_urb_dequeue")]
extern "C" fn xhci_urb_dequeue(hcd: *mut usb_hcd, urb: *mut urb, status: i32) -> i32 {
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
let mut ep_ring: core::mem::MaybeUninit<*mut xhci_ring> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut xhci_ring = core::ptr::addr_of_mut!(ep_ring) as *mut *mut xhci_ring;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
let mut td: core::mem::MaybeUninit<*mut xhci_td> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut xhci_td = core::ptr::addr_of_mut!(td) as *mut *mut xhci_td;
let mut urb_priv: core::mem::MaybeUninit<*mut urb_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut urb_priv = core::ptr::addr_of_mut!(urb_priv) as *mut *mut urb_priv;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
unsafe {
'__join_3: {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_16 = hcd_to_xhci(hcd);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_20);
trace_xhci_urb_dequeue(urb);
// Make sure the URB hasn't completed or been unlinked already
*__slate_slot_19 = usb_hcd_check_unlink_urb(hcd, urb, status);
if *__slate_slot_19 == 0 {
// give back URB now if we can't queue it for cancel
*__slate_slot_9 = *unsafe { (*(*__slate_slot_16)).devs }.wrapping_offset((*(*urb).dev).slot_id as isize);
*__slate_slot_15 = (*urb).hcpriv as *mut urb_priv;
if !((*__slate_slot_9).is_null() || (*__slate_slot_15).is_null()) {
*__slate_slot_13 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*urb).ep).desc));
*__slate_slot_11 = ((*(*__slate_slot_9)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_13 as isize);
*__slate_slot_12 = xhci_urb_to_transfer_ring(*__slate_slot_16, urb);
if !((*__slate_slot_11).is_null() || (*__slate_slot_12).is_null()) {
// If xHC is dead take it down and return ALL URBs in xhci_hc_died()
*__slate_slot_17 = readl(core::ptr::addr_of_mut!((*(*(*__slate_slot_16)).op_regs).status) as *const ());
if *__slate_slot_17 == !0 || ((*(*__slate_slot_16)).xhc_state as u64) & BIT!(0i32) != 0 {
xhci_hc_died(*__slate_slot_16);
break '__join_3;
} else {
// check ring is not re-allocated since URB was enqueued. If it is, then
// make sure none of the ring related pointers in this URB private data
// are touched, such as td_list, otherwise we overwrite freed data
if !td_on_ring((core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(0 as isize), *__slate_slot_12) {
while false {
}
_dev_err((*xhci_to_hcd(*__slate_slot_16)).self_.controller as *const device, c"Canceled URB td not found on endpoint ring".as_ptr() as *const u8);
*__slate_slot_18 = (*(*__slate_slot_15)).num_tds_done;
while *__slate_slot_18 < (*(*__slate_slot_15)).num_tds {
*__slate_slot_14 = (core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_18 as isize);
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list) as *const list_head) == 0 {
list_del_init(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list));
}
let __v8: i32 = *__slate_slot_18;
*__slate_slot_18 = __v8.wrapping_add(1);
}
} else {
if ((*(*__slate_slot_16)).xhc_state as u64) & BIT!(1i32) != 0 {
xhci_dbg_trace(*__slate_slot_16, Some(trace_xhci_dbg_cancel_urb), c"HC halted, freeing TD manually.".as_ptr() as *const u8);
*__slate_slot_18 = (*(*__slate_slot_15)).num_tds_done;
while *__slate_slot_18 < (*(*__slate_slot_15)).num_tds {
*__slate_slot_14 = (core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_18 as isize);
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_14)).td_list) as *const list_head) == 0 {
list_del_init(core::ptr::addr_of_mut!((*(*__slate_slot_14)).td_list));
}
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list) as *const list_head) == 0 {
list_del_init(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list));
}
let __v6: i32 = *__slate_slot_18;
*__slate_slot_18 = __v6.wrapping_add(1);
}
} else {
*__slate_slot_18 = (*(*__slate_slot_15)).num_tds_done;
if *__slate_slot_18 < (*(*__slate_slot_15)).num_tds {
xhci_dbg_trace(*__slate_slot_16, Some(trace_xhci_dbg_cancel_urb), c"Cancel URB %p, dev %s, ep 0x%x, starting at offset 0x%llx".as_ptr() as *const u8, urb, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, (*(*urb).ep).desc.bEndpointAddress as i32, xhci_trb_virt_to_dma((*(core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_18 as isize)).start_seg, (*(core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_18 as isize)).start_trb));
}
while *__slate_slot_18 < (*(*__slate_slot_15)).num_tds {
*__slate_slot_14 = (core::ptr::addr_of_mut!((*(*__slate_slot_15)).td) as *mut xhci_td).wrapping_offset(*__slate_slot_18 as isize);
// TD can already be on cancelled list if ep halted on it
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list) as *const list_head) != 0 {
(*(*__slate_slot_14)).cancel_status = 0;
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_14)).cancelled_td_list), core::ptr::addr_of_mut!((*(*__slate_slot_11)).cancelled_td_list));
}
let __v4: i32 = *__slate_slot_18;
*__slate_slot_18 = __v4.wrapping_add(1);
}
// These completion handlers will sort out cancelled TDs for us
if ((*(*__slate_slot_11)).ep_state as u64) & (BIT!(2i32) | BIT!(1i32) | BIT!(0i32)) != 0 {
break '__join_3;
} else {
// In this case no commands are pending but the endpoint is stopped
if ((*(*__slate_slot_11)).ep_state as u64) & (BIT!(8i32) | BIT!(9i32)) != 0 {
// and cancelled TDs can be given back right away
xhci_process_cancelled_tds(*__slate_slot_16, *__slate_slot_11);
break '__join_3;
} else {
// Otherwise, queue a new Stop Endpoint command
*__slate_slot_10 = xhci_alloc_command(*__slate_slot_16, false, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_10).is_null() {
*__slate_slot_19 = 12i32.wrapping_neg();
break '__join_3;
} else {
(*(*__slate_slot_11)).stop_time = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
let __v2: *mut xhci_virt_ep = *__slate_slot_11;
let __v1: u32 = (*__v2).ep_state;
let __v0: u32 = ((__v1 as u64) | BIT!(2i32)) as u32;
(*__v2).ep_state = __v0;
xhci_queue_stop_endpoint(*__slate_slot_16, *__slate_slot_10, (*(*urb).dev).slot_id, *__slate_slot_13, 0i32);
xhci_ring_cmd_db(*__slate_slot_16);
break '__join_3;
}
}
}
}
}
}
}
}
if !(*__slate_slot_15).is_null() {
xhci_urb_free_priv(*__slate_slot_15);
}
usb_hcd_unlink_urb_from_ep(hcd, urb);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_20);
usb_hcd_giveback_urb(hcd, urb, 108i32.wrapping_neg());
return *__slate_slot_19;
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_20);
return *__slate_slot_19;
}
}

/// Drop an endpoint from a new bandwidth configuration for this device.
/// Only one call to this function is allowed per endpoint before
/// check_bandwidth() or reset_bandwidth() must be called.
/// A call to xhci_drop_endpoint() followed by a call to xhci_add_endpoint() will
/// add the endpoint to the schedule with possibly new parameters denoted by a
/// different endpoint descriptor in usb_host_endpoint.
/// A call to xhci_add_endpoint() followed by a call to xhci_drop_endpoint() is
/// not allowed.
///
/// The USB core will not allow URBs to be queued to an endpoint that is being
/// disabled, so there's no need for mutual exclusion to protect
/// the xhci->devs[slot_id] structure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_drop_endpoint")]
extern "C" fn xhci_drop_endpoint(hcd: *mut usb_hcd, udev: *mut usb_device, ep: *mut usb_host_endpoint) -> i32 {
let mut ep_index: u32 = 0;
let mut drop_flag: u32 = 0;
let mut new_add_flags: u32 = 0;
let mut new_drop_flags: u32 = 0;
let ret: i32 = xhci_check_args(hcd, udev, ep, 1i32, true, c"xhci_drop_endpoint".as_ptr() as *const u8);
if ret <= 0 {
return ret;
}
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
return 19i32.wrapping_neg();
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s called for udev %p\n".as_ptr() as *const u8, c"xhci_drop_endpoint".as_ptr() as *mut u8, udev) };
}
drop_flag = xhci_get_endpoint_flag(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
if (drop_flag as u64) == BIT!(0i32) || (drop_flag as u64) == BIT!(1i32) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI %s - can't drop slot or ep 0 %#x\n".as_ptr() as *const u8, c"xhci_drop_endpoint".as_ptr() as *mut u8, drop_flag) };
}
return 0;
}
let in_ctx: *mut xhci_container_ctx = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).in_ctx };
let out_ctx: *mut xhci_container_ctx = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).out_ctx };
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx(in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_drop_endpoint".as_ptr() as *mut u8) };
return 0;
}
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, out_ctx, ep_index) };
// If the HC already knows the endpoint is disabled,
// or the HCD has noted it is disabled, ignore this request
let __v1: bool = GET_EP_CTX_STATE!(ep_ctx) == 0 || (unsafe { (*ctrl_ctx).drop_flags }) & xhci_get_endpoint_flag(unsafe { core::ptr::addr_of_mut!((*ep).desc) }) != 0;
if __v1 {
// Do not warn when called after a usb_device_reset
if !unsafe { (*((*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring }.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI %s called with disabled ep %p\n".as_ptr() as *const u8, c"xhci_drop_endpoint".as_ptr() as *mut u8, ep) };
}
return 0;
}
unsafe {
(*ctrl_ctx).drop_flags = (*ctrl_ctx).drop_flags | drop_flag;
}
new_drop_flags = unsafe { (*ctrl_ctx).drop_flags };
let __v8: u32 = (unsafe { (*ctrl_ctx).add_flags }) & !drop_flag;
unsafe {
(*ctrl_ctx).add_flags = __v8;
}
new_add_flags = unsafe { (*ctrl_ctx).add_flags };
unsafe { xhci_debugfs_remove_endpoint(xhci, *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize), ep_index as i32) };
unsafe { xhci_endpoint_zero(xhci, *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize), ep) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"drop ep 0x%x, slot id %d, new drop flags = %#x, new add flags = %#x\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as u32, (*udev).slot_id, new_drop_flags, new_add_flags) };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_drop_endpoint_942: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>, *mut ()>(Some(xhci_drop_endpoint)) };

/// Add an endpoint to a new possible bandwidth configuration for this device.
/// Only one call to this function is allowed per endpoint before
/// check_bandwidth() or reset_bandwidth() must be called.
/// A call to xhci_drop_endpoint() followed by a call to xhci_add_endpoint() will
/// add the endpoint to the schedule with possibly new parameters denoted by a
/// different endpoint descriptor in usb_host_endpoint.
/// A call to xhci_add_endpoint() followed by a call to xhci_drop_endpoint() is
/// not allowed.
///
/// The USB core will not allow URBs to be queued to an endpoint until the
/// configuration or alt setting is installed in the device, so there's no need
/// for mutual exclusion to protect the xhci->devs[slot_id] structure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_add_endpoint")]
extern "C" fn xhci_add_endpoint(hcd: *mut usb_hcd, udev: *mut usb_device, ep: *mut usb_host_endpoint) -> i32 {
let mut ep_index: u32 = 0;
let mut added_ctxs: u32 = 0;
let mut new_add_flags: u32 = 0;
let mut new_drop_flags: u32 = 0;
let ret: i32 = xhci_check_args(hcd, udev, ep, 1i32, true, c"xhci_add_endpoint".as_ptr() as *const u8);
if ret <= 0 {
// So we won't queue a reset ep command for a root hub
unsafe {
(*ep).hcpriv = core::ptr::null_mut::<()>();
}
return ret;
}
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
return 19i32.wrapping_neg();
}
added_ctxs = xhci_get_endpoint_flag(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
if (added_ctxs as u64) == BIT!(0i32) || (added_ctxs as u64) == BIT!(1i32) {
// FIXME when we have to issue an evaluate endpoint command to
// deal with ep0 max packet size changing once we get the
// descriptors
return 0;
}
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
let in_ctx: *mut xhci_container_ctx = unsafe { (*virt_dev).in_ctx };
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx(in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_add_endpoint".as_ptr() as *mut u8) };
return 0;
}
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
// If this endpoint is already in use, and the upper layers are trying
// to add it again without dropping it, reject the addition.
if !unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring }.is_null() && (unsafe { (*ctrl_ctx).drop_flags }) & added_ctxs == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Trying to add endpoint 0x%x without dropping it.\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as u32) };
return 22i32.wrapping_neg();
}
// If the HCD has already noted the endpoint is enabled,
// ignore this request.
if (unsafe { (*ctrl_ctx).add_flags }) & added_ctxs != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI %s called with enabled ep %p\n".as_ptr() as *const u8, c"xhci_add_endpoint".as_ptr() as *mut u8, ep) };
return 0;
}
// Configuration and alternate setting changes must be done in
// process context, not interrupt context (or so documenation
// for usb_set_interface() and usb_set_configuration() claim).
if (unsafe { xhci_endpoint_init(xhci, virt_dev, udev, ep, (BIT!(10i32) | BIT!(11i32)) as u32) }) < 0 {
return 12i32.wrapping_neg();
}
unsafe {
(*ctrl_ctx).add_flags = (*ctrl_ctx).add_flags | added_ctxs;
}
new_add_flags = unsafe { (*ctrl_ctx).add_flags };
// If xhci_endpoint_disable() was called for this endpoint, but the
// xHC hasn't been notified yet through the check_bandwidth() call,
// this re-adds a new state for the endpoint from the new endpoint
// descriptors.  We must drop and re-add this endpoint, so we leave the
// drop flags alone.
new_drop_flags = unsafe { (*ctrl_ctx).drop_flags };
// Store the usb_device pointer for later use
unsafe {
(*ep).hcpriv = udev as *mut ();
}
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*virt_dev).in_ctx, ep_index) };
trace_xhci_add_endpoint(ep_ctx);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_add_endpoint_943: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>, *mut ()>(Some(xhci_add_endpoint)) };

fn xhci_zero_in_ctx(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device) {
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*virt_dev).in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_zero_in_ctx".as_ptr() as *mut u8) };
return;
}
// When a device's add flag and drop flag are zero, any subsequent
// configure endpoint command will leave that endpoint's state
// untouched.  Make sure we don't leave any old state in the input
// endpoint contexts.
unsafe {
(*ctrl_ctx).drop_flags = 0;
}
unsafe {
(*ctrl_ctx).add_flags = 0;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*virt_dev).in_ctx) };
let __v3: u32 = (unsafe { (*slot_ctx).dev_info }) & 134217727u32;
unsafe {
(*slot_ctx).dev_info = __v3;
}
// Endpoint 0 is always valid
let __v5: u32 = unsafe { (*slot_ctx).dev_info };
let __v6: u32 = __v5 | (LAST_CTX!(1i32) as u32);
unsafe {
(*slot_ctx).dev_info = __v6;
}
for i in 1i32..31i32 {
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*virt_dev).in_ctx, i as u32) };
unsafe {
(*ep_ctx).ep_info = 0;
}
unsafe {
(*ep_ctx).ep_info2 = 0;
}
unsafe {
(*ep_ctx).deq = 0;
}
unsafe {
(*ep_ctx).tx_info = 0;
}
}
}

fn xhci_configure_endpoint_result(xhci: *mut xhci_hcd, udev: *mut usb_device, cmd_status: *mut u32) -> i32 {
let mut ret: i32 = 0;
match unsafe { *cmd_status } {
24 | 25 => {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Timeout while waiting for configure endpoint command\n".as_ptr() as *const u8) };
ret = 62i32.wrapping_neg();
}
7 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Not enough host controller resources for new device state.\n".as_ptr() as *const u8) };
ret = 12i32.wrapping_neg();
// FIXME: can we allocate more resources for the HC?
}
8 | 35 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Not enough bandwidth for new device state.\n".as_ptr() as *const u8) };
ret = 28i32.wrapping_neg();
// FIXME: can we go back to the old state?
}
5 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ERROR: Endpoint drop flag = 0, add flag = 1, and endpoint is not disabled.\n".as_ptr() as *const u8) };
// the HCD set up something wrong
ret = 22i32.wrapping_neg();
}
22 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ERROR: Incompatible device for endpoint configure command.\n".as_ptr() as *const u8) };
ret = 19i32.wrapping_neg();
}
1 => {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Successful Endpoint Configure command".as_ptr() as *const u8) };
ret = 0;
}
_ => {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR: unexpected command completion code 0x%x.\n".as_ptr() as *const u8, *cmd_status) };
ret = 22i32.wrapping_neg();
}
}
ret
}

fn xhci_evaluate_context_result(xhci: *mut xhci_hcd, udev: *mut usb_device, cmd_status: *mut u32) -> i32 {
let mut ret: i32 = 0;
match unsafe { *cmd_status } {
24 | 25 => {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Timeout while waiting for evaluate context command\n".as_ptr() as *const u8) };
ret = 62i32.wrapping_neg();
}
17 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"WARN: xHCI driver setup invalid evaluate context command.\n".as_ptr() as *const u8) };
ret = 22i32.wrapping_neg();
}
11 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"WARN: slot not enabled for evaluate context command.\n".as_ptr() as *const u8) };
ret = 22i32.wrapping_neg();
}
19 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"WARN: invalid context state for evaluate context command.\n".as_ptr() as *const u8) };
ret = 22i32.wrapping_neg();
}
22 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ERROR: Incompatible device for evaluate context command.\n".as_ptr() as *const u8) };
ret = 19i32.wrapping_neg();
}
29 => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"WARN: Max Exit Latency too large\n".as_ptr() as *const u8) };
// Max Exit Latency too large error
ret = 22i32.wrapping_neg();
}
1 => {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Successful evaluate context command".as_ptr() as *const u8) };
ret = 0;
}
_ => {
while false {
}
unsafe { _dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR: unexpected command completion code 0x%x.\n".as_ptr() as *const u8, *cmd_status) };
ret = 22i32.wrapping_neg();
}
}
ret
}

fn xhci_count_num_new_endpoints(xhci: *mut xhci_hcd, ctrl_ctx: *mut xhci_input_control_ctx) -> u32 {
let mut valid_add_flags: u32 = 0;
let mut valid_drop_flags: u32 = 0;
// Ignore the slot flag (bit 0), and the default control endpoint flag
// (bit 1).  The default control endpoint is added during the Address
// Device command and is never removed until the slot is disabled.
valid_add_flags = (unsafe { (*ctrl_ctx).add_flags }) >> 2i32;
valid_drop_flags = (unsafe { (*ctrl_ctx).drop_flags }) >> 2i32;
// Use hweight32 to count the number of ones in the add flags, or
// number of endpoints added.  Don't count endpoints that are changed
// (both added and dropped).
hweight32!(valid_add_flags).wrapping_sub(hweight32!(valid_add_flags & valid_drop_flags))
}

fn xhci_count_num_dropped_endpoints(xhci: *mut xhci_hcd, ctrl_ctx: *mut xhci_input_control_ctx) -> u32 {
let valid_add_flags: u32 = (unsafe { (*ctrl_ctx).add_flags }) >> 2i32;
let valid_drop_flags: u32 = (unsafe { (*ctrl_ctx).drop_flags }) >> 2i32;
hweight32!(valid_drop_flags).wrapping_sub(hweight32!(valid_add_flags & valid_drop_flags))
}

/// We need to reserve the new number of endpoints before the configure endpoint
/// command completes.  We can't subtract the dropped endpoints from the number
/// of active endpoints until the command completes because we can oversubscribe
/// the host in this case:
///
///  - the first configure endpoint command drops more endpoints than it adds
///  - a second configure endpoint command that adds more endpoints is queued
///  - the first configure endpoint command fails, so the config is unchanged
///  - the second command may succeed, even though there isn't enough resources
///
/// Must be called with xhci->lock held.
fn xhci_reserve_host_resources(xhci: *mut xhci_hcd, ctrl_ctx: *mut xhci_input_control_ctx) -> i32 {
let added_eps: u32 = xhci_count_num_new_endpoints(xhci, ctrl_ctx);
if unsafe { (*xhci).num_active_eps }.wrapping_add(added_eps) > unsafe { (*xhci).limit_active_eps } {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Not enough ep ctxs: %u active, need to add %u, limit is %u.".as_ptr() as *const u8, (*xhci).num_active_eps, added_eps, (*xhci).limit_active_eps) };
return 12i32.wrapping_neg();
}
let __v1: u32 = unsafe { (*xhci).num_active_eps };
unsafe {
(*xhci).num_active_eps = __v1.wrapping_add(added_eps);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Adding %u ep ctxs, %u now active.".as_ptr() as *const u8, added_eps, (*xhci).num_active_eps) };
0
}

/// The configure endpoint was failed by the xHC for some other reason, so we
/// need to revert the resources that failed configuration would have used.
///
/// Must be called with xhci->lock held.
fn xhci_free_host_resources(xhci: *mut xhci_hcd, ctrl_ctx: *mut xhci_input_control_ctx) {
let num_failed_eps: u32 = xhci_count_num_new_endpoints(xhci, ctrl_ctx);
let __v1: u32 = unsafe { (*xhci).num_active_eps };
unsafe {
(*xhci).num_active_eps = __v1.wrapping_sub(num_failed_eps);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Removing %u failed ep ctxs, %u now active.".as_ptr() as *const u8, num_failed_eps, (*xhci).num_active_eps) };
}

/// Now that the command has completed, clean up the active endpoint count by
/// subtracting out the endpoints that were dropped (but not changed).
///
/// Must be called with xhci->lock held.
fn xhci_finish_resource_reservation(xhci: *mut xhci_hcd, ctrl_ctx: *mut xhci_input_control_ctx) {
let num_dropped_eps: u32 = xhci_count_num_dropped_endpoints(xhci, ctrl_ctx);
let __v1: u32 = unsafe { (*xhci).num_active_eps };
unsafe {
(*xhci).num_active_eps = __v1.wrapping_sub(num_dropped_eps);
}
if num_dropped_eps != 0 {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Removing %u dropped ep ctxs, %u now active.".as_ptr() as *const u8, num_dropped_eps, (*xhci).num_active_eps) };
}
}

fn xhci_get_block_size(udev: *mut usb_device) -> u32 {
match (unsafe { (*udev).speed }) as u32 {
1 | 2 => {
return 1;
}
3 => {
return 4;
}
5 | 6 => {
return 16;
}
_ => {
return 1; // Should never happen
}
}
unsafe { core::mem::zeroed() }
}

fn xhci_get_largest_overhead(interval_bw: *mut xhci_interval_bw) -> u32 {
if (unsafe { *((*interval_bw).overhead.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) }) != 0 {
return 128;
}
if (unsafe { *((*interval_bw).overhead.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) }) != 0 {
return 20;
}
26
}

/// If we are changing a LS/FS device under a HS hub,
/// make sure (if we are activating a new TT) that the HS bus has enough
/// bandwidth for this new TT.
fn xhci_check_tt_bw_table(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, old_active_eps: i32) -> i32 {
// Find the bandwidth table for the root port this TT is attached to.
let bw_table: *mut xhci_interval_bw_table = unsafe { core::ptr::addr_of_mut!((*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*virt_dev).rhub_port).hw_portnum as isize)).bw_table) };
let tt_info: *mut xhci_tt_bw_info = unsafe { (*virt_dev).tt_info };
// If this TT already had active endpoints, the bandwidth for this TT
// has already been added.  Removing all periodic endpoints (and thus
// making the TT enactive) will only decrease the bandwidth used.
if old_active_eps != 0 {
return 0;
}
if old_active_eps == 0 && (unsafe { (*tt_info).active_eps }) != 0 {
if unsafe { (*bw_table).bw_used }.wrapping_add(125u32) > 1607 {
return 12i32.wrapping_neg();
}
return 0;
}
// Not sure why we would have no new active endpoints...
//
// Maybe because of an Evaluate Context change for a hub update or a
// control endpoint 0 max packet size change?
// FIXME: skip the bandwidth calculation in that case.
0
}

fn xhci_check_ss_bw(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device) -> i32 {
let mut bw_reserved: u32 = 391u32;
if (unsafe { (*(*virt_dev).bw_table).ss_bw_in }) > 3906u32.wrapping_sub(bw_reserved) {
return 12i32.wrapping_neg();
}
bw_reserved = 391u32;
if (unsafe { (*(*virt_dev).bw_table).ss_bw_out }) > 3906u32.wrapping_sub(bw_reserved) {
return 12i32.wrapping_neg();
}
0
}

/// This algorithm is a very conservative estimate of the worst-case scheduling
/// scenario for any one interval.  The hardware dynamically schedules the
/// packets, so we can't tell which microframe could be the limiting factor in
/// the bandwidth scheduling.  This only takes into account periodic endpoints.
///
/// Obviously, we can't solve an NP complete problem to find the minimum worst
/// case scenario.  Instead, we come up with an estimate that is no less than
/// the worst case bandwidth used for any one microframe, but may be an
/// over-estimate.
///
/// We walk the requirements for each endpoint by interval, starting with the
/// smallest interval, and place packets in the schedule where there is only one
/// possible way to schedule packets for that interval.  In order to simplify
/// this algorithm, we record the largest max packet size for each interval, and
/// assume all packets will be that size.
///
/// For interval 0, we obviously must schedule all packets for each interval.
/// The bandwidth for interval 0 is just the amount of data to be transmitted
/// (the sum of all max ESIT payload sizes, plus any overhead per packet times
/// the number of packets).
///
/// For interval 1, we have two possible microframes to schedule those packets
/// in.  For this algorithm, if we can schedule the same number of packets for
/// each possible scheduling opportunity (each microframe), we will do so.  The
/// remaining number of packets will be saved to be transmitted in the gaps in
/// the next interval's scheduling sequence.
///
/// As we move those remaining packets to be scheduled with interval 2 packets,
/// we have to double the number of remaining packets to transmit.  This is
/// because the intervals are actually powers of 2, and we would be transmitting
/// the previous interval's packets twice in this interval.  We also have to be
/// sure that when we look at the largest max packet size for this interval, we
/// also look at the largest max packet size for the remaining packets and take
/// the greater of the two.
///
/// The algorithm continues to evenly distribute packets in each scheduling
/// opportunity, and push the remaining packets out, until we get to the last
/// interval.  Then those packets and their associated overhead are just added
/// to the bandwidth used.
fn xhci_check_bw_table(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, old_active_eps: i32) -> i32 {
let mut bw_reserved: u32 = 0;
let mut max_bandwidth: u32 = 0;
let mut bw_used: u32 = 0;
let mut block_size: u32 = 0;
let mut bw_table: *mut xhci_interval_bw_table = unsafe { core::mem::zeroed() };
let mut packet_size: u32 = 0;
let mut overhead: u32 = 0;
let mut packets_transmitted: u32 = 0;
let mut packets_remaining: u32 = 0;
if ((unsafe { (*(*virt_dev).udev).speed }) as u32) >= 5 {
return xhci_check_ss_bw(xhci, virt_dev);
}
if ((unsafe { (*(*virt_dev).udev).speed }) as u32) == 3 {
max_bandwidth = 1607;
// Convert percent of bus BW reserved to blocks reserved
bw_reserved = 20u32.wrapping_mul(max_bandwidth).wrapping_add(100).wrapping_sub(1) / 100;
} else {
max_bandwidth = 1285;
bw_reserved = 10u32.wrapping_mul(max_bandwidth).wrapping_add(100).wrapping_sub(1) / 100;
}
bw_table = unsafe { (*virt_dev).bw_table };
// We need to translate the max packet size and max ESIT payloads into
// the units the hardware uses.
block_size = xhci_get_block_size(unsafe { (*virt_dev).udev });
// If we are manipulating a LS/FS device under a HS hub, double check
// that the HS bus has enough bandwidth if we are activing a new TT.
if !unsafe { (*virt_dev).tt_info }.is_null() {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Recalculating BW for rootport %u".as_ptr() as *const u8, unsafe { (*(*virt_dev).rhub_port).hw_portnum }.wrapping_add(1)) };
if xhci_check_tt_bw_table(xhci, virt_dev, old_active_eps) != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough bandwidth on HS bus for newly activated TT.\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Recalculating BW for TT slot %u port %u".as_ptr() as *const u8, (*(*virt_dev).tt_info).slot_id, (*(*virt_dev).tt_info).ttport) };
} else {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Recalculating BW for rootport %u".as_ptr() as *const u8, unsafe { (*(*virt_dev).rhub_port).hw_portnum }.wrapping_add(1)) };
}
// Add in how much bandwidth will be used for interval zero, or the
// rounded max ESIT payload + number of packets * largest overhead.
bw_used = (unsafe { (*bw_table).interval0_esit_payload }.wrapping_add(block_size).wrapping_sub(1) / block_size).wrapping_add(unsafe { (*((*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(0 as isize)).num_packets }.wrapping_mul(xhci_get_largest_overhead(unsafe { (*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw }.wrapping_offset(0 as isize))));
for i in 1u32..16u32 {
let mut bw_added: u32 = 0;
let mut largest_mps: u32 = 0;
let mut interval_overhead: u32 = 0;
// How many packets could we transmit in this interval?
// If packets didn't fit in the previous interval, we will need
// to transmit that many packets twice within this interval.
packets_remaining = 2u32.wrapping_mul(packets_remaining).wrapping_add(unsafe { (*((*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(i as isize)).num_packets });
// Find the largest max packet size of this or the previous
// interval.
if list_empty((unsafe { core::ptr::addr_of_mut!((*((*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(i as isize)).endpoints) }) as *const list_head) != 0 {
largest_mps = 0;
} else {
let ep_entry: *mut list_head = unsafe { (*((*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw).wrapping_offset(i as isize)).endpoints.next };
let virt_ep: *mut xhci_virt_ep = list_entry__41!(ep_entry, xhci_virt_ep);
// Convert to blocks, rounding up
largest_mps = unsafe { (*virt_ep).bw_info.max_packet_size }.wrapping_add(block_size).wrapping_sub(1) / block_size;
}
if largest_mps > packet_size {
packet_size = largest_mps;
}
// Use the larger overhead of this or the previous interval.
interval_overhead = xhci_get_largest_overhead(unsafe { (*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw }.wrapping_offset(i as isize));
if interval_overhead > overhead {
overhead = interval_overhead;
}
// How many packets can we evenly distribute across
// (1 << (i + 1)) possible scheduling opportunities?
packets_transmitted = packets_remaining >> i.wrapping_add(1);
// Add in the bandwidth used for those scheduled packets
bw_added = packets_transmitted.wrapping_mul(overhead.wrapping_add(packet_size));
// How many packets do we have remaining to transmit?
packets_remaining %= (1i32 << i.wrapping_add(1)) as u32;
// What largest max packet size should those packets have?
//
// If we've transmitted all packets, don't carry over the
// largest packet size.
if packets_remaining == 0 {
packet_size = 0;
overhead = 0;
} else {
if packets_transmitted > 0 {
// Otherwise if we do have remaining packets, and we've
// scheduled some packets in this interval, take the
// largest max packet size from endpoints with this
// interval.
packet_size = largest_mps;
overhead = interval_overhead;
}
}
// Otherwise carry over packet_size and overhead from the last
// time we had a remainder.
bw_used = bw_used.wrapping_add(bw_added);
if bw_used > max_bandwidth {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough bandwidth. Proposed: %u, Max: %u\n".as_ptr() as *const u8, bw_used, max_bandwidth) };
return 12i32.wrapping_neg();
}
}
// Ok, we know we have some packets left over after even-handedly
// scheduling interval 15.  We don't know which microframes they will
// fit into, so we over-schedule and say they will be scheduled every
// microframe.
if packets_remaining > 0 {
bw_used = bw_used.wrapping_add(overhead.wrapping_add(packet_size));
}
if unsafe { (*virt_dev).tt_info }.is_null() && ((unsafe { (*(*virt_dev).udev).speed }) as u32) == 3 {
// OK, we're manipulating a HS device attached to a
// root port bandwidth domain.  Include the number of active TTs
// in the bandwidth used.
bw_used = bw_used.wrapping_add(125u32.wrapping_mul(unsafe { (*unsafe { (*xhci).rh_bw }.wrapping_offset((*(*virt_dev).rhub_port).hw_portnum as isize)).num_active_tts }));
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Final bandwidth: %u, Limit: %u, Reserved: %u, Available: %u percent".as_ptr() as *const u8, bw_used, max_bandwidth, bw_reserved, max_bandwidth.wrapping_sub(bw_used).wrapping_sub(bw_reserved).wrapping_mul(100) / max_bandwidth) };
bw_used = bw_used.wrapping_add(bw_reserved);
if bw_used > max_bandwidth {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough bandwidth. Proposed: %u, Max: %u\n".as_ptr() as *const u8, bw_used, max_bandwidth) };
return 12i32.wrapping_neg();
}
unsafe {
(*bw_table).bw_used = bw_used;
}
0
}

fn xhci_is_async_ep(ep_type: u32) -> bool {
ep_type != 1 && ep_type != 3 && ep_type != 5 && ep_type != 7
}

fn xhci_is_sync_in_ep(ep_type: u32) -> bool {
ep_type == 5 || ep_type == 7
}

fn xhci_get_ss_bw_consumed(ep_bw: *mut xhci_bw_info) -> u32 {
let mps: u32 = unsafe { (*ep_bw).max_packet_size }.wrapping_add(16).wrapping_sub(1) / 16;
if (unsafe { (*ep_bw).ep_interval }) == 0 {
return 32u32.wrapping_add(unsafe { (*ep_bw).mult }.wrapping_mul(unsafe { (*ep_bw).num_packets }).wrapping_mul(8u32.wrapping_add(mps)));
}
unsafe { (*ep_bw).mult }.wrapping_mul(unsafe { (*ep_bw).num_packets }).wrapping_mul(8u32.wrapping_add(mps).wrapping_add(32)).wrapping_add((1i32 << unsafe { (*ep_bw).ep_interval }) as u32).wrapping_sub(1) / ((1i32 << unsafe { (*ep_bw).ep_interval }) as u32)
}

fn xhci_drop_ep_from_interval_table(xhci: *mut xhci_hcd, ep_bw: *mut xhci_bw_info, bw_table: *mut xhci_interval_bw_table, udev: *mut usb_device, virt_ep: *mut xhci_virt_ep, tt_info: *mut xhci_tt_bw_info) {
let mut normalized_interval: i32 = 0;
if xhci_is_async_ep(unsafe { (*ep_bw).r#type }) {
return;
}
if ((unsafe { (*udev).speed }) as u32) >= 5 {
if xhci_is_sync_in_ep(unsafe { (*ep_bw).r#type }) {
let __v0: u32 = xhci_get_ss_bw_consumed(ep_bw);
let __v1: *mut xhci_interval_bw_table = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).bw_table };
let __v2: u32 = unsafe { (*__v1).ss_bw_in };
unsafe {
(*__v1).ss_bw_in = __v2.wrapping_sub(__v0);
}
} else {
let __v4: u32 = xhci_get_ss_bw_consumed(ep_bw);
let __v5: *mut xhci_interval_bw_table = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).bw_table };
let __v6: u32 = unsafe { (*__v5).ss_bw_out };
unsafe {
(*__v5).ss_bw_out = __v6.wrapping_sub(__v4);
}
}
return;
}
// SuperSpeed endpoints never get added to intervals in the table, so
// this check is only valid for HS/FS/LS devices.
if list_empty((unsafe { core::ptr::addr_of_mut!((*virt_ep).bw_endpoint_list) }) as *const list_head) != 0 {
return;
}
// For LS/FS devices, we need to translate the interval expressed in
// microframes to frames.
if ((unsafe { (*udev).speed }) as u32) == 3 {
normalized_interval = (unsafe { (*ep_bw).ep_interval }) as i32;
} else {
normalized_interval = unsafe { (*ep_bw).ep_interval }.wrapping_sub(3) as i32;
}
if normalized_interval == 0 {
let __v9: u32 = unsafe { (*bw_table).interval0_esit_payload };
unsafe {
(*bw_table).interval0_esit_payload = __v9.wrapping_sub((*ep_bw).max_esit_payload);
}
}
let interval_bw: *mut xhci_interval_bw = unsafe { (*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw }.wrapping_offset(normalized_interval as isize);
let __v12: u32 = unsafe { (*interval_bw).num_packets };
unsafe {
(*interval_bw).num_packets = __v12.wrapping_sub((*ep_bw).num_packets);
}
match (unsafe { (*udev).speed }) as u32 {
1 => {
let __v14: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(0 as isize);
let __v15: u32 = unsafe { *__v14 };
unsafe {
*__v14 = __v15.wrapping_sub(1);
}
}
2 => {
let __v18: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(1 as isize);
let __v19: u32 = unsafe { *__v18 };
unsafe {
*__v18 = __v19.wrapping_sub(1);
}
}
3 => {
let __v21: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(2 as isize);
let __v22: u32 = unsafe { *__v21 };
unsafe {
*__v21 = __v22.wrapping_sub(1);
}
}
_ => {
return;
// Should never happen because only LS/FS/HS endpoints will get
// added to the endpoint list.
}
}
if !tt_info.is_null() {
let __v25: i32 = unsafe { (*tt_info).active_eps };
unsafe {
(*tt_info).active_eps = __v25.wrapping_sub(1);
}
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*virt_ep).bw_endpoint_list) });
}

fn xhci_add_ep_to_interval_table(xhci: *mut xhci_hcd, ep_bw: *mut xhci_bw_info, bw_table: *mut xhci_interval_bw_table, udev: *mut usb_device, virt_ep: *mut xhci_virt_ep, tt_info: *mut xhci_tt_bw_info) {
let mut interval_bw: *mut xhci_interval_bw = unsafe { core::mem::zeroed() };
let mut smaller_ep: *mut xhci_virt_ep = unsafe { core::mem::zeroed() };
let mut normalized_interval: i32 = 0;
if xhci_is_async_ep(unsafe { (*ep_bw).r#type }) {
return;
}
if ((unsafe { (*udev).speed }) as u32) == 5 {
if xhci_is_sync_in_ep(unsafe { (*ep_bw).r#type }) {
let __v0: u32 = xhci_get_ss_bw_consumed(ep_bw);
let __v1: *mut xhci_interval_bw_table = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).bw_table };
let __v2: u32 = unsafe { (*__v1).ss_bw_in };
unsafe {
(*__v1).ss_bw_in = __v2.wrapping_add(__v0);
}
} else {
let __v4: u32 = xhci_get_ss_bw_consumed(ep_bw);
let __v5: *mut xhci_interval_bw_table = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).bw_table };
let __v6: u32 = unsafe { (*__v5).ss_bw_out };
unsafe {
(*__v5).ss_bw_out = __v6.wrapping_add(__v4);
}
}
return;
}
// For LS/FS devices, we need to translate the interval expressed in
// microframes to frames.
if ((unsafe { (*udev).speed }) as u32) == 3 {
normalized_interval = (unsafe { (*ep_bw).ep_interval }) as i32;
} else {
normalized_interval = unsafe { (*ep_bw).ep_interval }.wrapping_sub(3) as i32;
}
if normalized_interval == 0 {
let __v9: u32 = unsafe { (*bw_table).interval0_esit_payload };
unsafe {
(*bw_table).interval0_esit_payload = __v9.wrapping_add((*ep_bw).max_esit_payload);
}
}
interval_bw = unsafe { (*bw_table).interval_bw.as_mut_ptr() as *mut xhci_interval_bw }.wrapping_offset(normalized_interval as isize);
let __v12: u32 = unsafe { (*interval_bw).num_packets };
unsafe {
(*interval_bw).num_packets = __v12.wrapping_add((*ep_bw).num_packets);
}
match (unsafe { (*udev).speed }) as u32 {
1 => {
let __v14: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(0 as isize);
let __v15: u32 = unsafe { *__v14 };
unsafe {
*__v14 = __v15.wrapping_add(1);
}
}
2 => {
let __v18: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(1 as isize);
let __v19: u32 = unsafe { *__v18 };
unsafe {
*__v18 = __v19.wrapping_add(1);
}
}
3 => {
let __v21: *mut u32 = unsafe { (*interval_bw).overhead.as_mut_ptr() as *mut u32 }.wrapping_offset(2 as isize);
let __v22: u32 = unsafe { *__v21 };
unsafe {
*__v21 = __v22.wrapping_add(1);
}
}
_ => {
return;
// Should never happen because only LS/FS/HS endpoints will get
// added to the endpoint list.
}
}
if !tt_info.is_null() {
let __v25: i32 = unsafe { (*tt_info).active_eps };
unsafe {
(*tt_info).active_eps = __v25.wrapping_add(1);
}
}
// Insert the endpoint into the list, largest max packet size first.
smaller_ep = list_first_entry__27!(unsafe { core::ptr::addr_of_mut!((*interval_bw).endpoints) });
while list_entry_is_head!(smaller_ep, unsafe { core::ptr::addr_of_mut!((*interval_bw).endpoints) }, bw_endpoint_list) == 0 {
if (unsafe { (*ep_bw).max_packet_size }) >= unsafe { (*smaller_ep).bw_info.max_packet_size } {
// Add the new ep before the smaller endpoint
list_add_tail(unsafe { core::ptr::addr_of_mut!((*virt_ep).bw_endpoint_list) }, unsafe { core::ptr::addr_of_mut!((*smaller_ep).bw_endpoint_list) });
return;
}
smaller_ep = list_next_entry__23!(smaller_ep, bw_endpoint_list);
}
// Add the new endpoint at the end of the list.
list_add_tail(unsafe { core::ptr::addr_of_mut!((*virt_ep).bw_endpoint_list) }, unsafe { core::ptr::addr_of_mut!((*interval_bw).endpoints) });
}

#[unsafe(no_mangle)]
extern "C" fn xhci_update_tt_active_eps(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, old_active_eps: i32) {
if unsafe { (*virt_dev).tt_info }.is_null() {
return;
}
let rh_bw_info: *mut xhci_root_port_bw_info = unsafe { (*xhci).rh_bw }.wrapping_offset((unsafe { (*(*virt_dev).rhub_port).hw_portnum }) as isize);
if old_active_eps == 0 && (unsafe { (*(*virt_dev).tt_info).active_eps }) != 0 {
let __v1: u32 = unsafe { (*rh_bw_info).num_active_tts };
unsafe {
(*rh_bw_info).num_active_tts = __v1.wrapping_add(1);
}
let __v4: u32 = unsafe { (*rh_bw_info).bw_table.bw_used };
unsafe {
(*rh_bw_info).bw_table.bw_used = __v4.wrapping_add(125u32);
}
} else {
if old_active_eps != 0 && (unsafe { (*(*virt_dev).tt_info).active_eps }) == 0 {
let __v7: u32 = unsafe { (*rh_bw_info).num_active_tts };
unsafe {
(*rh_bw_info).num_active_tts = __v7.wrapping_sub(1);
}
let __v10: u32 = unsafe { (*rh_bw_info).bw_table.bw_used };
unsafe {
(*rh_bw_info).bw_table.bw_used = __v10.wrapping_sub(125u32);
}
}
}
}

fn xhci_reserve_bandwidth(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, in_ctx: *mut xhci_container_ctx) -> i32 {
let mut ep_bw_info: __SlateAlign16<[xhci_bw_info; 31]> = __SlateAlign16(unsafe { core::mem::zeroed() });
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ep_bw_info.0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 744usize);
}
let mut i: i32 = 0;
let mut old_active_eps: i32 = 0;
if !unsafe { (*virt_dev).tt_info }.is_null() {
old_active_eps = unsafe { (*(*virt_dev).tt_info).active_eps };
}
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx(in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_reserve_bandwidth".as_ptr() as *mut u8) };
return 12i32.wrapping_neg();
}
while i < 31 {
if !(EP_IS_ADDED!(ctrl_ctx, i) == 0 && EP_IS_DROPPED!(ctrl_ctx, i) == 0) {
// Make a copy of the BW info in case we need to revert this
unsafe { memcpy((ep_bw_info.0.as_mut_ptr() as *mut xhci_bw_info).wrapping_offset(i as isize) as *mut (), core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) as *const (), core::mem::size_of::<xhci_bw_info>() as u64) };
// Drop the endpoint from the interval table if the endpoint is
// being dropped or changed.
if EP_IS_DROPPED!(ctrl_ctx, i) != 0 {
xhci_drop_ep_from_interval_table(xhci, unsafe { core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) }, unsafe { (*virt_dev).bw_table }, unsafe { (*virt_dev).udev }, unsafe { (*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize), unsafe { (*virt_dev).tt_info });
}
}
i = i.wrapping_add(1);
}
// Overwrite the information stored in the endpoints' bw_info
unsafe { xhci_update_bw_info(xhci, (*virt_dev).in_ctx, ctrl_ctx, virt_dev) };
i = 0;
while i < 31 {
// Add any changed or added endpoints to the interval table
if EP_IS_ADDED!(ctrl_ctx, i) != 0 {
xhci_add_ep_to_interval_table(xhci, unsafe { core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) }, unsafe { (*virt_dev).bw_table }, unsafe { (*virt_dev).udev }, unsafe { (*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize), unsafe { (*virt_dev).tt_info });
}
i = i.wrapping_add(1);
}
if xhci_check_bw_table(xhci, virt_dev, old_active_eps) == 0 {
// Ok, this fits in the bandwidth we have.
// Update the number of active TTs.
xhci_update_tt_active_eps(xhci, virt_dev, old_active_eps);
return 0;
}
// We don't have enough bandwidth for this, revert the stored info.
i = 0;
while i < 31 {
if !(EP_IS_ADDED!(ctrl_ctx, i) == 0 && EP_IS_DROPPED!(ctrl_ctx, i) == 0) {
// Drop the new copies of any added or changed endpoints from
// the interval table.
if EP_IS_ADDED!(ctrl_ctx, i) != 0 {
xhci_drop_ep_from_interval_table(xhci, unsafe { core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) }, unsafe { (*virt_dev).bw_table }, unsafe { (*virt_dev).udev }, unsafe { (*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize), unsafe { (*virt_dev).tt_info });
}
// Revert the endpoint back to its old information
unsafe { memcpy(core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) as *mut (), (ep_bw_info.0.as_mut_ptr() as *mut xhci_bw_info).wrapping_offset(i as isize) as *const (), core::mem::size_of::<xhci_bw_info>() as u64) };
// Add any changed or dropped endpoints back into the table
if EP_IS_DROPPED!(ctrl_ctx, i) != 0 {
xhci_add_ep_to_interval_table(xhci, unsafe { core::ptr::addr_of_mut!((*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).bw_info) }, unsafe { (*virt_dev).bw_table }, unsafe { (*virt_dev).udev }, unsafe { (*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize), unsafe { (*virt_dev).tt_info });
}
}
i = i.wrapping_add(1);
}
12i32.wrapping_neg()
}

/// Synchronous XHCI stop endpoint helper.  Issues the stop endpoint command and
/// waits for the command completion before returning.  This does not call
/// xhci_handle_cmd_stop_ep(), which has additional handling for 'context error'
/// cases, along with transfer ring cleanup.
///
/// xhci_stop_endpoint_sync() is intended to be utilized by clients that manage
/// their own transfer ring, such as offload situations.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_stop_endpoint_sync")]
extern "C" fn xhci_stop_endpoint_sync(xhci: *mut xhci_hcd, ep: *mut xhci_virt_ep, suspend: i32, gfp_flags: u32) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 19i32.wrapping_neg();
*__slate_slot_3 = xhci_alloc_command(xhci, true, gfp_flags);
if (*__slate_slot_3).is_null() {
return 12i32.wrapping_neg();
} else {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
// make sure endpoint exists and is running before stopping it
if !unsafe { (*ep).ring }.is_null() {
*__slate_slot_2 = xhci_get_ep_ctx(xhci, (*(*ep).vdev).out_ctx, (*ep).ep_index);
if GET_EP_CTX_STATE!(*__slate_slot_2) == 1 {
ret = xhci_queue_stop_endpoint(xhci, *__slate_slot_3, (*(*ep).vdev).slot_id, (*ep).ep_index, suspend);
}
}
if ret < 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
} else {
xhci_ring_cmd_db(xhci);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
wait_for_completion((*(*__slate_slot_3)).completion);
// No handling for COMP_CONTEXT_STATE_ERROR done at command completion
if (*(*__slate_slot_3)).status == 25 || (*(*__slate_slot_3)).status == 24 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Timeout while waiting for stop endpoint command\n".as_ptr() as *const u8);
ret = 62i32.wrapping_neg();
}
}
xhci_free_command(xhci, *__slate_slot_3);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_stop_endpoint_sync_944: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut xhci_hcd, *mut xhci_virt_ep, i32, u32) -> i32>, *mut ()>(Some(xhci_stop_endpoint_sync)) };

/// xhci_usb_endpoint_maxp - get endpoint max packet size
/// @host_ep: USB host endpoint to be checked
///
/// Returns max packet from the correct descriptor
#[unsafe(no_mangle)]
extern "C" fn xhci_usb_endpoint_maxp(udev: *mut usb_device, host_ep: *mut usb_host_endpoint) -> i32 {
if unsafe { usb_endpoint_is_hs_isoc_double(udev, host_ep as *const usb_host_endpoint) } {
return (unsafe { (*host_ep).eusb2_isoc_ep_comp.wMaxPacketSize }) as i32;
}
usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*host_ep).desc) }) as *const usb_endpoint_descriptor)
}

/// Issue a configure endpoint command or evaluate context command
/// and wait for it to finish.
fn xhci_configure_endpoint(xhci: *mut xhci_hcd, udev: *mut usb_device, command: *mut xhci_command, ctx_change: bool, must_succeed: bool) -> i32 {
let mut ret: i32 = 0;
let mut flags: u64 = 0;
if command.is_null() {
return 22i32.wrapping_neg();
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
if ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
return 108i32.wrapping_neg();
}
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*command).in_ctx) };
if ctrl_ctx.is_null() {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_configure_endpoint".as_ptr() as *mut u8) };
return 12i32.wrapping_neg();
}
let __v6: bool = (unsafe { (*xhci).quirks }) & BIT_ULL!(5i32) != 0 && xhci_reserve_host_resources(xhci, ctrl_ctx) != 0;
if __v6 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough host resources, active endpoint contexts = %u\n".as_ptr() as *const u8, (*xhci).num_active_eps) };
return 12i32.wrapping_neg();
}
let __v8: bool = (unsafe { (*xhci).quirks }) & BIT_ULL!(8i32) != 0 && !ctx_change && xhci_reserve_bandwidth(xhci, virt_dev, unsafe { (*command).in_ctx }) != 0;
if __v8 {
if (unsafe { (*xhci).quirks }) & BIT_ULL!(5i32) != 0 {
xhci_free_host_resources(xhci, ctrl_ctx);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough bandwidth\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*command).in_ctx) };
trace_xhci_configure_endpoint_ctrl_ctx(ctrl_ctx);
trace_xhci_configure_endpoint(slot_ctx);
if !ctx_change {
ret = unsafe { xhci_queue_configure_endpoint(xhci, command, (*(*command).in_ctx).dma, (*udev).slot_id as u32, must_succeed) };
} else {
ret = unsafe { xhci_queue_evaluate_context(xhci, command, (*(*command).in_ctx).dma, (*udev).slot_id as u32, must_succeed) };
}
if ret < 0 {
if (unsafe { (*xhci).quirks }) & BIT_ULL!(5i32) != 0 {
xhci_free_host_resources(xhci, ctrl_ctx);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"FIXME allocate a new ring segment".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
unsafe { xhci_ring_cmd_db(xhci) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// Wait for the configure endpoint command to complete
unsafe { wait_for_completion((*command).completion) };
if !ctx_change {
ret = xhci_configure_endpoint_result(xhci, udev, unsafe { core::ptr::addr_of_mut!((*command).status) });
} else {
ret = xhci_evaluate_context_result(xhci, udev, unsafe { core::ptr::addr_of_mut!((*command).status) });
}
if (unsafe { (*xhci).quirks }) & BIT_ULL!(5i32) != 0 {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// If the command failed, remove the reserved resources.
// Otherwise, clean up the estimate to include dropped eps.
if ret != 0 {
xhci_free_host_resources(xhci, ctrl_ctx);
} else {
xhci_finish_resource_reservation(xhci, ctrl_ctx);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
}
ret
}

fn xhci_check_bw_drop_ep_streams(xhci: *mut xhci_hcd, vdev: *mut xhci_virt_device, i: i32) {
let ep: *mut xhci_virt_ep = unsafe { (*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize);
if ((unsafe { (*ep).ep_state }) as u64) & BIT!(4i32) != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: endpoint 0x%02x has streams on set_interface, freeing streams.\n".as_ptr() as *const u8, xhci_get_endpoint_address(i as u32)) };
unsafe { xhci_free_stream_info(xhci, (*ep).stream_info) };
unsafe {
(*ep).stream_info = core::ptr::null_mut::<xhci_stream_info>();
}
let __v2: u32 = unsafe { (*ep).ep_state };
let __v3: u32 = ((__v2 as u64) & !BIT!(4i32)) as u32;
unsafe {
(*ep).ep_state = __v3;
}
}
}

/// Called after one or more calls to xhci_add_endpoint() or
/// xhci_drop_endpoint().  If this call fails, the USB core is expected
/// to call xhci_reset_bandwidth().
///
/// Since we are in the middle of changing either configuration or
/// installing a new alt setting, the USB core won't allow URBs to be
/// enqueued for any endpoint on the old config or interface.  Nothing
/// else should be touching the xhci->devs[slot_id] structure, so we
/// don't need to take the xhci->lock for manipulating that.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_check_bandwidth")]
extern "C" fn xhci_check_bandwidth(hcd: *mut usb_hcd, udev: *mut usb_device) -> i32 {
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
let mut ctrl_ctx: core::mem::MaybeUninit<*mut xhci_input_control_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut xhci_input_control_ctx = core::ptr::addr_of_mut!(ctrl_ctx) as *mut *mut xhci_input_control_ctx;
let mut virt_dev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(virt_dev) as *mut *mut xhci_virt_device;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 0;
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = xhci_check_args(hcd, udev, core::ptr::null_mut::<usb_host_endpoint>(), 0i32, true, c"xhci_check_bandwidth".as_ptr() as *const u8);
if ret <= 0 {
return ret;
} else {
*__slate_slot_26 = hcd_to_xhci(hcd);
if ((*(*__slate_slot_26)).xhc_state as u64) & BIT!(0i32) != 0 || ((*(*__slate_slot_26)).xhc_state as u64) & BIT!(2i32) != 0 {
return 19i32.wrapping_neg();
} else {
*__slate_slot_25 = *unsafe { (*(*__slate_slot_26)).devs }.wrapping_offset((*udev).slot_id as isize);
*__slate_slot_22 = xhci_alloc_command(*__slate_slot_26, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_22).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_22)).in_ctx = (*(*__slate_slot_25)).in_ctx;
// See section 4.6.6 - A0 = 1; A1 = D0 = D1 = 0
*__slate_slot_24 = xhci_get_input_control_ctx((*(*__slate_slot_22)).in_ctx);
if (*__slate_slot_24).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_26)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_check_bandwidth".as_ptr() as *mut u8);
ret = 12i32.wrapping_neg();
} else {
let __v21: *mut xhci_input_control_ctx = *__slate_slot_24;
let __v20: u32 = (*__v21).add_flags;
let __v19: u32 = __v20 | (BIT!(0i32) as u32);
(*__v21).add_flags = __v19;
let __v18: *mut xhci_input_control_ctx = *__slate_slot_24;
let __v17: u32 = (*__v18).add_flags;
let __v16: u32 = __v17 & (!BIT!(1i32) as u32);
(*__v18).add_flags = __v16;
let __v15: *mut xhci_input_control_ctx = *__slate_slot_24;
let __v14: u32 = (*__v15).drop_flags;
let __v13: u32 = __v14 & (!(BIT!(0i32) | BIT!(1i32)) as u32);
(*__v15).drop_flags = __v13;
// Don't issue the command if there's no endpoints to update.
if (*(*__slate_slot_24)).add_flags == (BIT!(0i32) as u32) && (*(*__slate_slot_24)).drop_flags == 0 {
ret = 0;
} else {
// Fix up Context Entries field. Minimum value is EP0 == BIT(1).
*__slate_slot_23 = xhci_get_slot_ctx(*__slate_slot_26, (*(*__slate_slot_25)).in_ctx);
*__slate_slot_28 = 31;
'__join_13: {
loop {
if *__slate_slot_28 >= 1 {
let le32: u32 = BIT!(*__slate_slot_28) as u32;
if !unsafe { (*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset((*__slate_slot_28).wrapping_sub(1) as isize)).ring }.is_null() && (*(*__slate_slot_24)).drop_flags & le32 == 0 || (*(*__slate_slot_24)).add_flags & le32 != 0 || *__slate_slot_28 == 1 {
break;
} else {
let __v5: i32 = *__slate_slot_28;
*__slate_slot_28 = __v5.wrapping_sub(1);
}
} else {
break '__join_13;
}
}
let __v11: *mut xhci_slot_ctx = *__slate_slot_23;
let __v9: u32 = (*__v11).dev_info & 134217727u32;
(*__v11).dev_info = __v9;
let __v8: *mut xhci_slot_ctx = *__slate_slot_23;
let __v7: u32 = (*__v8).dev_info;
let __v6: u32 = __v7 | (LAST_CTX!(*__slate_slot_28) as u32);
(*__v8).dev_info = __v6;
}
ret = xhci_configure_endpoint(*__slate_slot_26, udev, *__slate_slot_22, false, false);
if ret == 0 {
// Callee should call reset_bandwidth()
// Free any rings that were dropped, but not changed.
*__slate_slot_28 = 1;
while *__slate_slot_28 < 31 {
if (*(*__slate_slot_24)).drop_flags & ((1i32 << (*__slate_slot_28).wrapping_add(1)) as u32) != 0 && (*(*__slate_slot_24)).add_flags & ((1i32 << (*__slate_slot_28).wrapping_add(1)) as u32) == 0 {
xhci_free_endpoint_ring(*__slate_slot_26, *__slate_slot_25, *__slate_slot_28 as u32);
xhci_check_bw_drop_ep_streams(*__slate_slot_26, *__slate_slot_25, *__slate_slot_28);
}
let __v3: i32 = *__slate_slot_28;
*__slate_slot_28 = __v3.wrapping_add(1);
}
xhci_zero_in_ctx(*__slate_slot_26, *__slate_slot_25);
// Install any rings for completely new endpoints or changed endpoints,
// and free any old rings from changed endpoints.
*__slate_slot_28 = 1;
while *__slate_slot_28 < 31 {
if !unsafe { (*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_28 as isize)).new_ring }.is_null() {
// Only free the old ring if it exists.
// It may not if this is the first add of an endpoint.
if !unsafe { (*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_28 as isize)).ring }.is_null() {
xhci_free_endpoint_ring(*__slate_slot_26, *__slate_slot_25, *__slate_slot_28 as u32);
}
xhci_check_bw_drop_ep_streams(*__slate_slot_26, *__slate_slot_25, *__slate_slot_28);
(*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_28 as isize)).ring = (*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_28 as isize)).new_ring;
(*((*(*__slate_slot_25)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_28 as isize)).new_ring = core::ptr::null_mut::<xhci_ring>();
xhci_debugfs_create_endpoint(*__slate_slot_26, *__slate_slot_25, *__slate_slot_28);
}
let __v1: i32 = *__slate_slot_28;
*__slate_slot_28 = __v1.wrapping_add(1);
}
}
}
}
kfree((*(*__slate_slot_22)).completion as *const ());
kfree(*__slate_slot_22 as *const ());
return ret;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_check_bandwidth_945: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>, *mut ()>(Some(xhci_check_bandwidth)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_reset_bandwidth")]
extern "C" fn xhci_reset_bandwidth(hcd: *mut usb_hcd, udev: *mut usb_device) {
let ret: i32 = xhci_check_args(hcd, udev, core::ptr::null_mut::<usb_host_endpoint>(), 0i32, true, c"xhci_reset_bandwidth".as_ptr() as *const u8);
if ret <= 0 {
return;
}
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s called for udev %p\n".as_ptr() as *const u8, c"xhci_reset_bandwidth".as_ptr() as *mut u8, udev) };
}
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
// Free any rings allocated for added endpoints
for i in 0i32..31i32 {
if !unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).new_ring }.is_null() {
unsafe { xhci_debugfs_remove_endpoint(xhci, virt_dev, i) };
unsafe { xhci_ring_free(xhci, (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).new_ring) };
unsafe {
(*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).new_ring = core::ptr::null_mut::<xhci_ring>();
}
}
}
xhci_zero_in_ctx(xhci, virt_dev);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_reset_bandwidth_946: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>, *mut ()>(Some(xhci_reset_bandwidth)) };

/// Get the available bandwidth of the ports under the xhci roothub.
/// EIO means the command failed: command not implemented or unsupported
/// speed (TRB Error), some ASMedia complete with Parameter Error when
/// querying the root hub (slot_id = 0), or other error or timeout.
#[unsafe(no_mangle)]
extern "C" fn xhci_get_port_bandwidth(xhci: *mut xhci_hcd, ctx: *mut xhci_container_ctx, dev_speed: u8) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut xhci_command = core::ptr::addr_of_mut!(cmd) as *mut *mut xhci_command;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ctx.is_null() || xhci.is_null() {
return 22i32.wrapping_neg();
} else {
*__slate_slot_2 = xhci_alloc_command(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_2)).in_ctx = ctx;
// get xhci port bandwidth, refer to xhci rev1_2 protocol 4.6.15
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
*__slate_slot_0 = xhci_queue_get_port_bw(xhci, *__slate_slot_2, (*ctx).dma, dev_speed, false);
if *__slate_slot_0 != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
} else {
xhci_ring_cmd_db(xhci);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_1);
wait_for_completion((*(*__slate_slot_2)).completion);
if (*(*__slate_slot_2)).status != 1 {
*__slate_slot_0 = 5i32.wrapping_neg();
}
}
kfree((*(*__slate_slot_2)).completion as *const ());
kfree(*__slate_slot_2 as *const ());
return *__slate_slot_0;
}
}
}
}

fn xhci_setup_input_ctx_for_config_ep(xhci: *mut xhci_hcd, in_ctx: *mut xhci_container_ctx, out_ctx: *mut xhci_container_ctx, ctrl_ctx: *mut xhci_input_control_ctx, add_flags: u32, drop_flags: u32) {
unsafe {
(*ctrl_ctx).add_flags = add_flags;
}
unsafe {
(*ctrl_ctx).drop_flags = drop_flags;
}
unsafe { xhci_slot_copy(xhci, in_ctx, out_ctx) };
let __v1: u32 = unsafe { (*ctrl_ctx).add_flags };
let __v2: u32 = __v1 | (BIT!(0i32) as u32);
unsafe {
(*ctrl_ctx).add_flags = __v2;
}
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_endpoint_disable")]
extern "C" fn xhci_endpoint_disable(hcd: *mut usb_hcd, host_ep: *mut usb_host_endpoint) {
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut udev: core::mem::MaybeUninit<*mut usb_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut usb_device = core::ptr::addr_of_mut!(udev) as *mut *mut usb_device;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = hcd_to_xhci(hcd);
'__join_0: {
loop {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_5)).lock), *__slate_slot_1);
*__slate_slot_2 = (*host_ep).hcpriv as *mut usb_device;
if (*__slate_slot_2).is_null() || (*(*__slate_slot_2)).slot_id == 0 {
break '__join_0;
} else {
*__slate_slot_4 = *unsafe { (*(*__slate_slot_5)).devs }.wrapping_offset((*(*__slate_slot_2)).slot_id as isize);
if (*__slate_slot_4).is_null() {
break '__join_0;
} else {
*__slate_slot_0 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*host_ep).desc));
*__slate_slot_3 = ((*(*__slate_slot_4)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_0 as isize);
// wait for hub_tt_work to finish clearing hub TT
if ((*(*__slate_slot_3)).ep_state as u64) & BIT!(8i32) != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_5)).lock), *__slate_slot_1);
schedule_timeout_uninterruptible(1i64);
} else {
break;
}
}
}
}
if (*(*__slate_slot_3)).ep_state != 0 {
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_5)).lock), *__slate_slot_1);
}
}

// Called after usb core issues a clear halt control message.
// The host side of the halt should already be cleared by a reset endpoint
// command issued when the STALL event was received.
//
// The reset endpoint command may only be issued to endpoints in the halted
// state. For software that wishes to reset the data toggle or sequence number
// of an endpoint that isn't in the halted state this function will issue a
// configure endpoint command with the Drop and Add bits set for the target
// endpoint. Refer to the additional note in xhci spcification section 4.6.8.
//
// vdev may be lost due to xHC restore error and re-initialization during S3/S4
// resume. A new vdev will be allocated later by xhci_discover_or_reset_device()
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_endpoint_reset")]
extern "C" fn xhci_endpoint_reset(hcd: *mut usb_hcd, host_ep: *mut usb_host_endpoint) {
// Only interrupt and bulk ep's use data toggle, USB2 spec 5.5.4->
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ep_flag: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(ep_flag) as *mut u32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
let mut cfg_cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut xhci_command = core::ptr::addr_of_mut!(cfg_cmd) as *mut *mut xhci_command;
let mut stop_cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut xhci_command = core::ptr::addr_of_mut!(stop_cmd) as *mut *mut xhci_command;
let mut ctrl_ctx: core::mem::MaybeUninit<*mut xhci_input_control_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut xhci_input_control_ctx = core::ptr::addr_of_mut!(ctrl_ctx) as *mut *mut xhci_input_control_ctx;
let mut ep: core::mem::MaybeUninit<*mut xhci_virt_ep> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut xhci_virt_ep = core::ptr::addr_of_mut!(ep) as *mut *mut xhci_virt_ep;
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
let mut udev: core::mem::MaybeUninit<*mut usb_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut usb_device = core::ptr::addr_of_mut!(udev) as *mut *mut usb_device;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
unsafe {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_20 = hcd_to_xhci(hcd);
*__slate_slot_13 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*host_ep).desc));
// Usb core assumes a max packet value for ep0 on FS devices until the
// real value is read from the descriptor. Core resets Ep0 if values
// mismatch. Reconfigure the xhci ep0 endpoint context here in that case
if usb_endpoint_xfer_control(core::ptr::addr_of_mut!((*host_ep).desc) as *const usb_endpoint_descriptor) != 0 && *__slate_slot_13 == 0 {
*__slate_slot_19 = container_of__65!(host_ep, usb_device);
if ((*(*__slate_slot_19)).speed as u32) != 2 || (*(*__slate_slot_19)).slot_id == 0 {
return;
} else {
*__slate_slot_18 = *unsafe { (*(*__slate_slot_20)).devs }.wrapping_offset((*(*__slate_slot_19)).slot_id as isize);
if (*__slate_slot_18).is_null() || (*(*__slate_slot_18)).udev != *__slate_slot_19 {
return;
} else {
xhci_check_ep0_maxpacket(*__slate_slot_20, *__slate_slot_18);
// Nothing else should be done here for ep0 during ep reset
return;
}
}
} else {
if unsafe { (*host_ep).hcpriv }.is_null() {
return;
} else {
*__slate_slot_19 = (*host_ep).hcpriv as *mut usb_device;
*__slate_slot_18 = *unsafe { (*(*__slate_slot_20)).devs }.wrapping_offset((*(*__slate_slot_19)).slot_id as isize);
if (*(*__slate_slot_19)).slot_id == 0 || (*__slate_slot_18).is_null() {
return;
} else {
*__slate_slot_17 = ((*(*__slate_slot_18)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_13 as isize);
// Bail out if toggle is already being cleared by a endpoint reset
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
if ((*(*__slate_slot_17)).ep_state as u64) & BIT!(6i32) != 0 {
let __v9: *mut xhci_virt_ep = *__slate_slot_17;
let __v8: u32 = (*__v9).ep_state;
let __v7: u32 = ((__v8 as u64) & !BIT!(6i32)) as u32;
(*__v9).ep_state = __v7;
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
return;
} else {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
if usb_endpoint_xfer_control(core::ptr::addr_of_mut!((*host_ep).desc) as *const usb_endpoint_descriptor) != 0 {
__v6 = true;
} else {
__v6 = usb_endpoint_xfer_isoc(core::ptr::addr_of_mut!((*host_ep).desc) as *const usb_endpoint_descriptor) != 0;
}
if __v6 {
return;
} else {
*__slate_slot_11 = xhci_get_endpoint_flag(core::ptr::addr_of_mut!((*host_ep).desc));
if (*__slate_slot_11 as u64) == BIT!(0i32) || (*__slate_slot_11 as u64) == BIT!(1i32) {
return;
} else {
*__slate_slot_15 = xhci_alloc_command(*__slate_slot_20, true, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if (*__slate_slot_15).is_null() {
return;
} else {
*__slate_slot_14 = xhci_alloc_command_with_ctx(*__slate_slot_20, true, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if !(*__slate_slot_14).is_null() {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
// block queuing new trbs and ringing ep doorbell
let __v5: *mut xhci_virt_ep = *__slate_slot_17;
let __v4: u32 = (*__v5).ep_state;
let __v3: u32 = ((__v4 as u64) | BIT!(7i32)) as u32;
(*__v5).ep_state = __v3;
// Make sure endpoint ring is empty before resetting the toggle/seq.
// Driver is required to synchronously cancel all transfer request.
// Stop the endpoint to force xHC to update the output context
if list_empty(core::ptr::addr_of_mut!((*(*(*__slate_slot_17)).ring).td_list) as *const list_head) == 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*__slate_slot_19)).dev) as *const device, c"EP not empty, refuse reset\n".as_ptr() as *const u8);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
xhci_free_command(*__slate_slot_20, *__slate_slot_14);
} else {
*__slate_slot_10 = xhci_queue_stop_endpoint(*__slate_slot_20, *__slate_slot_15, (*(*__slate_slot_19)).slot_id, *__slate_slot_13, 0i32);
if *__slate_slot_10 < 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
xhci_free_command(*__slate_slot_20, *__slate_slot_14);
} else {
xhci_ring_cmd_db(*__slate_slot_20);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
wait_for_completion((*(*__slate_slot_15)).completion);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
// config ep command clears toggle if add and drop ep flags are set
*__slate_slot_16 = xhci_get_input_control_ctx((*(*__slate_slot_14)).in_ctx);
if (*__slate_slot_16).is_null() {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
xhci_free_command(*__slate_slot_20, *__slate_slot_14);
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_20)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_endpoint_reset".as_ptr() as *mut u8);
} else {
xhci_setup_input_ctx_for_config_ep(*__slate_slot_20, (*(*__slate_slot_14)).in_ctx, (*(*__slate_slot_18)).out_ctx, *__slate_slot_16, *__slate_slot_11, *__slate_slot_11);
xhci_endpoint_copy(*__slate_slot_20, (*(*__slate_slot_14)).in_ctx, (*(*__slate_slot_18)).out_ctx, *__slate_slot_13);
*__slate_slot_10 = xhci_queue_configure_endpoint(*__slate_slot_20, *__slate_slot_14, (*(*(*__slate_slot_14)).in_ctx).dma, (*(*__slate_slot_19)).slot_id as u32, false);
if *__slate_slot_10 < 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
xhci_free_command(*__slate_slot_20, *__slate_slot_14);
} else {
xhci_ring_cmd_db(*__slate_slot_20);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
wait_for_completion((*(*__slate_slot_14)).completion);
xhci_free_command(*__slate_slot_20, *__slate_slot_14);
}
}
}
}
}
xhci_free_command(*__slate_slot_20, *__slate_slot_15);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
if ((*(*__slate_slot_17)).ep_state as u64) & BIT!(7i32) != 0 {
let __v2: *mut xhci_virt_ep = *__slate_slot_17;
let __v1: u32 = (*__v2).ep_state;
let __v0: u32 = ((__v1 as u64) & !BIT!(7i32)) as u32;
(*__v2).ep_state = __v0;
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_20)).lock), *__slate_slot_12);
}
}
}
}
}
}
}
}
}

fn xhci_check_streams_endpoint(xhci: *mut xhci_hcd, udev: *mut usb_device, ep: *mut usb_host_endpoint, slot_id: u32) -> i32 {
let mut ret: i32 = 0;
let mut ep_index: u32 = 0;
let mut ep_state: u32 = 0;
if ep.is_null() {
return 22i32.wrapping_neg();
}
ret = xhci_check_args(xhci_to_hcd(xhci), udev, ep, 1i32, true, c"xhci_check_streams_endpoint".as_ptr() as *const u8);
if ret <= 0 {
return if ret != 0 { ret } else { 22i32.wrapping_neg() };
}
if usb_ss_max_streams((unsafe { core::ptr::addr_of_mut!((*ep).ss_ep_comp) }) as *const usb_ss_ep_comp_descriptor) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: SuperSpeed Endpoint Companion descriptor for ep 0x%x does not support streams\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as i32) };
return 22i32.wrapping_neg();
}
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
ep_state = unsafe { (*((*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ep_state };
if (ep_state as u64) & BIT!(4i32) != 0 || (ep_state as u64) & BIT!(3i32) != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN: SuperSpeed bulk endpoint 0x%x already has streams set up.\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as i32) };
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Send email to xHCI maintainer and ask for dynamic stream context array reallocation.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*(*((*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring).td_list) }) as *const list_head) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Cannot setup streams for SuperSpeed bulk endpoint 0x%x; URBs are pending.\n".as_ptr() as *const u8, (*ep).desc.bEndpointAddress as i32) };
return 22i32.wrapping_neg();
}
0
}

fn xhci_calculate_streams_entries(xhci: *mut xhci_hcd, num_streams: *mut u32, num_stream_ctxs: &mut u32) {
let mut max_streams: u32 = 0;
// The stream context array size must be a power of two
unsafe {
*(num_stream_ctxs as *mut u32) = roundup_pow_of_two!(unsafe { *num_streams }) as u32;
}
// Find out how many primary stream array entries the host controller
// supports.  Later we may use secondary stream arrays (similar to 2nd
// level page entries), but that's an optional feature for xHCI host
// controllers. xHCs must support at least 4 stream IDs.
let mut __v0: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_3511_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_947() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_948() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_3511_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) == 0 {
unsafe { __compiletime_assert_949() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_3511_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_3511_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_950() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_3511_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_3511_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_3511_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_3511_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_951() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*xhci).hcc_params }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_3511_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_3511_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32);
__v0 = __v2;
max_streams = (1i32 << __v0.wrapping_add(1)) as u32;
if (unsafe { *(num_stream_ctxs as *mut u32) }) > max_streams {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"xHCI HW only supports %u stream ctx entries.\n".as_ptr() as *const u8, max_streams) };
}
unsafe {
*(num_stream_ctxs as *mut u32) = max_streams;
}
unsafe {
*num_streams = max_streams;
}
}
}

/// Returns an error code if one of the endpoint already has streams.
/// This does not change any data structures, it only checks and gathers
/// information.
fn xhci_calculate_streams_and_bitmask(xhci: *mut xhci_hcd, udev: *mut usb_device, eps: *mut *mut usb_host_endpoint, num_eps: u32, num_streams: &mut u32, changed_ep_bitmask: &mut u32) -> i32 {
let mut max_streams: u32 = 0;
let mut endpoint_flag: u32 = 0;
let mut i: i32 = 0;
let mut ret: i32 = 0;
while (i as u32) < num_eps {
ret = xhci_check_streams_endpoint(xhci, udev, unsafe { *eps.wrapping_offset(i as isize) }, (unsafe { (*udev).slot_id }) as u32);
if ret < 0 {
return ret;
}
max_streams = usb_ss_max_streams((unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).ss_ep_comp) }) as *const usb_ss_ep_comp_descriptor) as u32;
if max_streams < unsafe { *(num_streams as *mut u32) }.wrapping_sub(1) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*xhci_to_hcd(xhci)).self_.controller as *const device, c"Ep 0x%x only supports %u stream IDs.\n".as_ptr() as *const u8, (*(*eps.wrapping_offset(i as isize))).desc.bEndpointAddress as i32, max_streams) };
}
unsafe {
*(num_streams as *mut u32) = max_streams.wrapping_add(1);
}
}
endpoint_flag = xhci_get_endpoint_flag(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).desc) });
if (unsafe { *(changed_ep_bitmask as *mut u32) }) & endpoint_flag != 0 {
return 22i32.wrapping_neg();
}
unsafe {
*changed_ep_bitmask = *changed_ep_bitmask | endpoint_flag;
}
i = i.wrapping_add(1);
}
0
}

fn xhci_calculate_no_streams_bitmask(xhci: *mut xhci_hcd, udev: *mut usb_device, eps: *mut *mut usb_host_endpoint, num_eps: u32) -> u32 {
let mut changed_ep_bitmask: u32 = 0;
let slot_id: u32 = (unsafe { (*udev).slot_id }) as u32;
let mut ep_index: u32 = 0;
let mut ep_state: u32 = 0;
let mut i: i32 = 0;
if unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) }.is_null() {
return 0;
}
while (i as u32) < num_eps {
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).desc) });
ep_state = unsafe { (*((*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ep_state };
// Are streams already being freed for the endpoint?
if (ep_state as u64) & BIT!(5i32) != 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Can't disable streams for endpoint 0x%x, streams are being disabled already\n".as_ptr() as *const u8, (*(*eps.wrapping_offset(i as isize))).desc.bEndpointAddress as i32) };
return 0;
}
// Are there actually any streams to free?
if (ep_state as u64) & BIT!(4i32) == 0 && (ep_state as u64) & BIT!(3i32) == 0 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN Can't disable streams for endpoint 0x%x, streams are already disabled!\n".as_ptr() as *const u8, (*(*eps.wrapping_offset(i as isize))).desc.bEndpointAddress as i32) };
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"WARN xhci_free_streams() called with non-streams endpoint\n".as_ptr() as *const u8) };
return 0;
}
let __v3: u32 = xhci_get_endpoint_flag(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).desc) });
changed_ep_bitmask |= __v3;
i = i.wrapping_add(1);
}
changed_ep_bitmask
}

// The USB device drivers use this function (through the HCD interface in USB
// core) to prepare a set of bulk endpoints to use streams.  Streams are used to
// coordinate mass storage command queueing across multiple endpoints (basically
// a stream ID == a task ID).
//
// Setting up streams involves allocating the same size stream context array
// for each endpoint and issuing a configure endpoint command for all endpoints.
//
// Don't allow the call to succeed if one endpoint only supports one stream
// (which means it doesn't support streams at all).
//
// Drivers may get less stream IDs than they asked for, if the host controller
// hardware or endpoints claim they can't support the number of requested
// stream IDs.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_alloc_streams")]
extern "C" fn xhci_alloc_streams(hcd: *mut usb_hcd, udev: *mut usb_device, eps: *mut *mut usb_host_endpoint, num_eps: u32, mut num_streams: u32, mem_flags: u32) -> i32 {
let mut ep_ctx: core::mem::MaybeUninit<*mut xhci_ep_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut xhci_ep_ctx = core::ptr::addr_of_mut!(ep_ctx) as *mut *mut xhci_ep_ctx;
// MaxPSASize value 0 (2 streams) means streams are not supported
let mut __v30: bool = unsafe { core::mem::zeroed() };
let mut changed_ep_bitmask: u32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut max_packet: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u32 = core::ptr::addr_of_mut!(max_packet) as *mut u32;
let mut num_stream_ctxs: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u32 = core::ptr::addr_of_mut!(num_stream_ctxs) as *mut u32;
let mut ep_index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u32 = core::ptr::addr_of_mut!(ep_index) as *mut u32;
let mut ctrl_ctx: core::mem::MaybeUninit<*mut xhci_input_control_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut xhci_input_control_ctx = core::ptr::addr_of_mut!(ctrl_ctx) as *mut *mut xhci_input_control_ctx;
let mut config_cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut *mut xhci_command = core::ptr::addr_of_mut!(config_cmd) as *mut *mut xhci_command;
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
changed_ep_bitmask = 0;
if eps.is_null() {
return 22i32.wrapping_neg();
} else {
// Add one to the number of streams requested to account for
// stream 0 that is reserved for xHCI usage.
num_streams = num_streams.wrapping_add(1);
*__slate_slot_41 = hcd_to_xhci(hcd);
if (*(*__slate_slot_41)).quirks & BIT_ULL!(19i32) != 0 {
__v30 = true;
} else {
while false {
}
while false {
}
while false {
}
loop {
if !((core::mem::size_of::<__SlateRecord_xhci_c_3641_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_3641_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & 0u64 != 0 {
__compiletime_assert_955();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_3641_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_3641_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_3641_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_3641_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_956();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
let __v28: u64 = (((*(*__slate_slot_41)).hcc_params as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_3641_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_3641_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 12i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)).trailing_zeros() as i32);
__v30 = 1i32 << __v28.wrapping_add(1) < 4;
}
if __v30 {
return 38i32.wrapping_neg();
} else {
*__slate_slot_39 = xhci_alloc_command_with_ctx(*__slate_slot_41, true, mem_flags);
if (*__slate_slot_39).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_38 = xhci_get_input_control_ctx((*(*__slate_slot_39)).in_ctx);
if (*__slate_slot_38).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_41)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_alloc_streams".as_ptr() as *mut u8);
xhci_free_command(*__slate_slot_41, *__slate_slot_39);
return 12i32.wrapping_neg();
} else {
// Check to make sure all endpoints are not already configured for
// streams.  While we're at it, find the maximum number of streams that
// all the endpoints will support and check for duplicate endpoints.
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
*__slate_slot_42 = xhci_calculate_streams_and_bitmask(*__slate_slot_41, udev, eps, num_eps, &mut (*core::ptr::addr_of_mut!(num_streams)), &mut (*core::ptr::addr_of_mut!(changed_ep_bitmask)));
if *__slate_slot_42 < 0 {
xhci_free_command(*__slate_slot_41, *__slate_slot_39);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
return *__slate_slot_42;
} else {
if num_streams <= 1 {
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_41)).self_.controller as *const device, c"WARN: endpoints can't handle more than one stream.\n".as_ptr() as *const u8);
xhci_free_command(*__slate_slot_41, *__slate_slot_39);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
return 22i32.wrapping_neg();
} else {
*__slate_slot_40 = *unsafe { (*(*__slate_slot_41)).devs }.wrapping_offset((*udev).slot_id as isize);
// Mark each endpoint as being in transition, so
// xhci_urb_enqueue() will reject all URBs.
*__slate_slot_43 = 0;
while (*__slate_slot_43 as u32) < num_eps {
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
let __v27: *mut xhci_virt_ep = ((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize);
let __v26: u32 = (*__v27).ep_state;
let __v25: u32 = ((__v26 as u64) | BIT!(3i32)) as u32;
(*__v27).ep_state = __v25;
let __v24: i32 = *__slate_slot_43;
*__slate_slot_43 = __v24.wrapping_add(1);
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
// Setup internal data structures and allocate HW data structures for
// streams (but don't install the HW structures in the input context
// until we're sure all memory allocation succeeded).
xhci_calculate_streams_entries(*__slate_slot_41, core::ptr::addr_of_mut!(num_streams), &mut (*core::ptr::addr_of_mut!(*__slate_slot_36)));
*__slate_slot_43 = 0;
'__join_3: {
while (*__slate_slot_43 as u32) < num_eps {
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
*__slate_slot_35 = usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc) as *const usb_endpoint_descriptor) as u32;
(*((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize)).stream_info = xhci_alloc_stream_info(*__slate_slot_41, *__slate_slot_36, num_streams, *__slate_slot_35, mem_flags);
if unsafe { (*((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize)).stream_info }.is_null() {
break '__join_3;
} else {
// Set maxPstreams in endpoint context and update deq ptr to
// point to stream context array. FIXME
let __v22: i32 = *__slate_slot_43;
*__slate_slot_43 = __v22.wrapping_add(1);
}
}
// Set up the input context for a configure endpoint command.
*__slate_slot_43 = 0;
while (*__slate_slot_43 as u32) < num_eps {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
*__slate_slot_20 = xhci_get_ep_ctx(*__slate_slot_41, (*(*__slate_slot_39)).in_ctx, *__slate_slot_37);
xhci_endpoint_copy(*__slate_slot_41, (*(*__slate_slot_39)).in_ctx, (*(*__slate_slot_40)).out_ctx, *__slate_slot_37);
xhci_setup_streams_ep_input_ctx(*__slate_slot_41, *__slate_slot_20, (*((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize)).stream_info);
let __v19: i32 = *__slate_slot_43;
*__slate_slot_43 = __v19.wrapping_add(1);
}
// Tell the HW to drop its old copy of the endpoint context info
// and add the updated copy from the input context.
xhci_setup_input_ctx_for_config_ep(*__slate_slot_41, (*(*__slate_slot_39)).in_ctx, (*(*__slate_slot_40)).out_ctx, *__slate_slot_38, changed_ep_bitmask, changed_ep_bitmask);
// Issue and wait for the configure endpoint command
*__slate_slot_42 = xhci_configure_endpoint(*__slate_slot_41, udev, *__slate_slot_39, false, false);
// xHC rejected the configure endpoint command for some reason, so we
// leave the old ring intact and free our internal streams data
// structure.
if *__slate_slot_42 >= 0 {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
*__slate_slot_43 = 0;
while (*__slate_slot_43 as u32) < num_eps {
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
let __v17: *mut xhci_virt_ep = ((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize);
let __v16: u32 = (*__v17).ep_state;
let __v15: u32 = ((__v16 as u64) & !BIT!(3i32)) as u32;
(*__v17).ep_state = __v15;
let __v14: *mut xhci_virt_ep = ((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize);
let __v13: u32 = (*__v14).ep_state;
let __v12: u32 = ((__v13 as u64) | BIT!(4i32)) as u32;
(*__v14).ep_state = __v12;
let __v11: i32 = *__slate_slot_43;
*__slate_slot_43 = __v11.wrapping_add(1);
}
xhci_free_command(*__slate_slot_41, *__slate_slot_39);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_41)).lock), *__slate_slot_34);
*__slate_slot_43 = 0;
while (*__slate_slot_43 as u32) < num_eps {
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
xhci_debugfs_create_stream_files(*__slate_slot_41, *__slate_slot_40, *__slate_slot_37 as i32);
let __v9: i32 = *__slate_slot_43;
*__slate_slot_43 = __v9.wrapping_add(1);
}
// Subtract 1 for stream 0, which drivers can't use
return num_streams.wrapping_sub(1) as i32;
}
}
*__slate_slot_43 = 0;
while (*__slate_slot_43 as u32) < num_eps {
*__slate_slot_37 = xhci_get_endpoint_index(core::ptr::addr_of_mut!((*(*eps.wrapping_offset(*__slate_slot_43 as isize))).desc));
xhci_free_stream_info(*__slate_slot_41, (*((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize)).stream_info);
(*((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize)).stream_info = core::ptr::null_mut::<xhci_stream_info>();
// FIXME Unset maxPstreams in endpoint context and
// update deq ptr to point to normal string ring.
let __v7: *mut xhci_virt_ep = ((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize);
let __v6: u32 = (*__v7).ep_state;
let __v5: u32 = ((__v6 as u64) & !BIT!(3i32)) as u32;
(*__v7).ep_state = __v5;
let __v4: *mut xhci_virt_ep = ((*(*__slate_slot_40)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_37 as isize);
let __v3: u32 = (*__v4).ep_state;
let __v2: u32 = ((__v3 as u64) & !BIT!(4i32)) as u32;
(*__v4).ep_state = __v2;
xhci_endpoint_zero(*__slate_slot_41, *__slate_slot_40, *eps.wrapping_offset(*__slate_slot_43 as isize));
let __v1: i32 = *__slate_slot_43;
*__slate_slot_43 = __v1.wrapping_add(1);
}
// If it didn't work, free the streams!
xhci_free_command(*__slate_slot_41, *__slate_slot_39);
return 12i32.wrapping_neg();
}
}
}
}
}
}
}
}

/// Transition the endpoint from using streams to being a "normal" endpoint
/// without streams.
///
/// Modify the endpoint context state, submit a configure endpoint command,
/// and free all endpoint rings for streams if that completes successfully.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_free_streams")]
extern "C" fn xhci_free_streams(hcd: *mut usb_hcd, udev: *mut usb_device, eps: *mut *mut usb_host_endpoint, num_eps: u32, mem_flags: u32) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
let mut stream_info: __SlateAlign16<[*mut xhci_stream_info; 31]> = __SlateAlign16([0 as *mut xhci_stream_info; 31]);
let mut ep_index: u32 = 0;
let mut flags: u64 = 0;
let mut changed_ep_bitmask: u32 = 0;
xhci = hcd_to_xhci(hcd);
let vdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
// Set up a configure endpoint command to remove the streams rings
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
changed_ep_bitmask = xhci_calculate_no_streams_bitmask(xhci, udev, eps, num_eps);
if changed_ep_bitmask == 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
return 22i32.wrapping_neg();
}
// Use the xhci_command structure from the first endpoint.  We may have
// allocated too many, but the driver may call xhci_free_streams() for
// each endpoint it grouped into one call to xhci_alloc_streams().
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(0 as isize))).desc) });
let command: *mut xhci_command = unsafe { (*(*((*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).stream_info).free_streams_command };
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*command).in_ctx) };
if ctrl_ctx.is_null() {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_free_streams".as_ptr() as *mut u8) };
return 22i32.wrapping_neg();
}
while (i as u32) < num_eps {
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).desc) });
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*command).in_ctx, ep_index) };
let __v6: *mut xhci_virt_ep = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
let __v7: u32 = unsafe { (*__v6).ep_state };
let __v8: u32 = ((__v7 as u64) | BIT!(5i32)) as u32;
unsafe {
(*__v6).ep_state = __v8;
}
unsafe { xhci_endpoint_copy(xhci, (*command).in_ctx, (*vdev).out_ctx, ep_index) };
unsafe { xhci_setup_no_streams_ep_input_ctx(ep_ctx, ((*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)) };
i = i.wrapping_add(1);
}
xhci_setup_input_ctx_for_config_ep(xhci, unsafe { (*command).in_ctx }, unsafe { (*vdev).out_ctx }, ctrl_ctx, changed_ep_bitmask, changed_ep_bitmask);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// Issue and wait for the configure endpoint command,
// which must succeed.
ret = xhci_configure_endpoint(xhci, udev, command, false, true);
// xHC rejected the configure endpoint command for some reason, so we
// leave the streams rings intact.
if ret < 0 {
return ret;
}
// dma_free_coherent() called by xhci_free_stream_info() may sleep,
// so save stream_info pointers and clear references under lock,
// then free the memory outside lock.
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
i = 0;
while (i as u32) < num_eps {
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*(*eps.wrapping_offset(i as isize))).desc) });
stream_info.0[i as usize] = unsafe { (*((*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).stream_info };
unsafe {
(*((*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).stream_info = core::ptr::null_mut::<xhci_stream_info>();
}
// FIXME Unset maxPstreams in endpoint context and
// update deq ptr to point to normal string ring.
let __v17: *mut xhci_virt_ep = unsafe { (*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
let __v18: u32 = unsafe { (*__v17).ep_state };
let __v19: u32 = ((__v18 as u64) & !BIT!(5i32)) as u32;
unsafe {
(*__v17).ep_state = __v19;
}
let __v20: *mut xhci_virt_ep = unsafe { (*vdev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
let __v21: u32 = unsafe { (*__v20).ep_state };
let __v22: u32 = ((__v21 as u64) & !BIT!(4i32)) as u32;
unsafe {
(*__v20).ep_state = __v22;
}
i = i.wrapping_add(1);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
i = 0;
while (i as u32) < num_eps {
unsafe { xhci_free_stream_info(xhci, stream_info.0[i as usize]) };
i = i.wrapping_add(1);
}
0
}

/// Deletes endpoint resources for endpoints that were active before a Reset
/// Device command, or a Disable Slot command.  The Reset Device command leaves
/// the control endpoint intact, whereas the Disable Slot command deletes it.
///
/// Must be called with xhci->lock held.
#[unsafe(no_mangle)]
extern "C" fn xhci_free_device_endpoint_resources(xhci: *mut xhci_hcd, virt_dev: *mut xhci_virt_device, drop_control_ep: bool) {
let mut i: i32 = if drop_control_ep { 0 } else { 1 };
let mut num_dropped_eps: u32 = 0;
let mut drop_flags: u32 = 0;
while i < 31 {
if !unsafe { (*((*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(i as isize)).ring }.is_null() {
drop_flags |= (1i32 << i) as u32;
num_dropped_eps = num_dropped_eps.wrapping_add(1);
}
i = i.wrapping_add(1);
}
let __v8: u32 = unsafe { (*xhci).num_active_eps };
unsafe {
(*xhci).num_active_eps = __v8.wrapping_sub(num_dropped_eps);
}
if num_dropped_eps != 0 {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Dropped %u ep ctxs, flags = 0x%x, %u now active.".as_ptr() as *const u8, num_dropped_eps, drop_flags, (*xhci).num_active_eps) };
}
}

/// This submits a Reset Device Command, which will set the device state to 0,
/// set the device address to 0, and disable all the endpoints except the default
/// control endpoint.  The USB core should come back and call
/// xhci_address_device(), and then re-set up the configuration.  If this is
/// called because of a usb_reset_and_verify_device(), then the old alternate
/// settings will be re-installed through the normal bandwidth allocation
/// functions.
///
/// Wait for the Reset Device command to finish.  Remove all structures
/// associated with the endpoints that were disabled.  Clear the input device
/// structure? Reset the control endpoint 0 max packet size?
///
/// If the virt_dev to be reset does not exist or does not match the udev,
/// it means the device is lost, possibly due to the xHC restore error and
/// re-initialization during S3/S4. In this case, call xhci_alloc_dev() to
/// re-allocate the device.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_discover_or_reset_device")]
extern "C" fn xhci_discover_or_reset_device(hcd: *mut usb_hcd, udev: *mut usb_device) -> i32 {
let mut ep: *mut xhci_virt_ep = unsafe { core::mem::zeroed() };
let mut old_active_eps: i32 = unsafe { core::mem::zeroed() };
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
let mut reset_device_cmd: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut xhci_command = core::ptr::addr_of_mut!(reset_device_cmd) as *mut *mut xhci_command;
let mut virt_dev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(virt_dev) as *mut *mut xhci_virt_device;
let mut slot_id: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(slot_id) as *mut u32;
let mut xhci: core::mem::MaybeUninit<*mut xhci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut xhci_hcd = core::ptr::addr_of_mut!(xhci) as *mut *mut xhci_hcd;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
old_active_eps = 0;
*__slate_slot_19 = xhci_check_args(hcd, udev, core::ptr::null_mut::<usb_host_endpoint>(), 0i32, false, c"xhci_discover_or_reset_device".as_ptr() as *const u8);
if *__slate_slot_19 <= 0 {
return *__slate_slot_19;
} else {
*__slate_slot_16 = hcd_to_xhci(hcd);
*__slate_slot_15 = (*udev).slot_id as u32;
*__slate_slot_14 = *unsafe { (*(*__slate_slot_16)).devs }.wrapping_offset(*__slate_slot_15 as isize);
if (*__slate_slot_14).is_null() {
*__slate_slot_19 = xhci_alloc_dev(hcd, udev);
if *__slate_slot_19 == 1 {
return 0;
} else {
return 22i32.wrapping_neg();
}
} else {
if !unsafe { (*(*__slate_slot_14)).tt_info }.is_null() {
old_active_eps = (*(*(*__slate_slot_14)).tt_info).active_eps;
}
if (*(*__slate_slot_14)).udev != udev {
// If the virt_dev and the udev does not match, this virt_dev
// may belong to another udev.
// Re-allocate the device.
*__slate_slot_19 = xhci_alloc_dev(hcd, udev);
if *__slate_slot_19 == 1 {
return 0;
} else {
return 22i32.wrapping_neg();
}
} else {
// If device is not setup, there is no point in resetting it
*__slate_slot_12 = xhci_get_slot_ctx(*__slate_slot_16, (*(*__slate_slot_14)).out_ctx);
if GET_SLOT_STATE!(unsafe { (*(*__slate_slot_12)).dev_state }) == 0 {
return 0;
} else {
if (*(*__slate_slot_16)).quirks & BIT_ULL!(49i32) != 0 {
// Obtaining a new device slot to inform the xHCI host that
// the USB device has been reset.
*__slate_slot_19 = xhci_disable_and_free_slot(*__slate_slot_16, (*udev).slot_id as u32);
if *__slate_slot_19 == 0 {
*__slate_slot_19 = xhci_alloc_dev(hcd, udev);
if *__slate_slot_19 == 1 {
*__slate_slot_19 = 0;
} else {
*__slate_slot_19 = 22i32.wrapping_neg();
}
}
return *__slate_slot_19;
} else {
trace_xhci_discover_or_reset_device(*__slate_slot_12);
// Allocate the command structure that holds the struct completion.
// Assume we're in process context, since the normal device reset
// process has to wait for the device anyway.  Storage devices are
// reset as part of error handling, so use GFP_NOIO instead of
// GFP_KERNEL.
*__slate_slot_13 = xhci_alloc_command(*__slate_slot_16, true, (BIT!(10i32) | BIT!(11i32)) as u32);
if (*__slate_slot_13).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_3: {
// Attempt to submit the Reset Device command to the command ring
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_17);
*__slate_slot_19 = xhci_queue_reset_device(*__slate_slot_16, *__slate_slot_13, *__slate_slot_15);
if *__slate_slot_19 != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_17);
} else {
'__join_24: {
'__join_22: {
xhci_ring_cmd_db(*__slate_slot_16);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_17);
// Wait for the Reset Device command to finish
wait_for_completion((*(*__slate_slot_13)).completion);
// The Reset Device command can't fail, according to the 0.95/0.96 spec,
// unless we tried to reset a slot ID that wasn't enabled,
// or the device wasn't in the addressed or configured state.
*__slate_slot_19 = (*(*__slate_slot_13)).status as i32;
let __t0: i32 = *__slate_slot_19;
if __t0 == 25 {
break '__join_24;
} else {
if __t0 == 24 {
break '__join_24;
} else {
if __t0 == 11 {
break '__join_22;
} else {
if __t0 == 19 {
break '__join_22;
} else {
if __t0 != 1 {
if xhci_is_vendor_info_code(*__slate_slot_16, *__slate_slot_19 as u32) == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_16)).self_.controller as *const device, c"Unknown completion code %u for reset device command.\n".as_ptr() as *const u8, *__slate_slot_19);
*__slate_slot_19 = 22i32.wrapping_neg();
break '__join_3;
}
}
}
}
}
}
// Free up host controller endpoint resources
if (*(*__slate_slot_16)).quirks & BIT_ULL!(5i32) != 0 {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_17);
// Don't delete the default control endpoint resources
xhci_free_device_endpoint_resources(*__slate_slot_16, *__slate_slot_14, false);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock), *__slate_slot_17);
}
// Everything but endpoint 0 is disabled, so free the rings.
*__slate_slot_18 = 1;
while *__slate_slot_18 < 31 {
ep = ((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize);
if ((*ep).ep_state as u64) & BIT!(4i32) != 0 {
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_16)).self_.controller as *const device, c"WARN: endpoint 0x%02x has streams on device reset, freeing streams.\n".as_ptr() as *const u8, xhci_get_endpoint_address(*__slate_slot_18 as u32));
xhci_free_stream_info(*__slate_slot_16, (*ep).stream_info);
(*ep).stream_info = core::ptr::null_mut::<xhci_stream_info>();
let __v8: u32 = (*ep).ep_state;
let __v7: u32 = ((__v8 as u64) & !BIT!(4i32)) as u32;
(*ep).ep_state = __v7;
}
if !unsafe { (*ep).ring }.is_null() {
if !unsafe { (*ep).sideband }.is_null() {
xhci_sideband_notify_ep_ring_free((*ep).sideband, *__slate_slot_18 as u32);
}
xhci_debugfs_remove_endpoint(*__slate_slot_16, *__slate_slot_14, *__slate_slot_18);
xhci_free_endpoint_ring(*__slate_slot_16, *__slate_slot_14, *__slate_slot_18 as u32);
}
if list_empty(core::ptr::addr_of_mut!((*((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize)).bw_endpoint_list) as *const list_head) == 0 {
xhci_drop_ep_from_interval_table(*__slate_slot_16, core::ptr::addr_of_mut!((*((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize)).bw_info), (*(*__slate_slot_14)).bw_table, udev, ((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize), (*(*__slate_slot_14)).tt_info);
}
xhci_clear_endpoint_bw_info(core::ptr::addr_of_mut!((*((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize)).bw_info));
let __v6: i32 = *__slate_slot_18;
*__slate_slot_18 = __v6.wrapping_add(1);
}
// If necessary, update the number of active TTs on this root port
xhci_update_tt_active_eps(*__slate_slot_16, *__slate_slot_14, old_active_eps);
*__slate_slot_19 = 0;
break '__join_3;
}
// 0.95 completion for bad slot ID
// 0.96 completion code for same thing
// Don't treat this as an error.  May change my mind later.
*__slate_slot_19 = 0;
break '__join_3;
}
while false {
}
_dev_warn((*xhci_to_hcd(*__slate_slot_16)).self_.controller as *const device, c"Timeout waiting for reset device command\n".as_ptr() as *const u8);
*__slate_slot_19 = 62i32.wrapping_neg();
}
}
*__slate_slot_18 = 0;
while *__slate_slot_18 < 31 {
let __v4: *mut xhci_virt_ep = ((*(*__slate_slot_14)).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(*__slate_slot_18 as isize);
let __v3: u32 = (*__v4).ep_state;
let __v2: u32 = ((__v3 as u64) & !BIT!(9i32)) as u32;
(*__v4).ep_state = __v2;
let __v1: i32 = *__slate_slot_18;
*__slate_slot_18 = __v1.wrapping_add(1);
}
xhci_free_command(*__slate_slot_16, *__slate_slot_13);
return *__slate_slot_19;
}
}
}
}
}
}
}
}

/// At this point, the struct usb_device is about to go away, the device has
/// disconnected, and all traffic has been stopped and the endpoints have been
/// disabled.  Free any HC data structures associated with that device.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_free_dev")]
extern "C" fn xhci_free_dev(hcd: *mut usb_hcd, udev: *mut usb_device) {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut flags: u64 = 0;
let mut ret: i32 = 0;
// We called pm_runtime_get_noresume when the device was attached.
// Decrement the counter here to allow controller to runtime suspend
// if no devices remain.
if (unsafe { (*xhci).quirks }) & BIT_ULL!(7i32) != 0 {
pm_runtime_put_noidle(unsafe { (*hcd).self_.controller });
}
ret = xhci_check_args(hcd, udev, core::ptr::null_mut::<usb_host_endpoint>(), 0i32, true, c"xhci_free_dev".as_ptr() as *const u8);
// If the host is halted due to driver unload, we still need to free the
// device.
if ret <= 0 && ret != 19i32.wrapping_neg() {
return;
}
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*virt_dev).out_ctx) };
trace_xhci_free_dev(slot_ctx);
// Stop any wayward timer functions (which may grab the lock)
for i in 0i32..31i32 {
let __v0: *mut xhci_virt_ep = unsafe { (*virt_dev).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(i as isize);
let __v1: u32 = unsafe { (*__v0).ep_state };
let __v2: u32 = ((__v1 as u64) & !BIT!(2i32)) as u32;
unsafe {
(*__v0).ep_state = __v2;
}
}
unsafe {
(*virt_dev).udev = core::ptr::null_mut::<usb_device>();
}
xhci_disable_slot(xhci, (unsafe { (*udev).slot_id }) as u32);
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_free_virt_device(xhci, virt_dev, (*udev).slot_id) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
}

#[unsafe(no_mangle)]
extern "C" fn xhci_disable_slot(xhci: *mut xhci_hcd, slot_id: u32) -> i32 {
let mut flags: u64 = 0;
let mut state: u32 = 0;
let mut ret: i32 = 0;
let command: *mut xhci_command = unsafe { xhci_alloc_command(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if command.is_null() {
return 12i32.wrapping_neg();
}
unsafe { xhci_debugfs_remove_slot(xhci, slot_id as i32) };
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// Don't disable the slot if the host controller is dead.
state = readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).op_regs).status) }) as *const ());
if state == 4294967295 || ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(0i32) != 0 || ((unsafe { (*xhci).xhc_state }) as u64) & BIT!(1i32) != 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_free_command(xhci, command) };
return 19i32.wrapping_neg();
}
ret = unsafe { xhci_queue_slot_control(xhci, command, 10u32, slot_id) };
if ret != 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_free_command(xhci, command) };
return ret;
}
unsafe { xhci_ring_cmd_db(xhci) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { wait_for_completion((*command).completion) };
if (unsafe { (*command).status }) != 1 {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Unsuccessful disable slot %u command, status %d\n".as_ptr() as *const u8, slot_id, (*command).status) };
}
unsafe { xhci_free_command(xhci, command) };
0
}

#[unsafe(no_mangle)]
extern "C" fn xhci_disable_and_free_slot(xhci: *mut xhci_hcd, slot_id: u32) -> i32 {
let vdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
let ret: i32 = xhci_disable_slot(xhci, slot_id);
unsafe { xhci_free_virt_device(xhci, vdev, slot_id as i32) };
ret
}

/// Checks if we have enough host controller resources for the default control
/// endpoint.
///
/// Must be called with xhci->lock held.
fn xhci_reserve_host_control_ep_resources(xhci: *mut xhci_hcd) -> i32 {
if unsafe { (*xhci).num_active_eps }.wrapping_add(1) > unsafe { (*xhci).limit_active_eps } {
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Not enough ep ctxs: %u active, need to add 1, limit is %u.".as_ptr() as *const u8, (*xhci).num_active_eps, (*xhci).limit_active_eps) };
return 12i32.wrapping_neg();
}
let __v1: u32 = unsafe { (*xhci).num_active_eps };
unsafe {
(*xhci).num_active_eps = __v1.wrapping_add(1);
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_quirks), c"Adding 1 ep ctx, %u now active.".as_ptr() as *const u8, (*xhci).num_active_eps) };
0
}

/// Returns 0 if the xHC ran out of device slots, the Enable Slot command
/// timed out, or allocating memory failed.  Returns 1 on success.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_alloc_dev")]
extern "C" fn xhci_alloc_dev(hcd: *mut usb_hcd, udev: *mut usb_device) -> i32 {
let mut command: core::mem::MaybeUninit<*mut xhci_command> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut xhci_command = core::ptr::addr_of_mut!(command) as *mut *mut xhci_command;
let mut slot_id: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(slot_id) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
let mut vdev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(vdev) as *mut *mut xhci_virt_device;
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
unsafe {
xhci = hcd_to_xhci(hcd);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = xhci_alloc_command(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_0).is_null() {
return 0;
} else {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
*__slate_slot_2 = xhci_queue_slot_control(xhci, *__slate_slot_0, 9u32, 0u32);
if *__slate_slot_2 != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
xhci_free_command(xhci, *__slate_slot_0);
return 0;
} else {
xhci_ring_cmd_db(xhci);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
wait_for_completion((*(*__slate_slot_0)).completion);
*__slate_slot_1 = (*(*__slate_slot_0)).slot_id;
if *__slate_slot_1 == 0 || (*(*__slate_slot_0)).status != 1 {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Error while assigning device slot ID: %s\n".as_ptr() as *const u8, xhci_trb_comp_code_string((*(*__slate_slot_0)).status as u8));
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Max number of devices this xHCI host supports is %u.\n".as_ptr() as *const u8, (*xhci).max_slots as i32);
xhci_free_command(xhci, *__slate_slot_0);
return 0;
} else {
'__join_0: {
xhci_free_command(xhci, *__slate_slot_0);
if (*xhci).quirks & BIT_ULL!(5i32) != 0 {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
*__slate_slot_2 = xhci_reserve_host_control_ep_resources(xhci);
if *__slate_slot_2 != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Not enough host resources, active endpoint contexts = %u\n".as_ptr() as *const u8, (*xhci).num_active_eps);
break '__join_0;
} else {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_3);
}
}
// Use GFP_NOIO, since this function can be called from
// xhci_discover_or_reset_device(), which may be called as part of
// mass storage driver error handling.
if xhci_alloc_virt_device(xhci, *__slate_slot_1, udev, (BIT!(10i32) | BIT!(11i32)) as u32) == 0 {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Could not allocate xHCI USB device data structures\n".as_ptr() as *const u8);
} else {
*__slate_slot_5 = *unsafe { (*xhci).devs }.wrapping_offset(*__slate_slot_1 as isize);
*__slate_slot_4 = xhci_get_slot_ctx(xhci, (*(*__slate_slot_5)).out_ctx);
trace_xhci_alloc_dev(*__slate_slot_4);
(*udev).slot_id = *__slate_slot_1;
xhci_debugfs_create_slot(xhci, *__slate_slot_1);
// If resetting upon resume, we can't put the controller into runtime
// suspend if there is a device attached.
if (*xhci).quirks & BIT_ULL!(7i32) != 0 {
pm_runtime_get_noresume((*hcd).self_.controller);
}
// Is this a LS or FS device under a HS hub?
//
// Hub or peripherial?
return 1;
}
}
xhci_disable_and_free_slot(xhci, (*udev).slot_id as u32);
return 0;
}
}
}
}
}

/// xhci_setup_device - issues an Address Device command to assign a unique
/// 			USB bus address.
/// @hcd: USB host controller data structure.
/// @udev: USB dev structure representing the connected device.
/// @setup: Enum specifying setup mode: address only or with context.
/// @timeout_ms: Max wait time (ms) for the command operation to complete.
///
/// Return: 0 if successful; otherwise, negative error code.
fn xhci_setup_device(hcd: *mut usb_hcd, udev: *mut usb_device, setup: u32, timeout_ms: u32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut command: *mut xhci_command = unsafe { core::mem::zeroed() };
let mut temp_64: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(temp_64) as *mut u64;
let mut ctrl_ctx: core::mem::MaybeUninit<*mut xhci_input_control_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut xhci_input_control_ctx = core::ptr::addr_of_mut!(ctrl_ctx) as *mut *mut xhci_input_control_ctx;
let mut slot_ctx: core::mem::MaybeUninit<*mut xhci_slot_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut xhci_slot_ctx = core::ptr::addr_of_mut!(slot_ctx) as *mut *mut xhci_slot_ctx;
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut virt_dev: core::mem::MaybeUninit<*mut xhci_virt_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut xhci_virt_device = core::ptr::addr_of_mut!(virt_dev) as *mut *mut xhci_virt_device;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut act: *const u8 = unsafe { core::mem::zeroed() };
unsafe {
'__join_2: {
act = (if (setup as u32) == 0 { c"context".as_ptr() as *mut u8 } else { c"address".as_ptr() as *mut u8 }) as *const u8;
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 0;
xhci = hcd_to_xhci(hcd);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
command = core::ptr::null_mut::<xhci_command>();
mutex_lock(core::ptr::addr_of_mut!((*xhci).mutex));
if (*xhci).xhc_state != 0 {
// dying, removing or halted
ret = 108i32.wrapping_neg();
} else {
if (*udev).slot_id == 0 {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Bad Slot ID %d".as_ptr() as *const u8, (*udev).slot_id);
ret = 22i32.wrapping_neg();
} else {
*__slate_slot_9 = *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize);
__ret_warn_on = (*__slate_slot_9).is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("957: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 957b - ., 3; .popsection\n# {0}", const 957 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xhci_setup_device_17, sym __slate_asm_string_xhci_setup_device_18, const 4347 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("958: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 958b - ., 4; .popsection\n# {0}", const 958 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
// In plug/unplug torture test with an NEC controller,
// a zero-dereference was observed once due to virt_dev = 0.
// Print useful debug rather than crash if it is observed again!
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Virt dev invalid for slot_id 0x%x!\n".as_ptr() as *const u8, (*udev).slot_id);
ret = 22i32.wrapping_neg();
} else {
*__slate_slot_6 = xhci_get_slot_ctx(xhci, (*(*__slate_slot_9)).out_ctx);
trace_xhci_setup_device_slot(*__slate_slot_6);
if (setup as u32) == 0 {
if GET_SLOT_STATE!(unsafe { (*(*__slate_slot_6)).dev_state }) == 1 {
break '__join_2;
}
}
command = xhci_alloc_command(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if command.is_null() {
ret = 12i32.wrapping_neg();
} else {
(*command).in_ctx = (*(*__slate_slot_9)).in_ctx;
(*command).timeout_ms = timeout_ms;
*__slate_slot_6 = xhci_get_slot_ctx(xhci, (*(*__slate_slot_9)).in_ctx);
*__slate_slot_5 = xhci_get_input_control_ctx((*(*__slate_slot_9)).in_ctx);
if (*__slate_slot_5).is_null() {
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_setup_device".as_ptr() as *mut u8);
ret = 22i32.wrapping_neg();
} else {
// If this is the first Set Address since device plug-in or
// virt_device realloaction after a resume with an xHCI power loss,
// then set up the slot context.
if (*(*__slate_slot_6)).dev_info == 0 {
xhci_setup_addressable_virt_dev(xhci, udev);
} else {
xhci_copy_ep0_dequeue_into_input_ctx(xhci, udev);
}
// Otherwise, update the control endpoint ring enqueue pointer.
(*(*__slate_slot_5)).add_flags = (BIT!(0i32) | BIT!(1i32)) as u32;
(*(*__slate_slot_5)).drop_flags = 0;
trace_xhci_address_ctx(xhci, (*(*__slate_slot_9)).in_ctx);
trace_xhci_address_ctrl_ctx(*__slate_slot_5);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_10);
trace_xhci_setup_device(*__slate_slot_9);
ret = xhci_queue_address_device(xhci, command, (*(*(*__slate_slot_9)).in_ctx).dma, (*udev).slot_id as u32, setup);
if ret != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_10);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"FIXME: allocate a command ring segment".as_ptr() as *const u8);
} else {
'__join_4: {
'__join_18: {
xhci_ring_cmd_db(xhci);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*xhci).lock), *__slate_slot_10);
// ctrl tx can take up to 5 sec; XXX: need more time for xHC?
wait_for_completion((*command).completion);
// FIXME: From section 4.3.4: "Software shall be responsible for timing
// the SetAddress() "recovery interval" required by USB and aborting the
// command on a timeout.
let __t0: u32 = (*command).status;
if __t0 == 25 {
break '__join_18;
} else {
if __t0 == 24 {
break '__join_18;
} else {
if __t0 != 19 {
if __t0 != 11 {
if __t0 == 4 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Device not responding to setup %s.\n".as_ptr() as *const u8, act);
mutex_unlock(core::ptr::addr_of_mut!((*xhci).mutex));
ret = xhci_disable_and_free_slot(xhci, (*udev).slot_id as u32);
if ret == 0 {
if xhci_alloc_dev(hcd, udev) == 1 {
xhci_setup_addressable_virt_dev(xhci, udev);
}
}
kfree((*command).completion as *const ());
kfree(command as *const ());
return 71i32.wrapping_neg();
} else {
if __t0 == 22 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"ERROR: Incompatible device for setup %s command\n".as_ptr() as *const u8, act);
ret = 19i32.wrapping_neg();
break '__join_4;
} else {
if __t0 == 1 {
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Successful setup %s command".as_ptr() as *const u8, act);
break '__join_4;
} else {
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"ERROR: unexpected setup %s command completion code 0x%x.\n".as_ptr() as *const u8, act, (*command).status);
trace_xhci_address_ctx(xhci, (*(*__slate_slot_9)).out_ctx);
ret = 22i32.wrapping_neg();
break '__join_4;
}
}
}
}
}
}
}
while false {
}
_dev_err((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Setup ERROR: setup %s command for slot %d.\n".as_ptr() as *const u8, act, (*udev).slot_id);
ret = 22i32.wrapping_neg();
break '__join_4;
}
while false {
}
_dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Timeout while waiting for setup device command\n".as_ptr() as *const u8);
ret = 62i32.wrapping_neg();
}
if ret == 0 {
*__slate_slot_4 = xhci_read_64(xhci as *const xhci_hcd, core::ptr::addr_of_mut!((*(*xhci).op_regs).dcbaa_ptr));
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Op regs DCBAA ptr = %#016llx".as_ptr() as *const u8, *__slate_slot_4);
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Slot ID %d dcbaa entry @%p = %#016llx".as_ptr() as *const u8, (*udev).slot_id, unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset((*udev).slot_id as isize), *unsafe { (*xhci).dcbaa.ctx_array }.wrapping_offset((*udev).slot_id as isize));
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Output Context DMA address = %#08llx".as_ptr() as *const u8, (*(*(*__slate_slot_9)).out_ctx).dma);
trace_xhci_address_ctx(xhci, (*(*__slate_slot_9)).in_ctx);
// USB core uses address 1 for the roothubs, so we add one to the
// address given back to us by the HC.
trace_xhci_address_ctx(xhci, (*(*__slate_slot_9)).out_ctx);
// Zero the input context control for later use
(*(*__slate_slot_5)).add_flags = 0;
(*(*__slate_slot_5)).drop_flags = 0;
*__slate_slot_6 = xhci_get_slot_ctx(xhci, (*(*__slate_slot_9)).out_ctx);
(*udev).devaddr = ((*(*__slate_slot_6)).dev_state & 255) as u8;
xhci_dbg_trace(xhci, Some(trace_xhci_dbg_address), c"Internal device address = %d".as_ptr() as *const u8, (*(*__slate_slot_6)).dev_state & 255);
}
}
}
}
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*xhci).mutex));
if !command.is_null() {
kfree((*command).completion as *const ());
kfree(command as *const ());
}
return ret;
}
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_address_device")]
extern "C" fn xhci_address_device(hcd: *mut usb_hcd, udev: *mut usb_device, timeout_ms: u32) -> i32 {
xhci_setup_device(hcd, udev, 1u32, timeout_ms)
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_enable_device")]
extern "C" fn xhci_enable_device(hcd: *mut usb_hcd, udev: *mut usb_device) -> i32 {
xhci_setup_device(hcd, udev, 0u32, 5000u32)
}

/// Transfer the port index into real index in the HW port status
/// registers. Caculate offset between the port's PORTSC register
/// and port status base. Divide the number of per port register
/// to get the real index. The raw port number bases 1.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_find_raw_port_number")]
extern "C" fn xhci_find_raw_port_number(hcd: *mut usb_hcd, port1: i32) -> i32 {
let rhub: *mut xhci_hub = unsafe { xhci_get_rhub(hcd) };
unsafe { (*(*unsafe { (*rhub).ports }.wrapping_offset(port1.wrapping_sub(1) as isize))).hw_portnum }.wrapping_add(1)
}

/// Issue an Evaluate Context command to change the Maximum Exit Latency in the
/// slot context.  If that succeeds, store the new MEL in the xhci_virt_device.
fn xhci_change_max_exit_latency(xhci: *mut xhci_hcd, udev: *mut usb_device, max_exit_latency: u16) -> i32 {
let mut flags: u64 = 0;
let mut ret: i32 = 0;
let command: *mut xhci_command = unsafe { xhci_alloc_command_with_ctx(xhci, true, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if command.is_null() {
return 12i32.wrapping_neg();
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
let virt_dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) };
// virt_dev might not exists yet if xHC resumed from hibernate (S4) and
// xHC was re-initialized. Exit latency will be set later after
// hub_port_finish_reset() is done and xhci->devs[] are re-allocated
if virt_dev.is_null() || (max_exit_latency as i32) == ((unsafe { (*virt_dev).current_mel }) as i32) {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_free_command(xhci, command) };
return 0;
}
// Attempt to issue an Evaluate Context command to change the MEL.
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*command).in_ctx) };
if ctrl_ctx.is_null() {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe { xhci_free_command(xhci, command) };
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_change_max_exit_latency".as_ptr() as *mut u8) };
return 12i32.wrapping_neg();
}
unsafe { xhci_slot_copy(xhci, (*command).in_ctx, (*virt_dev).out_ctx) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
let __v7: u32 = unsafe { (*ctrl_ctx).add_flags };
let __v8: u32 = __v7 | (BIT!(0i32) as u32);
unsafe {
(*ctrl_ctx).add_flags = __v8;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*command).in_ctx) };
let __v11: u32 = (unsafe { (*slot_ctx).dev_info2 }) & !65535;
unsafe {
(*slot_ctx).dev_info2 = __v11;
}
unsafe {
(*slot_ctx).dev_info2 = (*slot_ctx).dev_info2 | (max_exit_latency as u32);
}
unsafe {
(*slot_ctx).dev_state = 0;
}
unsafe { xhci_dbg_trace(xhci, Some(trace_xhci_dbg_context_change), c"Set up evaluate context for LPM MEL change.".as_ptr() as *const u8) };
// Issue and wait for the evaluate context command.
ret = xhci_configure_endpoint(xhci, udev, command, true, true);
if ret == 0 {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
unsafe {
(*virt_dev).current_mel = max_exit_latency;
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
}
unsafe { xhci_free_command(xhci, command) };
ret
}

/// BESL to HIRD Encoding array for USB2 LPM
static mut xhci_besl_encoding: __SlateAlign16<[i32; 16]> = __SlateAlign16([125i32, 150i32, 200i32, 300i32, 400i32, 500i32, 1000i32, 2000i32, 3000i32, 4000i32, 5000i32, 6000i32, 7000i32, 8000i32, 9000i32, 10000i32]);

// Calculate HIRD/BESL for USB2 PORTPMSC
fn xhci_calculate_hird_besl(xhci: *mut xhci_hcd, udev: *mut usb_device) -> i32 {
let mut u2del: i32 = 0;
let mut besl: i32 = 0;
let mut besl_host: i32 = 0;
let mut besl_device: i32 = 0;
let mut field: u32 = 0;
let mut __v0: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_c_4612_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_959() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_960() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_c_4612_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) == 0 {
unsafe { __compiletime_assert_961() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_c_4612_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) >> ((core::mem::size_of::<__SlateRecord_xhci_c_4612_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_962() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_c_4612_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_4612_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_c_4612_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_c_4612_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_963() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((unsafe { (*xhci).hcs_params3 }) as u64) & (core::mem::size_of::<__SlateRecord_xhci_c_4612_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))) >> ((core::mem::size_of::<__SlateRecord_xhci_c_4612_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)).trailing_zeros() as i32);
__v0 = __v2;
u2del = __v0 as i32;
field = unsafe { (*(*(*udev).bos).ext_cap).bmAttributes };
if field & 4u32 != 0 {
besl_host = 0;
while besl_host < 16 {
if (unsafe { xhci_besl_encoding.0[besl_host as usize] }) >= u2del {
break;
}
besl_host = besl_host.wrapping_add(1);
}
// Use baseline BESL value as default
if field & 8u32 != 0 {
besl_device = USB_GET_BESL_BASELINE!(field) as i32;
} else {
if field & 16u32 != 0 {
besl_device = USB_GET_BESL_DEEP!(field) as i32;
}
}
} else {
if u2del <= 50 {
besl_host = 0;
} else {
besl_host = (u2del.wrapping_sub(51) / 75).wrapping_add(1);
}
}
besl = besl_host.wrapping_add(besl_device);
if besl > 15 {
besl = 15;
}
besl
}

/// Calculate BESLD, L1 timeout and HIRDM for USB2 PORTHLPMC
fn xhci_calculate_usb2_hw_lpm_params(udev: *mut usb_device) -> i32 {
let field: u32 = unsafe { (*(*(*udev).bos).ext_cap).bmAttributes };
let mut l1: i32 = 0;
let mut besld: i32 = 0;
let mut hirdm: i32 = 0;
// xHCI l1 is set in steps of 256us, xHCI 1.0 section 5.4.11.2
l1 = (unsafe { (*udev).l1_params.timeout }) / 256;
// device has preferred BESLD
if field & 16u32 != 0 {
besld = USB_GET_BESL_DEEP!(field) as i32;
hirdm = 1;
}
PORT_BESLD!(besld) | PORT_L1_TIMEOUT!(l1) | PORT_HIRDM!(hirdm)
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_set_usb2_hardware_lpm")]
extern "C" fn xhci_set_usb2_hardware_lpm(hcd: *mut usb_hcd, udev: *mut usb_device, enable: i32) -> i32 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut ports: *mut *mut xhci_port = unsafe { core::mem::zeroed() };
let mut port_reg: *mut xhci_port_regs = unsafe { core::mem::zeroed() };
let mut pm_val: u32 = 0;
let mut hlpm_val: u32 = 0;
let mut field: u32 = 0;
let mut port_num: u32 = 0;
let mut flags: u64 = 0;
let mut hird: i32 = 0;
let mut exit_latency: i32 = 0;
let mut ret: i32 = 0;
if (unsafe { (*xhci).quirks }) & BIT_ULL!(29i32) != 0 {
return 1i32.wrapping_neg();
}
if (unsafe { (*hcd).speed }) >= 64 || ((unsafe { (*xhci).__slate_bits_0.__get_hw_lpm_support() }) as i32) == 0 || ((unsafe { (*udev).__slate_bits_0.__get_lpm_capable() }) as i32) == 0 {
return 1i32.wrapping_neg();
}
if unsafe { (*udev).parent }.is_null() || !unsafe { (*(*udev).parent).parent }.is_null() || ((unsafe { (*udev).descriptor.bDeviceClass }) as i32) == 9 {
return 1i32.wrapping_neg();
}
if ((unsafe { (*udev).__slate_bits_0.__get_usb2_hw_lpm_capable() }) as i32) != 1 {
return 1i32.wrapping_neg();
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
ports = unsafe { (*xhci).usb2_rhub.ports };
port_num = ((unsafe { (*udev).portnum }) as i32).wrapping_sub(1) as u32;
port_reg = unsafe { (*(*ports.wrapping_offset(port_num as isize))).port_reg };
pm_val = readl((unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *const ());
if enable != 0 {
// Host supports BESL timeout instead of HIRD
if ((unsafe { (*udev).__slate_bits_0.__get_usb2_hw_lpm_besl_capable() }) as i32) != 0 {
// if device doesn't have a preferred BESL value use a
// default one which works with mixed HIRD and BESL
// systems. See XHCI_DEFAULT_BESL definition in xhci.h
field = unsafe { (*(*(*udev).bos).ext_cap).bmAttributes };
if field & 4u32 != 0 && field & 8u32 != 0 {
hird = USB_GET_BESL_BASELINE!(field) as i32;
} else {
hird = (unsafe { (*udev).l1_params.besl }) as i32;
}
exit_latency = unsafe { xhci_besl_encoding.0[hird as usize] };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
ret = xhci_change_max_exit_latency(xhci, udev, exit_latency as u16);
if ret < 0 {
return ret;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
hlpm_val = xhci_calculate_usb2_hw_lpm_params(udev) as u32;
writel(hlpm_val, (unsafe { core::ptr::addr_of_mut!((*port_reg).porthlmpc) }) as *mut ());
// flush write
readl((unsafe { core::ptr::addr_of_mut!((*port_reg).porthlmpc) }) as *const ());
} else {
hird = xhci_calculate_hird_besl(xhci, udev);
}
let __v11: u32 = pm_val & 4294967055u32;
pm_val = __v11;
let __v12: u32 = pm_val;
pm_val = __v12 | ((PORT_HIRD!(hird) | 8i32 | PORT_L1DS!(unsafe { (*udev).slot_id })) as u32);
writel(pm_val, (unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *mut ());
pm_val = readl((unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *const ());
pm_val |= 65536u32;
writel(pm_val, (unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *mut ());
// flush write
readl((unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *const ());
} else {
let __v17: u32 = pm_val & 4294836231u32;
pm_val = __v17;
writel(pm_val, (unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *mut ());
// flush write
readl((unsafe { core::ptr::addr_of_mut!((*port_reg).portpmsc) }) as *const ());
if ((unsafe { (*udev).__slate_bits_0.__get_usb2_hw_lpm_besl_capable() }) as i32) != 0 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
xhci_change_max_exit_latency(xhci, udev, 0u16);
let mut __v18: i32;
let __timeout_us: u64 = 10000;
let __sleep_us: u64 = 100;
let __timeout: i64 = ktime_add_us(unsafe { ktime_get() }, __timeout_us);
let mut ___ret: i32 = 0;
might_sleep_if!(__sleep_us != 0);
loop {
let mut __expired: bool = false;
__expired = __timeout_us != 0 && ktime_compare(unsafe { ktime_get() }, __timeout) > 0;
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
pm_val = readl((unsafe { core::ptr::addr_of_mut!((*(*(*ports.wrapping_offset(port_num as isize))).port_reg).portsc) }) as *const ());
if pm_val & 480u32 == 0u32 {
___ret = 0;
break;
}
if __expired {
___ret = 110i32.wrapping_neg();
break;
}
if __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
cpu_relax();
}
__v18 = ___ret;
return 0;
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_update_device")]
extern "C" fn xhci_update_device(hcd: *mut usb_hcd, udev: *mut usb_device) -> i32 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut port: *mut xhci_port = unsafe { core::mem::zeroed() };
let mut capability: u32 = 0;
// Check if USB3 device at root port is tunneled over USB4
if (unsafe { (*hcd).speed }) >= 64 && unsafe { (*(*udev).parent).parent }.is_null() {
port = unsafe { *unsafe { (*xhci).usb3_rhub.ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
unsafe {
(*udev).tunnel_mode = xhci_port_is_tunneled(xhci, port);
}
if ((unsafe { (*udev).tunnel_mode }) as u32) == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"link tunnel state unknown\n".as_ptr() as *const u8) };
}
} else {
if ((unsafe { (*udev).tunnel_mode }) as u32) == 2 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"tunneled over USB4 link\n".as_ptr() as *const u8) };
}
} else {
if ((unsafe { (*udev).tunnel_mode }) as u32) == 1 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"native USB 3.x link\n".as_ptr() as *const u8) };
}
}
}
}
return 0;
}
if (unsafe { (*hcd).speed }) >= 64 || ((unsafe { (*udev).__slate_bits_0.__get_lpm_capable() }) as i32) == 0 || ((unsafe { (*xhci).__slate_bits_0.__get_hw_lpm_support() }) as i32) == 0 {
return 0;
}
// we only support lpm for non-hub device connected to root hub yet
if unsafe { (*udev).parent }.is_null() || !unsafe { (*(*udev).parent).parent }.is_null() || ((unsafe { (*udev).descriptor.bDeviceClass }) as i32) == 9 {
return 0;
}
port = unsafe { *unsafe { (*xhci).usb2_rhub.ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
capability = unsafe { (*(*port).port_cap).protocol_caps };
if capability & 524288u32 != 0 {
unsafe {
(*udev).__slate_bits_0.__set_usb2_hw_lpm_capable(1u32);
}
unsafe {
(*udev).l1_params.timeout = 512;
}
unsafe {
(*udev).l1_params.besl = 4;
}
if capability & 1048576u32 != 0 {
unsafe {
(*udev).__slate_bits_0.__set_usb2_hw_lpm_besl_capable(1u32);
}
}
}
0
}

// ---------------------- USB 3.0 Link PM functions ------------------------
/// Service interval in nanoseconds = 2^(bInterval - 1) * 125us * 1000ns / 1us
fn xhci_service_interval_to_ns(desc: *mut usb_endpoint_descriptor) -> u64 {
(1u64 << ((unsafe { (*desc).bInterval }) as i32).wrapping_sub(1)).wrapping_mul(125).wrapping_mul(1000)
}

fn xhci_get_timeout_no_hub_lpm(udev: *mut usb_device, state: u32) -> u16 {
let mut sel: u64 = 0;
let mut pel: u64 = 0;
let mut max_sel_pel: u32 = 0;
let mut state_name: *mut u8 = unsafe { core::mem::zeroed() };
match state as u32 {
1 => {
sel = (unsafe { (*udev).u1_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64; // Convert SEL and PEL stored in nanoseconds to microseconds
pel = (unsafe { (*udev).u1_params.pel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
max_sel_pel = 255;
state_name = c"U1".as_ptr() as *mut u8;
}
2 => {
sel = (unsafe { (*udev).u2_params.sel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
pel = (unsafe { (*udev).u2_params.pel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
max_sel_pel = 65535;
state_name = c"U2".as_ptr() as *mut u8;
}
_ => {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"%s: Can't get timeout for non-U1 or U2 state.\n".as_ptr() as *const u8, c"xhci_get_timeout_no_hub_lpm".as_ptr() as *mut u8) };
return 0;
}
}
if sel <= (max_sel_pel as u64) && pel <= (max_sel_pel as u64) {
return 255;
}
if sel > (max_sel_pel as u64) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Device-initiated %s disabled due to long SEL %llu ms\n".as_ptr() as *const u8, state_name, sel) };
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Device-initiated %s disabled due to long PEL %llu ms\n".as_ptr() as *const u8, state_name, pel) };
}
}
0
}

/// The U1 timeout should be the maximum of the following values:
/// - For control endpoints, U1 system exit latency (SEL) * 3
/// - For bulk endpoints, U1 SEL * 5
/// - For interrupt endpoints:
///   - Notification EPs, U1 SEL * 3
///   - Periodic EPs, max(105% of bInterval, U1 SEL * 2)
/// - For isochronous endpoints, max(105% of bInterval, U1 SEL * 2)
fn xhci_calculate_intel_u1_timeout(udev: *mut usb_device, desc: *mut usb_endpoint_descriptor) -> u64 {
let mut timeout_ns: u64 = 0;
let ep_type: i32 = usb_endpoint_type(desc as *const usb_endpoint_descriptor);
let mut intr_type: i32 = 0;
// Otherwise the calculation is the same as isoc eps
match ep_type {
0 => {
timeout_ns = unsafe { (*udev).u1_params.sel }.wrapping_mul(3) as u64;
}
2 => {
timeout_ns = unsafe { (*udev).u1_params.sel }.wrapping_mul(5) as u64;
}
3 => {
intr_type = usb_endpoint_interrupt_type(desc as *const usb_endpoint_descriptor);
if intr_type == 16i32 {
timeout_ns = unsafe { (*udev).u1_params.sel }.wrapping_mul(3) as u64;
} else {
// Otherwise the calculation is the same as isoc eps
timeout_ns = xhci_service_interval_to_ns(desc);
timeout_ns = DIV_ROUND_UP_ULL!(timeout_ns.wrapping_mul(105), 100i32);
if timeout_ns < (unsafe { (*udev).u1_params.sel }.wrapping_mul(2) as u64) {
timeout_ns = unsafe { (*udev).u1_params.sel }.wrapping_mul(2) as u64;
}
}
}
1 => {
timeout_ns = xhci_service_interval_to_ns(desc);
timeout_ns = DIV_ROUND_UP_ULL!(timeout_ns.wrapping_mul(105), 100i32);
if timeout_ns < (unsafe { (*udev).u1_params.sel }.wrapping_mul(2) as u64) {
timeout_ns = unsafe { (*udev).u1_params.sel }.wrapping_mul(2) as u64;
}
}
_ => {
return 0;
}
}
timeout_ns
}

/// Returns the hub-encoded U1 timeout value.
fn xhci_calculate_u1_timeout(xhci: *mut xhci_hcd, udev: *mut usb_device, desc: *mut usb_endpoint_descriptor) -> u16 {
let mut timeout_ns: u64 = 0;
// Prevent U1 if service interval is shorter than U1 exit latency
let __v0: bool = usb_endpoint_xfer_int(desc as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc(desc as *const usb_endpoint_descriptor) != 0;
if __v0 {
if xhci_service_interval_to_ns(desc) <= ((unsafe { (*udev).u1_params.mel }) as u64) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Disable U1, ESIT shorter than exit latency\n".as_ptr() as *const u8) };
}
return 0;
}
}
if (unsafe { (*xhci).quirks }) & (BIT_ULL!(12i32) | BIT_ULL!(46i32)) != 0 {
timeout_ns = xhci_calculate_intel_u1_timeout(udev, desc);
} else {
timeout_ns = (unsafe { (*udev).u1_params.sel }) as u64;
}
// The U1 timeout is encoded in 1us intervals.
// Don't return a timeout of zero, because that's USB3_LPM_DISABLED.
if timeout_ns == 0 {
timeout_ns = 1;
} else {
timeout_ns = DIV_ROUND_UP_ULL!(timeout_ns, 1000i32);
}
// If the necessary timeout value is bigger than what we can set in the
// USB 3.0 hub, we have to disable hub-initiated U1.
if timeout_ns <= 127 {
return timeout_ns as u16;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Hub-initiated U1 disabled due to long timeout %lluus\n".as_ptr() as *const u8, timeout_ns) };
}
xhci_get_timeout_no_hub_lpm(udev, 1u32)
}

/// The U2 timeout should be the maximum of:
/// - 10 ms (to avoid the bandwidth impact on the scheduler)
/// - largest bInterval of any active periodic endpoint (to avoid going
///   into lower power link states between intervals).
/// - the U2 Exit Latency of the device
fn xhci_calculate_intel_u2_timeout(udev: *mut usb_device, desc: *mut usb_endpoint_descriptor) -> u64 {
let mut timeout_ns: u64 = 10000000u64;
let mut u2_del_ns: u64 = 0;
let __v0: bool = usb_endpoint_xfer_int(desc as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc(desc as *const usb_endpoint_descriptor) != 0;
let __v1: bool = __v0 && xhci_service_interval_to_ns(desc) > timeout_ns;
if __v1 {
timeout_ns = xhci_service_interval_to_ns(desc);
}
u2_del_ns = (((unsafe { (*(*(*udev).bos).ss_cap).bU2DevExitLat }) as i32) as u64).wrapping_mul(1000);
if u2_del_ns > timeout_ns {
timeout_ns = u2_del_ns;
}
timeout_ns
}

/// Returns the hub-encoded U2 timeout value.
fn xhci_calculate_u2_timeout(xhci: *mut xhci_hcd, udev: *mut usb_device, desc: *mut usb_endpoint_descriptor) -> u16 {
let mut timeout_ns: u64 = 0;
// Prevent U2 if service interval is shorter than U2 exit latency
let __v0: bool = usb_endpoint_xfer_int(desc as *const usb_endpoint_descriptor) != 0 || usb_endpoint_xfer_isoc(desc as *const usb_endpoint_descriptor) != 0;
if __v0 {
if xhci_service_interval_to_ns(desc) <= ((unsafe { (*udev).u2_params.mel }) as u64) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Disable U2, ESIT shorter than exit latency\n".as_ptr() as *const u8) };
}
return 0;
}
}
if (unsafe { (*xhci).quirks }) & (BIT_ULL!(12i32) | BIT_ULL!(46i32)) != 0 {
timeout_ns = xhci_calculate_intel_u2_timeout(udev, desc);
} else {
timeout_ns = (unsafe { (*udev).u2_params.sel }) as u64;
}
// The U2 timeout is encoded in 256us intervals
timeout_ns = DIV_ROUND_UP_ULL!(timeout_ns, 256000i32);
// If the necessary timeout value is bigger than what we can set in the
// USB 3.0 hub, we have to disable hub-initiated U2.
if timeout_ns <= 254 {
return timeout_ns as u16;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Hub-initiated U2 disabled due to long timeout %lluus\n".as_ptr() as *const u8, timeout_ns.wrapping_mul(256)) };
}
xhci_get_timeout_no_hub_lpm(udev, 2u32)
}

fn xhci_call_host_update_timeout_for_endpoint(xhci: *mut xhci_hcd, udev: *mut usb_device, desc: *mut usb_endpoint_descriptor, state: u32, timeout: &mut u16) -> u16 {
if (state as u32) == 1 {
return xhci_calculate_u1_timeout(xhci, udev, desc);
} else {
if (state as u32) == 2 {
return xhci_calculate_u2_timeout(xhci, udev, desc);
}
}
0
}

fn xhci_update_timeout_for_endpoint(xhci: *mut xhci_hcd, udev: *mut usb_device, desc: *mut usb_endpoint_descriptor, state: u32, timeout: &mut u16) -> i32 {
let alt_timeout: u16 = xhci_call_host_update_timeout_for_endpoint(xhci, udev, desc, state, timeout);
// If we found we can't enable hub-initiated LPM, and
// the U1 or U2 exit latency was too high to allow
// device-initiated LPM as well, then we will disable LPM
// for this device, so stop searching any further.
if (alt_timeout as i32) == 0 {
unsafe {
*(timeout as *mut u16) = alt_timeout;
}
return 7i32.wrapping_neg();
}
if (alt_timeout as i32) > ((unsafe { *(timeout as *mut u16) }) as i32) {
unsafe {
*(timeout as *mut u16) = alt_timeout;
}
}
0
}

fn xhci_update_timeout_for_interface(xhci: *mut xhci_hcd, udev: *mut usb_device, alt: *mut usb_host_interface, state: u32, timeout: &mut u16) -> i32 {
let mut j: i32 = 0;
while j < ((unsafe { (*alt).desc.bNumEndpoints }) as i32) {
if xhci_update_timeout_for_endpoint(xhci, udev, unsafe { core::ptr::addr_of_mut!((*unsafe { (*alt).endpoint }.wrapping_offset(j as isize)).desc) }, state, timeout) != 0 {
return 7i32.wrapping_neg();
}
j = j.wrapping_add(1);
}
0
}

fn xhci_check_tier_policy(xhci: *mut xhci_hcd, udev: *mut usb_device, state: u32) -> i32 {
// roothub is tier1
let mut tier: i32 = unsafe { core::mem::zeroed() };
let mut parent: *mut usb_device = unsafe { core::mem::zeroed() };
unsafe {
parent = (*udev).parent;
tier = 1;
while !parent.is_null() {
parent = (*parent).parent;
tier = tier.wrapping_add(1);
}
if !((*xhci).quirks & BIT_ULL!(12i32) != 0 && tier > 3) {
if !((*xhci).quirks & BIT_ULL!(46i32) != 0 && tier > 2) {
return 0;
}
}
return 7i32.wrapping_neg();
}
}

/// Returns the U1 or U2 timeout that should be enabled.
/// If the tier check or timeout setting functions return with a non-zero exit
/// code, that means the timeout value has been finalized and we shouldn't look
/// at any more endpoints.
fn xhci_calculate_lpm_timeout(hcd: *mut usb_hcd, udev: *mut usb_device, state: u32) -> u16 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut state_name: *mut u8 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut timeout: u16 = 0;
if (state as u32) == 1 {
state_name = c"U1".as_ptr() as *mut u8;
} else {
if (state as u32) == 2 {
state_name = c"U2".as_ptr() as *mut u8;
} else {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Can't enable unknown link state %i\n".as_ptr() as *const u8, state as u32) };
return timeout;
}
}
// Gather some information about the currently installed configuration
// and alternate interface settings.
if xhci_update_timeout_for_endpoint(xhci, udev, unsafe { core::ptr::addr_of_mut!((*udev).ep0.desc) }, state, unsafe { &mut (*core::ptr::addr_of_mut!(timeout)) }) != 0 {
return timeout;
}
let config: *mut usb_host_config = unsafe { (*udev).actconfig };
if config.is_null() {
return timeout;
}
while i < ((unsafe { (*config).desc.bNumInterfaces }) as i32) {
let intf: *mut usb_interface = unsafe { *((*config).interface.as_mut_ptr() as *mut *mut usb_interface).wrapping_offset(i as isize) };
if !intf.is_null() {
// Check if any currently bound drivers want hub-initiated LPM
// disabled.
if !unsafe { (*intf).dev.driver }.is_null() {
let driver: *mut usb_driver = to_usb_driver!(unsafe { (*intf).dev.driver });
if !driver.is_null() && ((unsafe { (*driver).__slate_bits_0.__get_disable_hub_initiated_lpm() }) as i32) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Hub-initiated %s disabled at request of driver %s\n".as_ptr() as *const u8, state_name, (*driver).name) };
}
timeout = xhci_get_timeout_no_hub_lpm(udev, state);
if (timeout as i32) == 0 {
return timeout;
}
}
}
// Not sure how this could happen...
if !unsafe { (*intf).cur_altsetting }.is_null() {
if xhci_update_timeout_for_interface(xhci, udev, unsafe { (*intf).cur_altsetting }, state, unsafe { &mut (*core::ptr::addr_of_mut!(timeout)) }) != 0 {
return timeout;
}
}
}
i = i.wrapping_add(1);
}
timeout
}

fn calculate_max_exit_latency(udev: *mut usb_device, state_changed: u32, hub_encoded_timeout: u16) -> i32 {
let mut u1_mel_us: u64 = 0;
let mut u2_mel_us: u64 = 0;
let mut mel_us: u64 = 0;
let mut disabling_u1: bool = false;
let mut disabling_u2: bool = false;
let mut enabling_u1: bool = false;
let mut enabling_u2: bool = false;
disabling_u1 = (state_changed as u32) == 1 && (hub_encoded_timeout as i32) == 0;
disabling_u2 = (state_changed as u32) == 2 && (hub_encoded_timeout as i32) == 0;
enabling_u1 = (state_changed as u32) == 1 && (hub_encoded_timeout as i32) != 0;
enabling_u2 = (state_changed as u32) == 2 && (hub_encoded_timeout as i32) != 0;
// If U1 was already enabled and we're not disabling it,
// or we're going to enable U1, account for the U1 max exit latency.
if (unsafe { (*udev).u1_params.timeout }) != 0 && !disabling_u1 || enabling_u1 {
u1_mel_us = (unsafe { (*udev).u1_params.mel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
}
if (unsafe { (*udev).u2_params.timeout }) != 0 && !disabling_u2 || enabling_u2 {
u2_mel_us = (unsafe { (*udev).u2_params.mel }.wrapping_add(1000).wrapping_sub(1) / 1000) as u64;
}
let mut __v0: u64;
let __UNIQUE_ID_x__964: u64 = u1_mel_us;
let __UNIQUE_ID_y__965: u64 = u2_mel_us;
loop {
if __types_ok__3!(__UNIQUE_ID_x__964, __UNIQUE_ID_y__965) == 0 {
unsafe { __compiletime_assert_966() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__964 > __UNIQUE_ID_y__965 { __UNIQUE_ID_x__964 } else { __UNIQUE_ID_y__965 };
mel_us = __v0;
// xHCI host controller max exit latency field is only 16 bits wide.
if mel_us > 65535 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*udev).dev) as *const device, c"Link PM max exit latency of %lluus is too big.\n".as_ptr() as *const u8, mel_us) };
return 7i32.wrapping_neg();
}
mel_us as i32
}

/// Returns the USB3 hub-encoded value for the U1/U2 timeout.
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_enable_usb3_lpm_timeout")]
extern "C" fn xhci_enable_usb3_lpm_timeout(hcd: *mut usb_hcd, udev: *mut usb_device, state: u32) -> i32 {
let mut hub_encoded_timeout: u16 = 0;
let mut mel: i32 = 0;
let mut ret: i32 = 0;
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
// The LPM timeout values are pretty host-controller specific, so don't
// enable hub-initiated timeouts unless the vendor has provided
// information about their timeout algorithm.
if xhci.is_null() || (unsafe { (*xhci).quirks }) & BIT_ULL!(11i32) == 0 || unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) }.is_null() {
return 0;
}
if xhci_check_tier_policy(xhci, udev, state) < 0 {
return 0;
}
// If connected to root port then check port can handle lpm
if !unsafe { (*udev).parent }.is_null() && unsafe { (*(*udev).parent).parent }.is_null() {
let port: *mut xhci_port = unsafe { *unsafe { (*xhci).usb3_rhub.ports }.wrapping_offset(((*udev).portnum as i32).wrapping_sub(1) as isize) };
if ((unsafe { (*port).__slate_bits_0.__get_lpm_incapable() }) as i32) != 0 {
return 0;
}
}
hub_encoded_timeout = xhci_calculate_lpm_timeout(hcd, udev, state);
mel = calculate_max_exit_latency(udev, state, hub_encoded_timeout);
if mel < 0 {
// Max Exit Latency is too big, disable LPM.
hub_encoded_timeout = 0;
mel = 0;
}
ret = xhci_change_max_exit_latency(xhci, udev, mel as u16);
if ret != 0 {
return ret;
}
hub_encoded_timeout as i32
}

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_disable_usb3_lpm_timeout")]
extern "C" fn xhci_disable_usb3_lpm_timeout(hcd: *mut usb_hcd, udev: *mut usb_device, state: u32) -> i32 {
let mut mel: u16 = 0;
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
if xhci.is_null() || (unsafe { (*xhci).quirks }) & BIT_ULL!(11i32) == 0 || unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*udev).slot_id as isize) }.is_null() {
return 0;
}
mel = calculate_max_exit_latency(udev, state, 0u16) as u16;
xhci_change_max_exit_latency(xhci, udev, mel)
}

/// Once a hub descriptor is fetched for a device, we need to update the xHC's
/// internal data structures for the device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_update_hub_device")]
extern "C" fn xhci_update_hub_device(hcd: *mut usb_hcd, hdev: *mut usb_device, tt: *mut usb_tt, mem_flags: u32) -> i32 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
let mut flags: u64 = 0;
let mut think_time: u32 = 0;
let mut ret: i32 = 0;
// Ignore root hubs
if unsafe { (*hdev).parent }.is_null() {
return 0;
}
let vdev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset((*hdev).slot_id as isize) };
if vdev.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Cannot update hub desc for unknown device.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
let config_cmd: *mut xhci_command = unsafe { xhci_alloc_command_with_ctx(xhci, true, mem_flags) };
if config_cmd.is_null() {
return 12i32.wrapping_neg();
}
let ctrl_ctx: *mut xhci_input_control_ctx = unsafe { xhci_get_input_control_ctx((*config_cmd).in_ctx) };
if ctrl_ctx.is_null() {
while false {
}
unsafe { _dev_warn((*xhci_to_hcd(xhci)).self_.controller as *const device, c"%s: Could not get input context, bad type.\n".as_ptr() as *const u8, c"xhci_update_hub_device".as_ptr() as *mut u8) };
unsafe { xhci_free_command(xhci, config_cmd) };
return 12i32.wrapping_neg();
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
let __v7: bool = ((unsafe { (*hdev).speed }) as u32) == 3 && (unsafe { xhci_alloc_tt_info(xhci, vdev, hdev, tt, (BIT!(5i32) as u32) | (BIT!(11i32) as u32)) }) != 0;
if __v7 {
unsafe { xhci_free_command(xhci, config_cmd) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
return 12i32.wrapping_neg();
}
unsafe { xhci_slot_copy(xhci, (*config_cmd).in_ctx, (*vdev).out_ctx) };
let __v9: u32 = unsafe { (*ctrl_ctx).add_flags };
let __v10: u32 = __v9 | (BIT!(0i32) as u32);
unsafe {
(*ctrl_ctx).add_flags = __v10;
}
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*config_cmd).in_ctx) };
let __v12: u32 = unsafe { (*slot_ctx).dev_info };
let __v13: u32 = __v12 | (BIT!(26i32) as u32);
unsafe {
(*slot_ctx).dev_info = __v13;
}
// refer to section 6.2.2: MTT should be 0 for full speed hub,
// but it may be already set to 1 when setup an xHCI virtual
// device, so clear it anyway.
if (unsafe { (*tt).multi }) != 0 {
let __v15: u32 = unsafe { (*slot_ctx).dev_info };
let __v16: u32 = __v15 | (BIT!(25i32) as u32);
unsafe {
(*slot_ctx).dev_info = __v16;
}
} else {
if ((unsafe { (*hdev).speed }) as u32) == 2 {
let __v18: u32 = unsafe { (*slot_ctx).dev_info };
let __v19: u32 = __v18 & (!BIT!(25i32) as u32);
unsafe {
(*slot_ctx).dev_info = __v19;
}
}
}
if ((unsafe { (*xhci).hci_version }) as i32) > 149 {
let __v21: u32 = unsafe { (*slot_ctx).dev_info2 };
let __v22: u32 = __v21 | (XHCI_MAX_PORTS!(unsafe { (*hdev).maxchild }) as u32);
unsafe {
(*slot_ctx).dev_info2 = __v22;
}
// Set TT think time - convert from ns to FS bit times.
// 0 = 8 FS bit times, 1 = 16 FS bit times,
// 2 = 24 FS bit times, 3 = 32 FS bit times.
//
// xHCI 1.0: this field shall be 0 if the device is not a
// High-spped hub.
think_time = unsafe { (*tt).think_time };
if think_time != 0 {
think_time = (think_time / 666).wrapping_sub(1);
}
if ((unsafe { (*xhci).hci_version }) as i32) < 256 || ((unsafe { (*hdev).speed }) as u32) == 3 {
let __v24: u32 = unsafe { (*slot_ctx).tt_info };
let __v25: u32 = __v24 | TT_THINK_TIME!(think_time);
unsafe {
(*slot_ctx).tt_info = __v25;
}
}
}
unsafe {
(*slot_ctx).dev_state = 0;
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// Issue and wait for the configure endpoint or
// evaluate context command.
if ((unsafe { (*xhci).hci_version }) as i32) > 149 {
ret = xhci_configure_endpoint(xhci, hdev, config_cmd, false, false);
} else {
ret = xhci_configure_endpoint(xhci, hdev, config_cmd, true, false);
}
unsafe { xhci_free_command(xhci, config_cmd) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_update_hub_device_967: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_tt, u32) -> i32>, *mut ()>(Some(xhci_update_hub_device)) };

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_get_frame")]
extern "C" fn xhci_get_frame(hcd: *mut usb_hcd) -> i32 {
let xhci: *mut xhci_hcd = hcd_to_xhci(hcd);
// EHCI mods by the periodic size.  Why?
(readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).run_regs).microframe_index) }) as *const ()) >> 3i32) as i32
}

fn xhci_hcd_init_usb2_data(xhci: *mut xhci_hcd, hcd: *mut usb_hcd) {
unsafe {
(*xhci).usb2_rhub.hcd = hcd;
}
unsafe {
(*hcd).speed = 32;
}
unsafe {
(*(*hcd).self_.root_hub).speed = 3;
}
// USB 2.0 roothub under xHCI has an integrated TT,
// (rate matching hub) as opposed to having an OHCI/UHCI
// companion controller.
unsafe {
(*hcd).__slate_bits_0.__set_has_tt(1u32);
}
}

fn xhci_hcd_init_usb3_data(xhci: *mut xhci_hcd, hcd: *mut usb_hcd) {
let mut minor_rev: u32 = 0;
// Early xHCI 1.1 spec did not mention USB 3.1 capable hosts
// should return 0x31 for sbrn, or that the minor revision
// is a two digit BCD containig minor and sub-minor numbers.
// This was later clarified in xHCI 1.2.
//
// Some USB 3.1 capable hosts therefore have sbrn 0x30, and
// minor revision set to 0x1 instead of 0x10.
if ((unsafe { (*xhci).usb3_rhub.min_rev }) as i32) == 1 {
minor_rev = 1;
} else {
minor_rev = (((unsafe { (*xhci).usb3_rhub.min_rev }) as i32) / 16) as u32;
}
match minor_rev {
2 => {
unsafe {
(*hcd).speed = 96;
}
unsafe {
(*(*hcd).self_.root_hub).speed = 6;
}
unsafe {
(*(*hcd).self_.root_hub).rx_lanes = 2;
}
unsafe {
(*(*hcd).self_.root_hub).tx_lanes = 2;
}
unsafe {
(*(*hcd).self_.root_hub).ssp_rate = 3;
}
}
1 => {
unsafe {
(*hcd).speed = 80;
}
unsafe {
(*(*hcd).self_.root_hub).speed = 6;
}
unsafe {
(*(*hcd).self_.root_hub).ssp_rate = 1;
}
}
_ => {
}
}
while false {
}
unsafe { _dev_info((*xhci_to_hcd(xhci)).self_.controller as *const device, c"Host supports USB 3.%x %sSuperSpeed\n".as_ptr() as *const u8, minor_rev, if minor_rev != 0 { c"Enhanced ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe {
(*xhci).usb3_rhub.hcd = hcd;
}
}

static mut __key__5456: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5552: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_gen_setup_1012: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, Option<unsafe extern "C" fn(*mut device, *mut xhci_hcd)>) -> i32>, *mut ()>(Some(xhci_gen_setup)) };

#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_clear_tt_buffer_complete")]
extern "C" fn xhci_clear_tt_buffer_complete(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) {
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
let mut slot_id: u32 = 0;
let mut ep_index: u32 = 0;
let mut flags: u64 = 0;
xhci = hcd_to_xhci(hcd);
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
let udev: *mut usb_device = (unsafe { (*ep).hcpriv }) as *mut usb_device;
slot_id = (unsafe { (*udev).slot_id }) as u32;
ep_index = xhci_get_endpoint_index(unsafe { core::ptr::addr_of_mut!((*ep).desc) });
let __v5: *mut xhci_virt_ep = unsafe { (*(*unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize))).eps.as_mut_ptr() as *mut xhci_virt_ep }.wrapping_offset(ep_index as isize);
let __v6: u32 = unsafe { (*__v5).ep_state };
let __v7: u32 = ((__v6 as u64) & !BIT!(8i32)) as u32;
unsafe {
(*__v5).ep_state = __v7;
}
unsafe { xhci_ring_doorbell_for_active_rings(xhci, slot_id, ep_index) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
}

/// generic hardware linkage
/// basic lifecycle operations
/// set in xhci_init_driver()
/// managing i/o requests and associated device resources
/// scheduling support
/// root hub support
/// call back when device connected and addressed
static xhci_hc_driver: __SlateConst<hc_driver> = __SlateConst({
let mut __t0: hc_driver = unsafe { core::mem::zeroed() };
__t0.description = (b"xhci-hcd\0".as_ptr() as *mut u8) as *const u8;
__t0.product_desc = (b"xHCI Host Controller\0".as_ptr() as *mut u8) as *const u8;
__t0.hcd_priv_size = 720u64;
__t0.irq = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd) -> u32>>(xhci_irq as *const ()) };
__t0.flags = 327i32;
__t0.reset = None;
__t0.start = Some(xhci_run);
__t0.stop = Some(xhci_stop);
__t0.shutdown = Some(xhci_shutdown);
__t0.get_frame_number = Some(xhci_get_frame);
__t0.urb_enqueue = Some(xhci_urb_enqueue);
__t0.urb_dequeue = Some(xhci_urb_dequeue);
__t0.map_urb_for_dma = Some(xhci_map_urb_for_dma);
__t0.unmap_urb_for_dma = Some(xhci_unmap_urb_for_dma);
__t0.endpoint_disable = Some(xhci_endpoint_disable);
__t0.endpoint_reset = Some(xhci_endpoint_reset);
__t0.hub_status_data = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd, *mut u8) -> i32>>(xhci_hub_status_data as *const ()) };
__t0.hub_control = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>>(xhci_hub_control as *const ()) };
__t0.bus_suspend = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>>(xhci_bus_suspend as *const ()) };
__t0.bus_resume = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>>(xhci_bus_resume as *const ()) };
__t0.get_resuming_ports = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut usb_hcd) -> u64>>(xhci_get_resuming_ports as *const ()) };
__t0.clear_tt_buffer_complete = Some(xhci_clear_tt_buffer_complete);
__t0.alloc_dev = Some(xhci_alloc_dev);
__t0.free_dev = Some(xhci_free_dev);
__t0.alloc_streams = Some(xhci_alloc_streams);
__t0.free_streams = Some(xhci_free_streams);
__t0.add_endpoint = Some(xhci_add_endpoint);
__t0.drop_endpoint = Some(xhci_drop_endpoint);
__t0.check_bandwidth = Some(xhci_check_bandwidth);
__t0.reset_bandwidth = Some(xhci_reset_bandwidth);
__t0.address_device = Some(xhci_address_device);
__t0.enable_device = Some(xhci_enable_device);
__t0.update_hub_device = Some(xhci_update_hub_device);
__t0.reset_device = Some(xhci_discover_or_reset_device);
__t0.update_device = Some(xhci_update_device);
__t0.set_usb2_hw_lpm = Some(xhci_set_usb2_hardware_lpm);
__t0.enable_usb3_lpm_timeout = Some(xhci_enable_usb3_lpm_timeout);
__t0.disable_usb3_lpm_timeout = Some(xhci_disable_usb3_lpm_timeout);
__t0.find_raw_port_number = Some(xhci_find_raw_port_number);
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.xhci.xhci_init_driver")]
extern "C" fn xhci_init_driver(drv: *mut hc_driver, over: *const xhci_driver_overrides) {
loop {
if unlikely!(over.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("1013: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1013b - ., 3; .popsection\n# {0}", const 1013 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_xhci_init_driver_0, sym __slate_asm_string_xhci_init_driver_1, const 5667 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// Copy the generic table to drv then apply the overrides
unsafe {
*drv = xhci_hc_driver.0;
}
if !over.is_null() {
let __v6: u64 = unsafe { (*drv).hcd_priv_size };
unsafe {
(*drv).hcd_priv_size = __v6.wrapping_add((*over).extra_priv_size);
}
if (unsafe { (*over).reset }) != None {
unsafe {
(*drv).reset = (*over).reset;
}
}
if (unsafe { (*over).start }) != None {
unsafe {
(*drv).start = (*over).start;
}
}
if (unsafe { (*over).add_endpoint }) != None {
unsafe {
(*drv).add_endpoint = (*over).add_endpoint;
}
}
if (unsafe { (*over).drop_endpoint }) != None {
unsafe {
(*drv).drop_endpoint = (*over).drop_endpoint;
}
}
if (unsafe { (*over).check_bandwidth }) != None {
unsafe {
(*drv).check_bandwidth = (*over).check_bandwidth;
}
}
if (unsafe { (*over).reset_bandwidth }) != None {
unsafe {
(*drv).reset_bandwidth = (*over).reset_bandwidth;
}
}
if (unsafe { (*over).update_hub_device }) != None {
unsafe {
(*drv).update_hub_device = (*over).update_hub_device;
}
}
if (unsafe { (*over).hub_control }) != None {
unsafe {
(*drv).hub_control = (*over).hub_control;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_init_driver_1014: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hc_driver, *const xhci_driver_overrides)>, *mut ()>(Some(xhci_init_driver)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_1015: [u8; 63] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 39, 101, 88, 116, 101, 110, 115, 105, 98, 108, 101, 39, 32, 72, 111, 115, 116, 32, 67, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 40, 120, 72, 67, 41, 32, 68, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_1016: [u8; 28] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 97, 117, 116, 104, 111, 114, 61, 83, 97, 114, 97, 104, 32, 83, 104, 97, 114, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_1017: [u8; 40] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 117, 115, 98, 47, 104, 111, 115, 116, 47, 120, 104, 99, 105, 45, 104, 99, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_1018: [u8; 21] = [120, 104, 99, 105, 95, 104, 99, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[unsafe(link_section = ".init.text")]
extern "C" fn xhci_hcd_init() -> i32 {
// Check the compiler generated sizes of structures that must be laid
// out in specific ways for hardware access.
loop {
if ((const { core::mem::size_of::<xhci_doorbell_array>() }) as u64) != 1024u64 {
unsafe { __compiletime_assert_1019() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_slot_ctx>() }) as u64) != 32u64 {
unsafe { __compiletime_assert_1020() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_ep_ctx>() }) as u64) != 32u64 {
unsafe { __compiletime_assert_1021() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
// xhci_device_control has eight fields, and also
// embeds one xhci_slot_ctx and 31 xhci_ep_ctx
loop {
if ((const { core::mem::size_of::<xhci_stream_ctx>() }) as u64) != 16u64 {
unsafe { __compiletime_assert_1022() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_trb>() }) as u64) != 16u64 {
unsafe { __compiletime_assert_1023() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_erst_entry>() }) as u64) != 16u64 {
unsafe { __compiletime_assert_1024() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_cap_regs>() }) as u64) != 32u64 {
unsafe { __compiletime_assert_1025() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<xhci_intr_reg>() }) as u64) != 32u64 {
unsafe { __compiletime_assert_1026() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
// xhci_run_regs has eight fields and embeds 1024 xhci_intr_regs
loop {
if ((const { core::mem::size_of::<xhci_run_regs>() }) as u64) != 32800u64 {
unsafe { __compiletime_assert_1027() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if (unsafe { usb_disabled() }) != 0 {
return 19i32.wrapping_neg();
}
unsafe { xhci_debugfs_create_root() };
xhci_dbc_init();
0
}

/// If an init function is provided, an exit function must also be provided
/// to allow module unload.
#[unsafe(link_section = ".exit.text")]
extern "C" fn xhci_hcd_fini() {
unsafe { xhci_debugfs_remove_root() };
xhci_dbc_exit();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_xhci_hcd_init_1029: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(xhci_hcd_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_xhci_hcd_fini: Option<unsafe extern "C" fn()> = Some(xhci_hcd_fini);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_dma_map_single_attrs_3: [u8; 44] = *b"%s %s: rejecting DMA map of vmalloc memory\n\0";

static __slate_asm_string_dma_map_single_attrs_4: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_xhci_init_driver_0: [u8; 1] = *b"\0";

static __slate_asm_string_xhci_init_driver_1: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci.c\0";

static __slate_asm_string_xhci_setup_device_17: [u8; 1] = *b"\0";

static __slate_asm_string_xhci_setup_device_18: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/xhci.c\0";
