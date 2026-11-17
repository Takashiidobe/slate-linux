
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
//! xhci-debugfs.c - xHCI debugfs interface
//!
//! Copyright (C) 2017 Intel Corporation
//!
//! Author: Lu Baolu <baolu.lu@linux.intel.com>
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
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! label
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
//! 4th level page in 5-level paging case
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
//! Prevent speculative execution past this barrier.
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
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
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! Runtime PM flag argument bits
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
//! See the comment in asm-generic/io.h about ioremap_np().
//! HCCPARAMS offset from PCI base address
//! PORTSC - Port Status and Control Register - port_status_base bitmasks
//!
//! true: device connected
//! hc_capbase - bitmasks
//!
//! bits 7:0 - Capability Registers Length
//! max buffer size for trace and debug messages
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
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
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
// reg := -EIO
// reg := -EIO
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
// No such file or directory
// I/O error
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
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
// Atomic operations are already serializing on x86
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
// Operation is not supported
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// disable sas during sighandling
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
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
// Resource extended types
// No address assigned yet
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
// RTSOFF - run_regs_off - bitmasks
//
// bits 4:0 - Rsvd
//
// bits 31:5 - Runtime Register Space Offse
// bit 3 - Compliance Transition Capability, false: compliance is enabled by default
// bit 12 - HC support eUSB2V2 capability
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
macro_rules! ARRAY_SIZE__3 {
() => {
((core::mem::size_of::<[xhci_file_map; 3]>() as u64) / (core::mem::size_of::<xhci_file_map>() as u64)).wrapping_add(__must_be_array__2!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__4 {
() => {
((core::mem::size_of::<[xhci_file_map; 3]>() as u64) / (core::mem::size_of::<xhci_file_map>() as u64)).wrapping_add(__must_be_array__23!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__5 {
() => {
((core::mem::size_of::<[debugfs_reg32; 10]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__26!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__7 {
() => {
((core::mem::size_of::<[debugfs_reg32; 2]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__10!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[debugfs_reg32; 14]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__4!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__11 {
() => {
((core::mem::size_of::<[debugfs_reg32; 8]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__5!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[xhci_file_map; 4]>() as u64) / (core::mem::size_of::<xhci_file_map>() as u64)).wrapping_add(__must_be_array__18!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__16 {
() => {
((core::mem::size_of::<[xhci_file_map; 3]>() as u64) / (core::mem::size_of::<xhci_file_map>() as u64)).wrapping_add(__must_be_array__13!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__20 {
() => {
((core::mem::size_of::<[debugfs_reg32; 8]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__25!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__24 {
() => {
((core::mem::size_of::<[debugfs_reg32; 8]>() as u64) / (core::mem::size_of::<debugfs_reg32>() as u64)).wrapping_add(__must_be_array__9!() as u64)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
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

macro_rules! CTX_SIZE {
($_hcc:expr) => {
if (($_hcc) as u64) & BIT!(2i32) != 0 { 64i32 } else { 32i32 }
};
}

macro_rules! CTX_TO_EP_INTERVAL {
($p:expr) => {
($p) >> 16i32 & 255
};
}

macro_rules! CTX_TO_EP_MAXPSTREAMS {
($p:expr) => {
(($p) & 31744u32) >> 10i32
};
}

macro_rules! CTX_TO_EP_MULT {
($p:expr) => {
($p) >> 8i32 & 3
};
}

macro_rules! CTX_TO_EP_TYPE {
($p:expr) => {
($p) >> 3i32 & 7
};
}

macro_rules! CTX_TO_MAX_BURST {
($p:expr) => {
($p) >> 8i32 & 255
};
}

/// hosts with LEC=1 use bits 31:24 as ESIT high bits.
macro_rules! CTX_TO_MAX_ESIT_PAYLOAD_HI {
($p:expr) => {
($p) >> 24i32 & 255
};
}

macro_rules! CTX_TO_MAX_ESIT_PAYLOAD {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

macro_rules! DEVINFO_TO_MAX_PORTS {
($p:expr) => {
(($p) & 4278190080u32) >> 24i32
};
}

macro_rules! DEVINFO_TO_ROOT_HUB_PORT {
($p:expr) => {
($p) >> 16i32 & 255
};
}

macro_rules! DEV_PORT_SPEED {
($p:expr) => {
($p) >> 10i32 & 15
};
}

macro_rules! DEV_PORT_SPEED__2 {
($p:expr) => {
($p) >> 10i32 & 15
};
}

/// tx_info bitmasks
macro_rules! EP_AVG_TRB_LENGTH {
($p:expr) => {
($p) & 65535
};
}

/// Transfer event TRB length bit mask
macro_rules! EVENT_TRB_LEN {
($p:expr) => {
($p) & 16777215
};
}

macro_rules! GENMASK_INPUT_CHECK__25 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_800>() as i32
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
macro_rules! GENMASK_TYPE__25 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__25!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__25!(u64, $h, $l)
};
}

macro_rules! GET_COMP_CODE {
($p:expr) => {
(($p) & 4278190080u32) >> 24i32
};
}

macro_rules! GET_FRAME_ID {
($p:expr) => {
($p) >> 20i32 & 2047
};
}

macro_rules! GET_INTR_TARGET {
($p:expr) => {
($p) >> 22i32 & 1023
};
}

macro_rules! GET_SLOT_STATE {
($p:expr) => {
(($p) & 4160749568u32) >> 27i32
};
}

macro_rules! GET_TBC {
($p:expr) => {
($p) >> 7i32 & 3
};
}

macro_rules! GET_TD_SIZE {
($p:expr) => {
(($p) & 4063232) >> 17i32
};
}

macro_rules! GET_TLBPC {
($p:expr) => {
($p) >> 16i32 & 15
};
}

macro_rules! GET_TT_THINK_TIME {
($p:expr) => {
(($p) & 196608u32) >> 16i32
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

macro_rules! MAX_PACKET_DECODED {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// eUSB2v2 protocol PORTLI Port Link information, RsvdP for normal USB2
macro_rules! PORTLI_RDR {
($p:expr) => {
($p) & 15
};
}

macro_rules! PORTLI_TDR {
($p:expr) => {
($p) >> 4i32 & 15
};
}

/// USB3 Protocol PORTLI  Port Link Information
macro_rules! PORT_LEC {
($p:expr) => {
($p) & 65535
};
}

macro_rules! PORT_RX_LANES {
($p:expr) => {
($p) >> 16i32 & 15
};
}

macro_rules! PORT_TX_LANES {
($p:expr) => {
($p) >> 20i32 & 15
};
}

macro_rules! TRB_FIELD_TO_TYPE {
($p:expr) => {
(($p) & 64512) >> 10i32
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

/// Set Latency Tolerance Value
macro_rules! TRB_TO_BELT {
($p:expr) => {
(($p) & 268369920u32) >> 16i32
};
}

/// Get Port Bandwidth
macro_rules! TRB_TO_DEV_SPEED {
($p:expr) => {
(($p) & 983040u32) >> 16i32
};
}

/// Endpoint ID 1 - 31
macro_rules! TRB_TO_EP_ID {
($p:expr) => {
($p) >> 16i32 & 31
};
}

/// Force Header
macro_rules! TRB_TO_PACKET_TYPE {
($p:expr) => {
($p) & 31
};
}

macro_rules! TRB_TO_ROOTHUB_PORT {
($p:expr) => {
(($p) & 4278190080u32) >> 24i32
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

macro_rules! TRB_TO_VF_ID {
($p:expr) => {
(($p) & 16711680u32) >> 16i32
};
}

/// Force Event
macro_rules! TRB_TO_VF_INTR_TARGET {
($p:expr) => {
(($p) & 4290772992u32) >> 22i32
};
}

macro_rules! UL {
() => {
_UL!()
};
}

/// Extended capability register fields
macro_rules! XHCI_EXT_CAPS_ID {
($p:expr) => {
($p) >> 0i32 & 255
};
}

macro_rules! XHCI_EXT_CAPS_NEXT {
($p:expr) => {
($p) >> 8i32 & 255
};
}

macro_rules! XHCI_EXT_PORT_PSIC {
($x:expr) => {
($x) >> 28i32 & 15
};
}

/// HCCPARAMS contains the first extended capability pointer
macro_rules! XHCI_HCC_EXT_CAPS {
($p:expr) => {
($p) >> 16i32 & 65535
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__2 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_462>() as i32
};
}

macro_rules! __must_be_array__4 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_816>() as i32
};
}

macro_rules! __must_be_array__5 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_791>() as i32
};
}

macro_rules! __must_be_array__9 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_796>() as i32
};
}

macro_rules! __must_be_array__10 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_806>() as i32
};
}

macro_rules! __must_be_array__13 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_775>() as i32
};
}

macro_rules! __must_be_array__18 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_449>() as i32
};
}

macro_rules! __must_be_array__23 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_752>() as i32
};
}

macro_rules! __must_be_array__25 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_801>() as i32
};
}

macro_rules! __must_be_array__26 {
() => {
core::mem::size_of::<__SlateRecord_xhci_debugfs_c_811>() as i32
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
macro_rules! container_of__14 {
($ptr:expr) => {
{
let mut __v: *mut xhci_regset;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__43!() as isize)) }) as *mut ()) as *mut xhci_regset;
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

/// debugfs_create_file - create a file in the debugfs filesystem
/// @name: a pointer to a string containing the name of the file to create.
/// @mode: the permission that the file should have.
/// @parent: a pointer to the parent dentry for this file.  This should be a
///          directory dentry if set.  If this parameter is NULL, then the
///          file will be created in the root of the debugfs filesystem.
/// @data: a pointer to something that the caller will want to get to later
///        on.  The inode.i_private pointer will point to this value on
///        the open() call.
/// @fops: a pointer to a struct file_operations or struct debugfs_short_fops that
///        should be used for this file.
///
/// This is the basic "create a file" function for debugfs.  It allows for a
/// wide range of flexibility in creating a file, or a directory (if you want
/// to create a directory, the debugfs_create_dir() function is
/// recommended to be used instead.)
///
/// This function will return a pointer to a dentry if it succeeds.  This
/// pointer must be passed to the debugfs_remove() function when the file is
/// to be removed (no automatic cleanup happens if your module is unloaded,
/// you are responsible here.)  If an error occurs, ERR_PTR(-ERROR) will be
/// returned.
///
/// If debugfs is not enabled in the kernel, the value -%ENODEV will be
/// returned.
///
/// If fops points to a struct debugfs_short_fops, then simple_open() will be
/// used for the open, and only read/write/llseek are supported and are proxied,
/// so no module reference or release are needed.
///
/// NOTE: it's expected that most callers should _ignore_ the errors returned
/// by this function. Other debugfs functions handle the fact that the "dentry"
/// passed to them could be an error and they don't crash in that case.
/// Drivers should generally work fine even if debugfs fails to init anyway.
macro_rules! debugfs_create_file {
($name:expr, $mode:expr, $parent:expr, $data:expr, $fops:expr) => {
unsafe { debugfs_create_file_full(($name) as *const u8, ($mode) as u16, $parent, ($data) as *mut (), core::ptr::null::<()>(), $fops) };
};
}

macro_rules! debugfs_create_file_aux {
($name:expr, $mode:expr, $parent:expr, $data:expr, $aux:expr, $fops:expr) => {
unsafe { debugfs_create_file_full($name, ($mode) as u16, $parent, $data, ($aux) as *const (), $fops) };
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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
macro_rules! list_entry__39 {
($ptr:expr) => {
container_of__14!($ptr)
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
macro_rules! list_first_entry__21 {
($ptr:expr) => {
list_entry__39!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__13 {
($pos:expr, $member:ident) => {
list_entry__39!(unsafe { (*$pos).$member.next })
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

macro_rules! offsetof__43 {
() => {
core::mem::offset_of!(xhci_regset, list) as u64
};
}

macro_rules! offsetof__2 {
() => {
0u64
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

macro_rules! va_end {
() => {
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut usb_debug_root: *mut dentry;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn kstrtou16_from_user(arg0: *const u8, arg1: u64, arg2: u32, arg3: *mut u16) -> i32;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn vsnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, arg3: core::ffi::VaList<'static>) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_57();
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn usb_hcd_is_primary_hcd(arg0: *mut usb_hcd) -> i32;
    fn xhci_alloc_port_bw_ctx(arg0: *mut xhci_hcd, arg1: u32) -> *mut xhci_container_ctx;
    fn xhci_free_port_bw_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx);
    fn xhci_trb_virt_to_dma(arg0: *mut xhci_segment, arg1: *mut xhci_trb) -> u64;
    fn xhci_get_port_bandwidth(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: u8) -> i32;
    fn xhci_portsc_writel(arg0: *mut xhci_port, arg1: u32);
    fn xhci_portsc_readl(arg0: *mut xhci_port) -> u32;
    fn xhci_port_state_to_neutral(arg0: u32) -> u32;
    fn xhci_get_slot_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx) -> *mut xhci_slot_ctx;
    fn xhci_get_ep_ctx(arg0: *mut xhci_hcd, arg1: *mut xhci_container_ctx, arg2: u32) -> *mut xhci_ep_ctx;
    fn seq_read(arg0: *mut file, arg1: *mut u8, arg2: u64, arg3: *mut i64) -> i64;
    fn seq_lseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn single_open(arg0: *mut file, arg1: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn single_release(arg0: *mut inode, arg1: *mut file) -> i32;
    fn debugfs_create_file_full(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const file_operations) -> *mut dentry;
    fn debugfs_create_file_short(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const debugfs_short_fops) -> *mut dentry;
    fn debugfs_create_dir(arg0: *const u8, arg1: *mut dentry) -> *mut dentry;
    fn debugfs_remove(arg0: *mut dentry);
    fn debugfs_get_aux(arg0: *const file) -> *mut ();
    fn debugfs_create_regset32(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut debugfs_regset32);
    fn __compiletime_assert_469();
    fn __compiletime_assert_472();
    fn __compiletime_assert_473();
    fn __compiletime_assert_474();
    fn __compiletime_assert_475();
    fn __compiletime_assert_476();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
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
    __slate_bits_0: __slate_bits::__SlateBits95U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits96U0,
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


fn INIT_LIST_HEAD(list: *mut list_head) {
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
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
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
unsafe { __compiletime_assert_18() };
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
unsafe { __compiletime_assert_21() };
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
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_57() };
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
struct perf_event {
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
    __slate_bits_0: __slate_bits::__SlateBits106U0,
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct pollfd {
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
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
    __slate_bits_0: __slate_bits::__SlateBits265U0,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
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
struct dev_pagemap {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits383U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
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
struct __SlateRecord_slab_h_532 {
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("210: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 210b - ., 3; .popsection\n# {0}", const 210 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("211: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 211b - ., 4; .popsection\n# {0}", const 211 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
check_object_size(addr, bytes, is_source);
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[inline(always)]
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
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


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct ns_common {
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
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits457U0,
    vm_ops: *const vm_operations_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits459U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits484U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits484U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits484U2,
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
    __slate_bits_3: __slate_bits::__SlateBits484U3,
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
struct folio_queue {
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
struct sg_table {
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
struct mnt_idmap {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits268U0,
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
unsafe { __compiletime_assert_395() };
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
unsafe { __compiletime_assert_416() };
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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

fn file_dentry(file: *const file) -> *mut dentry {
let dentry: *mut dentry = unsafe { (*file).__slate_anon_10.f_path.dentry };
let mut __v0: i64;
let __ret_warn_on: i32 = (d_inode(dentry as *const dentry) != file_inode(file)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("429: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 429b - ., 3; .popsection\n# {0}", const 429 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_file_dentry_1, sym __slate_asm_string_file_dentry_2, const 1363 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("430: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 430b - ., 4; .popsection\n# {0}", const 430 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
dentry
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
struct ep_device {
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
struct mon_bus {
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
struct scatterlist {
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
    __slate_bits_0: __slate_bits::__SlateBits481U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn pm_runtime_put_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_idle(dev, 4i32) }
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
    __slate_bits_0: __slate_bits::__SlateBits709U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_sideband {
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
    __slate_bits_0: __slate_bits::__SlateBits713U0,
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
    __slate_bits_0: __slate_bits::__SlateBits703U0,
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
    __slate_bits_1: __slate_bits::__SlateBits703U1,
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
    __slate_bits_2: __slate_bits::__SlateBits703U2,
}


// urb->transfer_dma valid on submit
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
    __slate_bits_0: __slate_bits::__SlateBits721U0,
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
    __slate_bits_0: __slate_bits::__SlateBits726U0,
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
struct clk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reset_control {
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


// Extended capability IDs - ID 0 reserved
// IDs 6-9 reserved
fn xhci_find_next_ext_cap(base: *mut (), start: u32, id: i32) -> i32 {
let mut val: u32 = 0;
let mut next: u32 = 0;
let mut offset: u32 = start;
if start == 0 || start == 16 {
val = readl(((unsafe { (base as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if val == 4294967295u32 {
return 0;
}
offset = XHCI_HCC_EXT_CAPS!(val) << 2i32;
if offset == 0 {
return 0;
}
}
loop {
val = readl(((unsafe { (base as *const u8).wrapping_offset(offset as isize) }) as *mut ()) as *const ());
if val == 4294967295u32 {
return 0;
}
if offset != start && (id == 0 || XHCI_EXT_CAPS_ID!(val) == (id as u32)) {
return offset as i32;
}
next = XHCI_EXT_CAPS_NEXT!(val);
offset = offset.wrapping_add(next << 2i32);
if next == 0 {
break;
}
}
0
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_run_regs {
    microframe_index: u32,
    rsvd: [u32; 7],
    ir_set: [xhci_intr_reg; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_449 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_462 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_752 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_775 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_791 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_795_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_796 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_800 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_801 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_806 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_811 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_xhci_debugfs_c_816 {
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
    pub struct __SlateBits265U0 {
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
    pub struct __SlateBits268U0 {
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
    pub struct __SlateBits96U0 {
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
    pub struct __SlateBits106U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits457U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits459U0 {
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
    pub struct __SlateBits95U0 {
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
    pub struct __SlateBits383U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits484U0 {
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
    pub struct __SlateBits481U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits484U1 {
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
    pub struct __SlateBits484U2 {
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
    pub struct __SlateBits484U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits713U0 {
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
    pub struct __SlateBits709U0 {
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
    pub struct __SlateBits703U0 {
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
    pub struct __SlateBits703U1 {
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
    pub struct __SlateBits703U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits721U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits726U0 {
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
    pub struct __SlateBits757U0 {
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
    pub struct __SlateBits774U0 {
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
// Link TRB specific fields
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
fn xhci_trb_type_string(r#type: u8) -> *const u8 {
match r#type as i32 {
1 => {
return c"Normal".as_ptr() as *const u8;
}
2 => {
return c"Setup Stage".as_ptr() as *const u8;
}
3 => {
return c"Data Stage".as_ptr() as *const u8;
}
4 => {
return c"Status Stage".as_ptr() as *const u8;
}
5 => {
return c"Isoch".as_ptr() as *const u8;
}
6 => {
return c"Link".as_ptr() as *const u8;
}
7 => {
return c"Event Data".as_ptr() as *const u8;
}
8 => {
return c"No-Op".as_ptr() as *const u8;
}
9 => {
return c"Enable Slot Command".as_ptr() as *const u8;
}
10 => {
return c"Disable Slot Command".as_ptr() as *const u8;
}
11 => {
return c"Address Device Command".as_ptr() as *const u8;
}
12 => {
return c"Configure Endpoint Command".as_ptr() as *const u8;
}
13 => {
return c"Evaluate Context Command".as_ptr() as *const u8;
}
14 => {
return c"Reset Endpoint Command".as_ptr() as *const u8;
}
15 => {
return c"Stop Ring Command".as_ptr() as *const u8;
}
16 => {
return c"Set TR Dequeue Pointer Command".as_ptr() as *const u8;
}
17 => {
return c"Reset Device Command".as_ptr() as *const u8;
}
18 => {
return c"Force Event Command".as_ptr() as *const u8;
}
19 => {
return c"Negotiate Bandwidth Command".as_ptr() as *const u8;
}
20 => {
return c"Set Latency Tolerance Value Command".as_ptr() as *const u8;
}
21 => {
return c"Get Port Bandwidth Command".as_ptr() as *const u8;
}
22 => {
return c"Force Header Command".as_ptr() as *const u8;
}
23 => {
return c"No-Op Command".as_ptr() as *const u8;
}
32 => {
return c"Transfer Event".as_ptr() as *const u8;
}
33 => {
return c"Command Completion Event".as_ptr() as *const u8;
}
34 => {
return c"Port Status Change Event".as_ptr() as *const u8;
}
35 => {
return c"Bandwidth Request Event".as_ptr() as *const u8;
}
36 => {
return c"Doorbell Event".as_ptr() as *const u8;
}
37 => {
return c"Host Controller Event".as_ptr() as *const u8;
}
38 => {
return c"Device Notification Event".as_ptr() as *const u8;
}
39 => {
return c"MFINDEX Wrap Event".as_ptr() as *const u8;
}
48 => {
return c"NEC Command Completion Event".as_ptr() as *const u8;
}
49 => {
return c"NET Get Firmware Revision Command".as_ptr() as *const u8;
}
_ => {
return c"UNKNOWN".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

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
    __slate_bits_0: __slate_bits::__SlateBits757U0,
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
    __slate_bits_0: __slate_bits::__SlateBits774U0,
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

fn xhci_slot_state_string(state: u32) -> *mut u8 {
match state {
0 => {
return c"enabled/disabled".as_ptr() as *mut u8;
}
1 => {
return c"default".as_ptr() as *mut u8;
}
2 => {
return c"addressed".as_ptr() as *mut u8;
}
3 => {
return c"configured".as_ptr() as *mut u8;
}
_ => {
return c"reserved".as_ptr() as *mut u8;
}
}
unsafe { core::mem::zeroed() }
}

fn xhci_decode_trb(str: *mut u8, size: u64, field0: u32, field1: u32, field2: u32, field3: u32) -> *const u8 {
let r#type: i32 = TRB_FIELD_TO_TYPE!(field3) as i32;
match r#type {
6 => {
unsafe { snprintf(str, size, c"LINK %08x%08x intr %d type '%s' flags %c:%c:%c:%c".as_ptr() as *const u8, field1, field0, GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(4i32) != 0 { 67i32 } else { 99i32 }, if (field3 as u64) & BIT!(1i32) != 0 { 84i32 } else { 116i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
32..=39 => {
unsafe { snprintf(str, size, c"TRB %08x%08x status '%s' len %d slot %d ep %d type '%s' flags %c:%c".as_ptr() as *const u8, field1, field0, xhci_trb_comp_code_string(GET_COMP_CODE!(field2) as u8), EVENT_TRB_LEN!(field2), TRB_TO_SLOT_ID!(field3), TRB_TO_EP_ID!(field3), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(2i32) != 0 { 69i32 } else { 101i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
2 => {
unsafe { snprintf(str, size, c"bRequestType %02x bRequest %02x wValue %02x%02x wIndex %02x%02x wLength %d length %d TD size %d intr %d type '%s' flags %c:%c:%c".as_ptr() as *const u8, field0 & 255, (field0 & 65280) >> 8i32, (field0 & 4278190080) >> 24i32, (field0 & 16711680) >> 16i32, (field1 & 65280) >> 8i32, field1 & 255, (field1 & 4278190080) >> 16i32 | (field1 & 16711680) >> 16i32, TRB_LEN!(field2), GET_TD_SIZE!(field2), GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(6i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
3 => {
unsafe { snprintf(str, size, c"Buffer %08x%08x length %d TD size %d intr %d type '%s' flags %c:%c:%c:%c:%c:%c:%c".as_ptr() as *const u8, field1, field0, TRB_LEN!(field2), GET_TD_SIZE!(field2), GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(6i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(4i32) != 0 { 67i32 } else { 99i32 }, if (field3 as u64) & BIT!(3i32) != 0 { 83i32 } else { 115i32 }, if (field3 as u64) & BIT!(2i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(1i32) != 0 { 69i32 } else { 101i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
4 => {
unsafe { snprintf(str, size, c"Buffer %08x%08x length %d TD size %d intr %d type '%s' flags %c:%c:%c:%c".as_ptr() as *const u8, field1, field0, TRB_LEN!(field2), GET_TD_SIZE!(field2), GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(4i32) != 0 { 67i32 } else { 99i32 }, if (field3 as u64) & BIT!(1i32) != 0 { 69i32 } else { 101i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
1 | 7 | 8 => {
unsafe { snprintf(str, size, c"Buffer %08x%08x length %d TD size %d intr %d type '%s' flags %c:%c:%c:%c:%c:%c:%c:%c".as_ptr() as *const u8, field1, field0, TRB_LEN!(field2), GET_TD_SIZE!(field2), GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(9i32) != 0 { 66i32 } else { 98i32 }, if (field3 as u64) & BIT!(6i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(4i32) != 0 { 67i32 } else { 99i32 }, if (field3 as u64) & BIT!(3i32) != 0 { 83i32 } else { 115i32 }, if (field3 as u64) & BIT!(2i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(1i32) != 0 { 69i32 } else { 101i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
5 => {
unsafe { snprintf(str, size, c"Buffer %08x%08x length %d TD size/TBC %d intr %d type '%s' TBC %u TLBPC %u frame_id %u flags %c:%c:%c:%c:%c:%c:%c:%c:%c".as_ptr() as *const u8, field1, field0, TRB_LEN!(field2), GET_TD_SIZE!(field2), GET_INTR_TARGET!(field2), xhci_trb_type_string(r#type as u8), GET_TBC!(field3), GET_TLBPC!(field3), GET_FRAME_ID!(field3), if (field3 as u64) & BIT!(31i32) != 0 { 83i32 } else { 115i32 }, if (field3 as u64) & BIT!(9i32) != 0 { 66i32 } else { 98i32 }, if (field3 as u64) & BIT!(6i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(5i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(4i32) != 0 { 67i32 } else { 99i32 }, if (field3 as u64) & BIT!(3i32) != 0 { 83i32 } else { 115i32 }, if (field3 as u64) & BIT!(2i32) != 0 { 73i32 } else { 105i32 }, if (field3 as u64) & BIT!(1i32) != 0 { 69i32 } else { 101i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
9 | 23 => {
unsafe { snprintf(str, size, c"%s: flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
10 | 19 => {
unsafe { snprintf(str, size, c"%s: slot %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), TRB_TO_SLOT_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
11 => {
unsafe { snprintf(str, size, c"%s: ctx %08x%08x slot %d flags %c:%c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_SLOT_ID!(field3), if (field3 as u64) & BIT!(9i32) != 0 { 66i32 } else { 98i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
12 => {
unsafe { snprintf(str, size, c"%s: ctx %08x%08x slot %d flags %c:%c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_SLOT_ID!(field3), if (field3 as u64) & BIT!(9i32) != 0 { 68i32 } else { 100i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
13 => {
unsafe { snprintf(str, size, c"%s: ctx %08x%08x slot %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_SLOT_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
14 => {
unsafe { snprintf(str, size, c"%s: ctx %08x%08x slot %d ep %d flags %c:%c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_SLOT_ID!(field3), TRB_TO_EP_ID!(field3), if (field3 as u64) & BIT!(9i32) != 0 { 84i32 } else { 116i32 }, if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
15 => {
unsafe { snprintf(str, size, c"%s: slot %d sp %d ep %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), TRB_TO_SLOT_ID!(field3), TRB_TO_SUSPEND_PORT!(field3), TRB_TO_EP_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
16 => {
unsafe { snprintf(str, size, c"%s: deq %08x%08x stream %d slot %d ep %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_STREAM_ID!(field2), TRB_TO_SLOT_ID!(field3), TRB_TO_EP_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
17 => {
unsafe { snprintf(str, size, c"%s: slot %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), TRB_TO_SLOT_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
18 => {
unsafe { snprintf(str, size, c"%s: event %08x%08x vf intr %d vf id %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_VF_INTR_TARGET!(field2), TRB_TO_VF_ID!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
20 => {
unsafe { snprintf(str, size, c"%s: belt %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), TRB_TO_BELT!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
21 => {
unsafe { snprintf(str, size, c"%s: ctx %08x%08x slot %d speed %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field1, field0, TRB_TO_SLOT_ID!(field3), TRB_TO_DEV_SPEED!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
22 => {
unsafe { snprintf(str, size, c"%s: info %08x%08x%08x pkt type %d roothub port %d flags %c".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field2, field1, field0 & 4294967264, TRB_TO_PACKET_TYPE!(field0), TRB_TO_ROOTHUB_PORT!(field3), if (field3 as u64) & BIT!(0i32) != 0 { 67i32 } else { 99i32 }) };
}
_ => {
unsafe { snprintf(str, size, c"type '%s' -> raw %08x %08x %08x %08x".as_ptr() as *const u8, xhci_trb_type_string(r#type as u8), field0, field1, field2, field3) };
}
}
str as *const u8
}

fn xhci_decode_slot_context(str: *mut u8, info: u32, info2: u32, tt_info: u32, state: u32) -> *const u8 {
let speed: u32 = info & 15728640u32;
let hub: u32 = ((info as u64) & BIT!(26i32)) as u32;
let mtt: u32 = ((info as u64) & BIT!(25i32)) as u32;
let mut ret: i32 = 0;
let mut __v0: *mut u8;
let mut s: *mut u8 = unsafe { core::mem::zeroed() };
match speed {
1048576 => {
s = c"full-speed".as_ptr() as *mut u8;
}
2097152 => {
s = c"low-speed".as_ptr() as *mut u8;
}
3145728 => {
s = c"high-speed".as_ptr() as *mut u8;
}
4194304 => {
s = c"super-speed".as_ptr() as *mut u8;
}
5242880 => {
s = c"super-speed plus".as_ptr() as *mut u8;
}
_ => {
s = c"UNKNOWN speed".as_ptr() as *mut u8;
}
}
__v0 = s;
ret = unsafe { sprintf(str, c"RS %05x %s%s%s Ctx Entries %d MEL %d us Port# %d/%d".as_ptr() as *const u8, info & 1048575, __v0, if mtt != 0 { c" multi-TT".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if hub != 0 { c" Hub".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, (info & 4160749568u32) >> 27i32, info2 & 65535, DEVINFO_TO_ROOT_HUB_PORT!(info2), DEVINFO_TO_MAX_PORTS!(info2)) };
let __v2: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c" [TT Slot %d Port# %d TTT %d Intr %d] Addr %d State %s".as_ptr() as *const u8, tt_info & 255, (tt_info & 65280u32) >> 8i32, GET_TT_THINK_TIME!(tt_info), GET_INTR_TARGET!(tt_info), state & 255, xhci_slot_state_string(GET_SLOT_STATE!(state))) };
ret = ret.wrapping_add(__v2);
str as *const u8
}

fn xhci_portsc_link_state_string(portsc: u32) -> *const u8 {
match portsc & 480u32 {
0 => {
return c"U0".as_ptr() as *const u8;
}
32 => {
return c"U1".as_ptr() as *const u8;
}
64 => {
return c"U2".as_ptr() as *const u8;
}
96 => {
return c"U3".as_ptr() as *const u8;
}
128 => {
return c"Disabled".as_ptr() as *const u8;
}
160 => {
return c"RxDetect".as_ptr() as *const u8;
}
192 => {
return c"Inactive".as_ptr() as *const u8;
}
224 => {
return c"Polling".as_ptr() as *const u8;
}
256 => {
return c"Recovery".as_ptr() as *const u8;
}
288 => {
return c"Hot Reset".as_ptr() as *const u8;
}
320 => {
return c"Compliance mode".as_ptr() as *const u8;
}
352 => {
return c"Test mode".as_ptr() as *const u8;
}
480 => {
return c"Resume".as_ptr() as *const u8;
}
_ => {
}
}
c"Unknown".as_ptr() as *const u8
}

fn xhci_decode_portsc(str: *mut u8, portsc: u32) -> *const u8 {
let mut ret: i32 = unsafe { sprintf(str, c"0x%08x ".as_ptr() as *const u8, portsc) };
if portsc == !0 {
return str as *const u8;
}
let __v0: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"Speed=%d ".as_ptr() as *const u8, DEV_PORT_SPEED!(portsc)) };
ret = ret.wrapping_add(__v0);
let __v3: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"Link=%s ".as_ptr() as *const u8, xhci_portsc_link_state_string(portsc)) };
ret = ret.wrapping_add(__v3);
// RO/ROS: Read-only
if portsc & 1u32 != 0 {
let __v6: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"CCS ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v6);
}
if portsc & 8u32 != 0 {
let __v9: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"OCA ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v9);
}
// No set for USB2 ports
if portsc & 16777216u32 != 0 {
let __v12: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"CAS ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v12);
}
if portsc & 1073741824u32 != 0 {
let __v15: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"DR ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v15);
}
// RWS; writing 1 sets the bit, writing 0 clears the bit.
if portsc & 512u32 != 0 {
let __v18: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PP ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v18);
}
if portsc & 33554432u32 != 0 {
let __v21: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"WCE ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v21);
}
if portsc & 67108864u32 != 0 {
let __v24: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"WDE ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v24);
}
if portsc & 134217728u32 != 0 {
let __v27: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"WOE ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v27);
}
// RW; writing 1 sets the bit, writing 0 clears the bit
if portsc & 65536u32 != 0 {
let __v30: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"LWS ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v30);
}
// LWS 0 write is ignored
// RW1S; writing 1 sets the bit, writing 0 has no effect
if portsc & 16u32 != 0 {
let __v33: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PR ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v33);
}
if portsc & 2147483648u32 != 0 {
let __v36: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"WPR ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v36);
}
// RsvdZ for USB2 ports
// RW1CS; writing 1 clears the bit, writing 0 has no effect.
if portsc & 2u32 != 0 {
let __v39: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PED ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v39);
}
if portsc & 131072u32 != 0 {
let __v42: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"CSC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v42);
}
if portsc & 262144u32 != 0 {
let __v45: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PEC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v45);
}
// No set for USB3 ports
if portsc & 524288u32 != 0 {
let __v48: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"WRC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v48);
}
// RsvdZ for USB2 ports
if portsc & 1048576u32 != 0 {
let __v51: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"OCC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v51);
}
if portsc & 2097152u32 != 0 {
let __v54: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PRC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v54);
}
if portsc & 4194304u32 != 0 {
let __v57: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"PLC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v57);
}
if portsc & 8388608u32 != 0 {
let __v60: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"CEC ".as_ptr() as *const u8) };
ret = ret.wrapping_add(__v60);
}
// RsvdZ for USB2 ports
str as *const u8
}

fn xhci_ep_state_string(state: u8) -> *const u8 {
match state as i32 {
0 => {
return c"disabled".as_ptr() as *const u8;
}
1 => {
return c"running".as_ptr() as *const u8;
}
2 => {
return c"halted".as_ptr() as *const u8;
}
3 => {
return c"stopped".as_ptr() as *const u8;
}
4 => {
return c"error".as_ptr() as *const u8;
}
_ => {
return c"INVALID".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn xhci_ep_type_string(r#type: u8) -> *const u8 {
match r#type as i32 {
1 => {
return c"Isoc OUT".as_ptr() as *const u8;
}
2 => {
return c"Bulk OUT".as_ptr() as *const u8;
}
3 => {
return c"Int OUT".as_ptr() as *const u8;
}
4 => {
return c"Ctrl".as_ptr() as *const u8;
}
5 => {
return c"Isoc IN".as_ptr() as *const u8;
}
6 => {
return c"Bulk IN".as_ptr() as *const u8;
}
7 => {
return c"Int IN".as_ptr() as *const u8;
}
_ => {
return c"INVALID".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn xhci_decode_ep_context(str: *mut u8, info: u32, info2: u32, deq: u64, tx_info: u32) -> *const u8 {
let mut ret: i32 = 0;
let esit: u32 = CTX_TO_MAX_ESIT_PAYLOAD_HI!(info) << 16i32 | CTX_TO_MAX_ESIT_PAYLOAD!(tx_info);
let mut maxp: u16 = 0;
let mut avg: u16 = 0;
let mut max_pstr: u8 = 0;
let ep_state: u8 = (info & 7) as u8;
let mut interval: u8 = 0;
let mut ep_type: u8 = 0;
let mut burst: u8 = 0;
let mut cerr: u8 = 0;
let mut mult: u8 = 0;
let mut lsa: bool = false;
let mut hid: bool = false;
max_pstr = CTX_TO_EP_MAXPSTREAMS!(info) as u8;
interval = CTX_TO_EP_INTERVAL!(info) as u8;
mult = CTX_TO_EP_MULT!(info).wrapping_add(1) as u8;
lsa = (info as u64) & BIT!(15i32) != 0;
cerr = ((info2 & 6u32) >> 1i32) as u8;
ep_type = CTX_TO_EP_TYPE!(info2) as u8;
hid = info2 & 128u32 != 0;
burst = CTX_TO_MAX_BURST!(info2) as u8;
maxp = MAX_PACKET_DECODED!(info2) as u16;
avg = EP_AVG_TRB_LENGTH!(tx_info) as u16;
ret = unsafe { sprintf(str, c"State %s mult %d max P. Streams %d %s".as_ptr() as *const u8, xhci_ep_state_string(ep_state), mult as i32, max_pstr as i32, if lsa { c"LSA ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
let __v0: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"interval %d us max ESIT payload %d CErr %d ".as_ptr() as *const u8, (1i32 << (interval as i32)).wrapping_mul(125), esit, cerr as i32) };
ret = ret.wrapping_add(__v0);
let __v3: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"Type %s %sburst %d maxp %d deq %016llx ".as_ptr() as *const u8, xhci_ep_type_string(ep_type), if hid { c"HID".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, burst as i32, maxp as i32, deq) };
ret = ret.wrapping_add(__v3);
let __v6: i32 = unsafe { sprintf(str.wrapping_offset(ret as isize), c"avg trb len %d".as_ptr() as *const u8, avg as i32) };
ret = ret.wrapping_add(__v6);
str as *const u8
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


#[inline(always)]
fn seq_puts(m: *mut seq_file, s: *const u8) {
if !unsafe { __slate_intrinsic_70b71f185ebb8c4a(*s) } {
unsafe { __seq_puts(m, s) };
} else {
if (unsafe { *s.wrapping_offset(0 as isize) }) != 0 && (unsafe { *s.wrapping_offset(1 as isize) }) == 0 {
unsafe { seq_putc(m, *s.wrapping_offset(0 as isize)) };
} else {
unsafe { seq_write(m, s as *const (), __builtin_strlen(s)) };
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct debugfs_reg32 {
    name: *mut u8,
    offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct debugfs_regset32 {
    regs: *const debugfs_reg32,
    nregs: i32,
    base: *mut (),
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct debugfs_short_fops {
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_regset {
    name: [u8; 32],
    regset: debugfs_regset32,
    nregs: u64,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_file_map {
    name: *const u8,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_ep_priv {
    name: [u8; 32],
    root: *mut dentry,
    stream_info: *mut xhci_stream_info,
    show_ring: *mut xhci_ring,
    stream_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xhci_slot_priv {
    name: [u8; 32],
    root: *mut dentry,
    eps: [*mut xhci_ep_priv; 31],
    dev: *mut xhci_virt_device,
}


#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_init(xhci: *mut xhci_hcd) {
let dev: *mut device = unsafe { (*xhci_to_hcd(xhci)).self_.controller };
unsafe {
(*xhci).debugfs_root = debugfs_create_dir(dev_name(dev as *const device), xhci_debugfs_root);
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*xhci).regset_list) });
unsafe { xhci_debugfs_regset(xhci, 0u32, core::ptr::addr_of!(xhci_cap_regs.0) as *const debugfs_reg32, ARRAY_SIZE__11!(), (*xhci).debugfs_root, c"reg-cap".as_ptr() as *const u8) };
let mut __v0: u64;
loop {
if __bf_cast_unsigned__2!(((((const { core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) > (__bf_cast_unsigned!(18446744073709551615u64) as u64) {
unsafe { __compiletime_assert_472() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_473() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) == 0 {
unsafe { __compiletime_assert_474() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) >> ((core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_475() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_476() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((readl((unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).hc_capbase) }) as *const ()) as u64) & (core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))) >> ((core::mem::size_of::<__SlateRecord_xhci_debugfs_c_795_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)).trailing_zeros() as i32);
__v0 = __v2;
unsafe { xhci_debugfs_regset(xhci, __v0 as u32, core::ptr::addr_of!(xhci_op_regs.0) as *const debugfs_reg32, ARRAY_SIZE__24!(), (*xhci).debugfs_root, c"reg-op".as_ptr() as *const u8) };
unsafe { xhci_debugfs_regset(xhci, ((readl(core::ptr::addr_of_mut!((*(*xhci).cap_regs).run_regs_off) as *const ()) as u64) & GENMASK__2!(31i32, 5i32)) as u32, core::ptr::addr_of!(xhci_runtime_regs.0) as *const debugfs_reg32, ARRAY_SIZE__20!(), (*xhci).debugfs_root, c"reg-runtime".as_ptr() as *const u8) };
xhci_debugfs_extcap_regset(xhci, 1i32, unsafe { core::ptr::addr_of!(xhci_extcap_legsup.0) as *const debugfs_reg32 }, ARRAY_SIZE__7!(), c"reg-ext-legsup".as_ptr() as *const u8);
xhci_debugfs_extcap_regset(xhci, 2i32, unsafe { core::ptr::addr_of!(xhci_extcap_protocol.0) as *const debugfs_reg32 }, ARRAY_SIZE__5!(), c"reg-ext-protocol".as_ptr() as *const u8);
xhci_debugfs_extcap_regset(xhci, 10i32, unsafe { core::ptr::addr_of!(xhci_extcap_dbc.0) as *const debugfs_reg32 }, ARRAY_SIZE__9!(), c"reg-ext-dbc".as_ptr() as *const u8);
xhci_debugfs_create_ring_dir(xhci, unsafe { core::ptr::addr_of_mut!((*xhci).cmd_ring) }, c"command-ring".as_ptr() as *const u8, unsafe { (*xhci).debugfs_root });
xhci_debugfs_create_ring_dir(xhci, unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*xhci).interrupters }.wrapping_offset(0 as isize))).event_ring) }, c"event-ring".as_ptr() as *const u8, unsafe { (*xhci).debugfs_root });
unsafe {
(*xhci).debugfs_slots = debugfs_create_dir(c"devices".as_ptr() as *const u8, (*xhci).debugfs_root);
}
xhci_debugfs_create_ports(xhci, unsafe { (*xhci).debugfs_root });
xhci_debugfs_create_bandwidth(xhci, unsafe { (*xhci).debugfs_root });
}

static xhci_cap_regs: __SlateAlign16<[debugfs_reg32; 8]> = __SlateAlign16([debugfs_reg32 { name: b"CAPLENGTH\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"HCSPARAMS1\0".as_ptr() as *mut u8, offset: 4u64 }, debugfs_reg32 { name: b"HCSPARAMS2\0".as_ptr() as *mut u8, offset: 8u64 }, debugfs_reg32 { name: b"HCSPARAMS3\0".as_ptr() as *mut u8, offset: 12u64 }, debugfs_reg32 { name: b"HCCPARAMS1\0".as_ptr() as *mut u8, offset: 16u64 }, debugfs_reg32 { name: b"DOORBELLOFF\0".as_ptr() as *mut u8, offset: 20u64 }, debugfs_reg32 { name: b"RUNTIMEOFF\0".as_ptr() as *mut u8, offset: 24u64 }, debugfs_reg32 { name: b"HCCPARAMS2\0".as_ptr() as *mut u8, offset: 28u64 }]);

static xhci_op_regs: __SlateAlign16<[debugfs_reg32; 8]> = __SlateAlign16([debugfs_reg32 { name: b"USBCMD\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"USBSTS\0".as_ptr() as *mut u8, offset: 4u64 }, debugfs_reg32 { name: b"PAGESIZE\0".as_ptr() as *mut u8, offset: 8u64 }, debugfs_reg32 { name: b"DNCTRL\0".as_ptr() as *mut u8, offset: 20u64 }, debugfs_reg32 { name: b"CRCR\0".as_ptr() as *mut u8, offset: 24u64 }, debugfs_reg32 { name: b"DCBAAP_LOW\0".as_ptr() as *mut u8, offset: 48u64 }, debugfs_reg32 { name: b"DCBAAP_HIGH\0".as_ptr() as *mut u8, offset: 52u64 }, debugfs_reg32 { name: b"CONFIG\0".as_ptr() as *mut u8, offset: 56u64 }]);

static xhci_runtime_regs: __SlateAlign16<[debugfs_reg32; 8]> = __SlateAlign16([debugfs_reg32 { name: b"MFINDEX\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"IR0_IMAN\0".as_ptr() as *mut u8, offset: 32u64 }, debugfs_reg32 { name: b"IR0_IMOD\0".as_ptr() as *mut u8, offset: 36u64 }, debugfs_reg32 { name: b"IR0_ERSTSZ\0".as_ptr() as *mut u8, offset: 40u64 }, debugfs_reg32 { name: b"IR0_ERSTBA_LOW\0".as_ptr() as *mut u8, offset: 48u64 }, debugfs_reg32 { name: b"IR0_ERSTBA_HIGH\0".as_ptr() as *mut u8, offset: 52u64 }, debugfs_reg32 { name: b"IR0_ERDP_LOW\0".as_ptr() as *mut u8, offset: 56u64 }, debugfs_reg32 { name: b"IR0_ERDP_HIGH\0".as_ptr() as *mut u8, offset: 60u64 }]);

static xhci_extcap_legsup: __SlateAlign16<[debugfs_reg32; 2]> = __SlateAlign16([debugfs_reg32 { name: b"EXTCAP_USBLEGSUP\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"EXTCAP_USBLEGCTLSTS\0".as_ptr() as *mut u8, offset: 4u64 }]);

static xhci_extcap_protocol: __SlateAlign16<[debugfs_reg32; 10]> = __SlateAlign16([debugfs_reg32 { name: b"EXTCAP_REVISION\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"EXTCAP_NAME\0".as_ptr() as *mut u8, offset: 4u64 }, debugfs_reg32 { name: b"EXTCAP_PORTINFO\0".as_ptr() as *mut u8, offset: 8u64 }, debugfs_reg32 { name: b"EXTCAP_PORTTYPE\0".as_ptr() as *mut u8, offset: 12u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA1\0".as_ptr() as *mut u8, offset: 16u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA2\0".as_ptr() as *mut u8, offset: 20u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA3\0".as_ptr() as *mut u8, offset: 24u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA4\0".as_ptr() as *mut u8, offset: 28u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA5\0".as_ptr() as *mut u8, offset: 32u64 }, debugfs_reg32 { name: b"EXTCAP_MANTISSA6\0".as_ptr() as *mut u8, offset: 36u64 }]);

static xhci_extcap_dbc: __SlateAlign16<[debugfs_reg32; 14]> = __SlateAlign16([debugfs_reg32 { name: b"EXTCAP_DBC_CAPABILITY\0".as_ptr() as *mut u8, offset: 0u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_DOORBELL\0".as_ptr() as *mut u8, offset: 4u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_ERSTSIZE\0".as_ptr() as *mut u8, offset: 8u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_ERST_LOW\0".as_ptr() as *mut u8, offset: 16u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_ERST_HIGH\0".as_ptr() as *mut u8, offset: 20u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_ERDP_LOW\0".as_ptr() as *mut u8, offset: 24u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_ERDP_HIGH\0".as_ptr() as *mut u8, offset: 28u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_CONTROL\0".as_ptr() as *mut u8, offset: 32u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_STATUS\0".as_ptr() as *mut u8, offset: 36u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_PORTSC\0".as_ptr() as *mut u8, offset: 40u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_CONT_LOW\0".as_ptr() as *mut u8, offset: 48u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_CONT_HIGH\0".as_ptr() as *mut u8, offset: 52u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_DEVINFO1\0".as_ptr() as *mut u8, offset: 56u64 }, debugfs_reg32 { name: b"EXTCAP_DBC_DEVINFO2\0".as_ptr() as *mut u8, offset: 60u64 }]);

static mut xhci_debugfs_root: *mut dentry = unsafe { core::mem::zeroed() };

fn xhci_debugfs_alloc_regset(xhci: *mut xhci_hcd) -> *mut xhci_regset {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut regset: *mut xhci_regset = unsafe { core::mem::zeroed() };
let mut __v0: *mut xhci_regset;
let __obj_size: u64 = size_mul(core::mem::size_of::<xhci_regset>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut xhci_regset;
regset = __v0;
if regset.is_null() {
return core::ptr::null_mut::<xhci_regset>();
}
// The allocation and free of regset are executed in order.
// We needn't a lock here.
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*regset).list) });
list_add_tail(unsafe { core::ptr::addr_of_mut!((*regset).list) }, unsafe { core::ptr::addr_of_mut!((*xhci).regset_list) });
regset
}

fn xhci_debugfs_free_regset(regset: *mut xhci_regset) {
if regset.is_null() {
return;
}
list_del(unsafe { core::ptr::addr_of_mut!((*regset).list) });
unsafe { kfree(regset as *const ()) };
}

unsafe extern "C" fn xhci_debugfs_regset(xhci: *mut xhci_hcd, base: u32, regs: *const debugfs_reg32, nregs: u64, parent: *mut dentry, fmt: *const u8, mut __va_args: ...) {
let mut rgs: *mut xhci_regset = unsafe { core::mem::zeroed() };
let mut args: core::ffi::VaList<'static> = unsafe { core::mem::zeroed() };
let hcd: *mut usb_hcd = xhci_to_hcd(xhci);
rgs = xhci_debugfs_alloc_regset(xhci);
if rgs.is_null() {
return;
}
args = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone()) };
unsafe { vsnprintf((*rgs).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, fmt, args.clone()) };
va_end!();
let regset: *mut debugfs_regset32 = unsafe { core::ptr::addr_of_mut!((*rgs).regset) };
unsafe {
(*regset).regs = regs;
}
unsafe {
(*regset).nregs = nregs as i32;
}
unsafe {
(*regset).base = ((*hcd).regs as *const u8).wrapping_offset(base as isize) as *mut ();
}
unsafe {
(*regset).dev = (*hcd).self_.controller;
}
unsafe { debugfs_create_regset32(((*rgs).name.as_mut_ptr() as *mut u8) as *const u8, 292u16, parent, regset) };
}

fn xhci_debugfs_extcap_regset(xhci: *mut xhci_hcd, cap_id: i32, regs: *const debugfs_reg32, n: u64, cap_name: *const u8) {
let mut offset: u32 = 0;
let mut index: i32 = 0;
let mut psic: u64 = 0;
let mut nregs: u64 = n;
let base: *mut () = (unsafe { core::ptr::addr_of_mut!((*(*xhci).cap_regs).hc_capbase) }) as *mut ();
offset = xhci_find_next_ext_cap(base, 0u32, cap_id) as u32;
while offset != 0 {
if cap_id == 2 {
psic = XHCI_EXT_PORT_PSIC!(readl(((unsafe { (((base as *const u8).wrapping_offset(offset as isize) as *mut ()) as *const u8).wrapping_offset(8 as isize) }) as *mut ()) as *const ())) as u64;
let mut __v0: u64;
let __UNIQUE_ID_x__467: u64 = 4u64.wrapping_add(psic);
let __UNIQUE_ID_y__468: u64 = n;
loop {
if __types_ok__3!(__UNIQUE_ID_x__467, __UNIQUE_ID_y__468) == 0 {
unsafe { __compiletime_assert_469() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__467 < __UNIQUE_ID_y__468 { __UNIQUE_ID_x__467 } else { __UNIQUE_ID_y__468 };
nregs = __v0;
}
unsafe { xhci_debugfs_regset(xhci, offset, regs, nregs, (*xhci).debugfs_root, c"%s:%02d".as_ptr() as *const u8, cap_name, index) };
offset = xhci_find_next_ext_cap(base, offset, cap_id) as u32;
index = index.wrapping_add(1);
}
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ring_enqueue_show")]
extern "C" fn xhci_ring_enqueue_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut dma: u64 = 0;
let ring: *mut xhci_ring = unsafe { *((*s).private as *mut *mut xhci_ring) };
dma = unsafe { xhci_trb_virt_to_dma((*ring).enq_seg, (*ring).enqueue) };
unsafe { seq_printf(s, c"%pad\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(dma)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ring_dequeue_show")]
extern "C" fn xhci_ring_dequeue_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut dma: u64 = 0;
let ring: *mut xhci_ring = unsafe { *((*s).private as *mut *mut xhci_ring) };
dma = unsafe { xhci_trb_virt_to_dma((*ring).deq_seg, (*ring).dequeue) };
unsafe { seq_printf(s, c"%pad\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(dma)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ring_cycle_show")]
extern "C" fn xhci_ring_cycle_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let ring: *mut xhci_ring = unsafe { *((*s).private as *mut *mut xhci_ring) };
unsafe { seq_printf(s, c"%d\n".as_ptr() as *const u8, (*ring).cycle_state) };
0
}

fn xhci_ring_dump_segment(s: *mut seq_file, seg: *mut xhci_segment) {
let mut dma: u64 = 0;
let mut str: __SlateAlign16<[u8; 500]> = __SlateAlign16([0u8; 500]);
for i in 0i32..256i32 {
let trb: *mut xhci_trb = unsafe { (*seg).trbs }.wrapping_offset(i as isize);
dma = unsafe { (*seg).dma }.wrapping_add((i as u64).wrapping_mul(core::mem::size_of::<xhci_trb>() as u64));
unsafe { seq_printf(s, c"%2u %pad: %s\n".as_ptr() as *const u8, (*seg).num, core::ptr::addr_of_mut!(dma), xhci_decode_trb(str.0.as_mut_ptr() as *mut u8, 500u64, *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize), *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize), *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize), *((*trb).generic.field.as_mut_ptr() as *mut u32).wrapping_offset(3 as isize))) };
}
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ring_trb_show")]
extern "C" fn xhci_ring_trb_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let ring: *mut xhci_ring = unsafe { *((*s).private as *mut *mut xhci_ring) };
let mut seg: *mut xhci_segment = unsafe { (*ring).first_seg };
seg = unsafe { (*ring).first_seg };
while !seg.is_null() {
xhci_ring_dump_segment(s, seg);
seg = if (unsafe { (*seg).next }) != unsafe { (*ring).first_seg } { unsafe { (*seg).next } } else { core::ptr::null_mut::<xhci_segment>() };
}
0
}

static mut ring_files: __SlateAlign16<[xhci_file_map; 4]> = __SlateAlign16([xhci_file_map { name: (b"enqueue\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_ring_enqueue_show) }, xhci_file_map { name: (b"dequeue\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_ring_dequeue_show) }, xhci_file_map { name: (b"cycle\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_ring_cycle_show) }, xhci_file_map { name: (b"trbs\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_ring_trb_show) }]);

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ring_open")]
extern "C" fn xhci_ring_open(inode: *mut inode, file: *mut file) -> i32 {
let f_map: *const xhci_file_map = (unsafe { debugfs_get_aux(file as *const file) }) as *const xhci_file_map;
unsafe { single_open(file, (*f_map).show, (*inode).i_private) }
}

static xhci_ring_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(xhci_ring_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_slot_context_show")]
extern "C" fn xhci_slot_context_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let r#priv: *mut xhci_slot_priv = (unsafe { (*s).private }) as *mut xhci_slot_priv;
let dev: *mut xhci_virt_device = unsafe { (*r#priv).dev };
let mut str: __SlateAlign16<[u8; 500]> = __SlateAlign16([0u8; 500]);
let xhci: *mut xhci_hcd = hcd_to_xhci(bus_to_hcd(unsafe { (*(*dev).udev).bus }));
let slot_ctx: *mut xhci_slot_ctx = unsafe { xhci_get_slot_ctx(xhci, (*dev).out_ctx) };
unsafe { seq_printf(s, c"%pad: %s\n".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*dev).out_ctx).dma), xhci_decode_slot_context(str.0.as_mut_ptr() as *mut u8, (*slot_ctx).dev_info, (*slot_ctx).dev_info2, (*slot_ctx).tt_info, (*slot_ctx).dev_state)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_endpoint_context_show")]
extern "C" fn xhci_endpoint_context_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut dma: u64 = 0;
let r#priv: *mut xhci_slot_priv = (unsafe { (*s).private }) as *mut xhci_slot_priv;
let dev: *mut xhci_virt_device = unsafe { (*r#priv).dev };
let mut str: __SlateAlign16<[u8; 500]> = __SlateAlign16([0u8; 500]);
let xhci: *mut xhci_hcd = hcd_to_xhci(bus_to_hcd(unsafe { (*(*dev).udev).bus }));
for ep_index in 0i32..31i32 {
let ep_ctx: *mut xhci_ep_ctx = unsafe { xhci_get_ep_ctx(xhci, (*dev).out_ctx, ep_index as u32) };
dma = unsafe { (*(*dev).out_ctx).dma }.wrapping_add(ep_index.wrapping_add(1).wrapping_mul(CTX_SIZE!(unsafe { (*xhci).hcc_params })) as u64);
unsafe { seq_printf(s, c"%pad: %s, virt_state:%#x\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(dma), xhci_decode_ep_context(str.0.as_mut_ptr() as *mut u8, (*ep_ctx).ep_info, (*ep_ctx).ep_info2, (*ep_ctx).deq, (*ep_ctx).tx_info), (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ep_state) };
}
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_device_name_show")]
extern "C" fn xhci_device_name_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let r#priv: *mut xhci_slot_priv = (unsafe { (*s).private }) as *mut xhci_slot_priv;
let dev: *mut xhci_virt_device = unsafe { (*r#priv).dev };
unsafe { seq_printf(s, c"%s\n".as_ptr() as *const u8, dev_name(core::ptr::addr_of_mut!((*(*dev).udev).dev) as *const device)) };
0
}

static mut context_files: __SlateAlign16<[xhci_file_map; 3]> = __SlateAlign16([xhci_file_map { name: (b"name\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_device_name_show) }, xhci_file_map { name: (b"slot-context\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_slot_context_show) }, xhci_file_map { name: (b"ep-context\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_endpoint_context_show) }]);

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_context_open")]
extern "C" fn xhci_context_open(inode: *mut inode, file: *mut file) -> i32 {
let f_map: *const xhci_file_map = (unsafe { debugfs_get_aux(file as *const file) }) as *const xhci_file_map;
unsafe { single_open(file, (*f_map).show, (*inode).i_private) }
}

static xhci_context_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(xhci_context_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_portsc_show")]
extern "C" fn xhci_portsc_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let port: *mut xhci_port = (unsafe { (*s).private }) as *mut xhci_port;
let mut portsc: u32 = 0;
let mut str: __SlateAlign16<[u8; 500]> = __SlateAlign16([0u8; 500]);
portsc = unsafe { xhci_portsc_readl(port) };
unsafe { seq_printf(s, c"%s\n".as_ptr() as *const u8, xhci_decode_portsc(str.0.as_mut_ptr() as *mut u8, portsc)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_port_open")]
extern "C" fn xhci_port_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(xhci_portsc_show), (*inode).i_private) }
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_port_write")]
extern "C" fn xhci_port_write(file: *mut file, ubuf: *const u8, count: u64, ppos: *mut i64) -> i64 {
let s: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
let port: *mut xhci_port = (unsafe { (*s).private }) as *mut xhci_port;
let xhci: *mut xhci_hcd = hcd_to_xhci(unsafe { (*(*port).rhub).hcd });
let mut buf: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
let mut portsc: u32 = 0;
let mut flags: u64 = 0;
if copy_from_user(core::ptr::addr_of_mut!(buf.0) as *mut (), ubuf as *const (), min_t__4!(u64, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_sub(1), count)) != 0 {
return 14i32.wrapping_neg() as i64;
}
if (unsafe { strncmp(buf.0.as_mut_ptr() as *const u8, c"compliance".as_ptr() as *const u8, 10u64) }) == 0 {
// If CTC is clear, compliance is enabled by default
if ((unsafe { (*xhci).hcc_params2 }) as u64) & BIT!(3i32) == 0 {
return count as i64;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
// compliance mode can only be enabled on ports in RxDetect
portsc = unsafe { xhci_portsc_readl(port) };
if portsc & 480u32 != 160u32 {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
return 1i32.wrapping_neg() as i64;
}
portsc = unsafe { xhci_port_state_to_neutral(portsc) };
let __v9: u32 = portsc & 4294966815u32;
portsc = __v9;
portsc |= 65856u32;
unsafe { xhci_portsc_writel(port, portsc) };
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*xhci).lock) }, flags);
} else {
return 22i32.wrapping_neg() as i64;
}
count as i64
}

static port_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.write = Some(xhci_port_write);
__t0.open = Some(xhci_port_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_portli_show")]
extern "C" fn xhci_portli_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let port: *mut xhci_port = (unsafe { (*s).private }) as *mut xhci_port;
let mut xhci: *mut xhci_hcd = unsafe { core::mem::zeroed() };
let portli: u32 = readl((unsafe { core::ptr::addr_of_mut!((*(*port).port_reg).portli) }) as *const ());
// port without protocol capability isn't added to a roothub
if unsafe { (*port).rhub }.is_null() {
unsafe { seq_printf(s, c"0x%08x\n".as_ptr() as *const u8, portli) };
return 0;
}
xhci = hcd_to_xhci(unsafe { (*(*port).rhub).hcd });
// PORTLI fields are valid if port is a USB3 or eUSB2V2 port
if (unsafe { (*port).rhub }) == unsafe { core::ptr::addr_of_mut!((*xhci).usb3_rhub) } {
unsafe { seq_printf(s, c"0x%08x LEC=%u RLC=%u TLC=%u\n".as_ptr() as *const u8, portli, PORT_LEC!(portli), PORT_RX_LANES!(portli), PORT_TX_LANES!(portli)) };
} else {
if ((unsafe { (*xhci).hcc_params2 }) as u64) & BIT!(12i32) != 0 {
unsafe { seq_printf(s, c"0x%08x RDR=%u TDR=%u\n".as_ptr() as *const u8, portli, PORTLI_RDR!(portli), PORTLI_TDR!(portli)) };
} else {
unsafe { seq_printf(s, c"0x%08x RsvdP\n".as_ptr() as *const u8, portli) };
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_portli_open")]
extern "C" fn xhci_portli_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(xhci_portli_show), (*inode).i_private) }
}

static portli_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(xhci_portli_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

fn xhci_debugfs_create_files(xhci: *mut xhci_hcd, files: *mut xhci_file_map, nentries: u64, data: *mut (), parent: *mut dentry, fops: *const file_operations) {
let mut i: i32 = 0;
while (i as u64) < nentries {
debugfs_create_file_aux!(unsafe { (*files.wrapping_offset(i as isize)).name }, 292i32, parent, data, files.wrapping_offset(i as isize), fops);
i = i.wrapping_add(1);
}
}

fn xhci_debugfs_create_ring_dir(xhci: *mut xhci_hcd, ring: *mut *mut xhci_ring, name: *const u8, parent: *mut dentry) -> *mut dentry {
let dir: *mut dentry = unsafe { debugfs_create_dir(name, parent) };
xhci_debugfs_create_files(xhci, unsafe { core::ptr::addr_of_mut!(ring_files.0) as *mut xhci_file_map }, ARRAY_SIZE__14!(), ring as *mut (), dir, unsafe { core::ptr::addr_of!(xhci_ring_fops.0) });
dir
}

fn xhci_debugfs_create_context_files(xhci: *mut xhci_hcd, parent: *mut dentry, slot_id: i32) {
let dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
xhci_debugfs_create_files(xhci, unsafe { core::ptr::addr_of_mut!(context_files.0) as *mut xhci_file_map }, ARRAY_SIZE__3!(), unsafe { (*dev).debugfs_private }, parent, unsafe { core::ptr::addr_of!(xhci_context_fops.0) });
}

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_create_endpoint(xhci: *mut xhci_hcd, dev: *mut xhci_virt_device, ep_index: i32) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut epriv: *mut xhci_ep_priv = unsafe { core::mem::zeroed() };
let spriv: *mut xhci_slot_priv = (unsafe { (*dev).debugfs_private }) as *mut xhci_slot_priv;
if spriv.is_null() {
return;
}
if !unsafe { *((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) }.is_null() {
return;
}
let mut __v0: *mut xhci_ep_priv;
let __obj_size: u64 = size_mul(core::mem::size_of::<xhci_ep_priv>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut xhci_ep_priv;
epriv = __v0;
if epriv.is_null() {
return;
}
unsafe {
(*epriv).show_ring = (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).ring;
}
unsafe { snprintf((*epriv).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"ep%02d".as_ptr() as *const u8, ep_index) };
unsafe {
(*epriv).root = xhci_debugfs_create_ring_dir(xhci, core::ptr::addr_of_mut!((*epriv).show_ring), ((*epriv).name.as_mut_ptr() as *mut u8) as *const u8, (*spriv).root);
}
unsafe {
*((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) = epriv;
}
}

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_remove_endpoint(xhci: *mut xhci_hcd, dev: *mut xhci_virt_device, ep_index: i32) {
let spriv: *mut xhci_slot_priv = (unsafe { (*dev).debugfs_private }) as *mut xhci_slot_priv;
if spriv.is_null() || unsafe { *((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) }.is_null() {
return;
}
let epriv: *mut xhci_ep_priv = unsafe { *((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) };
unsafe { debugfs_remove((*epriv).root) };
unsafe {
*((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) = core::ptr::null_mut::<xhci_ep_priv>();
}
unsafe { kfree(epriv as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_stream_id_show")]
extern "C" fn xhci_stream_id_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let epriv: *mut xhci_ep_priv = (unsafe { (*s).private }) as *mut xhci_ep_priv;
if unsafe { (*epriv).stream_info }.is_null() {
return 1i32.wrapping_neg();
}
unsafe { seq_printf(s, c"Show stream ID %d trb ring, supported [1 - %d]\n".as_ptr() as *const u8, (*epriv).stream_id, unsafe { (*(*epriv).stream_info).num_streams }.wrapping_sub(1)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_stream_id_open")]
extern "C" fn xhci_stream_id_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(xhci_stream_id_show), (*inode).i_private) }
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_stream_id_write")]
extern "C" fn xhci_stream_id_write(file: *mut file, ubuf: *const u8, count: u64, ppos: *mut i64) -> i64 {
let s: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
let epriv: *mut xhci_ep_priv = (unsafe { (*s).private }) as *mut xhci_ep_priv;
let mut ret: i32 = 0;
let mut stream_id: u16 = 0; // MaxPStreams + 1 <= 16
if unsafe { (*epriv).stream_info }.is_null() {
return 1i32.wrapping_neg() as i64;
}
// Decimal number
ret = unsafe { kstrtou16_from_user(ubuf, count, 10u32, core::ptr::addr_of_mut!(stream_id)) };
if ret != 0 {
return ret as i64;
}
if (stream_id as i32) == 0 || ((stream_id as i32) as u32) >= unsafe { (*(*epriv).stream_info).num_streams } {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*epriv).stream_id = stream_id as u32;
}
unsafe {
(*epriv).show_ring = *unsafe { (*(*epriv).stream_info).stream_rings }.wrapping_offset((stream_id as i32) as isize);
}
count as i64
}

static stream_id_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.write = Some(xhci_stream_id_write);
__t0.open = Some(xhci_stream_id_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_stream_context_array_show")]
extern "C" fn xhci_stream_context_array_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let epriv: *mut xhci_ep_priv = (unsafe { (*s).private }) as *mut xhci_ep_priv;
let mut dma: u64 = 0;
let mut id: i32 = 0;
if unsafe { (*epriv).stream_info }.is_null() {
return 1i32.wrapping_neg();
}
unsafe { seq_printf(s, c"Allocated %d streams and %d stream context array entries\n".as_ptr() as *const u8, (*(*epriv).stream_info).num_streams, (*(*epriv).stream_info).num_stream_ctxs) };
while (id as u32) < unsafe { (*(*epriv).stream_info).num_stream_ctxs } {
let stream_ctx: *mut xhci_stream_ctx = unsafe { (*(*epriv).stream_info).stream_ctx_array }.wrapping_offset(id as isize);
dma = unsafe { (*(*epriv).stream_info).ctx_array_dma }.wrapping_add(id.wrapping_mul(16) as u64);
if (id as u32) < unsafe { (*(*epriv).stream_info).num_streams } {
unsafe { seq_printf(s, c"%pad stream id %d deq %016llx\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(dma), id, (*stream_ctx).stream_ring) };
} else {
unsafe { seq_printf(s, c"%pad stream context entry not used deq %016llx\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(dma), (*stream_ctx).stream_ring) };
}
id = id.wrapping_add(1);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_stream_context_array_open")]
extern "C" fn xhci_stream_context_array_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(xhci_stream_context_array_show), (*inode).i_private) }
}

static xhci_stream_context_array_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(xhci_stream_context_array_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_create_stream_files(xhci: *mut xhci_hcd, dev: *mut xhci_virt_device, ep_index: i32) {
let spriv: *mut xhci_slot_priv = (unsafe { (*dev).debugfs_private }) as *mut xhci_slot_priv;
if spriv.is_null() || unsafe { *((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) }.is_null() || unsafe { (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).stream_info }.is_null() {
return;
}
let epriv: *mut xhci_ep_priv = unsafe { *((*spriv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(ep_index as isize) };
unsafe {
(*epriv).stream_info = (*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(ep_index as isize)).stream_info;
}
// Show trb ring of stream ID 1 by default
unsafe {
(*epriv).stream_id = 1;
}
unsafe {
(*epriv).show_ring = *unsafe { (*(*epriv).stream_info).stream_rings }.wrapping_offset(1 as isize);
}
debugfs_create_file!(c"stream_id".as_ptr() as *mut u8, 420i32, unsafe { (*epriv).root }, epriv, unsafe { core::ptr::addr_of!(stream_id_fops.0) });
debugfs_create_file!(c"stream_context_array".as_ptr() as *mut u8, 292i32, unsafe { (*epriv).root }, epriv, unsafe { core::ptr::addr_of!(xhci_stream_context_array_fops.0) });
}

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_create_slot(xhci: *mut xhci_hcd, slot_id: i32) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
let mut __v0: *mut xhci_slot_priv;
let __obj_size: u64 = size_mul(core::mem::size_of::<xhci_slot_priv>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut xhci_slot_priv;
let r#priv: *mut xhci_slot_priv = __v0;
if r#priv.is_null() {
return;
}
unsafe { snprintf((*r#priv).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"%02d".as_ptr() as *const u8, slot_id) };
unsafe {
(*r#priv).root = debugfs_create_dir(((*r#priv).name.as_mut_ptr() as *mut u8) as *const u8, (*xhci).debugfs_slots);
}
unsafe {
(*r#priv).dev = dev;
}
unsafe {
(*dev).debugfs_private = r#priv as *mut ();
}
xhci_debugfs_create_ring_dir(xhci, unsafe { core::ptr::addr_of_mut!((*((*dev).eps.as_mut_ptr() as *mut xhci_virt_ep).wrapping_offset(0 as isize)).ring) }, c"ep00".as_ptr() as *const u8, unsafe { (*r#priv).root });
xhci_debugfs_create_context_files(xhci, unsafe { (*r#priv).root }, slot_id);
}

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_remove_slot(xhci: *mut xhci_hcd, slot_id: i32) {
let dev: *mut xhci_virt_device = unsafe { *unsafe { (*xhci).devs }.wrapping_offset(slot_id as isize) };
if dev.is_null() || unsafe { (*dev).debugfs_private }.is_null() {
return;
}
let r#priv: *mut xhci_slot_priv = (unsafe { (*dev).debugfs_private }) as *mut xhci_slot_priv;
unsafe { debugfs_remove((*r#priv).root) };
for i in 0i32..31i32 {
unsafe { kfree(*((*r#priv).eps.as_mut_ptr() as *mut *mut xhci_ep_priv).wrapping_offset(i as isize) as *const ()) };
}
unsafe { kfree(r#priv as *const ()) };
unsafe {
(*dev).debugfs_private = core::ptr::null_mut::<()>();
}
}

fn xhci_debugfs_create_ports(xhci: *mut xhci_hcd, mut parent: *mut dentry) {
let mut port_name: [u8; 8] = [0u8; 8];
parent = unsafe { debugfs_create_dir(c"ports".as_ptr() as *const u8, parent) };
let mut i: i32 = 0;
while i < ((unsafe { (*xhci).max_ports }) as i32) {
unsafe { scnprintf(port_name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 8]>() as u64, c"port%02d".as_ptr() as *const u8, i.wrapping_add(1)) };
let dir: *mut dentry = unsafe { debugfs_create_dir(port_name.as_mut_ptr() as *const u8, parent) };
let port: *mut xhci_port = unsafe { (*xhci).hw_ports }.wrapping_offset(i as isize);
debugfs_create_file!(c"portsc".as_ptr() as *mut u8, 420i32, dir, port, unsafe { core::ptr::addr_of!(port_fops.0) });
debugfs_create_file!(c"portli".as_ptr() as *mut u8, 292i32, dir, port, unsafe { core::ptr::addr_of!(portli_fops.0) });
i = i.wrapping_add(1);
}
}

fn xhci_port_bw_show(xhci: *mut xhci_hcd, dev_speed: u8, s: *mut seq_file) -> i32 {
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut ctx: core::mem::MaybeUninit<*mut xhci_container_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut xhci_container_ctx = core::ptr::addr_of_mut!(ctx) as *mut *mut xhci_container_ctx;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let hcd: *mut usb_hcd = xhci_to_hcd(xhci);
dev = (*hcd).self_.controller;
*__slate_slot_5 = pm_runtime_get_sync(dev);
if *__slate_slot_5 < 0 {
return *__slate_slot_5;
} else {
*__slate_slot_4 = xhci_alloc_port_bw_ctx(xhci, 0u32);
if (*__slate_slot_4).is_null() {
pm_runtime_put_sync(dev);
return 12i32.wrapping_neg();
} else {
// get roothub port bandwidth
*__slate_slot_5 = xhci_get_port_bandwidth(xhci, *__slate_slot_4, dev_speed);
if *__slate_slot_5 == 0 {
// print all roothub ports available bandwidth
// refer to xhci rev1_2 protocol 6.2.6 , byte 0 is reserved
*__slate_slot_6 = 1;
while *__slate_slot_6 <= (((*xhci).max_ports as i32) as u32) {
seq_printf(s, c"port[%d] available bw: %d%%.\n".as_ptr() as *const u8, *__slate_slot_6, *unsafe { (*(*__slate_slot_4)).bytes }.wrapping_offset(*__slate_slot_6 as isize) as i32);
let __v1: u32 = *__slate_slot_6;
*__slate_slot_6 = __v1.wrapping_add(1);
}
}
if *__slate_slot_5 == 5i32.wrapping_neg() {
seq_puts(s, c"Get Port Bandwidth failed\n".as_ptr() as *const u8);
*__slate_slot_5 = 0;
}
pm_runtime_put_sync(dev);
xhci_free_port_bw_ctx(xhci, *__slate_slot_4);
return *__slate_slot_5;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_ss_bw_show")]
extern "C" fn xhci_ss_bw_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut ret: i32 = 0;
let xhci: *mut xhci_hcd = (unsafe { (*s).private }) as *mut xhci_hcd;
ret = xhci_port_bw_show(xhci, DEV_PORT_SPEED__2!(4096i32) as u8, s);
ret
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_hs_bw_show")]
extern "C" fn xhci_hs_bw_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut ret: i32 = 0;
let xhci: *mut xhci_hcd = (unsafe { (*s).private }) as *mut xhci_hcd;
ret = xhci_port_bw_show(xhci, DEV_PORT_SPEED__2!(3072i32) as u8, s);
ret
}

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.xhci_fs_bw_show")]
extern "C" fn xhci_fs_bw_show(s: *mut seq_file, unused: *mut ()) -> i32 {
let mut ret: i32 = 0;
let xhci: *mut xhci_hcd = (unsafe { (*s).private }) as *mut xhci_hcd;
ret = xhci_port_bw_show(xhci, DEV_PORT_SPEED__2!(1024i32) as u8, s);
ret
}

static mut bw_context_files: __SlateAlign16<[xhci_file_map; 3]> = __SlateAlign16([xhci_file_map { name: (b"SS_BW\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_ss_bw_show) }, xhci_file_map { name: (b"HS_BW\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_hs_bw_show) }, xhci_file_map { name: (b"FS_BW\0".as_ptr() as *mut u8) as *const u8, show: Some(xhci_fs_bw_show) }]);

#[unsafe(link_section = ".text.slate_distinct.xhci_debugfs.bw_context_open")]
extern "C" fn bw_context_open(inode: *mut inode, file: *mut file) -> i32 {
let mut i: i32 = 0;
let mut f_map: *mut xhci_file_map = unsafe { core::mem::zeroed() };
let file_name: *const u8 = (unsafe { (*file_dentry(file as *const file)).d_shortname.string.as_mut_ptr() as *mut u8 }) as *const u8;
while (i as u64) < ARRAY_SIZE__4!() {
f_map = unsafe { core::ptr::addr_of_mut!(bw_context_files.0) as *mut xhci_file_map }.wrapping_offset(i as isize);
if (unsafe { strcmp((*f_map).name, file_name) }) == 0 {
break;
}
i = i.wrapping_add(1);
}
unsafe { single_open(file, (*f_map).show, (*inode).i_private) }
}

static bw_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(bw_context_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

fn xhci_debugfs_create_bandwidth(xhci: *mut xhci_hcd, mut parent: *mut dentry) {
parent = unsafe { debugfs_create_dir(c"port_bandwidth".as_ptr() as *const u8, parent) };
xhci_debugfs_create_files(xhci, unsafe { core::ptr::addr_of_mut!(bw_context_files.0) as *mut xhci_file_map }, ARRAY_SIZE__16!(), xhci as *mut (), parent, unsafe { core::ptr::addr_of!(bw_fops.0) });
}

#[unsafe(no_mangle)]
extern "C" fn xhci_debugfs_exit(xhci: *mut xhci_hcd) {
let mut rgs: *mut xhci_regset = unsafe { core::mem::zeroed() };
let mut tmp: *mut xhci_regset = unsafe { core::mem::zeroed() };
unsafe { debugfs_remove((*xhci).debugfs_root) };
unsafe {
(*xhci).debugfs_root = core::ptr::null_mut::<dentry>();
}
unsafe {
(*xhci).debugfs_slots = core::ptr::null_mut::<dentry>();
}
rgs = list_first_entry__21!(unsafe { core::ptr::addr_of_mut!((*xhci).regset_list) });
tmp = list_next_entry__13!(rgs, list);
while list_entry_is_head!(rgs, unsafe { core::ptr::addr_of_mut!((*xhci).regset_list) }, list) == 0 {
xhci_debugfs_free_regset(rgs);
rgs = tmp;
tmp = list_next_entry__13!(tmp, list);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn xhci_debugfs_create_root() {
unsafe {
xhci_debugfs_root = debugfs_create_dir(c"xhci".as_ptr() as *const u8, usb_debug_root);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".exit.text")]
extern "C" fn xhci_debugfs_remove_root() {
unsafe { debugfs_remove(xhci_debugfs_root) };
unsafe {
xhci_debugfs_root = core::ptr::null_mut::<dentry>();
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_file_dentry_1: [u8; 1] = *b"\0";

static __slate_asm_string_file_dentry_2: [u8; 48] = *b"/home/takashi/c-corpus/linux/include/linux/fs.h\0";
