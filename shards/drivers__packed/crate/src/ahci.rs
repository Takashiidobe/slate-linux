
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! These attributes define new context lock (Clang: capability) types.
//! Internal only.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! Turn individual warnings and errors on and off locally, depending
//! on version.
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//!  ahci.c - AHCI SATA support
//!
//!  Maintained by:  Tejun Heo <tj@kernel.org>
//!    		    Please ALWAYS copy linux-ide@vger.kernel.org
//! 		    on emails.
//!
//!  Copyright 2004-2005 Red Hat, Inc.
//!
//! libata documentation is available via 'make {ps|pdf}docs',
//! as Documentation/driver-api/libata.rst
//!
//! AHCI hardware documentation:
//! http://www.intel.com/technology/serialata/pdf/rev1_0.pdf
//! http://www.intel.com/technology/serialata/pdf/rev1_1.pdf
//! @a is a power of 2 value
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! ARRAY_SIZE - get the number of elements in array @arr
//! @arr: array to be sized
//! # chars in a path name including nul
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! PTR_IF - evaluate to @ptr if @cond is true, or to NULL otherwise.
//! @cond: A conditional, usually in a form of IS_ENABLED(CONFIG_FOO)
//! @ptr: A pointer to assign if @cond is true.
//!
//! PTR_IF(IS_ENABLED(CONFIG_FOO), ptr) evaluates to @ptr if CONFIG_FOO is set
//! to 'y' or 'm', or to NULL otherwise. The @ptr argument must be a pointer.
//!
//! The macro can be very useful to help compiler dropping dead code.
//!
//! For instance, consider the following::
//!
//!     #ifdef CONFIG_FOO_SUSPEND
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!     #endif
//!
//!     static struct pm_ops foo_ops = {
//!     #ifdef CONFIG_FOO_SUSPEND
//!         .suspend = foo_suspend,
//!     #endif
//!     };
//!
//! While this works, the foo_suspend() macro is compiled conditionally,
//! only when CONFIG_FOO_SUSPEND is set. This is problematic, as there could
//! be a build bug in this function, we wouldn't have a way to know unless
//! the configuration option is set.
//!
//! An alternative is to declare foo_suspend() always, but mark it
//! as __maybe_unused. This works, but the __maybe_unused attribute
//! is required to instruct the compiler that the function may not
//! be referenced anywhere, and is safe to remove without making
//! a fuss about it. This makes the programmer responsible for tagging
//! the functions that can be garbage-collected.
//!
//! With the macro it is possible to write the following::
//!
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!
//!     static struct pm_ops foo_ops = {
//!         .suspend = PTR_IF(IS_ENABLED(CONFIG_FOO_SUSPEND), foo_suspend),
//!     };
//!
//! The foo_suspend() function will now be automatically dropped by the
//! compiler, and it does not require any specific attribute.
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
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
//! number of env pointers
//! Generic info of form tag = "info"
//! Generate SD flag indexes
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! The Well Known LUNS (SAM-3) in our int representation of a LUN
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! DWORD 3
//! device class code
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
//! 31 pointers + header align the folio_batch structure to a power of two
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
// The below are used to annotate code being checked. Internal only.
// The "assert_capability" attribute is a bit confusingly named. It does not
// generate a check. Instead, it tells the analysis to *assume* the capability
// is held. This is used for augmenting runtime assertions, that can then help
// with patterns beyond the compiler's static reasoning abilities.
// __guarded_by() - struct member and globals attribute, declares variable
//                  only accessible within active context
// @...: context lock instance pointer(s)
//
// Declares that the struct member or global variable is only accessible within
// the context entered by the given context lock(s). Read operations on the data
// require shared access to at least one of the context locks, while write
// operations require exclusive access to all listed context locks.
//
// .. code-block:: c
//
// 	struct some_state {
// 		spinlock_t lock;
// 		long counter __guarded_by(&lock);
// 	};
//
// 	struct some_state {
// 		spinlock_t lock1, lock2;
// 		long counter __guarded_by(&lock1, &lock2);
// 	};
// context_lock_struct() - declare or define a context lock struct
// @name: struct name
//
// Helper to declare or define a struct type that is also a context lock.
//
// .. code-block:: c
//
// 	context_lock_struct(my_handle) {
// 		int foo;
// 		long bar;
// 	};
//
// 	struct some_state {
// 		...
// 	};
// 	// ... declared elsewhere ...
// 	context_lock_struct(some_state);
//
// Note: The implementation defines several helper functions that can acquire
// and release the context lock.
// disable_context_analysis() - disables context analysis
//
// Disables context analysis. Must be paired with a later
// enable_context_analysis().
// enable_context_analysis() - re-enables context analysis
//
// Re-enables context analysis. Must be paired with a prior
// disable_context_analysis().
// __no_context_analysis - function attribute, disables context analysis
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Prefer use of `context_unsafe()` where possible.
// context_unsafe() - disable context checking for contained code
//
// Disables context checking for contained statements or expression.
//
// .. code-block:: c
//
// 	struct some_data {
// 		spinlock_t lock;
// 		int counter __guarded_by(&lock);
// 	};
//
// 	int foo(struct some_data *d)
// 	{
// 		// ...
// 		// other code that is still checked ...
// 		// ...
// 		return context_unsafe(d->counter);
// 	}
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
// Severity used in pragma directives
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// informational
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
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

macro_rules! DMA_BIT_MASK {
($n:expr) => {
GENMASK_ULL__19!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! DMA_BIT_MASK__4 {
($n:expr) => {
GENMASK_ULL__3!($n.wrapping_sub(1), 0i32)
};
}

macro_rules! GENMASK_INPUT_CHECK__18 {
() => {
core::mem::size_of::<__SlateRecord_ahci_c_2035>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__27 {
() => {
core::mem::size_of::<__SlateRecord_ahci_c_2032>() as i32
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
(GENMASK_INPUT_CHECK__27!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__39 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__18!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__39!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__19 {
($h:expr, $l:expr) => {
GENMASK_TYPE__16!(u64, $h, $l)
};
}

macro_rules! IRQ_RETVAL {
($x:expr) => {
if $x { 1i32 } else { 0i32 }
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

macro_rules! PCI_SLOT {
($devfn:expr) => {
($devfn) >> 3i32 & 31
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
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

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__2 {
() => {
40u64
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

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
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

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! to_scsi_target {
($d:expr) => {
{
let mut __v: *mut scsi_target;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut scsi_target;
__v
}
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
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_charp: kernel_param_ops;
    static mut irq_stat: __SlateRecord_hardirq_h_65;
    static mut ata_dummy_port_ops: ata_port_operations;
    static mut sata_deb_timing_normal: [u32; 0];
    static mut sata_deb_timing_hotplug: [u32; 0];
    static mut kstat: kernel_stat;
    static mut ahci_ignore_sss: i32;
    static mut ahci_shost_groups: [*const attribute_group; 0];
    static mut ahci_sdev_groups: [*const attribute_group; 0];
    static mut ahci_ops: ata_port_operations;
    static mut ahci_pmp_retry_srst_ops: ata_port_operations;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn kstrtouint(arg0: *const u8, arg1: u32, arg2: *mut u32) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_34();
    fn __cmpxchg_wrong_size();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn strchr(arg0: *const u8, arg1: i32) -> *mut u8;
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn sysfs_add_file_to_group(arg0: *mut kobject, arg1: *const attribute, arg2: *const u8) -> i32;
    fn sysfs_remove_file_from_group(arg0: *mut kobject, arg1: *const attribute, arg2: *const u8);
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __modver_version_show(arg0: *const module_attribute, arg1: *mut module_kobject, arg2: *mut u8) -> i64;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn devm_kmalloc(arg0: *mut device, arg1: u64, arg2: u32) -> *mut ();
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pci_read_config_word(arg0: *const pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_word(arg0: *const pci_dev, arg1: i32, arg2: u16) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pcim_enable_device(arg0: *mut pci_dev) -> i32;
    fn pcim_pin_device(arg0: *mut pci_dev);
    fn pci_set_master(arg0: *mut pci_dev);
    fn __pci_register_driver(arg0: *mut pci_driver, arg1: *mut module, arg2: *const u8) -> i32;
    fn pci_unregister_driver(arg0: *mut pci_driver);
    fn pci_alloc_irq_vectors(arg0: *mut pci_dev, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn pci_free_irq_vectors(arg0: *mut pci_dev);
    fn pci_irq_vector(arg0: *mut pci_dev, arg1: u32) -> i32;
    fn pcim_request_all_regions(arg0: *mut pci_dev, arg1: *const u8) -> i32;
    fn pcim_iomap(arg0: *mut pci_dev, arg1: i32, arg2: u64) -> *mut ();
    fn dma_set_mask(arg0: *mut device, arg1: u64) -> i32;
    fn dma_set_coherent_mask(arg0: *mut device, arg1: u64) -> i32;
    fn dmi_check_system(arg0: *const dmi_system_id) -> i32;
    fn dmi_first_match(arg0: *const dmi_system_id) -> *const dmi_system_id;
    fn dmi_get_date(arg0: i32, arg1: *mut i32, arg2: *mut i32, arg3: *mut i32) -> bool;
    fn cdrom_multisession(arg0: *mut cdrom_device_info, arg1: *mut cdrom_multisession) -> i32;
    fn ata_wait_after_reset(arg0: *mut ata_link, arg1: u64, arg2: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn ata_host_alloc_pinfo(arg0: *mut device, arg1: *const *const ata_port_info, arg2: i32) -> *mut ata_host;
    fn ata_scsi_ioctl(arg0: *mut scsi_device, arg1: u32, arg2: *mut ()) -> i32;
    fn ata_scsi_queuecmd(arg0: *mut Scsi_Host, arg1: *mut scsi_cmnd) -> u32;
    fn ata_scsi_eh_timed_out(arg0: *mut scsi_cmnd) -> u32;
    fn ata_scsi_dma_need_drain(arg0: *mut request) -> bool;
    fn ata_host_suspend(arg0: *mut ata_host, arg1: pm_message);
    fn ata_host_resume(arg0: *mut ata_host);
    fn ata_msleep(arg0: *mut ata_port, arg1: u32);
    fn ata_std_bios_param(arg0: *mut scsi_device, arg1: *mut gendisk, arg2: u64, arg3: *mut i32) -> i32;
    fn ata_scsi_unlock_native_capacity(arg0: *mut scsi_device);
    fn ata_scsi_sdev_init(arg0: *mut scsi_device) -> i32;
    fn ata_scsi_sdev_configure(arg0: *mut scsi_device, arg1: *mut queue_limits) -> i32;
    fn ata_scsi_sdev_destroy(arg0: *mut scsi_device);
    fn ata_scsi_change_queue_depth(arg0: *mut scsi_device, arg1: i32) -> i32;
    fn sata_scr_read(arg0: *mut ata_link, arg1: i32, arg2: *mut u32) -> i32;
    fn sata_link_hardreset(arg0: *mut ata_link, arg1: *const u32, arg2: u64, arg3: *mut bool, arg4: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn ata_tf_to_fis(arg0: *const ata_taskfile, arg1: u8, arg2: i32, arg3: *mut u8);
    fn ata_pci_shutdown_one(arg0: *mut pci_dev);
    fn ata_pci_remove_one(arg0: *mut pci_dev);
    fn ata_acpi_gtm(arg0: *mut ata_port, arg1: *mut ata_acpi_gtm) -> i32;
    fn ata_port_pbar_desc(arg0: *mut ata_port, arg1: i32, arg2: i64, arg3: *const u8);
    fn ata_link_next(arg0: *mut ata_link, arg1: *mut ata_port, arg2: u32) -> *mut ata_link;
    fn ata_dev_next(arg0: *mut ata_device, arg1: *mut ata_link, arg2: u32) -> *mut ata_device;
    fn pm_suspend_default_s2idle() -> bool;
    fn ahci_dev_classify(arg0: *mut ata_port) -> u32;
    fn ahci_save_initial_config(arg0: *mut device, arg1: *mut ahci_host_priv);
    fn ahci_init_controller(arg0: *mut ata_host);
    fn ahci_reset_controller(arg0: *mut ata_host) -> i32;
    fn ahci_check_ready(arg0: *mut ata_link) -> i32;
    fn ahci_kick_engine(arg0: *mut ata_port) -> i32;
    fn ahci_set_em_messages(arg0: *mut ahci_host_priv, arg1: *mut ata_port_info);
    fn ahci_reset_em(arg0: *mut ata_host) -> i32;
    fn ahci_print_info(arg0: *mut ata_host, arg1: *const u8);
    fn ahci_host_activate(arg0: *mut ata_host, arg1: *const scsi_host_template) -> i32;
    fn ahci_handle_port_intr(arg0: *mut ata_host, arg1: u32) -> u32;
    fn __compiletime_assert_685();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
unsafe { __compiletime_assert_34() };
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
struct io_bitmap {
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
    __slate_bits_0: __slate_bits::__SlateBits122U0,
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
    __slate_bits_0: __slate_bits::__SlateBits127U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
    pkru: u32,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
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
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits280U0,
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
struct rq_flags {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits412U0,
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
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
struct irq_domain {
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct sock {
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

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits283U0,
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct io_cq {
    q: *mut request_queue,
    ioc: *mut io_context,
    __slate_anon_2: __SlateRecord_iocontext_h_83,
    __slate_anon_3: __SlateRecord_iocontext_h_87,
    flags: u32,
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


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
    __slate_bits_0: __slate_bits::__SlateBits586U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
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
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
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
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits652U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
    __slate_bits_0: __slate_bits::__SlateBits595U0,
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_version_attribute {
    mattr: module_attribute,
    module_name: *const u8,
    version: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
// Format: __mod_device_table__kmod_<modname>__<type>__<name>
// Parts of the string `__kmod_` and `__` are used as delimiters when parsing
// a symbol in file2alias.c
// Creates an alias so file2alias.c can find device table.
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
struct module_param_attrs {
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
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
    __slate_bits_0: __slate_bits::__SlateBits700U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits700U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits700U2,
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
    __slate_bits_3: __slate_bits::__SlateBits700U3,
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


fn devm_kzalloc(dev: *mut device, size: u64, gfp: u32) -> *mut () {
unsafe { devm_kmalloc(dev, size, gfp | (BIT!(8i32) as u32)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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


// module_driver() - Helper macro for drivers that don't do anything
// special in module init/exit. This eliminates a lot of boilerplate.
// Each module may only use this macro once, and calling it replaces
// module_init() and module_exit().
//
// @__driver: driver name
// @__register: register function for this driver type
// @__unregister: unregister function for this driver type
// @...: Additional arguments to be passed to __register and __unregister.
//
// Use this macro to construct bus specific macros for registering
// drivers, and do not use it on its own.
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


// DEVICE_ATTR_RO - Define a readable device attribute.
// @_name: Attribute name.
//
// Like DEVICE_ATTR(), but @_mode is 0444 and @_show is <_name>_show.
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
    __slate_bits_0: __slate_bits::__SlateBits697U0,
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

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_hardirq_h_65 {
    counts: [u32; 18],
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


fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


fn readq(mut addr: *const ()) -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq ({1:r}),{0:r}", lateout(reg) __t0, in(reg) addr as *mut u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

// Let people know that we have them
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
struct hotplug_slot {
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
    __slate_bits_0: __slate_bits::__SlateBits754U0,
    kobj: kobject,
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
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits749U0,
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
    __slate_bits_1: __slate_bits::__SlateBits749U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits749U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits749U3,
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
    __slate_bits_4: __slate_bits::__SlateBits749U4,
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
    __slate_bits_0: __slate_bits::__SlateBits442U0,
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
struct disk_stats {
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


// PCI_DEVICE - macro used to describe a specific PCI device
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device.  The subvendor and subdevice fields will be set to
// PCI_ANY_ID.
// PCI_DEVICE_SUB - macro used to describe a specific PCI device with subsystem
// @vend: the 16 bit PCI Vendor ID
// @dev: the 16 bit PCI Device ID
// @subvend: the 16 bit PCI Subvendor ID
// @subdev: the 16 bit PCI Subdevice ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific device with subsystem information.
// PCI_DEVICE_CLASS - macro used to describe a specific PCI device class
// @dev_class: the class, subclass, prog-if triple for this device
// @dev_class_mask: the class mask for this device
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI class.  The vendor, device, subvendor, and subdevice
// fields will be set to PCI_ANY_ID.
// PCI_VDEVICE - macro used to describe a specific PCI device in short form
// @vend: the vendor name
// @dev: the 16 bit PCI Device ID
//
// This macro is used to create a struct pci_device_id that matches a
// specific PCI device.  The subvendor, and subdevice fields will be set
// to PCI_ANY_ID. The macro allows the next field to follow as the device
// private data.
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// module_pci_driver() - Helper macro for registering a PCI driver
// @__pci_driver: pci_driver struct
//
// Helper macro for PCI drivers which do not do anything special in module
// init/exit. This eliminates a lot of boilerplate. Each module may only
// use this macro once, and calling it replaces module_init() and module_exit()
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
fn pci_get_drvdata(pdev: *mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*pdev).dev) }) as *const device)
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
fn dma_set_mask_and_coherent(dev: *mut device, mask: u64) -> i32 {
let rc: i32 = unsafe { dma_set_mask(dev, mask) };
if rc == 0 {
unsafe { dma_set_coherent_mask(dev, mask) };
}
rc
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_alloc_cache {
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
struct timer_rand_state {
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
struct blk_mq_ctx {
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
struct blk_holder_ops {
    mark_dead: Option<unsafe extern "C" fn(*mut block_device, bool)>,
    sync: Option<unsafe extern "C" fn(*mut block_device)>,
    freeze: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
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
struct bsg_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_strmatch {
    __slate_bits_0: __slate_bits::__SlateBits834U0,
    substr: [u8; 79],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device_handler {
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
struct scsi_sense_hdr {
    response_code: u8,
    sense_key: u8,
    asc: u8,
    ascq: u8,
    byte4: u8,
    byte5: u8,
    byte6: u8,
    additional_length: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits858U0,
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
    __slate_bits_0: __slate_bits::__SlateBits855U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_vpd {
    rcu: callback_head,
    len: i32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_586 {
    error: u8,
    feature: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_597 {
    status: u8,
    command: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits853U0,
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
    __slate_bits_0: __slate_bits::__SlateBits854U0,
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
struct ata_bmdma_prd {
    addr: u32,
    flags_len: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_msf0 {
    minute: u8,
    second: u8,
    frame: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union cdrom_addr {
    msf: cdrom_msf0,
    lba: i32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_771 {
    id: [u16; 256],
    gscr: [u32; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_multisession {
    addr: cdrom_addr,
    xa_flag: u8,
    addr_format: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_mcn {
    medium_catalog_number: [u8; 14],
}


// seconds per minute
// frames per second
// MSF numbering offset of first frame
#[repr(C)]
#[derive(Clone, Copy)]
struct packet_command {
    cmd: [u8; 12],
    buffer: *mut u8,
    buflen: u32,
    stat: i32,
    sshdr: *mut scsi_sense_hdr,
    data_direction: u8,
    quiet: i32,
    timeout: i32,
    reserved: [*mut (); 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
    ops: *const cdrom_device_ops,
    list: list_head,
    disk: *mut gendisk,
    handle: *mut (),
    mask: i32,
    speed: i32,
    capacity: i32,
    __slate_bits_0: __slate_bits::__SlateBits822U0,
    vfs_events: u32,
    ioctl_events: u32,
    use_count: i32,
    name: [u8; 20],
    __slate_bits_1: __slate_bits::__SlateBits822U1,
    cdda_method: i32,
    last_sense: u8,
    media_written: u8,
    mmc3_profile: u16,
    mrw_mode_page: i32,
    opened_for_data: bool,
    last_media_change_ms: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_ops {
    open: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut cdrom_device_info)>,
    drive_status: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut cdrom_device_info, u32, i32) -> u32>,
    tray_move: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    lock_door: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    select_speed: Option<unsafe extern "C" fn(*mut cdrom_device_info, u64) -> i32>,
    get_last_session: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut cdrom_multisession) -> i32>,
    get_mcn: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut cdrom_mcn) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut cdrom_device_info) -> i32>,
    audio_ioctl: Option<unsafe extern "C" fn(*mut cdrom_device_info, u32, *mut ()) -> i32>,
    generic_packet: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut packet_command) -> i32>,
    read_cdda_bpc: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut (), u32, u32, *mut u8) -> i32>,
    capability: i32,
}


// The SCSI spec says there could be 256 slots.
#[repr(C)]
#[derive(Clone, Copy)]
struct ata_taskfile {
    flags: u64,
    protocol: u8,
    ctl: u8,
    hob_feature: u8,
    hob_nsect: u8,
    hob_lbal: u8,
    hob_lbam: u8,
    hob_lbah: u8,
    __slate_anon_8: __SlateRecord_libata_h_586,
    nsect: u8,
    lbal: u8,
    lbam: u8,
    lbah: u8,
    device: u8,
    __slate_anon_14: __SlateRecord_libata_h_597,
    auxiliary: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct clk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ioports {
    cmd_addr: *mut (),
    data_addr: *mut (),
    error_addr: *mut (),
    feature_addr: *mut (),
    nsect_addr: *mut (),
    lbal_addr: *mut (),
    lbam_addr: *mut (),
    lbah_addr: *mut (),
    device_addr: *mut (),
    status_addr: *mut (),
    command_addr: *mut (),
    altstatus_addr: *mut (),
    ctl_addr: *mut (),
    bmdma_addr: *mut (),
    scr_addr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_host {
    lock: spinlock,
    dev: *mut device,
    iomap: *const *mut (),
    n_ports: u32,
    n_tags: u32,
    private_data: *mut (),
    ops: *mut ata_port_operations,
    flags: u64,
    kref: kref,
    eh_mutex: mutex,
    eh_owner: *mut task_struct,
    simplex_claimed: *mut ata_port,
    ports: [*mut ata_port; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_queued_cmd {
    ap: *mut ata_port,
    dev: *mut ata_device,
    scsicmd: *mut scsi_cmnd,
    scsidone: Option<unsafe extern "C" fn(*mut scsi_cmnd)>,
    tf: ata_taskfile,
    cdb: [u8; 16],
    flags: u64,
    tag: u32,
    hw_tag: u32,
    n_elem: u32,
    orig_n_elem: u32,
    dma_dir: i32,
    sect_size: u32,
    nbytes: u32,
    extrabytes: u32,
    curbytes: u32,
    sgent: scatterlist,
    sg: *mut scatterlist,
    cursg: *mut scatterlist,
    cursg_ofs: u32,
    err_mask: u32,
    result_tf: ata_taskfile,
    complete_fn: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    private_data: *mut (),
    lldd_task: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port_stats {
    unhandled_irq: u64,
    idle_irq: u64,
    rw_reqbuf: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ering_entry {
    eflags: u32,
    err_mask: u32,
    timestamp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ering {
    cursor: i32,
    ring: [ata_ering_entry; 32],
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
struct ata_cpr {
    num: u8,
    num_storage_elements: u8,
    start_lba: u64,
    num_lbas: u64,
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
struct ata_cpr_log {
    nr_cpr: u8,
    cpr: [ata_cpr; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_cdl {
    desc_log_buf: [u8; 512],
    ncq_sense_log_buf: [u8; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_device {
    link: *mut ata_link,
    devno: u32,
    quirks: u64,
    flags: u64,
    sdev: *mut scsi_device,
    private_data: *mut (),
    gtf_cache: *mut acpi_object,
    gtf_filter: u32,
    tdev: device,
    n_sectors: u64,
    n_native_sectors: u64,
    class: u32,
    unpark_deadline: u64,
    pio_mode: u8,
    dma_mode: u8,
    xfer_mode: u8,
    xfer_shift: u32,
    multi_count: u32,
    max_sectors: u32,
    cdb_len: u32,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    cylinders: u16,
    heads: u16,
    sectors: u16,
    __slate_anon_26: __SlateRecord_libata_h_771,
    gp_log_dir: [u8; 512],
    devslp_timing: [u8; 8],
    ncq_send_recv_cmds: [u8; 20],
    ncq_non_data_cmds: [u8; 64],
    zac_zoned_cap: u32,
    zac_zones_optimal_open: u32,
    zac_zones_optimal_nonseq: u32,
    zac_zones_max_open: u32,
    cpr_log: *mut ata_cpr_log,
    cdl: *mut ata_cdl,
    spdn_cnt: i32,
    ering: ata_ering,
    __slate_pad_39: [u8; 48],
    sector_buf: [u8; 512],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_eh_info {
    dev: *mut ata_device,
    serror: u32,
    err_mask: u32,
    action: u32,
    dev_action: [u32; 2],
    flags: u32,
    probe_mask: u32,
    desc: [u8; 80],
    desc_len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_eh_context {
    i: ata_eh_info,
    tries: [i32; 2],
    cmd_timeout_idx: [[i32; 9]; 2],
    classes: [u32; 2],
    did_probe_mask: u32,
    unloaded_mask: u32,
    saved_ncq_enabled: u32,
    saved_xfer_mode: [u8; 2],
    last_reset: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ata_acpi_drive {
    pio: u32,
    dma: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ata_acpi_gtm {
    drive: [ata_acpi_drive; 2],
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_link {
    ap: *mut ata_port,
    pmp: i32,
    tdev: device,
    active_tag: u32,
    sactive: u32,
    flags: u32,
    saved_scontrol: u32,
    hw_sata_spd_limit: u32,
    sata_spd_limit: u32,
    sata_spd: u32,
    lpm_policy: u32,
    deferred_qc_work: work_struct,
    deferred_qc: *mut ata_queued_cmd,
    eh_info: ata_eh_info,
    eh_context: ata_eh_context,
    device: [ata_device; 2],
    last_lpm_change: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port {
    scsi_host: *mut Scsi_Host,
    ops: *mut ata_port_operations,
    lock: *mut spinlock,
    flags: u64,
    pflags: u32,
    print_id: u32,
    port_no: u32,
    ioaddr: ata_ioports,
    ctl: u8,
    last_ctl: u8,
    sff_pio_task_link: *mut ata_link,
    sff_pio_task: delayed_work,
    bmdma_prd: *mut ata_bmdma_prd,
    bmdma_prd_dma: u64,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    cbl: u32,
    qcmd: [ata_queued_cmd; 33],
    qc_active: u64,
    nr_active_links: i32,
    link: ata_link,
    slave_link: *mut ata_link,
    nr_pmp_links: i32,
    pmp_link: *mut ata_link,
    excl_link: *mut ata_link,
    stats: ata_port_stats,
    host: *mut ata_host,
    dev: *mut device,
    tdev: device,
    scsi_scan_mutex: mutex,
    hotplug_task: delayed_work,
    scsi_rescan_task: delayed_work,
    hsm_task_state: u32,
    eh_done_q: list_head,
    eh_wait_q: wait_queue_head,
    eh_tries: i32,
    park_req_pending: completion,
    pm_mesg: pm_message,
    target_lpm_policy: u32,
    fastdrain_timer: timer_list,
    fastdrain_cnt: u32,
    cookie: u64,
    em_message_type: i32,
    private_data: *mut (),
    __acpi_init_gtm: ata_acpi_gtm,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_reset_operations {
    prereset: Option<unsafe extern "C" fn(*mut ata_link, u64) -> i32>,
    softreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>,
    hardreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>,
    postreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port_operations {
    qc_defer: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> i32>,
    check_atapi_dma: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> i32>,
    qc_prep: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>,
    qc_issue: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>,
    qc_fill_rtf: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    qc_ncq_fill_rtf: Option<unsafe extern "C" fn(*mut ata_port, u64)>,
    cable_detect: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    mode_filter: Option<unsafe extern "C" fn(*mut ata_device, u32) -> u32>,
    set_piomode: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_device)>,
    set_dmamode: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_device)>,
    set_mode: Option<unsafe extern "C" fn(*mut ata_link, *mut *mut ata_device) -> i32>,
    read_id: Option<unsafe extern "C" fn(*mut ata_device, *mut ata_taskfile, *mut u16) -> u32>,
    dev_config: Option<unsafe extern "C" fn(*mut ata_device)>,
    freeze: Option<unsafe extern "C" fn(*mut ata_port)>,
    thaw: Option<unsafe extern "C" fn(*mut ata_port)>,
    reset: ata_reset_operations,
    pmp_reset: ata_reset_operations,
    error_handler: Option<unsafe extern "C" fn(*mut ata_port)>,
    lost_interrupt: Option<unsafe extern "C" fn(*mut ata_port)>,
    post_internal_cmd: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    sched_eh: Option<unsafe extern "C" fn(*mut ata_port)>,
    end_eh: Option<unsafe extern "C" fn(*mut ata_port)>,
    scr_read: Option<unsafe extern "C" fn(*mut ata_link, u32, *mut u32) -> i32>,
    scr_write: Option<unsafe extern "C" fn(*mut ata_link, u32, u32) -> i32>,
    pmp_attach: Option<unsafe extern "C" fn(*mut ata_port)>,
    pmp_detach: Option<unsafe extern "C" fn(*mut ata_port)>,
    set_lpm: Option<unsafe extern "C" fn(*mut ata_link, u32, u32) -> i32>,
    port_suspend: Option<unsafe extern "C" fn(*mut ata_port, pm_message) -> i32>,
    port_resume: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    port_start: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    port_stop: Option<unsafe extern "C" fn(*mut ata_port)>,
    host_stop: Option<unsafe extern "C" fn(*mut ata_host)>,
    sff_dev_select: Option<unsafe extern "C" fn(*mut ata_port, u32)>,
    sff_set_devctl: Option<unsafe extern "C" fn(*mut ata_port, u8)>,
    sff_check_status: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    sff_check_altstatus: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    sff_tf_load: Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>,
    sff_tf_read: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_taskfile)>,
    sff_exec_command: Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>,
    sff_data_xfer: Option<unsafe extern "C" fn(*mut ata_queued_cmd, *mut u8, u32, i32) -> u32>,
    sff_irq_on: Option<unsafe extern "C" fn(*mut ata_port)>,
    sff_irq_check: Option<unsafe extern "C" fn(*mut ata_port) -> bool>,
    sff_irq_clear: Option<unsafe extern "C" fn(*mut ata_port)>,
    sff_drain_fifo: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_setup: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_start: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_stop: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_status: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    em_show: Option<unsafe extern "C" fn(*mut ata_port, *mut u8) -> i64>,
    em_store: Option<unsafe extern "C" fn(*mut ata_port, *const u8, u64) -> i64>,
    sw_activity_show: Option<unsafe extern "C" fn(*mut ata_device, *mut u8) -> i64>,
    sw_activity_store: Option<unsafe extern "C" fn(*mut ata_device, u32) -> i64>,
    transmit_led_message: Option<unsafe extern "C" fn(*mut ata_port, u32, i64) -> i64>,
    inherits: *const ata_port_operations,
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
struct ata_port_info {
    flags: u64,
    link_flags: u64,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    port_ops: *mut ata_port_operations,
    private_data: *mut (),
}


fn sata_ehc_deb_timing(ehc: *mut ata_eh_context) -> *const u32 {
if (unsafe { (*ehc).i.flags }) & 1 != 0 {
return unsafe { core::ptr::addr_of!(sata_deb_timing_hotplug) as *const u32 };
} else {
return unsafe { core::ptr::addr_of!(sata_deb_timing_normal) as *const u32 };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


// All sht initializers (BASE, PIO, BMDMA, NCQ) must be instantiated
// by the edge drivers.  Because the 'module' field of sht must be the
// edge driver's module reference, otherwise the driver can be unloaded
// even if the scsi_device is being accessed.
fn sata_pmp_attached(ap: *mut ata_port) -> bool {
(unsafe { (*ap).nr_pmp_links }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
}


fn ata_print_version_once(dev: *const device, version: *const u8) {
loop {
if !unsafe { __print_once } {
unsafe {
__print_once = true;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, dev, c"version %s\n".as_ptr() as *const u8, version) };
}
}
if true {
break;
}
}
}

#[unsafe(link_section = ".data..read_mostly")]
static mut __print_once: bool = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_phy_hdmi_h_27 {
    tmds_char_rate: u64,
    frl: __SlateRecord_phy_hdmi_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_phy_hdmi_h_29 {
    rate_per_lane: u8,
    lanes: u8,
}


// Shortcut notation for iterations
//
// ata_for_each_link() iterates over each link of @ap according to
// @mode.  @link points to the current link in the loop.  @link is
// NULL after loop termination.  ata_for_each_dev() works the same way
// except that it iterates over each device of @link.
//
// Note that the mode prefixes ATA_{L|D}ITER_ shouldn't need to be
// specified when using the following shorthand notations.  Only the
// mode itself (EDGE, HOST_FIRST, ENABLED, etc...) should be
// specified.  This not only increases brevity but also makes it
// impossible to use ATA_LITER_* for device iteration or vice-versa.
fn ata_tf_init(dev: *mut ata_device, tf: *mut ata_taskfile) {
unsafe { memset(tf as *mut (), 0i32, core::mem::size_of::<ata_taskfile>() as u64) };
unsafe {
(*tf).ctl = (*(*(*dev).link).ap).ctl;
}
if (unsafe { (*dev).devno }) == 0 {
unsafe {
(*tf).device = 160;
}
} else {
unsafe {
(*tf).device = 176u8;
}
}
}

// remap-device base relative to ahci-bar
fn ahci_remap_dcc(i: i32) -> u32 {
2176i32.wrapping_add(i.wrapping_mul(128)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct clk_bulk_data {
    id: *const u8,
    clk: *mut clk,
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
struct reset_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1614_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1618_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1624_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_1635_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2032 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2035 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ahci_c_2862_6 {
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
    pub struct __SlateBits554U0 {
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
    pub struct __SlateBits586U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits280U0 {
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
    pub struct __SlateBits652U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits948U0 {
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
    pub struct __SlateBits283U0 {
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
    pub struct __SlateBits700U0 {
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
    pub struct __SlateBits697U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits700U1 {
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
    pub struct __SlateBits700U2 {
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
    pub struct __SlateBits700U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits822U0 {
        #[bits(30)]
        pub options: u32,
        #[bits(2)]
        pub mc_flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits822U1 {
        #[bits(2)]
        pub sanyo_slot: u8,
        #[bits(1)]
        pub keeplocked: u8,
        #[bits(5)]
        pub reserved: u8,
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
    pub struct __SlateBits122U0 {
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
    pub struct __SlateBits128U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits595U0 {
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
    pub struct __SlateBits127U0 {
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
    pub struct __SlateBits412U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits854U0 {
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
    #[bitfields::bitfield([u8; 8], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits853U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits858U0 {
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
    pub struct __SlateBits855U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits442U0 {
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
    pub struct __SlateBits754U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits749U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits749U1 {
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
    pub struct __SlateBits749U2 {
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
    pub struct __SlateBits749U3 {
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
    pub struct __SlateBits749U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits834U0 {
        #[bits(7)]
        pub slot: u8,
        #[bits(1)]
        pub exact_match: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits971U0 {
        #[bits(1)]
        pub ssc: u8,
        #[bits(1)]
        pub set_rate: u8,
        #[bits(1)]
        pub set_lanes: u8,
        #[bits(1)]
        pub set_voltages: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits988U0 {
        #[bits(1)]
        pub ncq_saw_d2h: u32,
        #[bits(1)]
        pub ncq_saw_dmas: u32,
        #[bits(1)]
        pub ncq_saw_sdb: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits948U0,
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
    __slate_bits_0: __slate_bits::__SlateBits554U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_dp {
    link_rate: u32,
    lanes: u32,
    voltage: [u32; 4],
    pre: [u32; 4],
    __slate_bits_0: __slate_bits::__SlateBits971U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_hdmi {
    bpc: u32,
    __slate_anon_1: __SlateRecord_phy_hdmi_h_27,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_lvds {
    bits_per_lane_and_dclk_cycle: u32,
    differential_clk_rate: u64,
    lanes: u32,
    is_slave: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_mipi_dphy {
    clk_miss: u32,
    clk_post: u32,
    clk_pre: u32,
    clk_prepare: u32,
    clk_settle: u32,
    clk_term_en: u32,
    clk_trail: u32,
    clk_zero: u32,
    d_term_en: u32,
    eot: u32,
    hs_exit: u32,
    hs_prepare: u32,
    hs_settle: u32,
    hs_skip: u32,
    hs_trail: u32,
    hs_zero: u32,
    init: u32,
    lpx: u32,
    ta_get: u32,
    ta_go: u32,
    ta_sure: u32,
    wakeup: u32,
    hs_clk_rate: u64,
    lp_clk_rate: u64,
    lanes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_notify {
    ufs_state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_configure_opts {
    mipi_dphy: phy_configure_opts_mipi_dphy,
    dp: phy_configure_opts_dp,
    lvds: phy_configure_opts_lvds,
    hdmi: phy_configure_opts_hdmi,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_ops {
    init: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_on: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_off: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    set_mode: Option<unsafe extern "C" fn(*mut phy, u32, i32) -> i32>,
    set_media: Option<unsafe extern "C" fn(*mut phy, u32) -> i32>,
    set_speed: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    configure: Option<unsafe extern "C" fn(*mut phy, *mut phy_configure_opts) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut phy, u32, i32, *mut phy_configure_opts) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    calibrate: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    connect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    notify_phystate: Option<unsafe extern "C" fn(*mut phy, phy_notify) -> i32>,
    release: Option<unsafe extern "C" fn(*mut phy)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_attrs {
    bus_width: u32,
    max_link_rate: u32,
    mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy {
    dev: device,
    id: i32,
    ops: *const phy_ops,
    mutex: mutex,
    lockdep_key: lock_class_key,
    init_count: i32,
    power_count: i32,
    attrs: phy_attrs,
    pwr: *mut regulator,
    debugfs: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ahci_cmd_hdr {
    opts: u32,
    status: u32,
    tbl_addr: u32,
    tbl_addr_hi: u32,
    reserved: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ahci_em_priv {
    blink_policy: u32,
    timer: timer_list,
    saved_activity: u64,
    activity: u64,
    led_state: u64,
    link: *mut ata_link,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ahci_port_priv {
    active_link: *mut ata_link,
    cmd_slot: *mut ahci_cmd_hdr,
    cmd_slot_dma: u64,
    cmd_tbl: *mut (),
    cmd_tbl_dma: u64,
    rx_fis: *mut (),
    rx_fis_dma: u64,
    __slate_bits_0: __slate_bits::__SlateBits988U0,
    lock: spinlock,
    intr_mask: u32,
    fbs_supported: bool,
    fbs_enabled: bool,
    fbs_last_dev: i32,
    em_priv: [ahci_em_priv; 15],
    irq_desc: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ahci_host_priv {
    flags: u32,
    mask_port_map: u32,
    mask_port_ext: u32,
    mmio: *mut (),
    cap: u32,
    cap2: u32,
    version: u32,
    port_map: u32,
    saved_cap: u32,
    saved_cap2: u32,
    saved_port_map: u32,
    saved_port_cap: [u32; 32],
    em_loc: u32,
    em_buf_sz: u32,
    em_msg_type: u32,
    remapped_nvme: u32,
    got_runtime_pm: bool,
    n_clks: u32,
    clks: *mut clk_bulk_data,
    f_rsts: u32,
    rsts: *mut reset_control,
    target_pwrs: *mut *mut regulator,
    ahci_regulator: *mut regulator,
    phy_regulator: *mut regulator,
    nports: u32,
    plat_data: *mut (),
    irq: u32,
    start_engine: Option<unsafe extern "C" fn(*mut ata_port)>,
    stop_engine: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    irq_handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    get_irq_vector: Option<unsafe extern "C" fn(*mut ata_host, i32) -> i32>,
    phys: [*mut phy; 0],
}


// This must be instantiated by the edge drivers.  Read the comments
// for ATA_BASE_SHT
fn __ahci_port_base(hpriv: *mut ahci_host_priv, port_no: u32) -> *mut () {
let mmio: *mut () = unsafe { (*hpriv).mmio };
(unsafe { (((mmio as *const u8).wrapping_offset(256 as isize) as *mut ()) as *const u8).wrapping_offset(port_no.wrapping_mul(128) as isize) }) as *mut ()
}

core::arch::global_asm!(".section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_ahci__686_3166_ahci_pci_driver_init6:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym ahci_pci_driver_init, options(att_syntax));

fn ahci_port_base(ap: *mut ata_port) -> *mut () {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
__ahci_port_base(hpriv, unsafe { (*ap).port_no })
}

fn ahci_nr_ports(cap: u32) -> i32 {
(cap & 31).wrapping_add(1) as i32
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_init_one")]
extern "C" fn ahci_init_one(pdev: *mut pci_dev, ent: *const pci_device_id) -> i32 {
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__684: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__683: i32 = unsafe { core::mem::zeroed() };
let mut map: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u8 = core::ptr::addr_of_mut!(map) as *mut u8;
let mut ahci_pci_bar: i32 = unsafe { core::mem::zeroed() };
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_54: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut n_ports: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_55: *mut i32 = core::ptr::addr_of_mut!(n_ports) as *mut i32;
let mut host: core::mem::MaybeUninit<*mut ata_host> = core::mem::MaybeUninit::uninit();
let __slate_slot_56: *mut *mut ata_host = core::ptr::addr_of_mut!(host) as *mut *mut ata_host;
let mut hpriv: core::mem::MaybeUninit<*mut ahci_host_priv> = core::mem::MaybeUninit::uninit();
let __slate_slot_57: *mut *mut ahci_host_priv = core::ptr::addr_of_mut!(hpriv) as *mut *mut ahci_host_priv;
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut ppi: core::mem::MaybeUninit<__SlateAlign16<[*const ata_port_info; 2]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_59: *mut [*const ata_port_info; 2] = core::ptr::addr_of_mut!(ppi) as *mut [*const ata_port_info; 2];
let mut pi: core::mem::MaybeUninit<ata_port_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut ata_port_info = core::ptr::addr_of_mut!(pi) as *mut ata_port_info;
let mut board_id: u32 = unsafe { core::mem::zeroed() };
unsafe {
{
board_id = (*ent).driver_data as u32;
core::ptr::write(__slate_slot_60, ahci_port_info.0[board_id as usize]);
core::ptr::write(__slate_slot_59, [core::ptr::addr_of_mut!(*__slate_slot_60) as *const ata_port_info, core::ptr::null::<ata_port_info>()]);
dev = core::ptr::addr_of_mut!((*pdev).dev);
let __slate_auto_bytes_1 = __slate_slot_57 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_56 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_55 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_54 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ahci_pci_bar = 5;
let __ret_warn_on: i32 = false as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("681: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 681b - ., 3; .popsection\n# {0}", const 681 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ahci_init_one_62, sym __slate_asm_string_ahci_init_one_63, const 2893 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("682: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 682b - ., 4; .popsection\n# {0}", const 682 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
ata_print_version_once(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"3.0".as_ptr() as *const u8);
// The AHCI driver can only drive the SATA ports, the PATA driver
// can drive them all so if both drivers are selected make sure
// AHCI stays out of the way
if ((*pdev).vendor as i32) == 4523 && marvell_enable == 0 {
return 19i32.wrapping_neg();
} else {
// Apple BIOS on MCP89 prevents us using AHCI
if is_mcp89_apple(pdev) {
ahci_mcp89_apple_enable(pdev);
}
// Promise's PDC42819 is a SAS/SATA controller that has an AHCI mode.
// At the moment, we can only use the AHCI mode. Let the users know
// that for SAS drives they're out of luck.
if ((*pdev).vendor as i32) == 4186 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"PDC42819 can only drive SATA devices with this driver\n".as_ptr() as *const u8);
}
// Some devices use non-standard BARs
if ((*pdev).vendor as i32) == 4170 && ((*pdev).device as i32) == 52230 {
ahci_pci_bar = 0;
} else {
if ((*pdev).vendor as i32) == 7236 && ((*pdev).device as i32) == 32768 {
ahci_pci_bar = 2;
} else {
if ((*pdev).vendor as i32) == 6013 {
if ((*pdev).device as i32) == 40988 {
ahci_pci_bar = 0;
}
if ((*pdev).device as i32) == 41092 {
ahci_pci_bar = 4;
}
} else {
if ((*pdev).vendor as i32) == 20 {
if ((*pdev).device as i32) == 31240 {
ahci_pci_bar = 0;
}
}
}
}
}
// acquire resources
*__slate_slot_53 = pcim_enable_device(pdev);
if *__slate_slot_53 != 0 {
return *__slate_slot_53;
} else {
if ((*pdev).vendor as i32) == 32902 && (((*pdev).device as i32) == 9810 || ((*pdev).device as i32) == 9811) {
let __slate_auto_bytes = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// ICH6s share the same PCI ID for both piix and ahci
// modes.  Enabling ahci mode while MAP indicates
// combined mode is a bad idea.  Yield to ata_piix.
pci_read_config_byte(pdev as *const pci_dev, 144i32, core::ptr::addr_of_mut!(*__slate_slot_48));
if (*__slate_slot_48 as i32) & 3 != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"controller is in combined mode, can't enable AHCI mode\n".as_ptr() as *const u8);
return 19i32.wrapping_neg();
}
}
// AHCI controllers often implement SFF compatible interface.
// Grab all PCI BARs just in case.
*__slate_slot_53 = pcim_request_all_regions(pdev, c"ahci".as_ptr() as *const u8);
if *__slate_slot_53 == 16i32.wrapping_neg() {
pcim_pin_device(pdev);
}
if *__slate_slot_53 != 0 {
return *__slate_slot_53;
} else {
*__slate_slot_57 = devm_kzalloc(dev, 312u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut ahci_host_priv;
if (*__slate_slot_57).is_null() {
return 12i32.wrapping_neg();
} else {
let __v47: *mut ahci_host_priv = *__slate_slot_57;
(*__v47).flags = (((*__v47).flags as u64) | ((*__slate_slot_60).private_data as u64)) as u32;
// the Marvell "Thor" family needs HOST_IRQ_STAT cleared first
if board_id == 16 {
(*(*__slate_slot_57)).irq_handler = Some(ahci_mv_irq_handler);
}
// MCP65 revision A1 and A2 can't do MSI
if board_id == 13 && (((*pdev).revision as i32) == 161 || ((*pdev).revision as i32) == 162) {
let __v44: *mut ahci_host_priv = *__slate_slot_57;
(*__v44).flags = (*__v44).flags | 32;
}
// SB800 does NOT need the workaround to ignore SERR_INTERNAL
if board_id == 18 && ((*pdev).revision as i32) >= 64 {
let __v41: *mut ahci_host_priv = *__slate_slot_57;
let __v39: u32 = (*__v41).flags & !4;
(*__v41).flags = __v39;
}
// only some SB600s can do 64bit DMA
if ahci_sb600_enable_64bit(pdev) {
let __v38: *mut ahci_host_priv = *__slate_slot_57;
let __v36: u32 = (*__v38).flags & !8;
(*__v38).flags = __v36;
}
(*(*__slate_slot_57)).mmio = pcim_iomap(pdev, ahci_pci_bar, 0u64);
if unsafe { (*(*__slate_slot_57)).mmio }.is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_53 = ahci_validate_bar_size(pdev, ahci_pci_bar, *__slate_slot_57);
if *__slate_slot_53 != 0 {
return *__slate_slot_53;
} else {
// detect remapped nvme devices
ahci_remap_check(pdev, ahci_pci_bar, *__slate_slot_57);
sysfs_add_file_to_group(core::ptr::addr_of_mut!((*pdev).dev.kobj), core::ptr::addr_of_mut!(dev_attr_remapped_nvme.attr) as *const attribute, core::ptr::null::<u8>());
// save initial config
ahci_pci_save_initial_config(pdev, *__slate_slot_57);
// prepare host
if (*(*__slate_slot_57)).cap & 1073741824 != 0 {
(*__slate_slot_60).flags |= 1024;
// Auto-activate optimization is supposed to be
// supported on all AHCI controllers indicating NCQ
// capability, but it seems to be broken on some
// chipsets including NVIDIAs.
if (*(*__slate_slot_57)).flags & 8192 == 0 {
(*__slate_slot_60).flags |= 16384;
}
// All AHCI controllers should be forward-compatible
// with the new auxiliary field. This code should be
// conditionalized if any buggy AHCI controllers are
// encountered.
(*__slate_slot_60).flags |= 1048576;
}
if (*(*__slate_slot_57)).cap & 131072 != 0 {
(*__slate_slot_60).flags |= 524288;
}
ahci_set_em_messages(*__slate_slot_57, core::ptr::addr_of_mut!(*__slate_slot_60));
if ahci_broken_system_poweroff(pdev) {
(*__slate_slot_60).flags |= 2048;
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"quirky BIOS, skipping spindown on poweroff\n".as_ptr() as *const u8);
}
if ahci_broken_lpm(pdev) {
(*__slate_slot_60).flags |= 4;
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"BIOS update required for Link Power Management support\n".as_ptr() as *const u8);
}
if ahci_broken_suspend(pdev) {
let __v23: *mut ahci_host_priv = *__slate_slot_57;
(*__v23).flags = (*__v23).flags | 1024;
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"BIOS update required for suspend/resume\n".as_ptr() as *const u8);
}
if ahci_broken_online(pdev) {
let __v20: *mut ahci_host_priv = *__slate_slot_57;
(*__v20).flags = (*__v20).flags | 2048;
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"online status unreliable, applying workaround\n".as_ptr() as *const u8);
}
// Acer SA5-271 workaround modifies private_data
acer_sa5_271_workaround(*__slate_slot_57, pdev);
__UNIQUE_ID_x__683 = ahci_nr_ports((*(*__slate_slot_57)).cap);
__UNIQUE_ID_y__684 = fls((*(*__slate_slot_57)).port_map);
loop {
if __types_ok!(__UNIQUE_ID_x__683, __UNIQUE_ID_y__684) == 0 {
__compiletime_assert_685();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
// CAP.NP sometimes indicate the index of the last enabled
// port, at other times, that of the last possible port, so
// determining the maximum port number requires looking at
// both CAP.NP and port_map.
*__slate_slot_55 = if __UNIQUE_ID_x__683 > __UNIQUE_ID_y__684 { __UNIQUE_ID_x__683 } else { __UNIQUE_ID_y__684 };
*__slate_slot_56 = ata_host_alloc_pinfo(core::ptr::addr_of_mut!((*pdev).dev), (*__slate_slot_59).as_mut_ptr() as *const *const ata_port_info, *__slate_slot_55);
if (*__slate_slot_56).is_null() {
*__slate_slot_53 = 12i32.wrapping_neg();
} else {
(*(*__slate_slot_56)).private_data = *__slate_slot_57 as *mut ();
ahci_init_irq(pdev, *__slate_slot_55 as u32, *__slate_slot_57);
(*(*__slate_slot_57)).irq = pci_irq_vector(pdev, 0u32) as u32;
if (*(*__slate_slot_57)).cap & 134217728 == 0 || ahci_ignore_sss != 0 {
let __v14: *mut ata_host = *__slate_slot_56;
(*__v14).flags = (*__v14).flags | 4;
} else {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"SSS flag set, parallel bus scan disabled\n".as_ptr() as *const u8);
}
if (*(*__slate_slot_57)).cap & 8192 == 0 {
let __v11: *mut ata_host = *__slate_slot_56;
(*__v11).flags = (*__v11).flags | 16;
}
if (*(*__slate_slot_57)).cap & 16384 == 0 {
let __v8: *mut ata_host = *__slate_slot_56;
(*__v8).flags = (*__v8).flags | 32;
}
if (*(*__slate_slot_57)).cap2 & 8 == 0 {
let __v5: *mut ata_host = *__slate_slot_56;
(*__v5).flags = (*__v5).flags | 64;
}
if (*__slate_slot_60).flags & 2097152 != 0 {
ahci_reset_em(*__slate_slot_56);
}
*__slate_slot_54 = 0;
while (*__slate_slot_54 as u32) < (*(*__slate_slot_56)).n_ports {
ap = *(core::ptr::addr_of_mut!((*(*__slate_slot_56)).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_54 as isize);
ata_port_pbar_desc(ap, ahci_pci_bar, 1i32.wrapping_neg() as i64, c"abar".as_ptr() as *const u8);
ata_port_pbar_desc(ap, ahci_pci_bar, 256u32.wrapping_add(unsafe { (*ap).port_no }.wrapping_mul(128)) as i64, c"port".as_ptr() as *const u8);
// set enclosure management message type
if (*ap).flags & 2097152 != 0 {
(*ap).em_message_type = (*(*__slate_slot_57)).em_msg_type as i32;
}
// disabled/not-implemented port
if (*(*__slate_slot_57)).port_map & ((1i32 << *__slate_slot_54) as u32) == 0 {
(*ap).ops = core::ptr::addr_of_mut!(ata_dummy_port_ops);
} else {
ahci_mark_external_port(ap);
ahci_update_initial_lpm_policy(ap);
}
let __v1: i32 = *__slate_slot_54;
*__slate_slot_54 = __v1.wrapping_add(1);
}
// apply workaround for ASUS P5W DH Deluxe mainboard
ahci_p5wdh_workaround(*__slate_slot_56);
// apply gtf filter quirk
ahci_gtf_filter_workaround(*__slate_slot_56);
// initialize adapter
*__slate_slot_53 = ahci_configure_dma_masks(pdev, *__slate_slot_57);
if *__slate_slot_53 == 0 {
*__slate_slot_53 = ahci_pci_reset_controller(*__slate_slot_56);
if *__slate_slot_53 == 0 {
ahci_pci_init_controller(*__slate_slot_56);
ahci_pci_print_info(*__slate_slot_56);
pci_set_master(pdev);
*__slate_slot_53 = ahci_host_activate(*__slate_slot_56, core::ptr::addr_of!(ahci_sht.0));
if *__slate_slot_53 == 0 {
pm_runtime_put_noidle(core::ptr::addr_of_mut!((*pdev).dev));
return 0;
}
}
}
}
sysfs_remove_file_from_group(core::ptr::addr_of_mut!((*pdev).dev.kobj), core::ptr::addr_of_mut!(dev_attr_remapped_nvme.attr) as *const attribute, core::ptr::null::<u8>());
return *__slate_slot_53;
}
}
}
}
}
}
}
}

static ahci_sht: __SlateConst<scsi_host_template> = __SlateConst({
let mut __t0: scsi_host_template = unsafe { core::mem::zeroed() };
__t0.queuecommand = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>>(ata_scsi_queuecmd as *const ()) };
__t0.module = core::ptr::null_mut::<module>();
__t0.name = (b"ahci\0".as_ptr() as *mut u8) as *const u8;
__t0.ioctl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>>(ata_scsi_ioctl as *const ()) };
__t0.compat_ioctl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>>(ata_scsi_ioctl as *const ()) };
__t0.sdev_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device) -> i32>>(ata_scsi_sdev_init as *const ()) };
__t0.sdev_configure = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device, *mut queue_limits) -> i32>>(ata_scsi_sdev_configure as *const ()) };
__t0.sdev_destroy = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device)>>(ata_scsi_sdev_destroy as *const ()) };
__t0.change_queue_depth = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device, i32) -> i32>>(ata_scsi_change_queue_depth as *const ()) };
__t0.dma_need_drain = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut request) -> bool>>(ata_scsi_dma_need_drain as *const ()) };
__t0.bios_param = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device, *mut gendisk, u64, *mut i32) -> i32>>(ata_std_bios_param as *const ()) };
__t0.unlock_native_capacity = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_device)>>(ata_scsi_unlock_native_capacity as *const ()) };
__t0.eh_timed_out = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u32>>(ata_scsi_eh_timed_out as *const ()) };
__t0.proc_name = (b"ahci\0".as_ptr() as *mut u8) as *const u8;
__t0.can_queue = 32i32;
__t0.this_id = -1i32;
__t0.sg_tablesize = 168u16;
__t0.max_sectors = 65535u32;
__t0.dma_boundary = 4294967295u64;
__t0.__slate_bits_0.__set_tag_alloc_policy_rr(1i32 != 0i32);
__t0.__slate_bits_0.__set_emulated(1u32);
__t0.shost_groups = unsafe { core::ptr::addr_of_mut!(ahci_shost_groups) as *mut *const attribute_group };
__t0.sdev_groups = unsafe { core::ptr::addr_of_mut!(ahci_sdev_groups) as *mut *const attribute_group };
__t0
});

static mut ahci_vt8251_ops: ata_port_operations = {
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.reset = {
let mut __t1: ata_reset_operations = unsafe { core::mem::zeroed() };
__t1.hardreset = Some(ahci_vt8251_hardreset);
__t1
};
__t0.inherits = (unsafe { core::ptr::addr_of_mut!(ahci_ops) }) as *const ata_port_operations;
__t0
};

static mut ahci_p5wdh_ops: ata_port_operations = {
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.reset = {
let mut __t1: ata_reset_operations = unsafe { core::mem::zeroed() };
__t1.hardreset = Some(ahci_p5wdh_hardreset);
__t1
};
__t0.inherits = (unsafe { core::ptr::addr_of_mut!(ahci_ops) }) as *const ata_port_operations;
__t0
};

static mut ahci_avn_ops: ata_port_operations = {
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.reset = {
let mut __t1: ata_reset_operations = unsafe { core::mem::zeroed() };
__t1.hardreset = Some(ahci_avn_hardreset);
__t1
};
__t0.inherits = (unsafe { core::ptr::addr_of_mut!(ahci_ops) }) as *const ata_port_operations;
__t0
};

/// by features
/// by chipsets
/// JMicron JMB582/585: 64-bit DMA is broken, force 32-bit
/// for SB700 and SB800
static ahci_port_info: __SlateAlign16<[ata_port_info; 20]> = __SlateAlign16([{
let mut __t0: ata_port_info = unsafe { core::mem::zeroed() };
__t0.flags = 393346u64;
__t0.pio_mask = 31u32;
__t0.udma_mask = 127u32;
__t0.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t0
}, {
let mut __t1: ata_port_info = unsafe { core::mem::zeroed() };
__t1.flags = 393346u64;
__t1.pio_mask = 31u32;
__t1.udma_mask = 127u32;
__t1.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t1.private_data = 134217728u32 as *mut ();
__t1
}, {
let mut __t2: ata_port_info = unsafe { core::mem::zeroed() };
__t2.flags = 393346u64;
__t2.pio_mask = 31u32;
__t2.udma_mask = 127u32;
__t2.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t2.private_data = 2u32 as *mut ();
__t2
}, {
let mut __t3: ata_port_info = unsafe { core::mem::zeroed() };
__t3.flags = 393346u64;
__t3.link_flags = 2048u64;
__t3.pio_mask = 31u32;
__t3.udma_mask = 127u32;
__t3.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t3
}, {
let mut __t4: ata_port_info = unsafe { core::mem::zeroed() };
__t4.flags = 393346u64;
__t4.pio_mask = 31u32;
__t4.udma_mask = 127u32;
__t4.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t4.private_data = 32u32 as *mut ();
__t4
}, {
let mut __t5: ata_port_info = unsafe { core::mem::zeroed() };
__t5.flags = 393346u64;
__t5.pio_mask = 31u32;
__t5.udma_mask = 127u32;
__t5.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t5.private_data = 268435456u32 as *mut ();
__t5
}, {
let mut __t6: ata_port_info = unsafe { core::mem::zeroed() };
__t6.flags = 393346u64;
__t6.pio_mask = 31u32;
__t6.udma_mask = 127u32;
__t6.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t6.private_data = 268566528u32 as *mut ();
__t6
}, {
let mut __t7: ata_port_info = unsafe { core::mem::zeroed() };
__t7.flags = 393346u64;
__t7.pio_mask = 31u32;
__t7.udma_mask = 127u32;
__t7.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t7.private_data = 268439552u32 as *mut ();
__t7
}, {
let mut __t8: ata_port_info = unsafe { core::mem::zeroed() };
__t8.flags = 393346u64;
__t8.pio_mask = 31u32;
__t8.udma_mask = 127u32;
__t8.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t8.private_data = 16384u32 as *mut ();
__t8
}, {
let mut __t9: ata_port_info = unsafe { core::mem::zeroed() };
__t9.flags = 393346u64;
__t9.pio_mask = 31u32;
__t9.udma_mask = 127u32;
__t9.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t9.private_data = 536887296u32 as *mut ();
__t9
}, {
let mut __t10: ata_port_info = unsafe { core::mem::zeroed() };
__t10.flags = 393346u64;
__t10.pio_mask = 31u32;
__t10.udma_mask = 127u32;
__t10.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t10.private_data = 96u32 as *mut ();
__t10
}, {
let mut __t11: ata_port_info = unsafe { core::mem::zeroed() };
__t11.flags = 393346u64;
__t11.pio_mask = 31u32;
__t11.udma_mask = 127u32;
__t11.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_avn_ops) };
__t11.private_data = 268435456u32 as *mut ();
__t11
}, {
let mut __t12: ata_port_info = unsafe { core::mem::zeroed() };
__t12.flags = 393346u64;
__t12.pio_mask = 31u32;
__t12.udma_mask = 127u32;
__t12.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t12.private_data = 10u32 as *mut ();
__t12
}, {
let mut __t13: ata_port_info = unsafe { core::mem::zeroed() };
__t13.flags = 8781954u64;
__t13.pio_mask = 31u32;
__t13.udma_mask = 127u32;
__t13.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t13.private_data = 8768u32 as *mut ();
__t13
}, {
let mut __t14: ata_port_info = unsafe { core::mem::zeroed() };
__t14.flags = 393346u64;
__t14.pio_mask = 31u32;
__t14.udma_mask = 127u32;
__t14.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t14.private_data = 8256u32 as *mut ();
__t14
}, {
let mut __t15: ata_port_info = unsafe { core::mem::zeroed() };
__t15.flags = 393346u64;
__t15.pio_mask = 31u32;
__t15.udma_mask = 127u32;
__t15.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t15.private_data = 8192u32 as *mut ();
__t15
}, {
let mut __t16: ata_port_info = unsafe { core::mem::zeroed() };
__t16.flags = 130u64;
__t16.pio_mask = 31u32;
__t16.udma_mask = 127u32;
__t16.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_ops) };
__t16.private_data = 113u32 as *mut ();
__t16
}, {
let mut __t17: ata_port_info = unsafe { core::mem::zeroed() };
__t17.flags = 393346u64;
__t17.pio_mask = 31u32;
__t17.udma_mask = 127u32;
__t17.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_pmp_retry_srst_ops) };
__t17.private_data = 300u32 as *mut ();
__t17
}, {
let mut __t18: ata_port_info = unsafe { core::mem::zeroed() };
__t18.flags = 393346u64;
__t18.pio_mask = 31u32;
__t18.udma_mask = 127u32;
__t18.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_pmp_retry_srst_ops) };
__t18.private_data = 4u32 as *mut ();
__t18
}, {
let mut __t19: ata_port_info = unsafe { core::mem::zeroed() };
__t19.flags = 393346u64;
__t19.pio_mask = 31u32;
__t19.udma_mask = 127u32;
__t19.port_ops = unsafe { core::ptr::addr_of_mut!(ahci_vt8251_ops) };
__t19.private_data = 65u32 as *mut ();
__t19
}]);

/// Intel
/// Comet Lake PCH-H RAID
/// ICH6
/// ICH6M
/// ICH7
/// ICH7M
/// ICH7R
/// ULi M5288
/// ESB2
/// ESB2
/// ESB2
/// ICH7-M DH
/// ICH8
/// ICH8/Lewisburg RAID
/// ICH8
/// ICH8M
/// ICH8M
/// ICH9
/// ICH9
/// ICH9
/// ICH9
/// ICH9
/// ICH9M
/// ICH9M
/// ICH9M
/// ICH9M
/// ICH9M
/// ICH9
/// ICH9M
/// Tolapai
/// Tolapai
/// ICH10
/// ICH10
/// ICH10
/// PCH AHCI
/// PCH AHCI
/// PCH RAID
/// PCH RAID
/// PCH M AHCI
/// PCH RAID
/// PCH M RAID
/// PCH AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// DNV AHCI
/// CPT AHCI
/// CPT M AHCI
/// CPT RAID
/// CPT M RAID
/// CPT RAID
/// CPT RAID
/// PBG AHCI
/// PBG RAID
/// PBG RAID
/// DH89xxCC AHCI
/// Panther Point AHCI
/// Panther M AHCI
/// Panther Point RAID
/// Panther Point RAID
/// Panther Point RAID
/// Panther M RAID
/// Panther Point RAID
/// Lynx Point AHCI
/// Lynx M AHCI
/// Lynx Point RAID
/// Lynx M RAID
/// Lynx Point RAID
/// Lynx M RAID
/// Lynx Point RAID
/// Lynx M RAID
/// Lynx LP AHCI
/// Lynx LP AHCI
/// Lynx LP RAID
/// Lynx LP RAID
/// Lynx LP RAID
/// Lynx LP RAID
/// Lynx LP RAID
/// Lynx LP RAID
/// Cannon Lake PCH-LP AHCI
/// Avoton AHCI
/// Avoton AHCI
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton AHCI
/// Avoton AHCI
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Avoton RAID
/// Wellsburg/Lewisburg AHCI
/// *burg SATA0 'RAID'
/// *burg SATA1 'RAID'
/// *burg SATA2 'RAID'
/// Rocket Lake PCH-H RAID
/// Rocket Lake PCH-H RAID
/// Rocket Lake PCH-H RAID
/// Rocket Lake PCH-H RAID
/// Wellsburg AHCI
/// Wellsburg RAID
/// Wellsburg RAID
/// Wellsburg RAID
/// Wellsburg AHCI
/// Wellsburg RAID
/// Wellsburg RAID
/// Wellsburg RAID
/// Coleto Creek AHCI
/// Wildcat LP AHCI
/// Wildcat LP RAID
/// Wildcat LP RAID
/// Wildcat LP RAID
/// 9 Series AHCI
/// 9 Series M AHCI
/// 9 Series RAID
/// 9 Series M RAID
/// 9 Series RAID
/// 9 Series M RAID
/// 9 Series RAID
/// 9 Series M RAID
/// Sunrise LP AHCI
/// Sunrise LP RAID
/// Sunrise LP RAID
/// Sunrise Point-H AHCI
/// Sunrise M AHCI
/// Sunrise Point-H RAID
/// Sunrise Point-H RAID
/// Sunrise M RAID
/// Sunrise Point-H RAID
/// Lewisburg AHCI
/// Lewisburg RAID
/// Lewisburg RAID
/// Lewisburg RAID
/// Lewisburg AHCI
/// Lewisburg RAID
/// Lewisburg RAID
/// Lewisburg RAID
/// Cannon Lake PCH-H RAID
/// Comet Lake-H RAID
/// Comet Lake PCH-V RAID
/// Bay Trail AHCI
/// Bay Trail AHCI
/// Cherry Tr. AHCI
/// ApolloLake AHCI
/// Ice Lake LP AHCI
/// Comet Lake PCH-U AHCI
/// Comet Lake PCH RAID
/// Elkhart Lake IDs 0x4b60 & 0x4b62 https://sata-io.org/product/8803 not tested yet
/// Elkhart Lake AHCI
/// JMicron JMB582/585: force 32-bit DMA (broken 64-bit implementation)
/// JMicron 360/1/3/5/6, match class to avoid IDE function
/// JMicron 362B and 362C have an AHCI function with IDE class code
/// May need to update quirk_jmicron_async_suspend() for additions
/// ATI
/// ATI SB600
/// ATI SB700/800
/// ATI SB700/800
/// ATI SB700/800
/// ATI SB700/800
/// ATI SB700/800
/// ATI SB700/800
/// Amazon's Annapurna Labs support
/// AMD
/// AMD Hudson-2
/// AMD Hudson-2 (AHCI mode)
/// AMD CZ
/// AMD Green Sardine
/// AMD is using RAID class only for ahci controllers
/// Dell S140/S150
/// VIA
/// VIA VT8251
/// VIA VT8251
/// NVIDIA
/// MCP65
/// MCP65
/// MCP65
/// MCP65
/// MCP65
/// MCP65
/// MCP65
/// MCP65
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// MCP67
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// Linux ID
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP73
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP77
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP79
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// MCP89
/// SiS
/// SiS 966
/// SiS 968
/// SiS 968
/// ST Microelectronics
/// ST ConneXt
/// Marvell
/// 6145
/// 6121
/// 88se9128
/// 88se9125
/// 88se9170
/// 88se9172
/// 88se9182
/// 88se9172
/// 88se9172 on some Gigabyte
/// 88se91a2
/// TTI
/// highpoint rocketraid 642L
/// highpoint rocketraid 644L
/// Promise
/// PDC42819
/// FastTrak TX8660 ahci-mode
/// ASMedia
/// ASM1060
/// ASM1060
/// ASM1061
/// ASM1061/1062
/// ASM1061R
/// ASM1062R
/// ASM1062+JMB575
/// ASM1062A
/// ASM1064
/// ASM1164
/// ASM1165
/// ASM1166
/// Samsung SSDs found on some macbooks.  NCQ times out if MSI is
/// enabled.  https://bugzilla.kernel.org/show_bug.cgi?id=60731
/// Enmotus
/// Loongson
/// Generic, PCI class code for AHCI
/// terminate list
static ahci_pci_tbl: __SlateAlign16<[pci_device_id; 314]> = __SlateAlign16([{
let mut __t0: pci_device_id = unsafe { core::mem::zeroed() };
__t0.vendor = 32902u32;
__t0.device = 1750u32;
__t0.subvendor = 4294967295u32;
__t0.subdevice = 4294967295u32;
__t0.class = 0u32;
__t0.class_mask = 0u32;
__t0.driver_data = 5u64;
__t0
}, {
let mut __t1: pci_device_id = unsafe { core::mem::zeroed() };
__t1.vendor = 32902u32;
__t1.device = 9810u32;
__t1.subvendor = 4294967295u32;
__t1.subdevice = 4294967295u32;
__t1.class = 0u32;
__t1.class_mask = 0u32;
__t1.driver_data = 5u64;
__t1
}, {
let mut __t2: pci_device_id = unsafe { core::mem::zeroed() };
__t2.vendor = 32902u32;
__t2.device = 9811u32;
__t2.subvendor = 4294967295u32;
__t2.subdevice = 4294967295u32;
__t2.class = 0u32;
__t2.class_mask = 0u32;
__t2.driver_data = 5u64;
__t2
}, {
let mut __t3: pci_device_id = unsafe { core::mem::zeroed() };
__t3.vendor = 32902u32;
__t3.device = 10177u32;
__t3.subvendor = 4294967295u32;
__t3.subdevice = 4294967295u32;
__t3.class = 0u32;
__t3.class_mask = 0u32;
__t3.driver_data = 5u64;
__t3
}, {
let mut __t4: pci_device_id = unsafe { core::mem::zeroed() };
__t4.vendor = 32902u32;
__t4.device = 10181u32;
__t4.subvendor = 4294967295u32;
__t4.subdevice = 4294967295u32;
__t4.class = 0u32;
__t4.class_mask = 0u32;
__t4.driver_data = 5u64;
__t4
}, {
let mut __t5: pci_device_id = unsafe { core::mem::zeroed() };
__t5.vendor = 32902u32;
__t5.device = 10179u32;
__t5.subvendor = 4294967295u32;
__t5.subdevice = 4294967295u32;
__t5.class = 0u32;
__t5.class_mask = 0u32;
__t5.driver_data = 5u64;
__t5
}, {
let mut __t6: pci_device_id = unsafe { core::mem::zeroed() };
__t6.vendor = 4281u32;
__t6.device = 21128u32;
__t6.subvendor = 4294967295u32;
__t6.subdevice = 4294967295u32;
__t6.class = 0u32;
__t6.class_mask = 0u32;
__t6.driver_data = 2u64;
__t6
}, {
let mut __t7: pci_device_id = unsafe { core::mem::zeroed() };
__t7.vendor = 32902u32;
__t7.device = 9857u32;
__t7.subvendor = 4294967295u32;
__t7.subdevice = 4294967295u32;
__t7.class = 0u32;
__t7.class_mask = 0u32;
__t7.driver_data = 5u64;
__t7
}, {
let mut __t8: pci_device_id = unsafe { core::mem::zeroed() };
__t8.vendor = 32902u32;
__t8.device = 9858u32;
__t8.subvendor = 4294967295u32;
__t8.subdevice = 4294967295u32;
__t8.class = 0u32;
__t8.class_mask = 0u32;
__t8.driver_data = 5u64;
__t8
}, {
let mut __t9: pci_device_id = unsafe { core::mem::zeroed() };
__t9.vendor = 32902u32;
__t9.device = 9859u32;
__t9.subvendor = 4294967295u32;
__t9.subdevice = 4294967295u32;
__t9.class = 0u32;
__t9.class_mask = 0u32;
__t9.driver_data = 5u64;
__t9
}, {
let mut __t10: pci_device_id = unsafe { core::mem::zeroed() };
__t10.vendor = 32902u32;
__t10.device = 10182u32;
__t10.subvendor = 4294967295u32;
__t10.subdevice = 4294967295u32;
__t10.class = 0u32;
__t10.class_mask = 0u32;
__t10.driver_data = 5u64;
__t10
}, {
let mut __t11: pci_device_id = unsafe { core::mem::zeroed() };
__t11.vendor = 32902u32;
__t11.device = 10273u32;
__t11.subvendor = 4294967295u32;
__t11.subdevice = 4294967295u32;
__t11.class = 0u32;
__t11.class_mask = 0u32;
__t11.driver_data = 5u64;
__t11
}, {
let mut __t12: pci_device_id = unsafe { core::mem::zeroed() };
__t12.vendor = 32902u32;
__t12.device = 10274u32;
__t12.subvendor = 4294967295u32;
__t12.subdevice = 4294967295u32;
__t12.class = 0u32;
__t12.class_mask = 0u32;
__t12.driver_data = 7u64;
__t12
}, {
let mut __t13: pci_device_id = unsafe { core::mem::zeroed() };
__t13.vendor = 32902u32;
__t13.device = 10276u32;
__t13.subvendor = 4294967295u32;
__t13.subdevice = 4294967295u32;
__t13.class = 0u32;
__t13.class_mask = 0u32;
__t13.driver_data = 5u64;
__t13
}, {
let mut __t14: pci_device_id = unsafe { core::mem::zeroed() };
__t14.vendor = 32902u32;
__t14.device = 10281u32;
__t14.subvendor = 4294967295u32;
__t14.subdevice = 4294967295u32;
__t14.class = 0u32;
__t14.class_mask = 0u32;
__t14.driver_data = 5u64;
__t14
}, {
let mut __t15: pci_device_id = unsafe { core::mem::zeroed() };
__t15.vendor = 32902u32;
__t15.device = 10282u32;
__t15.subvendor = 4294967295u32;
__t15.subdevice = 4294967295u32;
__t15.class = 0u32;
__t15.class_mask = 0u32;
__t15.driver_data = 5u64;
__t15
}, {
let mut __t16: pci_device_id = unsafe { core::mem::zeroed() };
__t16.vendor = 32902u32;
__t16.device = 10530u32;
__t16.subvendor = 4294967295u32;
__t16.subdevice = 4294967295u32;
__t16.class = 0u32;
__t16.class_mask = 0u32;
__t16.driver_data = 5u64;
__t16
}, {
let mut __t17: pci_device_id = unsafe { core::mem::zeroed() };
__t17.vendor = 32902u32;
__t17.device = 10531u32;
__t17.subvendor = 4294967295u32;
__t17.subdevice = 4294967295u32;
__t17.class = 0u32;
__t17.class_mask = 0u32;
__t17.driver_data = 5u64;
__t17
}, {
let mut __t18: pci_device_id = unsafe { core::mem::zeroed() };
__t18.vendor = 32902u32;
__t18.device = 10532u32;
__t18.subvendor = 4294967295u32;
__t18.subdevice = 4294967295u32;
__t18.class = 0u32;
__t18.class_mask = 0u32;
__t18.driver_data = 5u64;
__t18
}, {
let mut __t19: pci_device_id = unsafe { core::mem::zeroed() };
__t19.vendor = 32902u32;
__t19.device = 10533u32;
__t19.subvendor = 4294967295u32;
__t19.subdevice = 4294967295u32;
__t19.class = 0u32;
__t19.class_mask = 0u32;
__t19.driver_data = 5u64;
__t19
}, {
let mut __t20: pci_device_id = unsafe { core::mem::zeroed() };
__t20.vendor = 32902u32;
__t20.device = 10535u32;
__t20.subvendor = 4294967295u32;
__t20.subdevice = 4294967295u32;
__t20.class = 0u32;
__t20.class_mask = 0u32;
__t20.driver_data = 5u64;
__t20
}, {
let mut __t21: pci_device_id = unsafe { core::mem::zeroed() };
__t21.vendor = 32902u32;
__t21.device = 10537u32;
__t21.subvendor = 4294967295u32;
__t21.subdevice = 4294967295u32;
__t21.class = 0u32;
__t21.class_mask = 0u32;
__t21.driver_data = 5u64;
__t21
}, {
let mut __t22: pci_device_id = unsafe { core::mem::zeroed() };
__t22.vendor = 32902u32;
__t22.device = 10538u32;
__t22.subvendor = 4294967295u32;
__t22.subdevice = 4294967295u32;
__t22.class = 0u32;
__t22.class_mask = 0u32;
__t22.driver_data = 5u64;
__t22
}, {
let mut __t23: pci_device_id = unsafe { core::mem::zeroed() };
__t23.vendor = 32902u32;
__t23.device = 10539u32;
__t23.subvendor = 4294967295u32;
__t23.subdevice = 4294967295u32;
__t23.class = 0u32;
__t23.class_mask = 0u32;
__t23.driver_data = 5u64;
__t23
}, {
let mut __t24: pci_device_id = unsafe { core::mem::zeroed() };
__t24.vendor = 32902u32;
__t24.device = 10540u32;
__t24.subvendor = 4294967295u32;
__t24.subdevice = 4294967295u32;
__t24.class = 0u32;
__t24.class_mask = 0u32;
__t24.driver_data = 5u64;
__t24
}, {
let mut __t25: pci_device_id = unsafe { core::mem::zeroed() };
__t25.vendor = 32902u32;
__t25.device = 10543u32;
__t25.subvendor = 4294967295u32;
__t25.subdevice = 4294967295u32;
__t25.class = 0u32;
__t25.class_mask = 0u32;
__t25.driver_data = 5u64;
__t25
}, {
let mut __t26: pci_device_id = unsafe { core::mem::zeroed() };
__t26.vendor = 32902u32;
__t26.device = 10573u32;
__t26.subvendor = 4294967295u32;
__t26.subdevice = 4294967295u32;
__t26.class = 0u32;
__t26.class_mask = 0u32;
__t26.driver_data = 5u64;
__t26
}, {
let mut __t27: pci_device_id = unsafe { core::mem::zeroed() };
__t27.vendor = 32902u32;
__t27.device = 10574u32;
__t27.subvendor = 4294967295u32;
__t27.subdevice = 4294967295u32;
__t27.class = 0u32;
__t27.class_mask = 0u32;
__t27.driver_data = 5u64;
__t27
}, {
let mut __t28: pci_device_id = unsafe { core::mem::zeroed() };
__t28.vendor = 32902u32;
__t28.device = 20522u32;
__t28.subvendor = 4294967295u32;
__t28.subdevice = 4294967295u32;
__t28.class = 0u32;
__t28.class_mask = 0u32;
__t28.driver_data = 5u64;
__t28
}, {
let mut __t29: pci_device_id = unsafe { core::mem::zeroed() };
__t29.vendor = 32902u32;
__t29.device = 20523u32;
__t29.subvendor = 4294967295u32;
__t29.subdevice = 4294967295u32;
__t29.class = 0u32;
__t29.class_mask = 0u32;
__t29.driver_data = 5u64;
__t29
}, {
let mut __t30: pci_device_id = unsafe { core::mem::zeroed() };
__t30.vendor = 32902u32;
__t30.device = 14853u32;
__t30.subvendor = 4294967295u32;
__t30.subdevice = 4294967295u32;
__t30.class = 0u32;
__t30.class_mask = 0u32;
__t30.driver_data = 5u64;
__t30
}, {
let mut __t31: pci_device_id = unsafe { core::mem::zeroed() };
__t31.vendor = 32902u32;
__t31.device = 14882u32;
__t31.subvendor = 4294967295u32;
__t31.subdevice = 4294967295u32;
__t31.class = 0u32;
__t31.class_mask = 0u32;
__t31.driver_data = 5u64;
__t31
}, {
let mut __t32: pci_device_id = unsafe { core::mem::zeroed() };
__t32.vendor = 32902u32;
__t32.device = 14885u32;
__t32.subvendor = 4294967295u32;
__t32.subdevice = 4294967295u32;
__t32.class = 0u32;
__t32.class_mask = 0u32;
__t32.driver_data = 5u64;
__t32
}, {
let mut __t33: pci_device_id = unsafe { core::mem::zeroed() };
__t33.vendor = 32902u32;
__t33.device = 15138u32;
__t33.subvendor = 4294967295u32;
__t33.subdevice = 4294967295u32;
__t33.class = 0u32;
__t33.class_mask = 0u32;
__t33.driver_data = 5u64;
__t33
}, {
let mut __t34: pci_device_id = unsafe { core::mem::zeroed() };
__t34.vendor = 32902u32;
__t34.device = 15139u32;
__t34.subvendor = 4294967295u32;
__t34.subdevice = 4294967295u32;
__t34.class = 0u32;
__t34.class_mask = 0u32;
__t34.driver_data = 5u64;
__t34
}, {
let mut __t35: pci_device_id = unsafe { core::mem::zeroed() };
__t35.vendor = 32902u32;
__t35.device = 15140u32;
__t35.subvendor = 4294967295u32;
__t35.subdevice = 4294967295u32;
__t35.class = 0u32;
__t35.class_mask = 0u32;
__t35.driver_data = 5u64;
__t35
}, {
let mut __t36: pci_device_id = unsafe { core::mem::zeroed() };
__t36.vendor = 32902u32;
__t36.device = 15141u32;
__t36.subvendor = 4294967295u32;
__t36.subdevice = 4294967295u32;
__t36.class = 0u32;
__t36.class_mask = 0u32;
__t36.driver_data = 5u64;
__t36
}, {
let mut __t37: pci_device_id = unsafe { core::mem::zeroed() };
__t37.vendor = 32902u32;
__t37.device = 15145u32;
__t37.subvendor = 4294967295u32;
__t37.subdevice = 4294967295u32;
__t37.class = 0u32;
__t37.class_mask = 0u32;
__t37.driver_data = 5u64;
__t37
}, {
let mut __t38: pci_device_id = unsafe { core::mem::zeroed() };
__t38.vendor = 32902u32;
__t38.device = 15147u32;
__t38.subvendor = 4294967295u32;
__t38.subdevice = 4294967295u32;
__t38.class = 0u32;
__t38.class_mask = 0u32;
__t38.driver_data = 5u64;
__t38
}, {
let mut __t39: pci_device_id = unsafe { core::mem::zeroed() };
__t39.vendor = 32902u32;
__t39.device = 15148u32;
__t39.subvendor = 4294967295u32;
__t39.subdevice = 4294967295u32;
__t39.class = 0u32;
__t39.class_mask = 0u32;
__t39.driver_data = 5u64;
__t39
}, {
let mut __t40: pci_device_id = unsafe { core::mem::zeroed() };
__t40.vendor = 32902u32;
__t40.device = 15151u32;
__t40.subvendor = 4294967295u32;
__t40.subdevice = 4294967295u32;
__t40.class = 0u32;
__t40.class_mask = 0u32;
__t40.driver_data = 5u64;
__t40
}, {
let mut __t41: pci_device_id = unsafe { core::mem::zeroed() };
__t41.vendor = 32902u32;
__t41.device = 6576u32;
__t41.subvendor = 4294967295u32;
__t41.subdevice = 4294967295u32;
__t41.class = 0u32;
__t41.class_mask = 0u32;
__t41.driver_data = 0u64;
__t41
}, {
let mut __t42: pci_device_id = unsafe { core::mem::zeroed() };
__t42.vendor = 32902u32;
__t42.device = 6577u32;
__t42.subvendor = 4294967295u32;
__t42.subdevice = 4294967295u32;
__t42.class = 0u32;
__t42.class_mask = 0u32;
__t42.driver_data = 0u64;
__t42
}, {
let mut __t43: pci_device_id = unsafe { core::mem::zeroed() };
__t43.vendor = 32902u32;
__t43.device = 6578u32;
__t43.subvendor = 4294967295u32;
__t43.subdevice = 4294967295u32;
__t43.class = 0u32;
__t43.class_mask = 0u32;
__t43.driver_data = 0u64;
__t43
}, {
let mut __t44: pci_device_id = unsafe { core::mem::zeroed() };
__t44.vendor = 32902u32;
__t44.device = 6579u32;
__t44.subvendor = 4294967295u32;
__t44.subdevice = 4294967295u32;
__t44.class = 0u32;
__t44.class_mask = 0u32;
__t44.driver_data = 0u64;
__t44
}, {
let mut __t45: pci_device_id = unsafe { core::mem::zeroed() };
__t45.vendor = 32902u32;
__t45.device = 6580u32;
__t45.subvendor = 4294967295u32;
__t45.subdevice = 4294967295u32;
__t45.class = 0u32;
__t45.class_mask = 0u32;
__t45.driver_data = 0u64;
__t45
}, {
let mut __t46: pci_device_id = unsafe { core::mem::zeroed() };
__t46.vendor = 32902u32;
__t46.device = 6581u32;
__t46.subvendor = 4294967295u32;
__t46.subdevice = 4294967295u32;
__t46.class = 0u32;
__t46.class_mask = 0u32;
__t46.driver_data = 0u64;
__t46
}, {
let mut __t47: pci_device_id = unsafe { core::mem::zeroed() };
__t47.vendor = 32902u32;
__t47.device = 6582u32;
__t47.subvendor = 4294967295u32;
__t47.subdevice = 4294967295u32;
__t47.class = 0u32;
__t47.class_mask = 0u32;
__t47.driver_data = 0u64;
__t47
}, {
let mut __t48: pci_device_id = unsafe { core::mem::zeroed() };
__t48.vendor = 32902u32;
__t48.device = 6583u32;
__t48.subvendor = 4294967295u32;
__t48.subdevice = 4294967295u32;
__t48.class = 0u32;
__t48.class_mask = 0u32;
__t48.driver_data = 0u64;
__t48
}, {
let mut __t49: pci_device_id = unsafe { core::mem::zeroed() };
__t49.vendor = 32902u32;
__t49.device = 6590u32;
__t49.subvendor = 4294967295u32;
__t49.subdevice = 4294967295u32;
__t49.class = 0u32;
__t49.class_mask = 0u32;
__t49.driver_data = 0u64;
__t49
}, {
let mut __t50: pci_device_id = unsafe { core::mem::zeroed() };
__t50.vendor = 32902u32;
__t50.device = 6591u32;
__t50.subvendor = 4294967295u32;
__t50.subdevice = 4294967295u32;
__t50.class = 0u32;
__t50.class_mask = 0u32;
__t50.driver_data = 0u64;
__t50
}, {
let mut __t51: pci_device_id = unsafe { core::mem::zeroed() };
__t51.vendor = 32902u32;
__t51.device = 6592u32;
__t51.subvendor = 4294967295u32;
__t51.subdevice = 4294967295u32;
__t51.class = 0u32;
__t51.class_mask = 0u32;
__t51.driver_data = 0u64;
__t51
}, {
let mut __t52: pci_device_id = unsafe { core::mem::zeroed() };
__t52.vendor = 32902u32;
__t52.device = 6593u32;
__t52.subvendor = 4294967295u32;
__t52.subdevice = 4294967295u32;
__t52.class = 0u32;
__t52.class_mask = 0u32;
__t52.driver_data = 0u64;
__t52
}, {
let mut __t53: pci_device_id = unsafe { core::mem::zeroed() };
__t53.vendor = 32902u32;
__t53.device = 6594u32;
__t53.subvendor = 4294967295u32;
__t53.subdevice = 4294967295u32;
__t53.class = 0u32;
__t53.class_mask = 0u32;
__t53.driver_data = 0u64;
__t53
}, {
let mut __t54: pci_device_id = unsafe { core::mem::zeroed() };
__t54.vendor = 32902u32;
__t54.device = 6595u32;
__t54.subvendor = 4294967295u32;
__t54.subdevice = 4294967295u32;
__t54.class = 0u32;
__t54.class_mask = 0u32;
__t54.driver_data = 0u64;
__t54
}, {
let mut __t55: pci_device_id = unsafe { core::mem::zeroed() };
__t55.vendor = 32902u32;
__t55.device = 6596u32;
__t55.subvendor = 4294967295u32;
__t55.subdevice = 4294967295u32;
__t55.class = 0u32;
__t55.class_mask = 0u32;
__t55.driver_data = 0u64;
__t55
}, {
let mut __t56: pci_device_id = unsafe { core::mem::zeroed() };
__t56.vendor = 32902u32;
__t56.device = 6597u32;
__t56.subvendor = 4294967295u32;
__t56.subdevice = 4294967295u32;
__t56.class = 0u32;
__t56.class_mask = 0u32;
__t56.driver_data = 0u64;
__t56
}, {
let mut __t57: pci_device_id = unsafe { core::mem::zeroed() };
__t57.vendor = 32902u32;
__t57.device = 6598u32;
__t57.subvendor = 4294967295u32;
__t57.subdevice = 4294967295u32;
__t57.class = 0u32;
__t57.class_mask = 0u32;
__t57.driver_data = 0u64;
__t57
}, {
let mut __t58: pci_device_id = unsafe { core::mem::zeroed() };
__t58.vendor = 32902u32;
__t58.device = 6599u32;
__t58.subvendor = 4294967295u32;
__t58.subdevice = 4294967295u32;
__t58.class = 0u32;
__t58.class_mask = 0u32;
__t58.driver_data = 0u64;
__t58
}, {
let mut __t59: pci_device_id = unsafe { core::mem::zeroed() };
__t59.vendor = 32902u32;
__t59.device = 6606u32;
__t59.subvendor = 4294967295u32;
__t59.subdevice = 4294967295u32;
__t59.class = 0u32;
__t59.class_mask = 0u32;
__t59.driver_data = 0u64;
__t59
}, {
let mut __t60: pci_device_id = unsafe { core::mem::zeroed() };
__t60.vendor = 32902u32;
__t60.device = 6607u32;
__t60.subvendor = 4294967295u32;
__t60.subdevice = 4294967295u32;
__t60.class = 0u32;
__t60.class_mask = 0u32;
__t60.driver_data = 0u64;
__t60
}, {
let mut __t61: pci_device_id = unsafe { core::mem::zeroed() };
__t61.vendor = 32902u32;
__t61.device = 7170u32;
__t61.subvendor = 4294967295u32;
__t61.subdevice = 4294967295u32;
__t61.class = 0u32;
__t61.class_mask = 0u32;
__t61.driver_data = 5u64;
__t61
}, {
let mut __t62: pci_device_id = unsafe { core::mem::zeroed() };
__t62.vendor = 32902u32;
__t62.device = 7171u32;
__t62.subvendor = 4294967295u32;
__t62.subdevice = 4294967295u32;
__t62.class = 0u32;
__t62.class_mask = 0u32;
__t62.driver_data = 5u64;
__t62
}, {
let mut __t63: pci_device_id = unsafe { core::mem::zeroed() };
__t63.vendor = 32902u32;
__t63.device = 7172u32;
__t63.subvendor = 4294967295u32;
__t63.subdevice = 4294967295u32;
__t63.class = 0u32;
__t63.class_mask = 0u32;
__t63.driver_data = 5u64;
__t63
}, {
let mut __t64: pci_device_id = unsafe { core::mem::zeroed() };
__t64.vendor = 32902u32;
__t64.device = 7173u32;
__t64.subvendor = 4294967295u32;
__t64.subdevice = 4294967295u32;
__t64.class = 0u32;
__t64.class_mask = 0u32;
__t64.driver_data = 5u64;
__t64
}, {
let mut __t65: pci_device_id = unsafe { core::mem::zeroed() };
__t65.vendor = 32902u32;
__t65.device = 7174u32;
__t65.subvendor = 4294967295u32;
__t65.subdevice = 4294967295u32;
__t65.class = 0u32;
__t65.class_mask = 0u32;
__t65.driver_data = 5u64;
__t65
}, {
let mut __t66: pci_device_id = unsafe { core::mem::zeroed() };
__t66.vendor = 32902u32;
__t66.device = 7175u32;
__t66.subvendor = 4294967295u32;
__t66.subdevice = 4294967295u32;
__t66.class = 0u32;
__t66.class_mask = 0u32;
__t66.driver_data = 5u64;
__t66
}, {
let mut __t67: pci_device_id = unsafe { core::mem::zeroed() };
__t67.vendor = 32902u32;
__t67.device = 7426u32;
__t67.subvendor = 4294967295u32;
__t67.subdevice = 4294967295u32;
__t67.class = 0u32;
__t67.class_mask = 0u32;
__t67.driver_data = 5u64;
__t67
}, {
let mut __t68: pci_device_id = unsafe { core::mem::zeroed() };
__t68.vendor = 32902u32;
__t68.device = 7428u32;
__t68.subvendor = 4294967295u32;
__t68.subdevice = 4294967295u32;
__t68.class = 0u32;
__t68.class_mask = 0u32;
__t68.driver_data = 5u64;
__t68
}, {
let mut __t69: pci_device_id = unsafe { core::mem::zeroed() };
__t69.vendor = 32902u32;
__t69.device = 7430u32;
__t69.subvendor = 4294967295u32;
__t69.subdevice = 4294967295u32;
__t69.class = 0u32;
__t69.class_mask = 0u32;
__t69.driver_data = 5u64;
__t69
}, {
let mut __t70: pci_device_id = unsafe { core::mem::zeroed() };
__t70.vendor = 32902u32;
__t70.device = 8995u32;
__t70.subvendor = 4294967295u32;
__t70.subdevice = 4294967295u32;
__t70.class = 0u32;
__t70.class_mask = 0u32;
__t70.driver_data = 5u64;
__t70
}, {
let mut __t71: pci_device_id = unsafe { core::mem::zeroed() };
__t71.vendor = 32902u32;
__t71.device = 7682u32;
__t71.subvendor = 4294967295u32;
__t71.subdevice = 4294967295u32;
__t71.class = 0u32;
__t71.class_mask = 0u32;
__t71.driver_data = 5u64;
__t71
}, {
let mut __t72: pci_device_id = unsafe { core::mem::zeroed() };
__t72.vendor = 32902u32;
__t72.device = 7683u32;
__t72.subvendor = 4294967295u32;
__t72.subdevice = 4294967295u32;
__t72.class = 0u32;
__t72.class_mask = 0u32;
__t72.driver_data = 5u64;
__t72
}, {
let mut __t73: pci_device_id = unsafe { core::mem::zeroed() };
__t73.vendor = 32902u32;
__t73.device = 7684u32;
__t73.subvendor = 4294967295u32;
__t73.subdevice = 4294967295u32;
__t73.class = 0u32;
__t73.class_mask = 0u32;
__t73.driver_data = 5u64;
__t73
}, {
let mut __t74: pci_device_id = unsafe { core::mem::zeroed() };
__t74.vendor = 32902u32;
__t74.device = 7685u32;
__t74.subvendor = 4294967295u32;
__t74.subdevice = 4294967295u32;
__t74.class = 0u32;
__t74.class_mask = 0u32;
__t74.driver_data = 5u64;
__t74
}, {
let mut __t75: pci_device_id = unsafe { core::mem::zeroed() };
__t75.vendor = 32902u32;
__t75.device = 7686u32;
__t75.subvendor = 4294967295u32;
__t75.subdevice = 4294967295u32;
__t75.class = 0u32;
__t75.class_mask = 0u32;
__t75.driver_data = 5u64;
__t75
}, {
let mut __t76: pci_device_id = unsafe { core::mem::zeroed() };
__t76.vendor = 32902u32;
__t76.device = 7687u32;
__t76.subvendor = 4294967295u32;
__t76.subdevice = 4294967295u32;
__t76.class = 0u32;
__t76.class_mask = 0u32;
__t76.driver_data = 5u64;
__t76
}, {
let mut __t77: pci_device_id = unsafe { core::mem::zeroed() };
__t77.vendor = 32902u32;
__t77.device = 7694u32;
__t77.subvendor = 4294967295u32;
__t77.subdevice = 4294967295u32;
__t77.class = 0u32;
__t77.class_mask = 0u32;
__t77.driver_data = 5u64;
__t77
}, {
let mut __t78: pci_device_id = unsafe { core::mem::zeroed() };
__t78.vendor = 32902u32;
__t78.device = 35842u32;
__t78.subvendor = 4294967295u32;
__t78.subdevice = 4294967295u32;
__t78.class = 0u32;
__t78.class_mask = 0u32;
__t78.driver_data = 5u64;
__t78
}, {
let mut __t79: pci_device_id = unsafe { core::mem::zeroed() };
__t79.vendor = 32902u32;
__t79.device = 35843u32;
__t79.subvendor = 4294967295u32;
__t79.subdevice = 4294967295u32;
__t79.class = 0u32;
__t79.class_mask = 0u32;
__t79.driver_data = 5u64;
__t79
}, {
let mut __t80: pci_device_id = unsafe { core::mem::zeroed() };
__t80.vendor = 32902u32;
__t80.device = 35844u32;
__t80.subvendor = 4294967295u32;
__t80.subdevice = 4294967295u32;
__t80.class = 0u32;
__t80.class_mask = 0u32;
__t80.driver_data = 5u64;
__t80
}, {
let mut __t81: pci_device_id = unsafe { core::mem::zeroed() };
__t81.vendor = 32902u32;
__t81.device = 35845u32;
__t81.subvendor = 4294967295u32;
__t81.subdevice = 4294967295u32;
__t81.class = 0u32;
__t81.class_mask = 0u32;
__t81.driver_data = 5u64;
__t81
}, {
let mut __t82: pci_device_id = unsafe { core::mem::zeroed() };
__t82.vendor = 32902u32;
__t82.device = 35846u32;
__t82.subvendor = 4294967295u32;
__t82.subdevice = 4294967295u32;
__t82.class = 0u32;
__t82.class_mask = 0u32;
__t82.driver_data = 5u64;
__t82
}, {
let mut __t83: pci_device_id = unsafe { core::mem::zeroed() };
__t83.vendor = 32902u32;
__t83.device = 35847u32;
__t83.subvendor = 4294967295u32;
__t83.subdevice = 4294967295u32;
__t83.class = 0u32;
__t83.class_mask = 0u32;
__t83.driver_data = 5u64;
__t83
}, {
let mut __t84: pci_device_id = unsafe { core::mem::zeroed() };
__t84.vendor = 32902u32;
__t84.device = 35854u32;
__t84.subvendor = 4294967295u32;
__t84.subdevice = 4294967295u32;
__t84.class = 0u32;
__t84.class_mask = 0u32;
__t84.driver_data = 5u64;
__t84
}, {
let mut __t85: pci_device_id = unsafe { core::mem::zeroed() };
__t85.vendor = 32902u32;
__t85.device = 35855u32;
__t85.subvendor = 4294967295u32;
__t85.subdevice = 4294967295u32;
__t85.class = 0u32;
__t85.class_mask = 0u32;
__t85.driver_data = 5u64;
__t85
}, {
let mut __t86: pci_device_id = unsafe { core::mem::zeroed() };
__t86.vendor = 32902u32;
__t86.device = 39938u32;
__t86.subvendor = 4294967295u32;
__t86.subdevice = 4294967295u32;
__t86.class = 0u32;
__t86.class_mask = 0u32;
__t86.driver_data = 5u64;
__t86
}, {
let mut __t87: pci_device_id = unsafe { core::mem::zeroed() };
__t87.vendor = 32902u32;
__t87.device = 39939u32;
__t87.subvendor = 4294967295u32;
__t87.subdevice = 4294967295u32;
__t87.class = 0u32;
__t87.class_mask = 0u32;
__t87.driver_data = 5u64;
__t87
}, {
let mut __t88: pci_device_id = unsafe { core::mem::zeroed() };
__t88.vendor = 32902u32;
__t88.device = 39940u32;
__t88.subvendor = 4294967295u32;
__t88.subdevice = 4294967295u32;
__t88.class = 0u32;
__t88.class_mask = 0u32;
__t88.driver_data = 5u64;
__t88
}, {
let mut __t89: pci_device_id = unsafe { core::mem::zeroed() };
__t89.vendor = 32902u32;
__t89.device = 39941u32;
__t89.subvendor = 4294967295u32;
__t89.subdevice = 4294967295u32;
__t89.class = 0u32;
__t89.class_mask = 0u32;
__t89.driver_data = 5u64;
__t89
}, {
let mut __t90: pci_device_id = unsafe { core::mem::zeroed() };
__t90.vendor = 32902u32;
__t90.device = 39942u32;
__t90.subvendor = 4294967295u32;
__t90.subdevice = 4294967295u32;
__t90.class = 0u32;
__t90.class_mask = 0u32;
__t90.driver_data = 5u64;
__t90
}, {
let mut __t91: pci_device_id = unsafe { core::mem::zeroed() };
__t91.vendor = 32902u32;
__t91.device = 39943u32;
__t91.subvendor = 4294967295u32;
__t91.subdevice = 4294967295u32;
__t91.class = 0u32;
__t91.class_mask = 0u32;
__t91.driver_data = 5u64;
__t91
}, {
let mut __t92: pci_device_id = unsafe { core::mem::zeroed() };
__t92.vendor = 32902u32;
__t92.device = 39950u32;
__t92.subvendor = 4294967295u32;
__t92.subdevice = 4294967295u32;
__t92.class = 0u32;
__t92.class_mask = 0u32;
__t92.driver_data = 5u64;
__t92
}, {
let mut __t93: pci_device_id = unsafe { core::mem::zeroed() };
__t93.vendor = 32902u32;
__t93.device = 39951u32;
__t93.subvendor = 4294967295u32;
__t93.subdevice = 4294967295u32;
__t93.class = 0u32;
__t93.class_mask = 0u32;
__t93.driver_data = 5u64;
__t93
}, {
let mut __t94: pci_device_id = unsafe { core::mem::zeroed() };
__t94.vendor = 32902u32;
__t94.device = 40403u32;
__t94.subvendor = 4294967295u32;
__t94.subdevice = 4294967295u32;
__t94.class = 0u32;
__t94.class_mask = 0u32;
__t94.driver_data = 5u64;
__t94
}, {
let mut __t95: pci_device_id = unsafe { core::mem::zeroed() };
__t95.vendor = 32902u32;
__t95.device = 7970u32;
__t95.subvendor = 4294967295u32;
__t95.subdevice = 4294967295u32;
__t95.class = 0u32;
__t95.class_mask = 0u32;
__t95.driver_data = 5u64;
__t95
}, {
let mut __t96: pci_device_id = unsafe { core::mem::zeroed() };
__t96.vendor = 32902u32;
__t96.device = 7971u32;
__t96.subvendor = 4294967295u32;
__t96.subdevice = 4294967295u32;
__t96.class = 0u32;
__t96.class_mask = 0u32;
__t96.driver_data = 5u64;
__t96
}, {
let mut __t97: pci_device_id = unsafe { core::mem::zeroed() };
__t97.vendor = 32902u32;
__t97.device = 7972u32;
__t97.subvendor = 4294967295u32;
__t97.subdevice = 4294967295u32;
__t97.class = 0u32;
__t97.class_mask = 0u32;
__t97.driver_data = 5u64;
__t97
}, {
let mut __t98: pci_device_id = unsafe { core::mem::zeroed() };
__t98.vendor = 32902u32;
__t98.device = 7973u32;
__t98.subvendor = 4294967295u32;
__t98.subdevice = 4294967295u32;
__t98.class = 0u32;
__t98.class_mask = 0u32;
__t98.driver_data = 5u64;
__t98
}, {
let mut __t99: pci_device_id = unsafe { core::mem::zeroed() };
__t99.vendor = 32902u32;
__t99.device = 7974u32;
__t99.subvendor = 4294967295u32;
__t99.subdevice = 4294967295u32;
__t99.class = 0u32;
__t99.class_mask = 0u32;
__t99.driver_data = 5u64;
__t99
}, {
let mut __t100: pci_device_id = unsafe { core::mem::zeroed() };
__t100.vendor = 32902u32;
__t100.device = 7975u32;
__t100.subvendor = 4294967295u32;
__t100.subdevice = 4294967295u32;
__t100.class = 0u32;
__t100.class_mask = 0u32;
__t100.driver_data = 5u64;
__t100
}, {
let mut __t101: pci_device_id = unsafe { core::mem::zeroed() };
__t101.vendor = 32902u32;
__t101.device = 7982u32;
__t101.subvendor = 4294967295u32;
__t101.subdevice = 4294967295u32;
__t101.class = 0u32;
__t101.class_mask = 0u32;
__t101.driver_data = 5u64;
__t101
}, {
let mut __t102: pci_device_id = unsafe { core::mem::zeroed() };
__t102.vendor = 32902u32;
__t102.device = 7983u32;
__t102.subvendor = 4294967295u32;
__t102.subdevice = 4294967295u32;
__t102.class = 0u32;
__t102.class_mask = 0u32;
__t102.driver_data = 5u64;
__t102
}, {
let mut __t103: pci_device_id = unsafe { core::mem::zeroed() };
__t103.vendor = 32902u32;
__t103.device = 7986u32;
__t103.subvendor = 4294967295u32;
__t103.subdevice = 4294967295u32;
__t103.class = 0u32;
__t103.class_mask = 0u32;
__t103.driver_data = 11u64;
__t103
}, {
let mut __t104: pci_device_id = unsafe { core::mem::zeroed() };
__t104.vendor = 32902u32;
__t104.device = 7987u32;
__t104.subvendor = 4294967295u32;
__t104.subdevice = 4294967295u32;
__t104.class = 0u32;
__t104.class_mask = 0u32;
__t104.driver_data = 11u64;
__t104
}, {
let mut __t105: pci_device_id = unsafe { core::mem::zeroed() };
__t105.vendor = 32902u32;
__t105.device = 7988u32;
__t105.subvendor = 4294967295u32;
__t105.subdevice = 4294967295u32;
__t105.class = 0u32;
__t105.class_mask = 0u32;
__t105.driver_data = 11u64;
__t105
}, {
let mut __t106: pci_device_id = unsafe { core::mem::zeroed() };
__t106.vendor = 32902u32;
__t106.device = 7989u32;
__t106.subvendor = 4294967295u32;
__t106.subdevice = 4294967295u32;
__t106.class = 0u32;
__t106.class_mask = 0u32;
__t106.driver_data = 11u64;
__t106
}, {
let mut __t107: pci_device_id = unsafe { core::mem::zeroed() };
__t107.vendor = 32902u32;
__t107.device = 7990u32;
__t107.subvendor = 4294967295u32;
__t107.subdevice = 4294967295u32;
__t107.class = 0u32;
__t107.class_mask = 0u32;
__t107.driver_data = 11u64;
__t107
}, {
let mut __t108: pci_device_id = unsafe { core::mem::zeroed() };
__t108.vendor = 32902u32;
__t108.device = 7991u32;
__t108.subvendor = 4294967295u32;
__t108.subdevice = 4294967295u32;
__t108.class = 0u32;
__t108.class_mask = 0u32;
__t108.driver_data = 11u64;
__t108
}, {
let mut __t109: pci_device_id = unsafe { core::mem::zeroed() };
__t109.vendor = 32902u32;
__t109.device = 7998u32;
__t109.subvendor = 4294967295u32;
__t109.subdevice = 4294967295u32;
__t109.class = 0u32;
__t109.class_mask = 0u32;
__t109.driver_data = 11u64;
__t109
}, {
let mut __t110: pci_device_id = unsafe { core::mem::zeroed() };
__t110.vendor = 32902u32;
__t110.device = 7999u32;
__t110.subvendor = 4294967295u32;
__t110.subdevice = 4294967295u32;
__t110.class = 0u32;
__t110.class_mask = 0u32;
__t110.driver_data = 11u64;
__t110
}, {
let mut __t111: pci_device_id = unsafe { core::mem::zeroed() };
__t111.vendor = 32902u32;
__t111.device = 10275u32;
__t111.subvendor = 4294967295u32;
__t111.subdevice = 4294967295u32;
__t111.class = 0u32;
__t111.class_mask = 0u32;
__t111.driver_data = 5u64;
__t111
}, {
let mut __t112: pci_device_id = unsafe { core::mem::zeroed() };
__t112.vendor = 32902u32;
__t112.device = 10278u32;
__t112.subvendor = 4294967295u32;
__t112.subdevice = 4294967295u32;
__t112.class = 0u32;
__t112.class_mask = 0u32;
__t112.driver_data = 5u64;
__t112
}, {
let mut __t113: pci_device_id = unsafe { core::mem::zeroed() };
__t113.vendor = 32902u32;
__t113.device = 10279u32;
__t113.subvendor = 4294967295u32;
__t113.subdevice = 4294967295u32;
__t113.class = 0u32;
__t113.class_mask = 0u32;
__t113.driver_data = 5u64;
__t113
}, {
let mut __t114: pci_device_id = unsafe { core::mem::zeroed() };
__t114.vendor = 32902u32;
__t114.device = 10287u32;
__t114.subvendor = 4294967295u32;
__t114.subdevice = 4294967295u32;
__t114.class = 0u32;
__t114.class_mask = 0u32;
__t114.driver_data = 5u64;
__t114
}, {
let mut __t115: pci_device_id = unsafe { core::mem::zeroed() };
__t115.vendor = 32902u32;
__t115.device = 17364u32;
__t115.subvendor = 4294967295u32;
__t115.subdevice = 4294967295u32;
__t115.class = 0u32;
__t115.class_mask = 0u32;
__t115.driver_data = 5u64;
__t115
}, {
let mut __t116: pci_device_id = unsafe { core::mem::zeroed() };
__t116.vendor = 32902u32;
__t116.device = 17365u32;
__t116.subvendor = 4294967295u32;
__t116.subdevice = 4294967295u32;
__t116.class = 0u32;
__t116.class_mask = 0u32;
__t116.driver_data = 5u64;
__t116
}, {
let mut __t117: pci_device_id = unsafe { core::mem::zeroed() };
__t117.vendor = 32902u32;
__t117.device = 17366u32;
__t117.subvendor = 4294967295u32;
__t117.subdevice = 4294967295u32;
__t117.class = 0u32;
__t117.class_mask = 0u32;
__t117.driver_data = 5u64;
__t117
}, {
let mut __t118: pci_device_id = unsafe { core::mem::zeroed() };
__t118.vendor = 32902u32;
__t118.device = 17367u32;
__t118.subvendor = 4294967295u32;
__t118.subdevice = 4294967295u32;
__t118.class = 0u32;
__t118.class_mask = 0u32;
__t118.driver_data = 5u64;
__t118
}, {
let mut __t119: pci_device_id = unsafe { core::mem::zeroed() };
__t119.vendor = 32902u32;
__t119.device = 36098u32;
__t119.subvendor = 4294967295u32;
__t119.subdevice = 4294967295u32;
__t119.class = 0u32;
__t119.class_mask = 0u32;
__t119.driver_data = 5u64;
__t119
}, {
let mut __t120: pci_device_id = unsafe { core::mem::zeroed() };
__t120.vendor = 32902u32;
__t120.device = 36100u32;
__t120.subvendor = 4294967295u32;
__t120.subdevice = 4294967295u32;
__t120.class = 0u32;
__t120.class_mask = 0u32;
__t120.driver_data = 5u64;
__t120
}, {
let mut __t121: pci_device_id = unsafe { core::mem::zeroed() };
__t121.vendor = 32902u32;
__t121.device = 36102u32;
__t121.subvendor = 4294967295u32;
__t121.subdevice = 4294967295u32;
__t121.class = 0u32;
__t121.class_mask = 0u32;
__t121.driver_data = 5u64;
__t121
}, {
let mut __t122: pci_device_id = unsafe { core::mem::zeroed() };
__t122.vendor = 32902u32;
__t122.device = 36110u32;
__t122.subvendor = 4294967295u32;
__t122.subdevice = 4294967295u32;
__t122.class = 0u32;
__t122.class_mask = 0u32;
__t122.driver_data = 5u64;
__t122
}, {
let mut __t123: pci_device_id = unsafe { core::mem::zeroed() };
__t123.vendor = 32902u32;
__t123.device = 36194u32;
__t123.subvendor = 4294967295u32;
__t123.subdevice = 4294967295u32;
__t123.class = 0u32;
__t123.class_mask = 0u32;
__t123.driver_data = 5u64;
__t123
}, {
let mut __t124: pci_device_id = unsafe { core::mem::zeroed() };
__t124.vendor = 32902u32;
__t124.device = 36196u32;
__t124.subvendor = 4294967295u32;
__t124.subdevice = 4294967295u32;
__t124.class = 0u32;
__t124.class_mask = 0u32;
__t124.driver_data = 5u64;
__t124
}, {
let mut __t125: pci_device_id = unsafe { core::mem::zeroed() };
__t125.vendor = 32902u32;
__t125.device = 36198u32;
__t125.subvendor = 4294967295u32;
__t125.subdevice = 4294967295u32;
__t125.class = 0u32;
__t125.class_mask = 0u32;
__t125.driver_data = 5u64;
__t125
}, {
let mut __t126: pci_device_id = unsafe { core::mem::zeroed() };
__t126.vendor = 32902u32;
__t126.device = 36206u32;
__t126.subvendor = 4294967295u32;
__t126.subdevice = 4294967295u32;
__t126.class = 0u32;
__t126.class_mask = 0u32;
__t126.driver_data = 5u64;
__t126
}, {
let mut __t127: pci_device_id = unsafe { core::mem::zeroed() };
__t127.vendor = 32902u32;
__t127.device = 9123u32;
__t127.subvendor = 4294967295u32;
__t127.subdevice = 4294967295u32;
__t127.class = 0u32;
__t127.class_mask = 0u32;
__t127.driver_data = 5u64;
__t127
}, {
let mut __t128: pci_device_id = unsafe { core::mem::zeroed() };
__t128.vendor = 32902u32;
__t128.device = 40067u32;
__t128.subvendor = 4294967295u32;
__t128.subdevice = 4294967295u32;
__t128.class = 0u32;
__t128.class_mask = 0u32;
__t128.driver_data = 5u64;
__t128
}, {
let mut __t129: pci_device_id = unsafe { core::mem::zeroed() };
__t129.vendor = 32902u32;
__t129.device = 40069u32;
__t129.subvendor = 4294967295u32;
__t129.subdevice = 4294967295u32;
__t129.class = 0u32;
__t129.class_mask = 0u32;
__t129.driver_data = 5u64;
__t129
}, {
let mut __t130: pci_device_id = unsafe { core::mem::zeroed() };
__t130.vendor = 32902u32;
__t130.device = 40071u32;
__t130.subvendor = 4294967295u32;
__t130.subdevice = 4294967295u32;
__t130.class = 0u32;
__t130.class_mask = 0u32;
__t130.driver_data = 5u64;
__t130
}, {
let mut __t131: pci_device_id = unsafe { core::mem::zeroed() };
__t131.vendor = 32902u32;
__t131.device = 40079u32;
__t131.subvendor = 4294967295u32;
__t131.subdevice = 4294967295u32;
__t131.class = 0u32;
__t131.class_mask = 0u32;
__t131.driver_data = 5u64;
__t131
}, {
let mut __t132: pci_device_id = unsafe { core::mem::zeroed() };
__t132.vendor = 32902u32;
__t132.device = 35970u32;
__t132.subvendor = 4294967295u32;
__t132.subdevice = 4294967295u32;
__t132.class = 0u32;
__t132.class_mask = 0u32;
__t132.driver_data = 5u64;
__t132
}, {
let mut __t133: pci_device_id = unsafe { core::mem::zeroed() };
__t133.vendor = 32902u32;
__t133.device = 35971u32;
__t133.subvendor = 4294967295u32;
__t133.subdevice = 4294967295u32;
__t133.class = 0u32;
__t133.class_mask = 0u32;
__t133.driver_data = 5u64;
__t133
}, {
let mut __t134: pci_device_id = unsafe { core::mem::zeroed() };
__t134.vendor = 32902u32;
__t134.device = 35972u32;
__t134.subvendor = 4294967295u32;
__t134.subdevice = 4294967295u32;
__t134.class = 0u32;
__t134.class_mask = 0u32;
__t134.driver_data = 5u64;
__t134
}, {
let mut __t135: pci_device_id = unsafe { core::mem::zeroed() };
__t135.vendor = 32902u32;
__t135.device = 35973u32;
__t135.subvendor = 4294967295u32;
__t135.subdevice = 4294967295u32;
__t135.class = 0u32;
__t135.class_mask = 0u32;
__t135.driver_data = 5u64;
__t135
}, {
let mut __t136: pci_device_id = unsafe { core::mem::zeroed() };
__t136.vendor = 32902u32;
__t136.device = 35974u32;
__t136.subvendor = 4294967295u32;
__t136.subdevice = 4294967295u32;
__t136.class = 0u32;
__t136.class_mask = 0u32;
__t136.driver_data = 5u64;
__t136
}, {
let mut __t137: pci_device_id = unsafe { core::mem::zeroed() };
__t137.vendor = 32902u32;
__t137.device = 35975u32;
__t137.subvendor = 4294967295u32;
__t137.subdevice = 4294967295u32;
__t137.class = 0u32;
__t137.class_mask = 0u32;
__t137.driver_data = 5u64;
__t137
}, {
let mut __t138: pci_device_id = unsafe { core::mem::zeroed() };
__t138.vendor = 32902u32;
__t138.device = 35982u32;
__t138.subvendor = 4294967295u32;
__t138.subdevice = 4294967295u32;
__t138.class = 0u32;
__t138.class_mask = 0u32;
__t138.driver_data = 5u64;
__t138
}, {
let mut __t139: pci_device_id = unsafe { core::mem::zeroed() };
__t139.vendor = 32902u32;
__t139.device = 35983u32;
__t139.subvendor = 4294967295u32;
__t139.subdevice = 4294967295u32;
__t139.class = 0u32;
__t139.class_mask = 0u32;
__t139.driver_data = 5u64;
__t139
}, {
let mut __t140: pci_device_id = unsafe { core::mem::zeroed() };
__t140.vendor = 32902u32;
__t140.device = 40195u32;
__t140.subvendor = 4294967295u32;
__t140.subdevice = 4294967295u32;
__t140.class = 0u32;
__t140.class_mask = 0u32;
__t140.driver_data = 5u64;
__t140
}, {
let mut __t141: pci_device_id = unsafe { core::mem::zeroed() };
__t141.vendor = 32902u32;
__t141.device = 40197u32;
__t141.subvendor = 4294967295u32;
__t141.subdevice = 4294967295u32;
__t141.class = 0u32;
__t141.class_mask = 0u32;
__t141.driver_data = 5u64;
__t141
}, {
let mut __t142: pci_device_id = unsafe { core::mem::zeroed() };
__t142.vendor = 32902u32;
__t142.device = 40199u32;
__t142.subvendor = 4294967295u32;
__t142.subdevice = 4294967295u32;
__t142.class = 0u32;
__t142.class_mask = 0u32;
__t142.driver_data = 5u64;
__t142
}, {
let mut __t143: pci_device_id = unsafe { core::mem::zeroed() };
__t143.vendor = 32902u32;
__t143.device = 41218u32;
__t143.subvendor = 4294967295u32;
__t143.subdevice = 4294967295u32;
__t143.class = 0u32;
__t143.class_mask = 0u32;
__t143.driver_data = 5u64;
__t143
}, {
let mut __t144: pci_device_id = unsafe { core::mem::zeroed() };
__t144.vendor = 32902u32;
__t144.device = 41219u32;
__t144.subvendor = 4294967295u32;
__t144.subdevice = 4294967295u32;
__t144.class = 0u32;
__t144.class_mask = 0u32;
__t144.driver_data = 5u64;
__t144
}, {
let mut __t145: pci_device_id = unsafe { core::mem::zeroed() };
__t145.vendor = 32902u32;
__t145.device = 41221u32;
__t145.subvendor = 4294967295u32;
__t145.subdevice = 4294967295u32;
__t145.class = 0u32;
__t145.class_mask = 0u32;
__t145.driver_data = 5u64;
__t145
}, {
let mut __t146: pci_device_id = unsafe { core::mem::zeroed() };
__t146.vendor = 32902u32;
__t146.device = 41222u32;
__t146.subvendor = 4294967295u32;
__t146.subdevice = 4294967295u32;
__t146.class = 0u32;
__t146.class_mask = 0u32;
__t146.driver_data = 5u64;
__t146
}, {
let mut __t147: pci_device_id = unsafe { core::mem::zeroed() };
__t147.vendor = 32902u32;
__t147.device = 41223u32;
__t147.subvendor = 4294967295u32;
__t147.subdevice = 4294967295u32;
__t147.class = 0u32;
__t147.class_mask = 0u32;
__t147.driver_data = 5u64;
__t147
}, {
let mut __t148: pci_device_id = unsafe { core::mem::zeroed() };
__t148.vendor = 32902u32;
__t148.device = 41231u32;
__t148.subvendor = 4294967295u32;
__t148.subdevice = 4294967295u32;
__t148.class = 0u32;
__t148.class_mask = 0u32;
__t148.driver_data = 5u64;
__t148
}, {
let mut __t149: pci_device_id = unsafe { core::mem::zeroed() };
__t149.vendor = 32902u32;
__t149.device = 41346u32;
__t149.subvendor = 4294967295u32;
__t149.subdevice = 4294967295u32;
__t149.class = 0u32;
__t149.class_mask = 0u32;
__t149.driver_data = 5u64;
__t149
}, {
let mut __t150: pci_device_id = unsafe { core::mem::zeroed() };
__t150.vendor = 32902u32;
__t150.device = 41350u32;
__t150.subvendor = 4294967295u32;
__t150.subdevice = 4294967295u32;
__t150.class = 0u32;
__t150.class_mask = 0u32;
__t150.driver_data = 5u64;
__t150
}, {
let mut __t151: pci_device_id = unsafe { core::mem::zeroed() };
__t151.vendor = 32902u32;
__t151.device = 41426u32;
__t151.subvendor = 4294967295u32;
__t151.subdevice = 4294967295u32;
__t151.class = 0u32;
__t151.class_mask = 0u32;
__t151.driver_data = 5u64;
__t151
}, {
let mut __t152: pci_device_id = unsafe { core::mem::zeroed() };
__t152.vendor = 32902u32;
__t152.device = 41430u32;
__t152.subvendor = 4294967295u32;
__t152.subdevice = 4294967295u32;
__t152.class = 0u32;
__t152.class_mask = 0u32;
__t152.driver_data = 5u64;
__t152
}, {
let mut __t153: pci_device_id = unsafe { core::mem::zeroed() };
__t153.vendor = 32902u32;
__t153.device = 41474u32;
__t153.subvendor = 4294967295u32;
__t153.subdevice = 4294967295u32;
__t153.class = 0u32;
__t153.class_mask = 0u32;
__t153.driver_data = 5u64;
__t153
}, {
let mut __t154: pci_device_id = unsafe { core::mem::zeroed() };
__t154.vendor = 32902u32;
__t154.device = 41478u32;
__t154.subvendor = 4294967295u32;
__t154.subdevice = 4294967295u32;
__t154.class = 0u32;
__t154.class_mask = 0u32;
__t154.driver_data = 5u64;
__t154
}, {
let mut __t155: pci_device_id = unsafe { core::mem::zeroed() };
__t155.vendor = 32902u32;
__t155.device = 41554u32;
__t155.subvendor = 4294967295u32;
__t155.subdevice = 4294967295u32;
__t155.class = 0u32;
__t155.class_mask = 0u32;
__t155.driver_data = 5u64;
__t155
}, {
let mut __t156: pci_device_id = unsafe { core::mem::zeroed() };
__t156.vendor = 32902u32;
__t156.device = 41558u32;
__t156.subvendor = 4294967295u32;
__t156.subdevice = 4294967295u32;
__t156.class = 0u32;
__t156.class_mask = 0u32;
__t156.driver_data = 5u64;
__t156
}, {
let mut __t157: pci_device_id = unsafe { core::mem::zeroed() };
__t157.vendor = 32902u32;
__t157.device = 41814u32;
__t157.subvendor = 4294967295u32;
__t157.subdevice = 4294967295u32;
__t157.class = 0u32;
__t157.class_mask = 0u32;
__t157.driver_data = 5u64;
__t157
}, {
let mut __t158: pci_device_id = unsafe { core::mem::zeroed() };
__t158.vendor = 32902u32;
__t158.device = 1751u32;
__t158.subvendor = 4294967295u32;
__t158.subdevice = 4294967295u32;
__t158.class = 0u32;
__t158.class_mask = 0u32;
__t158.driver_data = 5u64;
__t158
}, {
let mut __t159: pci_device_id = unsafe { core::mem::zeroed() };
__t159.vendor = 32902u32;
__t159.device = 41862u32;
__t159.subvendor = 4294967295u32;
__t159.subdevice = 4294967295u32;
__t159.class = 0u32;
__t159.class_mask = 0u32;
__t159.driver_data = 5u64;
__t159
}, {
let mut __t160: pci_device_id = unsafe { core::mem::zeroed() };
__t160.vendor = 32902u32;
__t160.device = 3874u32;
__t160.subvendor = 4294967295u32;
__t160.subdevice = 4294967295u32;
__t160.class = 0u32;
__t160.class_mask = 0u32;
__t160.driver_data = 5u64;
__t160
}, {
let mut __t161: pci_device_id = unsafe { core::mem::zeroed() };
__t161.vendor = 32902u32;
__t161.device = 3875u32;
__t161.subvendor = 4294967295u32;
__t161.subdevice = 4294967295u32;
__t161.class = 0u32;
__t161.class_mask = 0u32;
__t161.driver_data = 6u64;
__t161
}, {
let mut __t162: pci_device_id = unsafe { core::mem::zeroed() };
__t162.vendor = 32902u32;
__t162.device = 8867u32;
__t162.subvendor = 4294967295u32;
__t162.subdevice = 4294967295u32;
__t162.class = 0u32;
__t162.class_mask = 0u32;
__t162.driver_data = 5u64;
__t162
}, {
let mut __t163: pci_device_id = unsafe { core::mem::zeroed() };
__t163.vendor = 32902u32;
__t163.device = 23267u32;
__t163.subvendor = 4294967295u32;
__t163.subdevice = 4294967295u32;
__t163.class = 0u32;
__t163.class_mask = 0u32;
__t163.driver_data = 5u64;
__t163
}, {
let mut __t164: pci_device_id = unsafe { core::mem::zeroed() };
__t164.vendor = 32902u32;
__t164.device = 13523u32;
__t164.subvendor = 4294967295u32;
__t164.subdevice = 4294967295u32;
__t164.class = 0u32;
__t164.class_mask = 0u32;
__t164.driver_data = 5u64;
__t164
}, {
let mut __t165: pci_device_id = unsafe { core::mem::zeroed() };
__t165.vendor = 32902u32;
__t165.device = 723u32;
__t165.subvendor = 4294967295u32;
__t165.subdevice = 4294967295u32;
__t165.class = 0u32;
__t165.class_mask = 0u32;
__t165.driver_data = 5u64;
__t165
}, {
let mut __t166: pci_device_id = unsafe { core::mem::zeroed() };
__t166.vendor = 32902u32;
__t166.device = 727u32;
__t166.subvendor = 4294967295u32;
__t166.subdevice = 4294967295u32;
__t166.class = 0u32;
__t166.class_mask = 0u32;
__t166.driver_data = 5u64;
__t166
}, {
let mut __t167: pci_device_id = unsafe { core::mem::zeroed() };
__t167.vendor = 32902u32;
__t167.device = 19299u32;
__t167.subvendor = 4294967295u32;
__t167.subdevice = 4294967295u32;
__t167.class = 0u32;
__t167.class_mask = 0u32;
__t167.driver_data = 5u64;
__t167
}, {
let mut __t168: pci_device_id = unsafe { core::mem::zeroed() };
__t168.vendor = 6523u32;
__t168.device = 1410u32;
__t168.subvendor = 4294967295u32;
__t168.subdevice = 4294967295u32;
__t168.class = 0u32;
__t168.class_mask = 0u32;
__t168.driver_data = 12u64;
__t168
}, {
let mut __t169: pci_device_id = unsafe { core::mem::zeroed() };
__t169.vendor = 6523u32;
__t169.device = 1413u32;
__t169.subvendor = 4294967295u32;
__t169.subdevice = 4294967295u32;
__t169.class = 0u32;
__t169.class_mask = 0u32;
__t169.driver_data = 12u64;
__t169
}, {
let mut __t170: pci_device_id = unsafe { core::mem::zeroed() };
__t170.vendor = 6523u32;
__t170.device = 4294967295u32;
__t170.subvendor = 4294967295u32;
__t170.subdevice = 4294967295u32;
__t170.class = 67073u32;
__t170.class_mask = 16777215u32;
__t170.driver_data = 2u64;
__t170
}, {
let mut __t171: pci_device_id = unsafe { core::mem::zeroed() };
__t171.vendor = 6523u32;
__t171.device = 9058u32;
__t171.subvendor = 4294967295u32;
__t171.subdevice = 4294967295u32;
__t171.class = 0u32;
__t171.class_mask = 0u32;
__t171.driver_data = 2u64;
__t171
}, {
let mut __t172: pci_device_id = unsafe { core::mem::zeroed() };
__t172.vendor = 6523u32;
__t172.device = 9071u32;
__t172.subvendor = 4294967295u32;
__t172.subdevice = 4294967295u32;
__t172.class = 0u32;
__t172.class_mask = 0u32;
__t172.driver_data = 2u64;
__t172
}, {
let mut __t173: pci_device_id = unsafe { core::mem::zeroed() };
__t173.vendor = 4098u32;
__t173.device = 17280u32;
__t173.subvendor = 4294967295u32;
__t173.subdevice = 4294967295u32;
__t173.class = 0u32;
__t173.class_mask = 0u32;
__t173.driver_data = 17u64;
__t173
}, {
let mut __t174: pci_device_id = unsafe { core::mem::zeroed() };
__t174.vendor = 4098u32;
__t174.device = 17296u32;
__t174.subvendor = 4294967295u32;
__t174.subdevice = 4294967295u32;
__t174.class = 0u32;
__t174.class_mask = 0u32;
__t174.driver_data = 18u64;
__t174
}, {
let mut __t175: pci_device_id = unsafe { core::mem::zeroed() };
__t175.vendor = 4098u32;
__t175.device = 17297u32;
__t175.subvendor = 4294967295u32;
__t175.subdevice = 4294967295u32;
__t175.class = 0u32;
__t175.class_mask = 0u32;
__t175.driver_data = 18u64;
__t175
}, {
let mut __t176: pci_device_id = unsafe { core::mem::zeroed() };
__t176.vendor = 4098u32;
__t176.device = 17298u32;
__t176.subvendor = 4294967295u32;
__t176.subdevice = 4294967295u32;
__t176.class = 0u32;
__t176.class_mask = 0u32;
__t176.driver_data = 18u64;
__t176
}, {
let mut __t177: pci_device_id = unsafe { core::mem::zeroed() };
__t177.vendor = 4098u32;
__t177.device = 17299u32;
__t177.subvendor = 4294967295u32;
__t177.subdevice = 4294967295u32;
__t177.class = 0u32;
__t177.class_mask = 0u32;
__t177.driver_data = 18u64;
__t177
}, {
let mut __t178: pci_device_id = unsafe { core::mem::zeroed() };
__t178.vendor = 4098u32;
__t178.device = 17300u32;
__t178.subvendor = 4294967295u32;
__t178.subdevice = 4294967295u32;
__t178.class = 0u32;
__t178.class_mask = 0u32;
__t178.driver_data = 18u64;
__t178
}, {
let mut __t179: pci_device_id = unsafe { core::mem::zeroed() };
__t179.vendor = 4098u32;
__t179.device = 17301u32;
__t179.subvendor = 4294967295u32;
__t179.subdevice = 4294967295u32;
__t179.class = 0u32;
__t179.class_mask = 0u32;
__t179.driver_data = 18u64;
__t179
}, {
let mut __t180: pci_device_id = unsafe { core::mem::zeroed() };
__t180.vendor = 7222u32;
__t180.device = 49u32;
__t180.subvendor = 4294967295u32;
__t180.subdevice = 4294967295u32;
__t180.class = 67073u32;
__t180.class_mask = 16777215u32;
__t180.driver_data = 10u64;
__t180
}, {
let mut __t181: pci_device_id = unsafe { core::mem::zeroed() };
__t181.vendor = 4130u32;
__t181.device = 30720u32;
__t181.subvendor = 4294967295u32;
__t181.subdevice = 4294967295u32;
__t181.class = 0u32;
__t181.class_mask = 0u32;
__t181.driver_data = 0u64;
__t181
}, {
let mut __t182: pci_device_id = unsafe { core::mem::zeroed() };
__t182.vendor = 4130u32;
__t182.device = 30721u32;
__t182.subvendor = 4294967295u32;
__t182.subdevice = 4294967295u32;
__t182.class = 0u32;
__t182.class_mask = 0u32;
__t182.driver_data = 3u64;
__t182
}, {
let mut __t183: pci_device_id = unsafe { core::mem::zeroed() };
__t183.vendor = 4130u32;
__t183.device = 30976u32;
__t183.subvendor = 4294967295u32;
__t183.subdevice = 4294967295u32;
__t183.class = 0u32;
__t183.class_mask = 0u32;
__t183.driver_data = 0u64;
__t183
}, {
let mut __t184: pci_device_id = unsafe { core::mem::zeroed() };
__t184.vendor = 4130u32;
__t184.device = 30977u32;
__t184.subvendor = 4294967295u32;
__t184.subdevice = 4294967295u32;
__t184.class = 0u32;
__t184.class_mask = 0u32;
__t184.driver_data = 0u64;
__t184
}, {
let mut __t185: pci_device_id = unsafe { core::mem::zeroed() };
__t185.vendor = 4130u32;
__t185.device = 4294967295u32;
__t185.subvendor = 4294967295u32;
__t185.subdevice = 4294967295u32;
__t185.class = 66560u32;
__t185.class_mask = 16777215u32;
__t185.driver_data = 0u64;
__t185
}, {
let mut __t186: pci_device_id = unsafe { core::mem::zeroed() };
__t186.vendor = 32902u32;
__t186.device = 4294967295u32;
__t186.subvendor = 4136u32;
__t186.subdevice = 4294967295u32;
__t186.class = 66560u32;
__t186.class_mask = 16777215u32;
__t186.driver_data = 5u64;
__t186
}, {
let mut __t187: pci_device_id = unsafe { core::mem::zeroed() };
__t187.vendor = 4358u32;
__t187.device = 13129u32;
__t187.subvendor = 4294967295u32;
__t187.subdevice = 4294967295u32;
__t187.class = 0u32;
__t187.class_mask = 0u32;
__t187.driver_data = 19u64;
__t187
}, {
let mut __t188: pci_device_id = unsafe { core::mem::zeroed() };
__t188.vendor = 4358u32;
__t188.device = 25223u32;
__t188.subvendor = 4294967295u32;
__t188.subdevice = 4294967295u32;
__t188.class = 0u32;
__t188.class_mask = 0u32;
__t188.driver_data = 19u64;
__t188
}, {
let mut __t189: pci_device_id = unsafe { core::mem::zeroed() };
__t189.vendor = 4318u32;
__t189.device = 1100u32;
__t189.subvendor = 4294967295u32;
__t189.subdevice = 4294967295u32;
__t189.class = 0u32;
__t189.class_mask = 0u32;
__t189.driver_data = 13u64;
__t189
}, {
let mut __t190: pci_device_id = unsafe { core::mem::zeroed() };
__t190.vendor = 4318u32;
__t190.device = 1101u32;
__t190.subvendor = 4294967295u32;
__t190.subdevice = 4294967295u32;
__t190.class = 0u32;
__t190.class_mask = 0u32;
__t190.driver_data = 13u64;
__t190
}, {
let mut __t191: pci_device_id = unsafe { core::mem::zeroed() };
__t191.vendor = 4318u32;
__t191.device = 1102u32;
__t191.subvendor = 4294967295u32;
__t191.subdevice = 4294967295u32;
__t191.class = 0u32;
__t191.class_mask = 0u32;
__t191.driver_data = 13u64;
__t191
}, {
let mut __t192: pci_device_id = unsafe { core::mem::zeroed() };
__t192.vendor = 4318u32;
__t192.device = 1103u32;
__t192.subvendor = 4294967295u32;
__t192.subdevice = 4294967295u32;
__t192.class = 0u32;
__t192.class_mask = 0u32;
__t192.driver_data = 13u64;
__t192
}, {
let mut __t193: pci_device_id = unsafe { core::mem::zeroed() };
__t193.vendor = 4318u32;
__t193.device = 1116u32;
__t193.subvendor = 4294967295u32;
__t193.subdevice = 4294967295u32;
__t193.class = 0u32;
__t193.class_mask = 0u32;
__t193.driver_data = 13u64;
__t193
}, {
let mut __t194: pci_device_id = unsafe { core::mem::zeroed() };
__t194.vendor = 4318u32;
__t194.device = 1117u32;
__t194.subvendor = 4294967295u32;
__t194.subdevice = 4294967295u32;
__t194.class = 0u32;
__t194.class_mask = 0u32;
__t194.driver_data = 13u64;
__t194
}, {
let mut __t195: pci_device_id = unsafe { core::mem::zeroed() };
__t195.vendor = 4318u32;
__t195.device = 1118u32;
__t195.subvendor = 4294967295u32;
__t195.subdevice = 4294967295u32;
__t195.class = 0u32;
__t195.class_mask = 0u32;
__t195.driver_data = 13u64;
__t195
}, {
let mut __t196: pci_device_id = unsafe { core::mem::zeroed() };
__t196.vendor = 4318u32;
__t196.device = 1119u32;
__t196.subvendor = 4294967295u32;
__t196.subdevice = 4294967295u32;
__t196.class = 0u32;
__t196.class_mask = 0u32;
__t196.driver_data = 13u64;
__t196
}, {
let mut __t197: pci_device_id = unsafe { core::mem::zeroed() };
__t197.vendor = 4318u32;
__t197.device = 1360u32;
__t197.subvendor = 4294967295u32;
__t197.subdevice = 4294967295u32;
__t197.class = 0u32;
__t197.class_mask = 0u32;
__t197.driver_data = 13u64;
__t197
}, {
let mut __t198: pci_device_id = unsafe { core::mem::zeroed() };
__t198.vendor = 4318u32;
__t198.device = 1361u32;
__t198.subvendor = 4294967295u32;
__t198.subdevice = 4294967295u32;
__t198.class = 0u32;
__t198.class_mask = 0u32;
__t198.driver_data = 13u64;
__t198
}, {
let mut __t199: pci_device_id = unsafe { core::mem::zeroed() };
__t199.vendor = 4318u32;
__t199.device = 1362u32;
__t199.subvendor = 4294967295u32;
__t199.subdevice = 4294967295u32;
__t199.class = 0u32;
__t199.class_mask = 0u32;
__t199.driver_data = 13u64;
__t199
}, {
let mut __t200: pci_device_id = unsafe { core::mem::zeroed() };
__t200.vendor = 4318u32;
__t200.device = 1363u32;
__t200.subvendor = 4294967295u32;
__t200.subdevice = 4294967295u32;
__t200.class = 0u32;
__t200.class_mask = 0u32;
__t200.driver_data = 13u64;
__t200
}, {
let mut __t201: pci_device_id = unsafe { core::mem::zeroed() };
__t201.vendor = 4318u32;
__t201.device = 1364u32;
__t201.subvendor = 4294967295u32;
__t201.subdevice = 4294967295u32;
__t201.class = 0u32;
__t201.class_mask = 0u32;
__t201.driver_data = 13u64;
__t201
}, {
let mut __t202: pci_device_id = unsafe { core::mem::zeroed() };
__t202.vendor = 4318u32;
__t202.device = 1365u32;
__t202.subvendor = 4294967295u32;
__t202.subdevice = 4294967295u32;
__t202.class = 0u32;
__t202.class_mask = 0u32;
__t202.driver_data = 13u64;
__t202
}, {
let mut __t203: pci_device_id = unsafe { core::mem::zeroed() };
__t203.vendor = 4318u32;
__t203.device = 1366u32;
__t203.subvendor = 4294967295u32;
__t203.subdevice = 4294967295u32;
__t203.class = 0u32;
__t203.class_mask = 0u32;
__t203.driver_data = 13u64;
__t203
}, {
let mut __t204: pci_device_id = unsafe { core::mem::zeroed() };
__t204.vendor = 4318u32;
__t204.device = 1367u32;
__t204.subvendor = 4294967295u32;
__t204.subdevice = 4294967295u32;
__t204.class = 0u32;
__t204.class_mask = 0u32;
__t204.driver_data = 13u64;
__t204
}, {
let mut __t205: pci_device_id = unsafe { core::mem::zeroed() };
__t205.vendor = 4318u32;
__t205.device = 1368u32;
__t205.subvendor = 4294967295u32;
__t205.subdevice = 4294967295u32;
__t205.class = 0u32;
__t205.class_mask = 0u32;
__t205.driver_data = 13u64;
__t205
}, {
let mut __t206: pci_device_id = unsafe { core::mem::zeroed() };
__t206.vendor = 4318u32;
__t206.device = 1369u32;
__t206.subvendor = 4294967295u32;
__t206.subdevice = 4294967295u32;
__t206.class = 0u32;
__t206.class_mask = 0u32;
__t206.driver_data = 13u64;
__t206
}, {
let mut __t207: pci_device_id = unsafe { core::mem::zeroed() };
__t207.vendor = 4318u32;
__t207.device = 1370u32;
__t207.subvendor = 4294967295u32;
__t207.subdevice = 4294967295u32;
__t207.class = 0u32;
__t207.class_mask = 0u32;
__t207.driver_data = 13u64;
__t207
}, {
let mut __t208: pci_device_id = unsafe { core::mem::zeroed() };
__t208.vendor = 4318u32;
__t208.device = 1371u32;
__t208.subvendor = 4294967295u32;
__t208.subdevice = 4294967295u32;
__t208.class = 0u32;
__t208.class_mask = 0u32;
__t208.driver_data = 13u64;
__t208
}, {
let mut __t209: pci_device_id = unsafe { core::mem::zeroed() };
__t209.vendor = 4318u32;
__t209.device = 1408u32;
__t209.subvendor = 4294967295u32;
__t209.subdevice = 4294967295u32;
__t209.class = 0u32;
__t209.class_mask = 0u32;
__t209.driver_data = 13u64;
__t209
}, {
let mut __t210: pci_device_id = unsafe { core::mem::zeroed() };
__t210.vendor = 4318u32;
__t210.device = 1409u32;
__t210.subvendor = 4294967295u32;
__t210.subdevice = 4294967295u32;
__t210.class = 0u32;
__t210.class_mask = 0u32;
__t210.driver_data = 13u64;
__t210
}, {
let mut __t211: pci_device_id = unsafe { core::mem::zeroed() };
__t211.vendor = 4318u32;
__t211.device = 1410u32;
__t211.subvendor = 4294967295u32;
__t211.subdevice = 4294967295u32;
__t211.class = 0u32;
__t211.class_mask = 0u32;
__t211.driver_data = 13u64;
__t211
}, {
let mut __t212: pci_device_id = unsafe { core::mem::zeroed() };
__t212.vendor = 4318u32;
__t212.device = 1411u32;
__t212.subvendor = 4294967295u32;
__t212.subdevice = 4294967295u32;
__t212.class = 0u32;
__t212.class_mask = 0u32;
__t212.driver_data = 13u64;
__t212
}, {
let mut __t213: pci_device_id = unsafe { core::mem::zeroed() };
__t213.vendor = 4318u32;
__t213.device = 1412u32;
__t213.subvendor = 4294967295u32;
__t213.subdevice = 4294967295u32;
__t213.class = 0u32;
__t213.class_mask = 0u32;
__t213.driver_data = 13u64;
__t213
}, {
let mut __t214: pci_device_id = unsafe { core::mem::zeroed() };
__t214.vendor = 4318u32;
__t214.device = 1413u32;
__t214.subvendor = 4294967295u32;
__t214.subdevice = 4294967295u32;
__t214.class = 0u32;
__t214.class_mask = 0u32;
__t214.driver_data = 13u64;
__t214
}, {
let mut __t215: pci_device_id = unsafe { core::mem::zeroed() };
__t215.vendor = 4318u32;
__t215.device = 1414u32;
__t215.subvendor = 4294967295u32;
__t215.subdevice = 4294967295u32;
__t215.class = 0u32;
__t215.class_mask = 0u32;
__t215.driver_data = 13u64;
__t215
}, {
let mut __t216: pci_device_id = unsafe { core::mem::zeroed() };
__t216.vendor = 4318u32;
__t216.device = 1415u32;
__t216.subvendor = 4294967295u32;
__t216.subdevice = 4294967295u32;
__t216.class = 0u32;
__t216.class_mask = 0u32;
__t216.driver_data = 13u64;
__t216
}, {
let mut __t217: pci_device_id = unsafe { core::mem::zeroed() };
__t217.vendor = 4318u32;
__t217.device = 1416u32;
__t217.subvendor = 4294967295u32;
__t217.subdevice = 4294967295u32;
__t217.class = 0u32;
__t217.class_mask = 0u32;
__t217.driver_data = 13u64;
__t217
}, {
let mut __t218: pci_device_id = unsafe { core::mem::zeroed() };
__t218.vendor = 4318u32;
__t218.device = 1417u32;
__t218.subvendor = 4294967295u32;
__t218.subdevice = 4294967295u32;
__t218.class = 0u32;
__t218.class_mask = 0u32;
__t218.driver_data = 13u64;
__t218
}, {
let mut __t219: pci_device_id = unsafe { core::mem::zeroed() };
__t219.vendor = 4318u32;
__t219.device = 1418u32;
__t219.subvendor = 4294967295u32;
__t219.subdevice = 4294967295u32;
__t219.class = 0u32;
__t219.class_mask = 0u32;
__t219.driver_data = 13u64;
__t219
}, {
let mut __t220: pci_device_id = unsafe { core::mem::zeroed() };
__t220.vendor = 4318u32;
__t220.device = 1419u32;
__t220.subvendor = 4294967295u32;
__t220.subdevice = 4294967295u32;
__t220.class = 0u32;
__t220.class_mask = 0u32;
__t220.driver_data = 13u64;
__t220
}, {
let mut __t221: pci_device_id = unsafe { core::mem::zeroed() };
__t221.vendor = 4318u32;
__t221.device = 1420u32;
__t221.subvendor = 4294967295u32;
__t221.subdevice = 4294967295u32;
__t221.class = 0u32;
__t221.class_mask = 0u32;
__t221.driver_data = 13u64;
__t221
}, {
let mut __t222: pci_device_id = unsafe { core::mem::zeroed() };
__t222.vendor = 4318u32;
__t222.device = 1421u32;
__t222.subvendor = 4294967295u32;
__t222.subdevice = 4294967295u32;
__t222.class = 0u32;
__t222.class_mask = 0u32;
__t222.driver_data = 13u64;
__t222
}, {
let mut __t223: pci_device_id = unsafe { core::mem::zeroed() };
__t223.vendor = 4318u32;
__t223.device = 1422u32;
__t223.subvendor = 4294967295u32;
__t223.subdevice = 4294967295u32;
__t223.class = 0u32;
__t223.class_mask = 0u32;
__t223.driver_data = 13u64;
__t223
}, {
let mut __t224: pci_device_id = unsafe { core::mem::zeroed() };
__t224.vendor = 4318u32;
__t224.device = 1423u32;
__t224.subvendor = 4294967295u32;
__t224.subdevice = 4294967295u32;
__t224.class = 0u32;
__t224.class_mask = 0u32;
__t224.driver_data = 13u64;
__t224
}, {
let mut __t225: pci_device_id = unsafe { core::mem::zeroed() };
__t225.vendor = 4318u32;
__t225.device = 2032u32;
__t225.subvendor = 4294967295u32;
__t225.subdevice = 4294967295u32;
__t225.class = 0u32;
__t225.class_mask = 0u32;
__t225.driver_data = 13u64;
__t225
}, {
let mut __t226: pci_device_id = unsafe { core::mem::zeroed() };
__t226.vendor = 4318u32;
__t226.device = 2033u32;
__t226.subvendor = 4294967295u32;
__t226.subdevice = 4294967295u32;
__t226.class = 0u32;
__t226.class_mask = 0u32;
__t226.driver_data = 13u64;
__t226
}, {
let mut __t227: pci_device_id = unsafe { core::mem::zeroed() };
__t227.vendor = 4318u32;
__t227.device = 2034u32;
__t227.subvendor = 4294967295u32;
__t227.subdevice = 4294967295u32;
__t227.class = 0u32;
__t227.class_mask = 0u32;
__t227.driver_data = 13u64;
__t227
}, {
let mut __t228: pci_device_id = unsafe { core::mem::zeroed() };
__t228.vendor = 4318u32;
__t228.device = 2035u32;
__t228.subvendor = 4294967295u32;
__t228.subdevice = 4294967295u32;
__t228.class = 0u32;
__t228.class_mask = 0u32;
__t228.driver_data = 13u64;
__t228
}, {
let mut __t229: pci_device_id = unsafe { core::mem::zeroed() };
__t229.vendor = 4318u32;
__t229.device = 2036u32;
__t229.subvendor = 4294967295u32;
__t229.subdevice = 4294967295u32;
__t229.class = 0u32;
__t229.class_mask = 0u32;
__t229.driver_data = 13u64;
__t229
}, {
let mut __t230: pci_device_id = unsafe { core::mem::zeroed() };
__t230.vendor = 4318u32;
__t230.device = 2037u32;
__t230.subvendor = 4294967295u32;
__t230.subdevice = 4294967295u32;
__t230.class = 0u32;
__t230.class_mask = 0u32;
__t230.driver_data = 13u64;
__t230
}, {
let mut __t231: pci_device_id = unsafe { core::mem::zeroed() };
__t231.vendor = 4318u32;
__t231.device = 2038u32;
__t231.subvendor = 4294967295u32;
__t231.subdevice = 4294967295u32;
__t231.class = 0u32;
__t231.class_mask = 0u32;
__t231.driver_data = 13u64;
__t231
}, {
let mut __t232: pci_device_id = unsafe { core::mem::zeroed() };
__t232.vendor = 4318u32;
__t232.device = 2039u32;
__t232.subvendor = 4294967295u32;
__t232.subdevice = 4294967295u32;
__t232.class = 0u32;
__t232.class_mask = 0u32;
__t232.driver_data = 13u64;
__t232
}, {
let mut __t233: pci_device_id = unsafe { core::mem::zeroed() };
__t233.vendor = 4318u32;
__t233.device = 2040u32;
__t233.subvendor = 4294967295u32;
__t233.subdevice = 4294967295u32;
__t233.class = 0u32;
__t233.class_mask = 0u32;
__t233.driver_data = 13u64;
__t233
}, {
let mut __t234: pci_device_id = unsafe { core::mem::zeroed() };
__t234.vendor = 4318u32;
__t234.device = 2041u32;
__t234.subvendor = 4294967295u32;
__t234.subdevice = 4294967295u32;
__t234.class = 0u32;
__t234.class_mask = 0u32;
__t234.driver_data = 13u64;
__t234
}, {
let mut __t235: pci_device_id = unsafe { core::mem::zeroed() };
__t235.vendor = 4318u32;
__t235.device = 2042u32;
__t235.subvendor = 4294967295u32;
__t235.subdevice = 4294967295u32;
__t235.class = 0u32;
__t235.class_mask = 0u32;
__t235.driver_data = 13u64;
__t235
}, {
let mut __t236: pci_device_id = unsafe { core::mem::zeroed() };
__t236.vendor = 4318u32;
__t236.device = 2043u32;
__t236.subvendor = 4294967295u32;
__t236.subdevice = 4294967295u32;
__t236.class = 0u32;
__t236.class_mask = 0u32;
__t236.driver_data = 13u64;
__t236
}, {
let mut __t237: pci_device_id = unsafe { core::mem::zeroed() };
__t237.vendor = 4318u32;
__t237.device = 2768u32;
__t237.subvendor = 4294967295u32;
__t237.subdevice = 4294967295u32;
__t237.class = 0u32;
__t237.class_mask = 0u32;
__t237.driver_data = 14u64;
__t237
}, {
let mut __t238: pci_device_id = unsafe { core::mem::zeroed() };
__t238.vendor = 4318u32;
__t238.device = 2769u32;
__t238.subvendor = 4294967295u32;
__t238.subdevice = 4294967295u32;
__t238.class = 0u32;
__t238.class_mask = 0u32;
__t238.driver_data = 14u64;
__t238
}, {
let mut __t239: pci_device_id = unsafe { core::mem::zeroed() };
__t239.vendor = 4318u32;
__t239.device = 2770u32;
__t239.subvendor = 4294967295u32;
__t239.subdevice = 4294967295u32;
__t239.class = 0u32;
__t239.class_mask = 0u32;
__t239.driver_data = 14u64;
__t239
}, {
let mut __t240: pci_device_id = unsafe { core::mem::zeroed() };
__t240.vendor = 4318u32;
__t240.device = 2771u32;
__t240.subvendor = 4294967295u32;
__t240.subdevice = 4294967295u32;
__t240.class = 0u32;
__t240.class_mask = 0u32;
__t240.driver_data = 14u64;
__t240
}, {
let mut __t241: pci_device_id = unsafe { core::mem::zeroed() };
__t241.vendor = 4318u32;
__t241.device = 2772u32;
__t241.subvendor = 4294967295u32;
__t241.subdevice = 4294967295u32;
__t241.class = 0u32;
__t241.class_mask = 0u32;
__t241.driver_data = 14u64;
__t241
}, {
let mut __t242: pci_device_id = unsafe { core::mem::zeroed() };
__t242.vendor = 4318u32;
__t242.device = 2773u32;
__t242.subvendor = 4294967295u32;
__t242.subdevice = 4294967295u32;
__t242.class = 0u32;
__t242.class_mask = 0u32;
__t242.driver_data = 14u64;
__t242
}, {
let mut __t243: pci_device_id = unsafe { core::mem::zeroed() };
__t243.vendor = 4318u32;
__t243.device = 2774u32;
__t243.subvendor = 4294967295u32;
__t243.subdevice = 4294967295u32;
__t243.class = 0u32;
__t243.class_mask = 0u32;
__t243.driver_data = 14u64;
__t243
}, {
let mut __t244: pci_device_id = unsafe { core::mem::zeroed() };
__t244.vendor = 4318u32;
__t244.device = 2775u32;
__t244.subvendor = 4294967295u32;
__t244.subdevice = 4294967295u32;
__t244.class = 0u32;
__t244.class_mask = 0u32;
__t244.driver_data = 14u64;
__t244
}, {
let mut __t245: pci_device_id = unsafe { core::mem::zeroed() };
__t245.vendor = 4318u32;
__t245.device = 2776u32;
__t245.subvendor = 4294967295u32;
__t245.subdevice = 4294967295u32;
__t245.class = 0u32;
__t245.class_mask = 0u32;
__t245.driver_data = 14u64;
__t245
}, {
let mut __t246: pci_device_id = unsafe { core::mem::zeroed() };
__t246.vendor = 4318u32;
__t246.device = 2777u32;
__t246.subvendor = 4294967295u32;
__t246.subdevice = 4294967295u32;
__t246.class = 0u32;
__t246.class_mask = 0u32;
__t246.driver_data = 14u64;
__t246
}, {
let mut __t247: pci_device_id = unsafe { core::mem::zeroed() };
__t247.vendor = 4318u32;
__t247.device = 2778u32;
__t247.subvendor = 4294967295u32;
__t247.subdevice = 4294967295u32;
__t247.class = 0u32;
__t247.class_mask = 0u32;
__t247.driver_data = 14u64;
__t247
}, {
let mut __t248: pci_device_id = unsafe { core::mem::zeroed() };
__t248.vendor = 4318u32;
__t248.device = 2779u32;
__t248.subvendor = 4294967295u32;
__t248.subdevice = 4294967295u32;
__t248.class = 0u32;
__t248.class_mask = 0u32;
__t248.driver_data = 14u64;
__t248
}, {
let mut __t249: pci_device_id = unsafe { core::mem::zeroed() };
__t249.vendor = 4318u32;
__t249.device = 2740u32;
__t249.subvendor = 4294967295u32;
__t249.subdevice = 4294967295u32;
__t249.class = 0u32;
__t249.class_mask = 0u32;
__t249.driver_data = 14u64;
__t249
}, {
let mut __t250: pci_device_id = unsafe { core::mem::zeroed() };
__t250.vendor = 4318u32;
__t250.device = 2741u32;
__t250.subvendor = 4294967295u32;
__t250.subdevice = 4294967295u32;
__t250.class = 0u32;
__t250.class_mask = 0u32;
__t250.driver_data = 14u64;
__t250
}, {
let mut __t251: pci_device_id = unsafe { core::mem::zeroed() };
__t251.vendor = 4318u32;
__t251.device = 2742u32;
__t251.subvendor = 4294967295u32;
__t251.subdevice = 4294967295u32;
__t251.class = 0u32;
__t251.class_mask = 0u32;
__t251.driver_data = 14u64;
__t251
}, {
let mut __t252: pci_device_id = unsafe { core::mem::zeroed() };
__t252.vendor = 4318u32;
__t252.device = 2743u32;
__t252.subvendor = 4294967295u32;
__t252.subdevice = 4294967295u32;
__t252.class = 0u32;
__t252.class_mask = 0u32;
__t252.driver_data = 14u64;
__t252
}, {
let mut __t253: pci_device_id = unsafe { core::mem::zeroed() };
__t253.vendor = 4318u32;
__t253.device = 2744u32;
__t253.subvendor = 4294967295u32;
__t253.subdevice = 4294967295u32;
__t253.class = 0u32;
__t253.class_mask = 0u32;
__t253.driver_data = 14u64;
__t253
}, {
let mut __t254: pci_device_id = unsafe { core::mem::zeroed() };
__t254.vendor = 4318u32;
__t254.device = 2745u32;
__t254.subvendor = 4294967295u32;
__t254.subdevice = 4294967295u32;
__t254.class = 0u32;
__t254.class_mask = 0u32;
__t254.driver_data = 14u64;
__t254
}, {
let mut __t255: pci_device_id = unsafe { core::mem::zeroed() };
__t255.vendor = 4318u32;
__t255.device = 2746u32;
__t255.subvendor = 4294967295u32;
__t255.subdevice = 4294967295u32;
__t255.class = 0u32;
__t255.class_mask = 0u32;
__t255.driver_data = 14u64;
__t255
}, {
let mut __t256: pci_device_id = unsafe { core::mem::zeroed() };
__t256.vendor = 4318u32;
__t256.device = 2747u32;
__t256.subvendor = 4294967295u32;
__t256.subdevice = 4294967295u32;
__t256.class = 0u32;
__t256.class_mask = 0u32;
__t256.driver_data = 14u64;
__t256
}, {
let mut __t257: pci_device_id = unsafe { core::mem::zeroed() };
__t257.vendor = 4318u32;
__t257.device = 2748u32;
__t257.subvendor = 4294967295u32;
__t257.subdevice = 4294967295u32;
__t257.class = 0u32;
__t257.class_mask = 0u32;
__t257.driver_data = 14u64;
__t257
}, {
let mut __t258: pci_device_id = unsafe { core::mem::zeroed() };
__t258.vendor = 4318u32;
__t258.device = 2749u32;
__t258.subvendor = 4294967295u32;
__t258.subdevice = 4294967295u32;
__t258.class = 0u32;
__t258.class_mask = 0u32;
__t258.driver_data = 14u64;
__t258
}, {
let mut __t259: pci_device_id = unsafe { core::mem::zeroed() };
__t259.vendor = 4318u32;
__t259.device = 2750u32;
__t259.subvendor = 4294967295u32;
__t259.subdevice = 4294967295u32;
__t259.class = 0u32;
__t259.class_mask = 0u32;
__t259.driver_data = 14u64;
__t259
}, {
let mut __t260: pci_device_id = unsafe { core::mem::zeroed() };
__t260.vendor = 4318u32;
__t260.device = 2751u32;
__t260.subvendor = 4294967295u32;
__t260.subdevice = 4294967295u32;
__t260.class = 0u32;
__t260.class_mask = 0u32;
__t260.driver_data = 14u64;
__t260
}, {
let mut __t261: pci_device_id = unsafe { core::mem::zeroed() };
__t261.vendor = 4318u32;
__t261.device = 3460u32;
__t261.subvendor = 4294967295u32;
__t261.subdevice = 4294967295u32;
__t261.class = 0u32;
__t261.class_mask = 0u32;
__t261.driver_data = 15u64;
__t261
}, {
let mut __t262: pci_device_id = unsafe { core::mem::zeroed() };
__t262.vendor = 4318u32;
__t262.device = 3461u32;
__t262.subvendor = 4294967295u32;
__t262.subdevice = 4294967295u32;
__t262.class = 0u32;
__t262.class_mask = 0u32;
__t262.driver_data = 15u64;
__t262
}, {
let mut __t263: pci_device_id = unsafe { core::mem::zeroed() };
__t263.vendor = 4318u32;
__t263.device = 3462u32;
__t263.subvendor = 4294967295u32;
__t263.subdevice = 4294967295u32;
__t263.class = 0u32;
__t263.class_mask = 0u32;
__t263.driver_data = 15u64;
__t263
}, {
let mut __t264: pci_device_id = unsafe { core::mem::zeroed() };
__t264.vendor = 4318u32;
__t264.device = 3463u32;
__t264.subvendor = 4294967295u32;
__t264.subdevice = 4294967295u32;
__t264.class = 0u32;
__t264.class_mask = 0u32;
__t264.driver_data = 15u64;
__t264
}, {
let mut __t265: pci_device_id = unsafe { core::mem::zeroed() };
__t265.vendor = 4318u32;
__t265.device = 3464u32;
__t265.subvendor = 4294967295u32;
__t265.subdevice = 4294967295u32;
__t265.class = 0u32;
__t265.class_mask = 0u32;
__t265.driver_data = 15u64;
__t265
}, {
let mut __t266: pci_device_id = unsafe { core::mem::zeroed() };
__t266.vendor = 4318u32;
__t266.device = 3465u32;
__t266.subvendor = 4294967295u32;
__t266.subdevice = 4294967295u32;
__t266.class = 0u32;
__t266.class_mask = 0u32;
__t266.driver_data = 15u64;
__t266
}, {
let mut __t267: pci_device_id = unsafe { core::mem::zeroed() };
__t267.vendor = 4318u32;
__t267.device = 3466u32;
__t267.subvendor = 4294967295u32;
__t267.subdevice = 4294967295u32;
__t267.class = 0u32;
__t267.class_mask = 0u32;
__t267.driver_data = 15u64;
__t267
}, {
let mut __t268: pci_device_id = unsafe { core::mem::zeroed() };
__t268.vendor = 4318u32;
__t268.device = 3467u32;
__t268.subvendor = 4294967295u32;
__t268.subdevice = 4294967295u32;
__t268.class = 0u32;
__t268.class_mask = 0u32;
__t268.driver_data = 15u64;
__t268
}, {
let mut __t269: pci_device_id = unsafe { core::mem::zeroed() };
__t269.vendor = 4318u32;
__t269.device = 3468u32;
__t269.subvendor = 4294967295u32;
__t269.subdevice = 4294967295u32;
__t269.class = 0u32;
__t269.class_mask = 0u32;
__t269.driver_data = 15u64;
__t269
}, {
let mut __t270: pci_device_id = unsafe { core::mem::zeroed() };
__t270.vendor = 4318u32;
__t270.device = 3469u32;
__t270.subvendor = 4294967295u32;
__t270.subdevice = 4294967295u32;
__t270.class = 0u32;
__t270.class_mask = 0u32;
__t270.driver_data = 15u64;
__t270
}, {
let mut __t271: pci_device_id = unsafe { core::mem::zeroed() };
__t271.vendor = 4318u32;
__t271.device = 3470u32;
__t271.subvendor = 4294967295u32;
__t271.subdevice = 4294967295u32;
__t271.class = 0u32;
__t271.class_mask = 0u32;
__t271.driver_data = 15u64;
__t271
}, {
let mut __t272: pci_device_id = unsafe { core::mem::zeroed() };
__t272.vendor = 4318u32;
__t272.device = 3471u32;
__t272.subvendor = 4294967295u32;
__t272.subdevice = 4294967295u32;
__t272.class = 0u32;
__t272.class_mask = 0u32;
__t272.driver_data = 15u64;
__t272
}, {
let mut __t273: pci_device_id = unsafe { core::mem::zeroed() };
__t273.vendor = 4153u32;
__t273.device = 4484u32;
__t273.subvendor = 4294967295u32;
__t273.subdevice = 4294967295u32;
__t273.class = 0u32;
__t273.class_mask = 0u32;
__t273.driver_data = 0u64;
__t273
}, {
let mut __t274: pci_device_id = unsafe { core::mem::zeroed() };
__t274.vendor = 4153u32;
__t274.device = 4485u32;
__t274.subvendor = 4294967295u32;
__t274.subdevice = 4294967295u32;
__t274.class = 0u32;
__t274.class_mask = 0u32;
__t274.driver_data = 0u64;
__t274
}, {
let mut __t275: pci_device_id = unsafe { core::mem::zeroed() };
__t275.vendor = 4153u32;
__t275.device = 390u32;
__t275.subvendor = 4294967295u32;
__t275.subdevice = 4294967295u32;
__t275.class = 0u32;
__t275.class_mask = 0u32;
__t275.driver_data = 0u64;
__t275
}, {
let mut __t276: pci_device_id = unsafe { core::mem::zeroed() };
__t276.vendor = 4170u32;
__t276.device = 52230u32;
__t276.subvendor = 4294967295u32;
__t276.subdevice = 4294967295u32;
__t276.class = 0u32;
__t276.class_mask = 0u32;
__t276.driver_data = 0u64;
__t276
}, {
let mut __t277: pci_device_id = unsafe { core::mem::zeroed() };
__t277.vendor = 4523u32;
__t277.device = 24901u32;
__t277.subvendor = 4294967295u32;
__t277.subdevice = 4294967295u32;
__t277.class = 0u32;
__t277.class_mask = 0u32;
__t277.driver_data = 16u64;
__t277
}, {
let mut __t278: pci_device_id = unsafe { core::mem::zeroed() };
__t278.vendor = 4523u32;
__t278.device = 24865u32;
__t278.subvendor = 4294967295u32;
__t278.subdevice = 4294967295u32;
__t278.class = 0u32;
__t278.class_mask = 0u32;
__t278.driver_data = 16u64;
__t278
}, {
let mut __t279: pci_device_id = unsafe { core::mem::zeroed() };
__t279.vendor = 6987u32;
__t279.device = 37155u32;
__t279.subvendor = 4294967295u32;
__t279.subdevice = 4294967295u32;
__t279.class = 67073u32;
__t279.class_mask = 16777215u32;
__t279.driver_data = 8u64;
__t279
}, {
let mut __t280: pci_device_id = unsafe { core::mem::zeroed() };
__t280.vendor = 6987u32;
__t280.device = 37157u32;
__t280.subvendor = 4294967295u32;
__t280.subdevice = 4294967295u32;
__t280.driver_data = 8u64;
__t280
}, {
let mut __t281: pci_device_id = unsafe { core::mem::zeroed() };
__t281.vendor = 6987u32;
__t281.device = 37240u32;
__t281.subvendor = 6987u32;
__t281.subdevice = 37232u32;
__t281.driver_data = 8u64;
__t281
}, {
let mut __t282: pci_device_id = unsafe { core::mem::zeroed() };
__t282.vendor = 6987u32;
__t282.device = 37242u32;
__t282.subvendor = 4294967295u32;
__t282.subdevice = 4294967295u32;
__t282.driver_data = 8u64;
__t282
}, {
let mut __t283: pci_device_id = unsafe { core::mem::zeroed() };
__t283.vendor = 6987u32;
__t283.device = 37234u32;
__t283.subvendor = 4294967295u32;
__t283.subdevice = 4294967295u32;
__t283.driver_data = 8u64;
__t283
}, {
let mut __t284: pci_device_id = unsafe { core::mem::zeroed() };
__t284.vendor = 6987u32;
__t284.device = 37250u32;
__t284.subvendor = 4294967295u32;
__t284.subdevice = 4294967295u32;
__t284.driver_data = 8u64;
__t284
}, {
let mut __t285: pci_device_id = unsafe { core::mem::zeroed() };
__t285.vendor = 6987u32;
__t285.device = 37266u32;
__t285.subvendor = 4294967295u32;
__t285.subdevice = 4294967295u32;
__t285.driver_data = 8u64;
__t285
}, {
let mut __t286: pci_device_id = unsafe { core::mem::zeroed() };
__t286.vendor = 6987u32;
__t286.device = 37280u32;
__t286.subvendor = 4294967295u32;
__t286.subdevice = 4294967295u32;
__t286.driver_data = 8u64;
__t286
}, {
let mut __t287: pci_device_id = unsafe { core::mem::zeroed() };
__t287.vendor = 6987u32;
__t287.device = 37282u32;
__t287.subvendor = 4294967295u32;
__t287.subdevice = 4294967295u32;
__t287.driver_data = 8u64;
__t287
}, {
let mut __t288: pci_device_id = unsafe { core::mem::zeroed() };
__t288.vendor = 6987u32;
__t288.device = 37283u32;
__t288.subvendor = 4294967295u32;
__t288.subdevice = 4294967295u32;
__t288.driver_data = 8u64;
__t288
}, {
let mut __t289: pci_device_id = unsafe { core::mem::zeroed() };
__t289.vendor = 6987u32;
__t289.device = 37397u32;
__t289.subvendor = 4294967295u32;
__t289.subdevice = 4294967295u32;
__t289.driver_data = 9u64;
__t289
}, {
let mut __t290: pci_device_id = unsafe { core::mem::zeroed() };
__t290.vendor = 6987u32;
__t290.device = 37424u32;
__t290.subvendor = 4294967295u32;
__t290.subdevice = 4294967295u32;
__t290.driver_data = 8u64;
__t290
}, {
let mut __t291: pci_device_id = unsafe { core::mem::zeroed() };
__t291.vendor = 6987u32;
__t291.device = 37429u32;
__t291.subvendor = 4294967295u32;
__t291.subdevice = 4294967295u32;
__t291.driver_data = 3u64;
__t291
}, {
let mut __t292: pci_device_id = unsafe { core::mem::zeroed() };
__t292.vendor = 4355u32;
__t292.device = 1602u32;
__t292.subvendor = 4294967295u32;
__t292.subdevice = 4294967295u32;
__t292.driver_data = 8u64;
__t292
}, {
let mut __t293: pci_device_id = unsafe { core::mem::zeroed() };
__t293.vendor = 4355u32;
__t293.device = 1605u32;
__t293.subvendor = 4294967295u32;
__t293.subdevice = 4294967295u32;
__t293.driver_data = 8u64;
__t293
}, {
let mut __t294: pci_device_id = unsafe { core::mem::zeroed() };
__t294.vendor = 4186u32;
__t294.device = 16160u32;
__t294.subvendor = 4294967295u32;
__t294.subdevice = 4294967295u32;
__t294.class = 0u32;
__t294.class_mask = 0u32;
__t294.driver_data = 0u64;
__t294
}, {
let mut __t295: pci_device_id = unsafe { core::mem::zeroed() };
__t295.vendor = 4186u32;
__t295.device = 14209u32;
__t295.subvendor = 4294967295u32;
__t295.subdevice = 4294967295u32;
__t295.class = 0u32;
__t295.class_mask = 0u32;
__t295.driver_data = 0u64;
__t295
}, {
let mut __t296: pci_device_id = unsafe { core::mem::zeroed() };
__t296.vendor = 6945u32;
__t296.device = 1537u32;
__t296.subvendor = 4294967295u32;
__t296.subdevice = 4294967295u32;
__t296.class = 0u32;
__t296.class_mask = 0u32;
__t296.driver_data = 1u64;
__t296
}, {
let mut __t297: pci_device_id = unsafe { core::mem::zeroed() };
__t297.vendor = 6945u32;
__t297.device = 1538u32;
__t297.subvendor = 4294967295u32;
__t297.subdevice = 4294967295u32;
__t297.class = 0u32;
__t297.class_mask = 0u32;
__t297.driver_data = 1u64;
__t297
}, {
let mut __t298: pci_device_id = unsafe { core::mem::zeroed() };
__t298.vendor = 6945u32;
__t298.device = 1553u32;
__t298.subvendor = 4294967295u32;
__t298.subdevice = 4294967295u32;
__t298.class = 0u32;
__t298.class_mask = 0u32;
__t298.driver_data = 1u64;
__t298
}, {
let mut __t299: pci_device_id = unsafe { core::mem::zeroed() };
__t299.vendor = 6945u32;
__t299.device = 1554u32;
__t299.subvendor = 4294967295u32;
__t299.subdevice = 4294967295u32;
__t299.class = 0u32;
__t299.class_mask = 0u32;
__t299.driver_data = 1u64;
__t299
}, {
let mut __t300: pci_device_id = unsafe { core::mem::zeroed() };
__t300.vendor = 6945u32;
__t300.device = 1569u32;
__t300.subvendor = 4294967295u32;
__t300.subdevice = 4294967295u32;
__t300.class = 0u32;
__t300.class_mask = 0u32;
__t300.driver_data = 1u64;
__t300
}, {
let mut __t301: pci_device_id = unsafe { core::mem::zeroed() };
__t301.vendor = 6945u32;
__t301.device = 1570u32;
__t301.subvendor = 4294967295u32;
__t301.subdevice = 4294967295u32;
__t301.class = 0u32;
__t301.class_mask = 0u32;
__t301.driver_data = 1u64;
__t301
}, {
let mut __t302: pci_device_id = unsafe { core::mem::zeroed() };
__t302.vendor = 6945u32;
__t302.device = 1572u32;
__t302.subvendor = 4294967295u32;
__t302.subdevice = 4294967295u32;
__t302.class = 0u32;
__t302.class_mask = 0u32;
__t302.driver_data = 1u64;
__t302
}, {
let mut __t303: pci_device_id = unsafe { core::mem::zeroed() };
__t303.vendor = 6945u32;
__t303.device = 4194u32;
__t303.subvendor = 4294967295u32;
__t303.subdevice = 4294967295u32;
__t303.class = 0u32;
__t303.class_mask = 0u32;
__t303.driver_data = 0u64;
__t303
}, {
let mut __t304: pci_device_id = unsafe { core::mem::zeroed() };
__t304.vendor = 6945u32;
__t304.device = 4196u32;
__t304.subvendor = 4294967295u32;
__t304.subdevice = 4294967295u32;
__t304.class = 0u32;
__t304.class_mask = 0u32;
__t304.driver_data = 0u64;
__t304
}, {
let mut __t305: pci_device_id = unsafe { core::mem::zeroed() };
__t305.vendor = 6945u32;
__t305.device = 4452u32;
__t305.subvendor = 4294967295u32;
__t305.subdevice = 4294967295u32;
__t305.class = 0u32;
__t305.class_mask = 0u32;
__t305.driver_data = 0u64;
__t305
}, {
let mut __t306: pci_device_id = unsafe { core::mem::zeroed() };
__t306.vendor = 6945u32;
__t306.device = 4453u32;
__t306.subvendor = 4294967295u32;
__t306.subdevice = 4294967295u32;
__t306.class = 0u32;
__t306.class_mask = 0u32;
__t306.driver_data = 0u64;
__t306
}, {
let mut __t307: pci_device_id = unsafe { core::mem::zeroed() };
__t307.vendor = 6945u32;
__t307.device = 4454u32;
__t307.subvendor = 4294967295u32;
__t307.subdevice = 4294967295u32;
__t307.class = 0u32;
__t307.class_mask = 0u32;
__t307.driver_data = 0u64;
__t307
}, {
let mut __t308: pci_device_id = unsafe { core::mem::zeroed() };
__t308.vendor = 5197u32;
__t308.device = 5632u32;
__t308.subvendor = 4294967295u32;
__t308.subdevice = 4294967295u32;
__t308.class = 0u32;
__t308.class_mask = 0u32;
__t308.driver_data = 4u64;
__t308
}, {
let mut __t309: pci_device_id = unsafe { core::mem::zeroed() };
__t309.vendor = 5197u32;
__t309.device = 43008u32;
__t309.subvendor = 4294967295u32;
__t309.subdevice = 4294967295u32;
__t309.class = 0u32;
__t309.class_mask = 0u32;
__t309.driver_data = 4u64;
__t309
}, {
let mut __t310: pci_device_id = unsafe { core::mem::zeroed() };
__t310.vendor = 7236u32;
__t310.device = 32768u32;
__t310.subvendor = 4294967295u32;
__t310.subdevice = 4294967295u32;
__t310.driver_data = 0u64;
__t310
}, {
let mut __t311: pci_device_id = unsafe { core::mem::zeroed() };
__t311.vendor = 20u32;
__t311.device = 31240u32;
__t311.subvendor = 4294967295u32;
__t311.subdevice = 4294967295u32;
__t311.class = 0u32;
__t311.class_mask = 0u32;
__t311.driver_data = 0u64;
__t311
}, {
let mut __t312: pci_device_id = unsafe { core::mem::zeroed() };
__t312.vendor = 4294967295u32;
__t312.device = 4294967295u32;
__t312.subvendor = 4294967295u32;
__t312.subdevice = 4294967295u32;
__t312.class = 67073u32;
__t312.class_mask = 16777215u32;
__t312.driver_data = 0u64;
__t312
}, {
let mut __t313: pci_device_id = unsafe { core::mem::zeroed() };
__t313
}]);

static ahci_pci_pm_ops: __SlateConst<dev_pm_ops> = __SlateConst({
let mut __t0: dev_pm_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(ahci_pci_device_suspend);
__t0.resume = Some(ahci_pci_device_resume);
__t0.freeze = Some(ahci_pci_device_suspend);
__t0.thaw = Some(ahci_pci_device_resume);
__t0.poweroff = Some(ahci_pci_device_suspend);
__t0.restore = Some(ahci_pci_device_resume);
__t0.runtime_suspend = Some(ahci_pci_device_runtime_suspend);
__t0.runtime_resume = Some(ahci_pci_device_runtime_resume);
__t0.runtime_idle = None;
__t0
});

static mut ahci_pci_driver: pci_driver = {
let mut __t0: pci_driver = unsafe { core::mem::zeroed() };
__t0.name = (b"ahci\0".as_ptr() as *mut u8) as *const u8;
__t0.id_table = unsafe { core::ptr::addr_of!(ahci_pci_tbl.0) as *const pci_device_id };
__t0.probe = Some(ahci_init_one);
__t0.remove = Some(ahci_remove_one);
__t0.shutdown = Some(ahci_shutdown_one);
__t0.driver = {
let mut __t1: device_driver = unsafe { core::mem::zeroed() };
__t1.pm = unsafe { core::ptr::addr_of!(ahci_pci_pm_ops.0) };
__t1
};
__t0
};

static mut marvell_enable: i32 = 1;

fn __check_marvell_enable() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(marvell_enable) }
}

static __param_str_marvell_enable: __SlateAlign16<[u8; 20]> = __SlateAlign16([97u8, 104u8, 99u8, 105u8, 46u8, 109u8, 97u8, 114u8, 118u8, 101u8, 108u8, 108u8, 95u8, 101u8, 110u8, 97u8, 98u8, 108u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_marvell_enable: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_marvell_enable.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ahci_c_1614>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1614_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1614_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1614_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1614_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1614_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(marvell_enable) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_673: [u8; 33] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 97, 114, 118, 101, 108, 108, 95, 101, 110, 97, 98, 108, 101, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_674: [u8; 61] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 109, 97, 114, 118, 101, 108, 108, 95, 101, 110, 97, 98, 108, 101, 58, 77, 97, 114, 118, 101, 108, 108, 32, 83, 65, 84, 65, 32, 118, 105, 97, 32, 65, 72, 67, 73, 32, 40, 49, 32, 61, 32, 101, 110, 97, 98, 108, 101, 100, 41, 0];

static mut mobile_lpm_policy: i32 = 1i32.wrapping_neg();

fn __check_mobile_lpm_policy() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(mobile_lpm_policy) }
}

static __param_str_mobile_lpm_policy: __SlateAlign16<[u8; 23]> = __SlateAlign16([97u8, 104u8, 99u8, 105u8, 46u8, 109u8, 111u8, 98u8, 105u8, 108u8, 101u8, 95u8, 108u8, 112u8, 109u8, 95u8, 112u8, 111u8, 108u8, 105u8, 99u8, 121u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_mobile_lpm_policy: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_mobile_lpm_policy.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ahci_c_1618>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1618_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1618_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1618_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1618_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1618_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(mobile_lpm_policy) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_675: [u8; 36] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 111, 98, 105, 108, 101, 95, 108, 112, 109, 95, 112, 111, 108, 105, 99, 121, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_676: [u8; 144] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 109, 111, 98, 105, 108, 101, 95, 108, 112, 109, 95, 112, 111, 108, 105, 99, 121, 58, 68, 101, 102, 97, 117, 108, 116, 32, 76, 80, 77, 32, 112, 111, 108, 105, 99, 121, 46, 32, 68, 101, 115, 112, 105, 116, 101, 32, 105, 116, 115, 32, 110, 97, 109, 101, 44, 32, 116, 104, 105, 115, 32, 112, 97, 114, 97, 109, 101, 116, 101, 114, 32, 97, 112, 112, 108, 105, 101, 115, 32, 116, 111, 32, 97, 108, 108, 32, 99, 104, 105, 112, 115, 101, 116, 115, 44, 32, 105, 110, 99, 108, 117, 100, 105, 110, 103, 32, 100, 101, 115, 107, 116, 111, 112, 32, 97, 110, 100, 32, 115, 101, 114, 118, 101, 114, 32, 99, 104, 105, 112, 115, 101, 116, 115, 0];

static mut ahci_mask_port_map: *mut u8 = unsafe { core::mem::zeroed() };

fn __check_mask_port_map() -> *mut *mut u8 {
unsafe { core::ptr::addr_of_mut!(ahci_mask_port_map) }
}

static __param_str_mask_port_map: __SlateAlign16<[u8; 19]> = __SlateAlign16([97u8, 104u8, 99u8, 105u8, 46u8, 109u8, 97u8, 115u8, 107u8, 95u8, 112u8, 111u8, 114u8, 116u8, 95u8, 109u8, 97u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_mask_port_map: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_mask_port_map.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_charp) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ahci_c_1624>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1624_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1624_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1624_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1624_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1624_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ahci_mask_port_map) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_677: [u8; 34] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 97, 115, 107, 95, 112, 111, 114, 116, 95, 109, 97, 112, 58, 99, 104, 97, 114, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_678: [u8; 346] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 109, 97, 115, 107, 95, 112, 111, 114, 116, 95, 109, 97, 112, 58, 51, 50, 45, 98, 105, 116, 115, 32, 112, 111, 114, 116, 32, 109, 97, 112, 32, 109, 97, 115, 107, 115, 32, 116, 111, 32, 105, 103, 110, 111, 114, 101, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 115, 32, 112, 111, 114, 116, 115, 46, 32, 86, 97, 108, 105, 100, 32, 118, 97, 108, 117, 101, 115, 32, 97, 114, 101, 58, 32, 34, 60, 109, 97, 115, 107, 62, 34, 32, 116, 111, 32, 97, 112, 112, 108, 121, 32, 116, 104, 101, 32, 115, 97, 109, 101, 32, 109, 97, 115, 107, 32, 116, 111, 32, 97, 108, 108, 32, 65, 72, 67, 73, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 100, 101, 118, 105, 99, 101, 115, 44, 32, 97, 110, 100, 32, 34, 60, 112, 99, 105, 95, 100, 101, 118, 62, 61, 60, 109, 97, 115, 107, 62, 44, 60, 112, 99, 105, 95, 100, 101, 118, 62, 61, 60, 109, 97, 115, 107, 62, 44, 46, 46, 46, 34, 32, 116, 111, 32, 115, 112, 101, 99, 105, 102, 121, 32, 100, 105, 102, 102, 101, 114, 101, 110, 116, 32, 109, 97, 115, 107, 115, 32, 102, 111, 114, 32, 116, 104, 101, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 115, 32, 115, 112, 101, 99, 105, 102, 105, 101, 100, 44, 32, 119, 104, 101, 114, 101, 32, 60, 112, 99, 105, 95, 100, 101, 118, 62, 32, 105, 115, 32, 116, 104, 101, 32, 80, 67, 73, 32, 73, 68, 32, 111, 102, 32, 97, 110, 32, 65, 72, 67, 73, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 105, 110, 32, 116, 104, 101, 32, 102, 111, 114, 109, 32, 34, 100, 111, 109, 97, 105, 110, 58, 98, 117, 115, 58, 100, 101, 118, 46, 102, 117, 110, 99, 34, 0];

static mut ahci_mask_port_ext: *mut u8 = unsafe { core::mem::zeroed() };

fn __check_mask_port_ext() -> *mut *mut u8 {
unsafe { core::ptr::addr_of_mut!(ahci_mask_port_ext) }
}

static __param_str_mask_port_ext: __SlateAlign16<[u8; 19]> = __SlateAlign16([97u8, 104u8, 99u8, 105u8, 46u8, 109u8, 97u8, 115u8, 107u8, 95u8, 112u8, 111u8, 114u8, 116u8, 95u8, 101u8, 120u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_mask_port_ext: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_mask_port_ext.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_charp) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ahci_c_1635>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1635_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1635_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1635_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1635_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_1635_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ahci_mask_port_ext) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_679: [u8; 34] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 97, 115, 107, 95, 112, 111, 114, 116, 95, 101, 120, 116, 58, 99, 104, 97, 114, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_680: [u8; 359] = [97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 109, 97, 115, 107, 95, 112, 111, 114, 116, 95, 101, 120, 116, 58, 51, 50, 45, 98, 105, 116, 115, 32, 109, 97, 115, 107, 32, 116, 111, 32, 105, 103, 110, 111, 114, 101, 32, 116, 104, 101, 32, 101, 120, 116, 101, 114, 110, 97, 108, 47, 104, 111, 116, 112, 108, 117, 103, 32, 99, 97, 112, 97, 98, 105, 108, 105, 116, 121, 32, 111, 102, 32, 112, 111, 114, 116, 115, 46, 32, 86, 97, 108, 105, 100, 32, 118, 97, 108, 117, 101, 115, 32, 97, 114, 101, 58, 32, 34, 60, 109, 97, 115, 107, 62, 34, 32, 116, 111, 32, 97, 112, 112, 108, 121, 32, 116, 104, 101, 32, 115, 97, 109, 101, 32, 109, 97, 115, 107, 32, 116, 111, 32, 97, 108, 108, 32, 65, 72, 67, 73, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 100, 101, 118, 105, 99, 101, 115, 44, 32, 97, 110, 100, 32, 34, 60, 112, 99, 105, 95, 100, 101, 118, 62, 61, 60, 109, 97, 115, 107, 62, 44, 60, 112, 99, 105, 95, 100, 101, 118, 62, 61, 60, 109, 97, 115, 107, 62, 44, 46, 46, 46, 34, 32, 116, 111, 32, 115, 112, 101, 99, 105, 102, 121, 32, 100, 105, 102, 102, 101, 114, 101, 110, 116, 32, 109, 97, 115, 107, 115, 32, 102, 111, 114, 32, 116, 104, 101, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 115, 32, 115, 112, 101, 99, 105, 102, 105, 101, 100, 44, 32, 119, 104, 101, 114, 101, 32, 60, 112, 99, 105, 95, 100, 101, 118, 62, 32, 105, 115, 32, 116, 104, 101, 32, 80, 67, 73, 32, 73, 68, 32, 111, 102, 32, 97, 110, 32, 65, 72, 67, 73, 32, 99, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 105, 110, 32, 116, 104, 101, 32, 102, 111, 114, 109, 32, 34, 100, 111, 109, 97, 105, 110, 58, 98, 117, 115, 58, 100, 101, 118, 46, 102, 117, 110, 99, 34, 0];

fn ahci_port_mask(dev: *mut device, mask_s: *mut u8) -> u32 {
let mut mask: u32 = 0;
if (unsafe { kstrtouint(mask_s as *const u8, 0u32, core::ptr::addr_of_mut!(mask)) }) != 0 {
while false {
}
unsafe { _dev_err(dev as *const device, c"Invalid port map mask\n".as_ptr() as *const u8) };
return 0;
}
mask
}

fn ahci_get_port_mask(dev: *mut device, mask_p: *mut u8) -> u32 {
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut mask: u32 = unsafe { core::mem::zeroed() };
let mut name: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut u8 = core::ptr::addr_of_mut!(name) as *mut *mut u8;
let mut mask_s: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut u8 = core::ptr::addr_of_mut!(mask_s) as *mut *mut u8;
let mut str: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut u8 = core::ptr::addr_of_mut!(str) as *mut *mut u8;
let mut end: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u8 = core::ptr::addr_of_mut!(end) as *mut *mut u8;
let mut param: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut u8 = core::ptr::addr_of_mut!(param) as *mut *mut u8;
unsafe {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
mask = 0;
if mask_p.is_null() {
__v6 = true;
} else {
__v6 = strlen(mask_p as *const u8) == 0;
}
if __v6 {
return 0;
} else {
*__slate_slot_10 = kstrdup(mask_p as *const u8, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if (*__slate_slot_10).is_null() {
return 0;
} else {
// Handle single mask case
if strchr(*__slate_slot_10 as *const u8, 61i32).is_null() {
mask = ahci_port_mask(dev, *__slate_slot_10);
} else {
// Mask list case: parse the parameter to get the mask only if
// the device name matches.
*__slate_slot_12 = *__slate_slot_10;
*__slate_slot_11 = (*__slate_slot_12).wrapping_offset(strlen(*__slate_slot_12 as *const u8) as isize);
while !(*__slate_slot_12).is_null() && *__slate_slot_12 < *__slate_slot_11 && *(*__slate_slot_12) != 0 {
*__slate_slot_8 = *__slate_slot_12;
*__slate_slot_12 = strchr(*__slate_slot_8 as *const u8, 61i32);
if (*__slate_slot_12).is_null() {
break;
} else {
*(*__slate_slot_12) = 0;
let __v5: *mut u8 = *__slate_slot_12;
*__slate_slot_12 = __v5.wrapping_offset(1 as isize);
if *__slate_slot_12 >= *__slate_slot_11 {
break;
} else {
if strcmp(dev_name(dev as *const device), *__slate_slot_8 as *const u8) != 0 {
*__slate_slot_12 = strchr(*__slate_slot_12 as *const u8, 44i32);
if !(*__slate_slot_12).is_null() {
let __v3: *mut u8 = *__slate_slot_12;
*__slate_slot_12 = __v3.wrapping_offset(1 as isize);
}
} else {
*__slate_slot_9 = *__slate_slot_12;
*__slate_slot_12 = strchr(*__slate_slot_9 as *const u8, 44i32);
if !(*__slate_slot_12).is_null() {
*(*__slate_slot_12) = 0;
let __v1: *mut u8 = *__slate_slot_12;
*__slate_slot_12 = __v1.wrapping_offset(1 as isize);
}
mask = ahci_port_mask(dev, *__slate_slot_9);
}
}
}
}
}
kfree(*__slate_slot_10 as *const ());
return mask;
}
}
}
}

fn ahci_pci_save_initial_config(pdev: *mut pci_dev, hpriv: *mut ahci_host_priv) {
if ((unsafe { (*pdev).vendor }) as i32) == 6523 && ((unsafe { (*pdev).device }) as i32) == 9057 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"JMB361 has only one port\n".as_ptr() as *const u8) };
unsafe {
(*hpriv).saved_port_map = 1;
}
}
// Temporary Marvell 6145 hack: PATA port presence
// is asserted through the standard AHCI port
// presence register, as bit 4 (counting from 0)
if (unsafe { (*hpriv).flags }) & 16 != 0 {
if ((unsafe { (*pdev).device }) as i32) == 24865 {
unsafe {
(*hpriv).mask_port_map = 3;
}
} else {
unsafe {
(*hpriv).mask_port_map = 15;
}
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Disabling your PATA port. Use the boot option 'ahci.marvell_enable=0' to avoid this.\n".as_ptr() as *const u8) };
}
// Handle port map masks passed as module parameter.
unsafe {
(*hpriv).mask_port_map = ahci_get_port_mask(core::ptr::addr_of_mut!((*pdev).dev), ahci_mask_port_map);
}
unsafe {
(*hpriv).mask_port_ext = ahci_get_port_mask(core::ptr::addr_of_mut!((*pdev).dev), ahci_mask_port_ext);
}
unsafe { ahci_save_initial_config(core::ptr::addr_of_mut!((*pdev).dev), hpriv) };
}

fn ahci_pci_reset_controller(host: *mut ata_host) -> i32 {
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*host).dev });
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let rc: i32 = unsafe { ahci_reset_controller(host) };
if rc != 0 {
return rc;
}
// If platform firmware failed to enable ports, try to enable
// them here.
ahci_intel_pcs_quirk(pdev, hpriv);
0
}

fn ahci_pci_init_controller(host: *mut ata_host) {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*host).dev });
let mut tmp: u32 = 0;
let mut mv: i32 = 0;
if (unsafe { (*hpriv).flags }) & 16 != 0 {
if ((unsafe { (*pdev).device }) as i32) == 24865 {
mv = 2;
} else {
mv = 4;
}
let port_mmio: *mut () = __ahci_port_base(hpriv, mv as u32);
writel(0u32, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
// clear port IRQ
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"PORT_IRQ_STAT 0x%x\n".as_ptr() as *const u8, tmp) };
}
if tmp != 0 {
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ());
}
}
unsafe { ahci_init_controller(host) };
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_vt8251_hardreset")]
extern "C" fn ahci_vt8251_hardreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mut online: bool = false;
let mut rc: i32 = 0;
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
rc = unsafe { sata_link_hardreset(link, sata_ehc_deb_timing(core::ptr::addr_of_mut!((*link).eh_context)), deadline, core::ptr::addr_of_mut!(online), None) };
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
// vt8251 doesn't clear BSY on signature FIS reception,
// request follow-up softreset.
if online { 11i32.wrapping_neg() } else { rc }
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_p5wdh_hardreset")]
extern "C" fn ahci_p5wdh_hardreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let d2h_fis: *mut u8 = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *mut u8;
let mut tf: ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut online: bool = false;
let mut rc: i32 = 0;
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
// clear D2H reception area to properly wait for D2H FIS
ata_tf_init(unsafe { (*link).device.as_mut_ptr() as *mut ata_device }, core::ptr::addr_of_mut!(tf));
unsafe {
tf.__slate_anon_14.status = 128;
}
unsafe { ata_tf_to_fis(core::ptr::addr_of_mut!(tf) as *const ata_taskfile, 0u8, 0i32, d2h_fis) };
rc = unsafe { sata_link_hardreset(link, sata_ehc_deb_timing(core::ptr::addr_of_mut!((*link).eh_context)), deadline, core::ptr::addr_of_mut!(online), None) };
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
// The pseudo configuration device on SIMG4726 attached to
// ASUS P5W-DH Deluxe doesn't send signature FIS after
// hardreset if no device is attached to the first downstream
// port && the pseudo device locks up on SRST w/ PMP==0.  To
// work around this, wait for !BSY only briefly.  If BSY isn't
// cleared, perform CLO and proceed to IDENTIFY (achieved by
// ATA_LFLAG_NO_SRST and ATA_LFLAG_ASSUME_ATA).
//
// Wait for two seconds.  Devices attached to downstream port
// which can't process the following IDENTIFY after this will
// have to be reset again.  For most cases, this should
// suffice while making probing snappish enough.
if online {
rc = unsafe { ata_wait_after_reset(link, core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(2000u64), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut ata_link) -> i32>>(ahci_check_ready as *const ()) }) };
if rc != 0 {
unsafe { ahci_kick_engine(ap) };
}
}
rc
}

/// ahci_avn_hardreset - attempt more aggressive recovery of Avoton ports.
///
/// It has been observed with some SSDs that the timing of events in the
/// link synchronization phase can leave the port in a state that can not
/// be recovered by a SATA-hard-reset alone.  The failing signature is
/// SStatus.DET stuck at 1 ("Device presence detected but Phy
/// communication not established").  It was found that unloading and
/// reloading the driver when this problem occurs allows the drive
/// connection to be recovered (DET advanced to 0x3).  The critical
/// component of reloading the driver is that the port state machines are
/// reset by bouncing "port enable" in the AHCI PCS configuration
/// register.  So, reproduce that effect by bouncing a port whenever we
/// see DET==1 after a reset.
#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_avn_hardreset")]
extern "C" fn ahci_avn_hardreset(link: *mut ata_link, class: *mut u32, mut deadline: u64) -> i32 {
let timing: *const u32 = sata_ehc_deb_timing(unsafe { core::ptr::addr_of_mut!((*link).eh_context) });
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let d2h_fis: *mut u8 = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *mut u8;
let tmo: u64 = deadline.wrapping_sub(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
let mut tf: ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut online: bool = false;
let mut rc: i32 = 0;
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
for i in 0i32..2i32 {
let mut val: u16 = 0;
let mut sstatus: u32 = 0;
let port: i32 = (unsafe { (*ap).port_no }) as i32;
let host: *mut ata_host = unsafe { (*ap).host };
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*host).dev });
// clear D2H reception area to properly wait for D2H FIS
ata_tf_init(unsafe { (*link).device.as_mut_ptr() as *mut ata_device }, core::ptr::addr_of_mut!(tf));
unsafe {
tf.__slate_anon_14.status = 128;
}
unsafe { ata_tf_to_fis(core::ptr::addr_of_mut!(tf) as *const ata_taskfile, 0u8, 0i32, d2h_fis) };
rc = unsafe { sata_link_hardreset(link, timing, deadline, core::ptr::addr_of_mut!(online), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut ata_link) -> i32>>(ahci_check_ready as *const ()) }) };
if (unsafe { sata_scr_read(link, 0i32, core::ptr::addr_of_mut!(sstatus)) }) != 0 || sstatus & 15 != 1 {
break;
}
loop {
if sata_pmp_attached(unsafe { (*link).ap }) || !unsafe { (*(*link).ap).slave_link }.is_null() {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016ata%u.%02u: avn bounce port%d\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp, port) };
} else {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016ata%u: avn bounce port%d\n".as_ptr() as *const u8, (*(*link).ap).print_id, port) };
}
if true {
break;
}
}
unsafe { pci_read_config_word(pdev as *const pci_dev, 146i32, core::ptr::addr_of_mut!(val)) };
let __v8: u16 = ((val as i32) & !(1i32 << port)) as u16;
val = __v8;
unsafe { pci_write_config_word(pdev as *const pci_dev, 146i32, val) };
unsafe { ata_msleep(ap, 1000u32) };
val = ((val as i32) | 1i32 << port) as u16;
unsafe { pci_write_config_word(pdev as *const pci_dev, 146i32, val) };
deadline = deadline.wrapping_add(tmo);
}
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
if online {
unsafe {
*class = ahci_dev_classify(ap);
}
}
rc
}

fn ahci_pci_disable_interrupts(host: *mut ata_host) {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut ctl: u32 = 0;
// AHCI spec rev1.1 section 8.3.3:
// Software must disable interrupts prior to requesting a
// transition of the HBA to D3 state.
ctl = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ());
let __v1: u32 = ctl & !2;
ctl = __v1;
writel(ctl, (unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ()); // flush
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_pci_device_runtime_suspend")]
extern "C" fn ahci_pci_device_runtime_suspend(dev: *mut device) -> i32 {
let pdev: *mut pci_dev = to_pci_dev!(dev);
let host: *mut ata_host = pci_get_drvdata(pdev) as *mut ata_host;
ahci_pci_disable_interrupts(host);
0
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_pci_device_runtime_resume")]
extern "C" fn ahci_pci_device_runtime_resume(dev: *mut device) -> i32 {
let pdev: *mut pci_dev = to_pci_dev!(dev);
let host: *mut ata_host = pci_get_drvdata(pdev) as *mut ata_host;
let rc: i32 = ahci_pci_reset_controller(host);
if rc != 0 {
return rc;
}
ahci_pci_init_controller(host);
0
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_pci_device_suspend")]
extern "C" fn ahci_pci_device_suspend(dev: *mut device) -> i32 {
let mut __slate_compound_2: pm_message = unsafe { core::mem::zeroed() };
let pdev: *mut pci_dev = to_pci_dev!(dev);
let host: *mut ata_host = pci_get_drvdata(pdev) as *mut ata_host;
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
if (unsafe { (*hpriv).flags }) & 1024 != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"BIOS update required for suspend/resume\n".as_ptr() as *const u8) };
return 5i32.wrapping_neg();
}
ahci_pci_disable_interrupts(host);
unsafe { ata_host_suspend(host, *{
__slate_compound_2 = pm_message { event: 2 };
core::ptr::addr_of_mut!(__slate_compound_2)
}) };
0
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_pci_device_resume")]
extern "C" fn ahci_pci_device_resume(dev: *mut device) -> i32 {
let pdev: *mut pci_dev = to_pci_dev!(dev);
let host: *mut ata_host = pci_get_drvdata(pdev) as *mut ata_host;
let mut rc: i32 = 0;
// Apple BIOS helpfully mangles the registers on resume
if is_mcp89_apple(pdev) {
ahci_mcp89_apple_enable(pdev);
}
if (unsafe { (*pdev).dev.power.power_state.event }) == 2 {
rc = ahci_pci_reset_controller(host);
if rc != 0 {
return rc;
}
ahci_pci_init_controller(host);
}
unsafe { ata_host_resume(host) };
0
}

fn ahci_configure_dma_masks(pdev: *mut pci_dev, hpriv: *mut ahci_host_priv) -> i32 {
let mut dma_bits: i32 = 0;
let mut rc: i32 = 0;
if (unsafe { (*hpriv).cap }) & 2147483648 != 0 {
dma_bits = 64;
if (unsafe { (*hpriv).flags }) & 134217728 != 0 {
dma_bits = 43;
}
} else {
dma_bits = 32;
}
// If the device fixup already set the dma_mask to some non-standard
// value, don't extend it here. This happens on STA2X11, for example.
//
// XXX: manipulating the DMA mask from platform code is completely
// bogus, platform code should use dev->bus_dma_limit instead..
if (unsafe { (*pdev).dma_mask }) != 0 && (unsafe { (*pdev).dma_mask }) < DMA_BIT_MASK!(32i32) {
return 0;
}
rc = dma_set_mask_and_coherent(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, DMA_BIT_MASK__4!(dma_bits));
if rc != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"DMA enable failed\n".as_ptr() as *const u8) };
}
rc
}

fn ahci_pci_print_info(host: *mut ata_host) {
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*host).dev });
let mut cc: u16 = 0;
let mut scc_s: *const u8 = unsafe { core::mem::zeroed() };
unsafe { pci_read_config_word(pdev as *const pci_dev, 10i32, core::ptr::addr_of_mut!(cc)) };
if (cc as i32) == 257 {
scc_s = c"IDE".as_ptr() as *const u8;
} else {
if (cc as i32) == 262 {
scc_s = c"SATA".as_ptr() as *const u8;
} else {
if (cc as i32) == 260 {
scc_s = c"RAID".as_ptr() as *const u8;
} else {
scc_s = c"unknown".as_ptr() as *const u8;
}
}
}
unsafe { ahci_print_info(host, scc_s) };
}

/// On ASUS P5W DH Deluxe, the second port of PCI device 00:1f.2 is
/// hardwired to on-board SIMG 4726.  The chipset is ICH8 and doesn't
/// support PMP and the 4726 either directly exports the device
/// attached to the first downstream port or acts as a hardware storage
/// controller and emulate a single ATA device (can be RAID 0/1 or some
/// other configuration).
///
/// When there's no device attached to the first downstream port of the
/// 4726, "Config Disk" appears, which is a pseudo ATA device to
/// configure the 4726.  However, ATA emulation of the device is very
/// lame.  It doesn't send signature D2H Reg FIS after the initial
/// hardreset, pukes on SRST w/ PMP==0 and has bunch of other issues.
///
/// The following function works around the problem by always using
/// hardreset on the port and not depending on receiving signature FIS
/// afterward.  If signature FIS isn't received soon, ATA class is
/// assumed without follow-up softreset.
fn ahci_p5wdh_workaround(host: *mut ata_host) {
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*host).dev });
let __v1: bool = ((unsafe { (*(*pdev).bus).number }) as i32) == 0 && (unsafe { (*pdev).devfn }) == (PCI_DEVFN!(31i32, 2i32) as u32) && (unsafe { dmi_check_system(core::ptr::addr_of!(sysids__2080.0) as *const dmi_system_id) }) != 0;
if __v1 {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(1 as isize) };
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"enabling ASUS P5W DH Deluxe on-board SIMG4726 workaround\n".as_ptr() as *const u8) };
unsafe {
(*ap).ops = core::ptr::addr_of_mut!(ahci_p5wdh_ops);
}
unsafe {
(*ap).link.flags = (*ap).link.flags | 12u32;
}
}
}

static sysids__2080: __SlateAlign16<[dmi_system_id; 2]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"P5W DH Deluxe\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 69u8, 75u8, 32u8, 67u8, 79u8, 77u8, 80u8, 85u8, 84u8, 69u8, 82u8, 32u8, 73u8, 78u8, 67u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [80u8, 53u8, 87u8, 32u8, 68u8, 72u8, 32u8, 68u8, 101u8, 108u8, 117u8, 120u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4
}]);

/// Macbook7,1 firmware forcibly disables MCP89 AHCI and changes PCI ID when
/// booting in BIOS compatibility mode.  We restore the registers but not ID.
fn ahci_mcp89_apple_enable(pdev: *mut pci_dev) {
let mut val: u32 = 0;
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016ahci: enabling MCP89 AHCI mode\n".as_ptr() as *const u8) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 248i32, core::ptr::addr_of_mut!(val)) };
val |= 134217728u32;
// the following changes the device ID, but appears not to affect function
//
// val = (val & ~0xf0000000) | 0x80000000;
unsafe { pci_write_config_dword(pdev as *const pci_dev, 248i32, val) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 1356i32, core::ptr::addr_of_mut!(val)) };
val |= 4096u32;
unsafe { pci_write_config_dword(pdev as *const pci_dev, 1356i32, val) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 1188i32, core::ptr::addr_of_mut!(val)) };
val &= 255;
val |= 17170688;
unsafe { pci_write_config_dword(pdev as *const pci_dev, 1188i32, val) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 1356i32, core::ptr::addr_of_mut!(val)) };
let __v11: u32 = val & 4294963199u32;
val = __v11;
unsafe { pci_write_config_dword(pdev as *const pci_dev, 1356i32, val) };
unsafe { pci_read_config_dword(pdev as *const pci_dev, 248i32, core::ptr::addr_of_mut!(val)) };
let __v13: u32 = val & 4160749567u32;
val = __v13;
unsafe { pci_write_config_dword(pdev as *const pci_dev, 248i32, val) };
}

fn is_mcp89_apple(pdev: *mut pci_dev) -> bool {
((unsafe { (*pdev).vendor }) as i32) == 4318 && ((unsafe { (*pdev).device }) as i32) == 3461 && ((unsafe { (*pdev).subsystem_vendor }) as i32) == 4203 && ((unsafe { (*pdev).subsystem_device }) as i32) == 52105
}

/// only some SB600 ahci controllers can do 64bit DMA
fn ahci_sb600_enable_64bit(pdev: *mut pci_dev) -> bool {
let mut buf: core::mem::MaybeUninit<[u8; 9]> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut [u8; 9] = core::ptr::addr_of_mut!(buf) as *mut [u8; 9];
let mut date: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(date) as *mut i32;
let mut month: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(month) as *mut i32;
let mut year: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(year) as *mut i32;
let mut r#match: core::mem::MaybeUninit<*const dmi_system_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *const dmi_system_id = core::ptr::addr_of_mut!(r#match) as *mut *const dmi_system_id;
unsafe {
// The oldest version known to be broken is 0901 and
// working is 1501 which was released on 2007-10-26.
// Enable 64bit DMA on 1501 and anything newer.
//
// Please read bko#9412 for more info.
// yyyymmdd
// All BIOS versions for the MSI K9A2 Platinum (MS-7376)
// support 64bit DMA.
//
// BIOS versions earlier than 1.5 had the Manufacturer DMI
// fields as "MICRO-STAR INTERANTIONAL CO.,LTD".
// This spelling mistake was fixed in BIOS version 1.5, so
// 1.5 and later have the Manufacturer as
// "MICRO-STAR INTERNATIONAL CO.,LTD".
// So try to match on DMI_BOARD_VENDOR of "MICRO-STAR INTER".
//
// BIOS versions earlier than 1.9 had a Board Product Name
// DMI field of "MS-7376". This was changed to be
// "K9A2 Platinum (MS-7376)" in version 1.9, but we can still
// match on DMI_BOARD_NAME of "MS-7376".
// All BIOS versions for the MSI K9AGM2 (MS-7327) support
// 64bit DMA.
//
// This board also had the typo mentioned above in the
// Manufacturer DMI field (fixed in BIOS version 1.5), so
// match on DMI_BOARD_VENDOR of "MICRO-STAR INTER" again.
// All BIOS versions for the Asus M3A support 64bit DMA.
// (all release versions from 0301 to 1206 were tested)
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 9usize);
*__slate_slot_4 = dmi_first_match(core::ptr::addr_of!(sysids__2150.0) as *const dmi_system_id);
if ((*(*pdev).bus).number as i32) != 0 || (*pdev).devfn != (PCI_DEVFN!(18i32, 0i32) as u32) || (*__slate_slot_4).is_null() {
return false;
} else {
if !unsafe { (*(*__slate_slot_4)).driver_data }.is_null() {
dmi_get_date(3i32, core::ptr::addr_of_mut!(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_2), core::ptr::addr_of_mut!(*__slate_slot_1));
snprintf((*__slate_slot_0).as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 9]>() as u64, c"%04d%02d%02d".as_ptr() as *const u8, *__slate_slot_3, *__slate_slot_2, *__slate_slot_1);
if strcmp((*__slate_slot_0).as_mut_ptr() as *const u8, (*(*__slate_slot_4)).driver_data as *const u8) < 0 {
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"%s: BIOS too old, forcing 32bit DMA, update BIOS\n".as_ptr() as *const u8, (*(*__slate_slot_4)).ident);
return false;
}
}
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"%s: enabling 64bit DMA\n".as_ptr() as *const u8, (*(*__slate_slot_4)).ident);
return true;
}
}
}

static sysids__2150: __SlateAlign16<[dmi_system_id; 5]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"ASUS M2A-VM\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(13u8);
__t2.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 101u8, 75u8, 32u8, 67u8, 111u8, 109u8, 112u8, 117u8, 116u8, 101u8, 114u8, 32u8, 73u8, 78u8, 67u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(14u8);
__t3.substr = [77u8, 50u8, 65u8, 45u8, 86u8, 77u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = (b"20071026\0".as_ptr() as *mut u8) as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.ident = (b"MSI K9A2 Platinum\0".as_ptr() as *mut u8) as *const u8;
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(13u8);
__t6.substr = [77u8, 73u8, 67u8, 82u8, 79u8, 45u8, 83u8, 84u8, 65u8, 82u8, 32u8, 73u8, 78u8, 84u8, 69u8, 82u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(14u8);
__t7.substr = [77u8, 83u8, 45u8, 55u8, 51u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8.ident = (b"MSI K9AGM2\0".as_ptr() as *mut u8) as *const u8;
__t8.matches = {
let mut __t9: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t9[0] = {
let mut __t10: dmi_strmatch = unsafe { core::mem::zeroed() };
__t10.__slate_bits_0.__set_slot(13u8);
__t10.substr = [77u8, 73u8, 67u8, 82u8, 79u8, 45u8, 83u8, 84u8, 65u8, 82u8, 32u8, 73u8, 78u8, 84u8, 69u8, 82u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10
};
__t9[1] = {
let mut __t11: dmi_strmatch = unsafe { core::mem::zeroed() };
__t11.__slate_bits_0.__set_slot(14u8);
__t11.substr = [77u8, 83u8, 45u8, 55u8, 51u8, 50u8, 55u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
};
__t9
};
__t8
}, {
let mut __t12: dmi_system_id = unsafe { core::mem::zeroed() };
__t12.ident = (b"ASUS M3A\0".as_ptr() as *mut u8) as *const u8;
__t12.matches = {
let mut __t13: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t13[0] = {
let mut __t14: dmi_strmatch = unsafe { core::mem::zeroed() };
__t14.__slate_bits_0.__set_slot(13u8);
__t14.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 101u8, 75u8, 32u8, 67u8, 111u8, 109u8, 112u8, 117u8, 116u8, 101u8, 114u8, 32u8, 73u8, 78u8, 67u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14
};
__t13[1] = {
let mut __t15: dmi_strmatch = unsafe { core::mem::zeroed() };
__t15.__slate_bits_0.__set_slot(14u8);
__t15.substr = [77u8, 51u8, 65u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15
};
__t13
};
__t12
}, {
let mut __t16: dmi_system_id = unsafe { core::mem::zeroed() };
__t16
}]);

fn ahci_broken_system_poweroff(pdev: *mut pci_dev) -> bool {
// PCI slot number of the controller
// PCI slot number of the controller
// terminate list
let dmi: *const dmi_system_id = unsafe { dmi_first_match(core::ptr::addr_of!(broken_systems.0) as *const dmi_system_id) };
if !dmi.is_null() {
let slot: u64 = (unsafe { (*dmi).driver_data }) as u64;
// apply the quirk only to on-board controllers
return slot == (PCI_SLOT!(unsafe { (*pdev).devfn }) as u64);
}
false
}

static broken_systems: __SlateAlign16<[dmi_system_id; 3]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"HP Compaq nx6310\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [72u8, 80u8, 32u8, 67u8, 111u8, 109u8, 112u8, 97u8, 113u8, 32u8, 110u8, 120u8, 54u8, 51u8, 49u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = 31u64 as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.ident = (b"HP Compaq 6720s\0".as_ptr() as *mut u8) as *const u8;
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(6u8);
__t6.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(7u8);
__t7.substr = [72u8, 80u8, 32u8, 67u8, 111u8, 109u8, 112u8, 97u8, 113u8, 32u8, 54u8, 55u8, 50u8, 48u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4.driver_data = 31u64 as *mut ();
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8
}]);

fn ahci_broken_suspend(pdev: *mut pci_dev) -> bool {
// On HP dv[4-6] and HDX18 with earlier BIOSen, link
// to the harddisk doesn't become online after
// resuming from STR.  Warn and fail suspend.
//
// http://bugzilla.kernel.org/show_bug.cgi?id=12276
//
// Use dates instead of versions to match as HP is
// apparently recycling both product and version
// strings.
//
// http://bugzilla.kernel.org/show_bug.cgi?id=15462
// F.30
// F.16
// F.21
// F.23
// Acer eMachines G725 has the same problem.  BIOS
// V1.03 is known to be broken.  V3.04 is known to
// work.  Between, there are V1.06, V2.06 and V3.03
// that we don't have much idea about.  For now,
// assume that anything older than V3.04 is broken.
//
// http://bugzilla.kernel.org/show_bug.cgi?id=15104
// V3.04
// terminate list
let dmi: *const dmi_system_id = unsafe { dmi_first_match(core::ptr::addr_of!(sysids__2287.0) as *const dmi_system_id) };
let mut year: i32 = 0;
let mut month: i32 = 0;
let mut date: i32 = 0;
let mut buf: [u8; 9] = [0u8; 9];
if dmi.is_null() || (unsafe { (*(*pdev).bus).number }) != 0 || (unsafe { (*pdev).devfn }) != (PCI_DEVFN!(31i32, 2i32) as u32) {
return false;
}
unsafe { dmi_get_date(3i32, core::ptr::addr_of_mut!(year), core::ptr::addr_of_mut!(month), core::ptr::addr_of_mut!(date)) };
unsafe { snprintf(buf.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 9]>() as u64, c"%04d%02d%02d".as_ptr() as *const u8, year, month, date) };
(unsafe { strcmp(buf.as_mut_ptr() as *const u8, (*dmi).driver_data as *const u8) }) < 0
}

static sysids__2287: __SlateAlign16<[dmi_system_id; 6]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"dv4\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [72u8, 80u8, 32u8, 80u8, 97u8, 118u8, 105u8, 108u8, 105u8, 111u8, 110u8, 32u8, 100u8, 118u8, 52u8, 32u8, 78u8, 111u8, 116u8, 101u8, 98u8, 111u8, 111u8, 107u8, 32u8, 80u8, 67u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = (b"20090105\0".as_ptr() as *mut u8) as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.ident = (b"dv5\0".as_ptr() as *mut u8) as *const u8;
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(6u8);
__t6.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(7u8);
__t7.substr = [72u8, 80u8, 32u8, 80u8, 97u8, 118u8, 105u8, 108u8, 105u8, 111u8, 110u8, 32u8, 100u8, 118u8, 53u8, 32u8, 78u8, 111u8, 116u8, 101u8, 98u8, 111u8, 111u8, 107u8, 32u8, 80u8, 67u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4.driver_data = (b"20090506\0".as_ptr() as *mut u8) as *mut ();
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8.ident = (b"dv6\0".as_ptr() as *mut u8) as *const u8;
__t8.matches = {
let mut __t9: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t9[0] = {
let mut __t10: dmi_strmatch = unsafe { core::mem::zeroed() };
__t10.__slate_bits_0.__set_slot(6u8);
__t10.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10
};
__t9[1] = {
let mut __t11: dmi_strmatch = unsafe { core::mem::zeroed() };
__t11.__slate_bits_0.__set_slot(7u8);
__t11.substr = [72u8, 80u8, 32u8, 80u8, 97u8, 118u8, 105u8, 108u8, 105u8, 111u8, 110u8, 32u8, 100u8, 118u8, 54u8, 32u8, 78u8, 111u8, 116u8, 101u8, 98u8, 111u8, 111u8, 107u8, 32u8, 80u8, 67u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
};
__t9
};
__t8.driver_data = (b"20090423\0".as_ptr() as *mut u8) as *mut ();
__t8
}, {
let mut __t12: dmi_system_id = unsafe { core::mem::zeroed() };
__t12.ident = (b"HDX18\0".as_ptr() as *mut u8) as *const u8;
__t12.matches = {
let mut __t13: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t13[0] = {
let mut __t14: dmi_strmatch = unsafe { core::mem::zeroed() };
__t14.__slate_bits_0.__set_slot(6u8);
__t14.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14
};
__t13[1] = {
let mut __t15: dmi_strmatch = unsafe { core::mem::zeroed() };
__t15.__slate_bits_0.__set_slot(7u8);
__t15.substr = [72u8, 80u8, 32u8, 72u8, 68u8, 88u8, 49u8, 56u8, 32u8, 78u8, 111u8, 116u8, 101u8, 98u8, 111u8, 111u8, 107u8, 32u8, 80u8, 67u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15
};
__t13
};
__t12.driver_data = (b"20090430\0".as_ptr() as *mut u8) as *mut ();
__t12
}, {
let mut __t16: dmi_system_id = unsafe { core::mem::zeroed() };
__t16.ident = (b"G725\0".as_ptr() as *mut u8) as *const u8;
__t16.matches = {
let mut __t17: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t17[0] = {
let mut __t18: dmi_strmatch = unsafe { core::mem::zeroed() };
__t18.__slate_bits_0.__set_slot(6u8);
__t18.substr = [101u8, 77u8, 97u8, 99u8, 104u8, 105u8, 110u8, 101u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t18
};
__t17[1] = {
let mut __t19: dmi_strmatch = unsafe { core::mem::zeroed() };
__t19.__slate_bits_0.__set_slot(7u8);
__t19.substr = [101u8, 77u8, 97u8, 99u8, 104u8, 105u8, 110u8, 101u8, 115u8, 32u8, 71u8, 55u8, 50u8, 53u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t19
};
__t17
};
__t16.driver_data = (b"20091216\0".as_ptr() as *mut u8) as *mut ();
__t16
}, {
let mut __t20: dmi_system_id = unsafe { core::mem::zeroed() };
__t20
}]);

fn ahci_broken_lpm(pdev: *mut pci_dev) -> bool {
// Platforms with LPM problems.
// If driver_data is NULL, there is no existing BIOS version with
// functioning LPM.
// If driver_data is non-NULL, then driver_data contains the DMI BIOS
// build date of the first BIOS version with functioning LPM (i.e. older
// BIOS versions have broken LPM).
// 1.31
// 1.28
// 1.33
// 2.35
// 320 is broken, there is no known good version.
// AMD 500 Series Chipset SATA Controller [1022:43eb]
// on this motherboard timeouts on ports 5 and 6 when
// LPM is enabled, at least with WDC WD20EFAX-68FB5N0
// hard drives. LPM with the same drive works fine on
// all other ports on the same controller.
// 3621 is broken, there is no known good version.
// terminate list
let dmi: *const dmi_system_id = unsafe { dmi_first_match(core::ptr::addr_of!(sysids__2379.0) as *const dmi_system_id) };
let mut year: i32 = 0;
let mut month: i32 = 0;
let mut date: i32 = 0;
let mut buf: [u8; 9] = [0u8; 9];
if dmi.is_null() {
return false;
}
if unsafe { (*dmi).driver_data }.is_null() {
return true;
}
unsafe { dmi_get_date(3i32, core::ptr::addr_of_mut!(year), core::ptr::addr_of_mut!(month), core::ptr::addr_of_mut!(date)) };
unsafe { snprintf(buf.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 9]>() as u64, c"%04d%02d%02d".as_ptr() as *const u8, year, month, date) };
(unsafe { strcmp(buf.as_mut_ptr() as *const u8, (*dmi).driver_data as *const u8) }) < 0
}

static sysids__2379: __SlateAlign16<[dmi_system_id; 7]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [76u8, 69u8, 78u8, 79u8, 86u8, 79u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(8u8);
__t3.substr = [84u8, 104u8, 105u8, 110u8, 107u8, 80u8, 97u8, 100u8, 32u8, 88u8, 50u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = (b"20180406\0".as_ptr() as *mut u8) as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(6u8);
__t6.substr = [76u8, 69u8, 78u8, 79u8, 86u8, 79u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(8u8);
__t7.substr = [84u8, 104u8, 105u8, 110u8, 107u8, 80u8, 97u8, 100u8, 32u8, 76u8, 52u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4.driver_data = (b"20180420\0".as_ptr() as *mut u8) as *mut ();
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8.matches = {
let mut __t9: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t9[0] = {
let mut __t10: dmi_strmatch = unsafe { core::mem::zeroed() };
__t10.__slate_bits_0.__set_slot(6u8);
__t10.substr = [76u8, 69u8, 78u8, 79u8, 86u8, 79u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10
};
__t9[1] = {
let mut __t11: dmi_strmatch = unsafe { core::mem::zeroed() };
__t11.__slate_bits_0.__set_slot(8u8);
__t11.substr = [84u8, 104u8, 105u8, 110u8, 107u8, 80u8, 97u8, 100u8, 32u8, 84u8, 52u8, 53u8, 48u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
};
__t9
};
__t8.driver_data = (b"20180315\0".as_ptr() as *mut u8) as *mut ();
__t8
}, {
let mut __t12: dmi_system_id = unsafe { core::mem::zeroed() };
__t12.matches = {
let mut __t13: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t13[0] = {
let mut __t14: dmi_strmatch = unsafe { core::mem::zeroed() };
__t14.__slate_bits_0.__set_slot(6u8);
__t14.substr = [76u8, 69u8, 78u8, 79u8, 86u8, 79u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14
};
__t13[1] = {
let mut __t15: dmi_strmatch = unsafe { core::mem::zeroed() };
__t15.__slate_bits_0.__set_slot(8u8);
__t15.substr = [84u8, 104u8, 105u8, 110u8, 107u8, 80u8, 97u8, 100u8, 32u8, 87u8, 53u8, 52u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15
};
__t13
};
__t12.driver_data = (b"20180409\0".as_ptr() as *mut u8) as *mut ();
__t12
}, {
let mut __t16: dmi_system_id = unsafe { core::mem::zeroed() };
__t16.matches = {
let mut __t17: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t17[0] = {
let mut __t18: dmi_strmatch = unsafe { core::mem::zeroed() };
__t18.__slate_bits_0.__set_slot(6u8);
__t18.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 101u8, 75u8, 32u8, 67u8, 79u8, 77u8, 80u8, 85u8, 84u8, 69u8, 82u8, 32u8, 73u8, 78u8, 67u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t18
};
__t17[1] = {
let mut __t19: dmi_strmatch = unsafe { core::mem::zeroed() };
__t19.__slate_bits_0.__set_slot(7u8);
__t19.substr = [65u8, 83u8, 85u8, 83u8, 80u8, 82u8, 79u8, 32u8, 68u8, 56u8, 52u8, 48u8, 77u8, 66u8, 95u8, 77u8, 56u8, 52u8, 48u8, 83u8, 65u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t19
};
__t17
};
__t16
}, {
let mut __t20: dmi_system_id = unsafe { core::mem::zeroed() };
__t20.matches = {
let mut __t21: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t21[0] = {
let mut __t22: dmi_strmatch = unsafe { core::mem::zeroed() };
__t22.__slate_bits_0.__set_slot(13u8);
__t22.substr = [65u8, 83u8, 85u8, 83u8, 84u8, 101u8, 75u8, 32u8, 67u8, 79u8, 77u8, 80u8, 85u8, 84u8, 69u8, 82u8, 32u8, 73u8, 78u8, 67u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t22
};
__t21[1] = {
let mut __t23: dmi_strmatch = unsafe { core::mem::zeroed() };
__t23.__slate_bits_0.__set_slot(14u8);
__t23.substr = [82u8, 79u8, 71u8, 32u8, 83u8, 84u8, 82u8, 73u8, 88u8, 32u8, 66u8, 53u8, 53u8, 48u8, 45u8, 70u8, 32u8, 71u8, 65u8, 77u8, 73u8, 78u8, 71u8, 32u8, 40u8, 87u8, 73u8, 45u8, 70u8, 73u8, 41u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t23
};
__t21
};
__t20
}, {
let mut __t24: dmi_system_id = unsafe { core::mem::zeroed() };
__t24
}]);

fn ahci_broken_online(pdev: *mut pci_dev) -> bool {
// There are several gigabyte boards which use
// SIMG5723s configured as hardware RAID.  Certain
// 5723 firmware revisions shipped there keep the link
// online but fail to answer properly to SRST or
// IDENTIFY when no device is attached downstream
// causing libata to retry quite a few times leading
// to excessive detection delay.
//
// As these firmwares respond to the second reset try
// with invalid device signature, considering unknown
// sig as offline works around the problem acceptably.
// terminate list
let dmi: *const dmi_system_id = unsafe { dmi_first_match(core::ptr::addr_of!(sysids__2453.0) as *const dmi_system_id) };
let mut val: u32 = 0;
if dmi.is_null() {
return false;
}
val = ((unsafe { (*dmi).driver_data }) as u64) as u32;
(((unsafe { (*(*pdev).bus).number }) as i32) as u32) == val >> 8i32 && (unsafe { (*pdev).devfn }) == val & 255
}

static sysids__2453: __SlateAlign16<[dmi_system_id; 3]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"EP45-DQ6\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(13u8);
__t2.substr = [71u8, 105u8, 103u8, 97u8, 98u8, 121u8, 116u8, 101u8, 32u8, 84u8, 101u8, 99u8, 104u8, 110u8, 111u8, 108u8, 111u8, 103u8, 121u8, 32u8, 67u8, 111u8, 46u8, 44u8, 32u8, 76u8, 116u8, 100u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(14u8);
__t3.substr = [69u8, 80u8, 52u8, 53u8, 45u8, 68u8, 81u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = 2560u64 as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.ident = (b"EP45-DS5\0".as_ptr() as *mut u8) as *const u8;
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(13u8);
__t6.substr = [71u8, 105u8, 103u8, 97u8, 98u8, 121u8, 116u8, 101u8, 32u8, 84u8, 101u8, 99u8, 104u8, 110u8, 111u8, 108u8, 111u8, 103u8, 121u8, 32u8, 67u8, 111u8, 46u8, 44u8, 32u8, 76u8, 116u8, 100u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(14u8);
__t7.substr = [69u8, 80u8, 52u8, 53u8, 45u8, 68u8, 83u8, 53u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4.driver_data = 768u64 as *mut ();
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8
}]);

fn ahci_gtf_filter_workaround(host: *mut ata_host) {
// Aspire 3810T issues a bunch of SATA enable commands
// via _GTF including an invalid one and one which is
// rejected by the device.  Among the successful ones
// is FPDMA non-zero offset enable which when enabled
// only on the drive side leads to NCQ command
// failures.  Filter it out.
let dmi: *const dmi_system_id = unsafe { dmi_first_match(core::ptr::addr_of!(sysids__2502.0) as *const dmi_system_id) };
let mut filter: u32 = 0;
let mut i: i32 = 0;
if dmi.is_null() {
return;
}
filter = ((unsafe { (*dmi).driver_data }) as u64) as u32;
while false {
}
unsafe { _dev_info((*host).dev as *const device, c"applying extra ACPI _GTF filter 0x%x for %s\n".as_ptr() as *const u8, filter, (*dmi).ident) };
while (i as u32) < unsafe { (*host).n_ports } {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) };
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
dev = unsafe { ata_dev_next(core::ptr::null_mut::<ata_device>(), link, 2u32) };
while !dev.is_null() {
unsafe {
(*dev).gtf_filter = (*dev).gtf_filter | filter;
}
dev = unsafe { ata_dev_next(dev, link, 2u32) };
}
link = unsafe { ata_link_next(link, ap, 0u32) };
}
i = i.wrapping_add(1);
}
}

static sysids__2502: __SlateAlign16<[dmi_system_id; 2]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"Aspire 3810T\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [65u8, 99u8, 101u8, 114u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [65u8, 115u8, 112u8, 105u8, 114u8, 101u8, 32u8, 51u8, 56u8, 49u8, 48u8, 84u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0.driver_data = 8i32 as *mut ();
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4
}]);

/// On the Acer Aspire Switch Alpha 12, sometimes all SATA ports are detected
/// as DUMMY, or detected but eventually get a "link down" and never get up
/// again. When this happens, CAP.NP may hold a value of 0x00 or 0x01, and the
/// port_map may hold a value of 0x00.
///
/// Overriding CAP.NP to 0x02 and the port_map to 0x7 will reveal all 3 ports
/// and can significantly reduce the occurrence of the problem.
///
/// https://bugzilla.kernel.org/show_bug.cgi?id=189471
fn acer_sa5_271_workaround(hpriv: *mut ahci_host_priv, pdev: *mut pci_dev) {
if (unsafe { dmi_check_system(core::ptr::addr_of!(sysids__2561.0) as *const dmi_system_id) }) != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"enabling Acer Switch Alpha 12 workaround\n".as_ptr() as *const u8) };
if (unsafe { (*hpriv).saved_cap }) & 3342139136 == 3342139136 {
unsafe {
(*hpriv).port_map = 7;
}
unsafe {
(*hpriv).cap = 3342139138;
}
}
}
}

static sysids__2561: __SlateAlign16<[dmi_system_id; 2]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"Acer Switch Alpha 12\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [65u8, 99u8, 101u8, 114u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [83u8, 119u8, 105u8, 116u8, 99u8, 104u8, 32u8, 83u8, 65u8, 53u8, 45u8, 50u8, 55u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4
}]);

/// The Marvell 88SE6111/6121/6145 ("Thor") family stops reporting interrupts
/// for a port when HOST_IRQ_STAT is cleared while PxIS still holds bits: PxIS
/// keeps its content, HOST_IRQ_STAT reads back as 0, the port is never looked
/// at again and the command in flight only ends in a timeout.  On a 88SE6121
/// this makes every SATA-2 or SATA-3 disk fail to IDENTIFY, while SATA-1 disks
/// happen to win the race often enough to work.
///
/// Clearing the host status before servicing the ports avoids it.  Marvell's
/// own driver for these chips does the same and says so ("clear global before
/// channel"), and ahci_xgene handles its broken edge latch the same way.  The
/// price is at most one spurious interrupt per valid one, which is why this is
/// not the generic behaviour - see AHCI 1.1 section 10.6.2.
///
/// Link: https://bugzilla.kernel.org/show_bug.cgi?id=216094
#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_mv_irq_handler")]
extern "C" fn ahci_mv_irq_handler(irq: i32, dev_instance: *mut ()) -> u32 {
let host: *mut ata_host = dev_instance as *mut ata_host;
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut rc: u32 = 0;
let irq_stat__0: u32 = readl(((unsafe { (mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ()) as *const ());
let mut irq_masked: u32 = 0;
if irq_stat__0 == 0 {
return 0;
}
irq_masked = irq_stat__0 & unsafe { (*hpriv).port_map };
spin_lock(unsafe { core::ptr::addr_of_mut!((*host).lock) });
// Use the unmasked value to clear the interrupt, as a spurious pending
// event on a dummy port might cause a screaming IRQ.
writel(irq_stat__0, (unsafe { (mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
rc = unsafe { ahci_handle_port_intr(host, irq_masked) };
spin_unlock(unsafe { core::ptr::addr_of_mut!((*host).lock) });
IRQ_RETVAL!(rc != 0) as u32
}

fn ahci_remap_check(pdev: *mut pci_dev, bar: i32, hpriv: *mut ahci_host_priv) {
let mut cap: u32 = 0;
// Check if this device might have remapped nvme devices.
let __v0: bool = ((unsafe { (*pdev).vendor }) as i32) != 32902 || pci_resource_len!(pdev, bar) < 524288;
let __v2: bool = __v0 || bar != 5 || readl(((unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(164 as isize) }) as *mut ()) as *const ()) & 1 == 0;
if __v2 {
return;
}
cap = readq(((unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(2048 as isize) }) as *mut ()) as *const ()) as u32;
for i in 0i32..3i32 {
if cap & ((1i32 << i) as u32) != 0 {
if readl(((unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(ahci_remap_dcc(i) as isize) }) as *mut ()) as *const ()) == 67586 {
// We've found a remapped device
let __v5: u32 = unsafe { (*hpriv).remapped_nvme };
unsafe {
(*hpriv).remapped_nvme = __v5.wrapping_add(1);
}
}
}
}
if (unsafe { (*hpriv).remapped_nvme }) == 0 {
return;
}
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Found %u remapped NVMe devices.\n".as_ptr() as *const u8, (*hpriv).remapped_nvme) };
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"Switch your BIOS from RAID to AHCI mode to use them.\n".as_ptr() as *const u8) };
// Don't rely on the msi-x capability in the remap case,
// share the legacy interrupt across ahci and remapped devices.
unsafe {
(*hpriv).flags = (*hpriv).flags | 32;
}
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_get_irq_vector")]
extern "C" fn ahci_get_irq_vector(host: *mut ata_host, port: i32) -> i32 {
unsafe { pci_irq_vector(to_pci_dev!(unsafe { (*host).dev }), port as u32) }
}

fn ahci_init_irq(pdev: *mut pci_dev, n_ports: u32, hpriv: *mut ahci_host_priv) {
let mut nvec: i32 = 0;
if (unsafe { (*hpriv).flags }) & 32 != 0 {
unsafe { pci_alloc_irq_vectors(pdev, 1u32, 1u32, 1u32) };
return;
}
// If number of MSIs is less than number of ports then Sharing Last
// Message mode could be enforced. In this case assume that advantage
// of multiple MSIs is negated and use single MSI mode instead.
if n_ports > 1 {
nvec = unsafe { pci_alloc_irq_vectors(pdev, n_ports, 2147483647u32, 6u32) };
if nvec > 0 {
if readl(((unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ()) & 4 == 0 {
unsafe {
(*hpriv).get_irq_vector = Some(ahci_get_irq_vector);
}
unsafe {
(*hpriv).flags = (*hpriv).flags | 1048576;
}
return;
}
// Fallback to single MSI mode if the controller
// enforced MRSM mode.
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016ahci: MRSM is on, fallback to single MSI\n".as_ptr() as *const u8) };
unsafe { pci_free_irq_vectors(pdev) };
}
}
// If the host is not capable of supporting per-port vectors, fall
// back to single MSI before finally attempting single MSI-X or
// a legacy INTx.
nvec = unsafe { pci_alloc_irq_vectors(pdev, 1u32, 1u32, 2u32) };
if nvec == 1 {
return;
}
unsafe { pci_alloc_irq_vectors(pdev, 1u32, 1u32, 5u32) };
}

fn ahci_mark_external_port(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut tmp: u32 = 0;
// Mark external ports (hotplug-capable, eSATA), unless we were asked to
// ignore this feature.
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
if tmp & 2097152 != 0 && (unsafe { (*hpriv).cap }) & 32 != 0 || tmp & 262144 != 0 {
if (unsafe { (*hpriv).mask_port_ext }) & 1u32 << unsafe { (*ap).port_no } != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016ata%u: Ignoring external/hotplug capability\n".as_ptr() as *const u8, (*ap).print_id) };
return;
}
unsafe {
(*ap).pflags = (*ap).pflags | 4194304;
}
}
}

fn ahci_update_initial_lpm_policy(ap: *mut ata_port) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut policy: i32 = unsafe { core::mem::zeroed() };
let mut hpriv: *mut ahci_host_priv = unsafe { core::mem::zeroed() };
unsafe {
hpriv = (*(*ap).host).private_data as *mut ahci_host_priv;
policy = 3;
// AHCI contains a known incompatibility between LPM and hot-plug
// removal events, see 7.3.1 Hot Plug Removal Detection and Power
// Management Interaction in AHCI 1.3.1. Therefore, do not enable
// LPM if the port advertises itself as an external port.
if (*ap).pflags & 4194304 != 0 {
(*ap).flags = (*ap).flags | 4;
(*ap).target_lpm_policy = 1;
return;
} else {
// If no Partial or no Slumber, we cannot support DIPM.
if (*(*ap).host).flags & 16 != 0 || (*(*ap).host).flags & 32 != 0 {
(*ap).flags = (*ap).flags | 8388608;
}
// If no LPM states are supported by the HBA, do not bother with LPM
if (*(*ap).host).flags & 16 != 0 && (*(*ap).host).flags & 32 != 0 && (*(*ap).host).flags & 64 != 0 {
(*ap).flags = (*ap).flags | 4;
(*ap).target_lpm_policy = 1;
return;
} else {
// user modified policy via module param
if mobile_lpm_policy != 1i32.wrapping_neg() {
policy = mobile_lpm_policy;
} else {
if policy > 2 {
__v0 = pm_suspend_default_s2idle();
} else {
__v0 = false;
}
if __v0 {
if (*hpriv).cap & 8192 != 0 {
policy = 4;
} else {
if (*hpriv).cap & 16384 != 0 {
policy = 5;
}
}
}
}
if policy >= 0 && policy <= 5 {
(*ap).target_lpm_policy = policy as u32;
}
}
}
}
}

fn ahci_intel_pcs_quirk(pdev: *mut pci_dev, hpriv: *mut ahci_host_priv) {
let mut tmp16: u16 = 0;
if (unsafe { (*hpriv).flags }) & 268435456 == 0 {
return;
}
// port_map is determined from PORTS_IMPL PCI register which is
// implemented as write or write-once register.  If the register
// isn't programmed, ahci automatically generates it from number
// of ports, which is good enough for PCS programming. It is
// otherwise expected that platform firmware enables the ports
// before the OS boots.
unsafe { pci_read_config_word(pdev as *const pci_dev, 146i32, core::ptr::addr_of_mut!(tmp16)) };
if ((tmp16 as i32) as u32) & unsafe { (*hpriv).port_map } != unsafe { (*hpriv).port_map } {
tmp16 = (((tmp16 as i32) as u32) | unsafe { (*hpriv).port_map }) as u16;
unsafe { pci_write_config_word(pdev as *const pci_dev, 146i32, tmp16) };
}
}

#[unsafe(link_section = ".text.slate_distinct.ahci.remapped_nvme_show")]
extern "C" fn remapped_nvme_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let host: *mut ata_host = dev_get_drvdata(dev as *const device) as *mut ata_host;
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (*hpriv).remapped_nvme) }) as i64
}

static mut dev_attr_remapped_nvme: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"remapped_nvme\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_ahci_c_2862>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_2862_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_2862_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_2862_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_2862_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ahci_c_2862_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(remapped_nvme_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

fn ahci_validate_bar_size(pdev: *mut pci_dev, bar: i32, hpriv: *mut ahci_host_priv) -> i32 {
let cap: u32 = readl(((unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(0 as isize) }) as *mut ()) as *const ());
let max_ports: u32 = ahci_nr_ports(cap) as u32;
let last_port_end: u32 = 256u32.wrapping_add(max_ports.wrapping_mul(128));
let mut bar_size: u64 = pci_resource_len!(pdev, bar);
if (last_port_end as u64) > bar_size {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"BAR%d too small for %u ports (last port ends at %#x, BAR %pa)\n".as_ptr() as *const u8, bar, max_ports, last_port_end, core::ptr::addr_of_mut!(bar_size)) };
return 19i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_shutdown_one")]
extern "C" fn ahci_shutdown_one(pdev: *mut pci_dev) {
unsafe { ata_pci_shutdown_one(pdev) };
}

#[unsafe(link_section = ".text.slate_distinct.ahci.ahci_remove_one")]
extern "C" fn ahci_remove_one(pdev: *mut pci_dev) {
unsafe { sysfs_remove_file_from_group(core::ptr::addr_of_mut!((*pdev).dev.kobj), core::ptr::addr_of_mut!(dev_attr_remapped_nvme.attr) as *const attribute, core::ptr::null::<u8>()) };
pm_runtime_get_noresume(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
unsafe { ata_pci_remove_one(pdev) };
}

#[unsafe(link_section = ".init.text")]
extern "C" fn ahci_pci_driver_init() -> i32 {
unsafe { __pci_register_driver(core::ptr::addr_of_mut!(ahci_pci_driver), core::ptr::null_mut::<module>(), c"ahci".as_ptr() as *const u8) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_pci_driver_init_687: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(ahci_pci_driver_init)) };

#[unsafe(link_section = ".exit.text")]
extern "C" fn ahci_pci_driver_exit() {
unsafe { pci_unregister_driver(core::ptr::addr_of_mut!(ahci_pci_driver)) };
}

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_ahci_pci_driver_exit: Option<unsafe extern "C" fn()> = Some(ahci_pci_driver_exit);

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_688: [u8; 24] = [97, 104, 99, 105, 46, 97, 117, 116, 104, 111, 114, 61, 74, 101, 102, 102, 32, 71, 97, 114, 122, 105, 107, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_689: [u8; 44] = [97, 104, 99, 105, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 65, 72, 67, 73, 32, 83, 65, 84, 65, 32, 108, 111, 119, 45, 108, 101, 118, 101, 108, 32, 100, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_690: [u8; 27] = [97, 104, 99, 105, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 97, 116, 97, 47, 97, 104, 99, 105, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_691: [u8; 17] = [97, 104, 99, 105, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_692: [u8; 17] = [97, 104, 99, 105, 46, 118, 101, 114, 115, 105, 111, 110, 61, 51, 46, 48, 0];

#[used(compiler)]
#[unsafe(link_section = "__modver")]
static __modver_attr: __SlateConst<module_version_attribute> = __SlateConst(module_version_attribute { mattr: {
let mut __t0: module_attribute = unsafe { core::mem::zeroed() };
__t0.attr = {
let mut __t1: attribute = unsafe { core::mem::zeroed() };
__t1.name = (b"version\0".as_ptr() as *mut u8) as *const u8;
__t1.mode = 292u16;
__t1
};
__t0.show = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>>(__modver_version_show as *const ()) };
__t0
}, module_name: (b"ahci\0".as_ptr() as *mut u8) as *const u8, version: (b"3.0\0".as_ptr() as *mut u8) as *const u8 });

static __slate_asm_string_ahci_init_one_62: [u8; 1] = *b"\0";

static __slate_asm_string_ahci_init_one_63: [u8; 48] = *b"/home/takashi/c-corpus/linux/drivers/ata/ahci.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
