
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
//! scan.c - support for transforming the ACPI namespace into individual objects
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
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
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
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
//! restart by calling sys_restart_syscall
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
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! SCHED_ISO: reserved but not implemented yet
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! 0xFFFFFFFF
//! Multiple APIC Description Table
//! Serial Port Console Redirection table
//! Quirk for broken IBM BIOSes
//! See the comment in asm-generic/io.h about ioremap_np().
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! 8237 DMA controllers
//! We are using top down, so it is safe to use 0 here
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-weak-function-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-weak-variable-attribute
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
// Invalid system call number
// No data available
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// Driver requests probe retry
// Operation is not supported
// strscpy - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer (optional)
//
// Copy the source string @src, or as much of it as fits, into the
// destination @dst buffer. The behavior is undefined if the string
// buffers overlap. The destination @dst buffer is always NUL terminated,
// unless it's zero-sized.
//
// The size argument @... is only required when @dst is not an array, or
// when the copy needs to be smaller than sizeof(@dst).
//
// Preferred to strncpy() since it always returns a valid string, and
// doesn't unnecessarily force the tail of the destination buffer to be
// zero padded. If padding is desired please use strscpy_pad().
//
// Returns the number of characters copied in @dst (not including the
// trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
// truncated.
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// min_not_zero - return the minimum that is _not_ zero, unless both are zero
// @x: value1
// @y: value2
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Environmental exceptions
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// DMA_ATTR_CC_SHARED: Indicates the DMA mapping is shared (decrypted) for
// confidential computing guests. For normal system memory the caller must have
// called set_memory_decrypted(), and pgprot_decrypted must be used when
// creating CPU PTEs for the mapping. The same shared semantic may be passed
// to the vIOMMU when it sets up the IOPTE. For MMIO use together with
// DMA_ATTR_MMIO to indicate shared MMIO. Unless DMA_ATTR_MMIO is provided
// a struct page is required.
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
macro_rules! ACPI_CAST_PTR {
($t:ty, $p:expr) => {
(($p) as u64) as *mut $t
};
}

macro_rules! ACPI_CAST_PTR__2 {
($p:expr) => {
(($p) as u64) as *mut ()
};
}

macro_rules! ACPI_COMPANION_SET {
($dev:expr, $adev:expr) => {
let __v: *mut fwnode_handle = if !$adev.is_null() {
acpi_fwnode_handle($adev)
} else {
core::ptr::null_mut::<fwnode_handle>()
};
unsafe { set_primary_fwnode($dev, __v) };
};
}

macro_rules! ACPI_COMPANION {
($dev:expr) => {
to_acpi_device_node!(unsafe { (*$dev).fwnode })
};
}

macro_rules! ACPI_COMPARE_NAMESEG {
($a:expr, $b:expr) => {
(unsafe { *ACPI_CAST_PTR!(u32, $a) }) == unsafe { *ACPI_CAST_PTR!(u32, $b) }
};
}

macro_rules! ACPI_FAILURE {
($a:expr) => {
$a
};
}

macro_rules! ACPI_FREE {
($a:expr) => {
acpi_os_free(($a) as *mut ());
};
}

macro_rules! ACPI_FREE__2 {
($a:expr) => {
acpi_os_free($a);
};
}

/// Success is always zero, failure is non-zero
macro_rules! ACPI_SUCCESS {
($a:expr) => {
($a) == 0
};
}

macro_rules! ACPI_TO_POINTER {
($i:expr) => {
ACPI_CAST_PTR__2!(($i) as u64)
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

macro_rules! EXCEP_CTL {
($code:expr) => {
(($code) | 16384) as u32
};
}

/// Macros to insert the exception code classes
macro_rules! EXCEP_ENV {
($code:expr) => {
(($code) | 0) as u32
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

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! XA_FLAGS_MARK {
($mark:expr) => {
16777216u32 << (($mark) as u32)
};
}

macro_rules! ZERO_PAGE {
($vaddr:expr) => {
_zero_page(($vaddr) as u64)
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

macro_rules! __must_be_array__4 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1188>() as i32
};
}

macro_rules! __must_be_array__17 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1191>() as i32
};
}

macro_rules! __must_be_array__19 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1179>() as i32
};
}

macro_rules! __must_be_array__33 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1185>() as i32
};
}

macro_rules! __must_be_array__38 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1202>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__6 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1202_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__7 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1185_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__12 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1191_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__13 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1202_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__18 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1179_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__22 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1188_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__23 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1191_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__25 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1185_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__32 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1179_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__45 {
() => {
core::mem::size_of::<__SlateRecord_scan_c_1188_3>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
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
macro_rules! container_of__16 {
($ptr:expr) => {
{
let mut __v: *mut acpi_dep_data;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__48!() as isize)) }) as *mut ()) as *mut acpi_dep_data;
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
($ptr:expr) => {
{
let mut __v: *mut resource_entry;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut resource_entry;
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
macro_rules! container_of__27 {
($ptr:expr) => {
{
let mut __v: *mut acpi_scan_system_dev;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__41!() as isize)) }) as *mut ()) as *mut acpi_scan_system_dev;
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
macro_rules! container_of__30 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__50!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr) => {
{
let mut __v: *mut acpi_hardware_id;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!() as isize)) }) as *mut ()) as *mut acpi_hardware_id;
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
macro_rules! container_of__41 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__35!($type) as isize)) }) as *mut ()) as *mut $type;
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
let mut __v: *mut acpi_scan_handler;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__26!() as isize)) }) as *mut ()) as *mut acpi_scan_handler;
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
let mut __v: *mut acpi_device_bus_id;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__14!() as isize)) }) as *mut ()) as *mut acpi_device_bus_id;
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
macro_rules! container_of__6 {
($ptr:expr) => {
{
let mut __v: *mut acpi_device_physical_node;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut acpi_device_physical_node;
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
macro_rules! list_entry__14 {
($ptr:expr) => {
container_of__43!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__17 {
($ptr:expr) => {
container_of__16!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of__3!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__35 {
($ptr:expr, $type:ty) => {
container_of__41!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__36 {
($ptr:expr, $type:ty) => {
container_of__30!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr) => {
container_of__5!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__26 {
($ptr:expr) => {
container_of__27!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr) => {
container_of__6!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr) => {
container_of__11!($ptr)
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
macro_rules! list_first_entry__10 {
($ptr:expr) => {
list_entry__17!(unsafe { (*$ptr).next })
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
list_entry__26!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__28 {
($ptr:expr) => {
list_entry__14!(unsafe { (*$ptr).next })
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

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__3!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__18 {
($ptr:expr) => {
list_entry__9!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__4 {
($ptr:expr) => {
list_entry__2!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__34 {
($ptr:expr, $type:ty) => {
list_entry__35!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__15 {
($pos:expr, $member:ident) => {
list_entry__9!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__2!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__10 {
($pos:expr, $member:ident) => {
list_entry__17!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__3!(unsafe { (*$pos).$member.next })
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

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__32 {
($pos:expr, $member:ident) => {
list_entry__14!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__25 {
($pos:expr, $member:ident) => {
list_entry__26!(unsafe { (*$pos).$member.next })
};
}

macro_rules! mutex_lock_nested {
($lock:expr) => {
unsafe { mutex_lock($lock) };
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__2 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fwnode) as u64
};
}

macro_rules! offsetof__26 {
() => {
core::mem::offset_of!(acpi_scan_handler, list_node) as u64
};
}

macro_rules! offsetof__35 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, del_list) as u64
};
}

macro_rules! offsetof__41 {
() => {
core::mem::offset_of!(acpi_scan_system_dev, node) as u64
};
}

macro_rules! offsetof__4 {
() => {
core::mem::offset_of!(acpi_device_physical_node, node) as u64
};
}

macro_rules! offsetof__48 {
() => {
core::mem::offset_of!(acpi_dep_data, node) as u64
};
}

macro_rules! offsetof__50 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, list) as u64
};
}

macro_rules! offsetof__14 {
() => {
core::mem::offset_of!(acpi_device_bus_id, node) as u64
};
}

macro_rules! offsetof__5 {
() => {
core::mem::offset_of!(acpi_hardware_id, list) as u64
};
}

macro_rules! offsetof__6 {
() => {
core::mem::offset_of!(resource_entry, node) as u64
};
}

macro_rules! request_mem_region__2 {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(iomem_resource), $start, $n, $name, 0i32) }
};
}

/// Convenience shorthand with allocation
macro_rules! request_region__2 {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(ioport_resource), $start, $n, $name, 0i32) }
};
}

macro_rules! test_bit__5 {
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

macro_rules! to_acpi_device__2 {
($d:expr) => {
{
let mut __v: *mut acpi_device;
__v = ((unsafe { (($d) as *const u8).wrapping_offset(-(offsetof!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v
}
};
}

macro_rules! to_acpi_device {
($d:expr) => {
{
let mut __v: *mut acpi_device;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v
}
};
}

macro_rules! to_acpi_device_node {
($__fwnode:expr) => {
{
let mut __v: *mut acpi_device;
let __v2: *mut fwnode_handle = $__fwnode;
let __v3: *mut acpi_device = if unsafe { is_acpi_device_node(__v2 as *const fwnode_handle) } {
let mut __v4: *mut acpi_device;
__v4 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__2!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v4
} else {
core::ptr::null_mut::<acpi_device>()
};
__v = __v3;
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

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut ioport_resource: resource;
    static mut iomem_resource: resource;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut acpi_gbl_reduced_hardware: u8;
    static mut acpi_gbl_FADT: acpi_table_fadt;
    static mut acpi_device_fwnode_ops: fwnode_operations;
    static mut acpi_bus_type: bus_type;
    static mut acpi_disabled: i32;
    static mut __zero_page: *mut page;
    static mut iommu_probe_device_lock: mutex;
    static mut dma_dummy_ops: dma_map_ops;
    static mut x86_apple_machine: bool;
    static mut acpi_root: *mut acpi_device;
    static mut acpi_groups: [*const attribute_group; 0];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn kfree_const(arg0: *const ());
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn kstrdup_const(arg0: *const u8, arg1: u32) -> *const u8;
    fn match_string(arg0: *const *const u8, arg1: u64, arg2: *const u8) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_46();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn blocking_notifier_chain_register(arg0: *mut blocking_notifier_head, arg1: *mut notifier_block) -> i32;
    fn blocking_notifier_chain_unregister(arg0: *mut blocking_notifier_head, arg1: *mut notifier_block) -> i32;
    fn blocking_notifier_call_chain(arg0: *mut blocking_notifier_head, arg1: u64, arg2: *mut ()) -> i32;
    fn ida_alloc_range(arg0: *mut ida, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn ida_free(arg0: *mut ida, arg1: u32);
    fn kobject_uevent(arg0: *mut kobject, arg1: u32) -> i32;
    fn kobject_uevent_env(arg0: *mut kobject, arg1: u32, arg2: *mut *mut u8) -> i32;
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn device_initialize(arg0: *mut device);
    fn device_add(arg0: *mut device) -> i32;
    fn device_del(arg0: *mut device);
    fn lock_device_hotplug();
    fn unlock_device_hotplug();
    fn device_offline(arg0: *mut device) -> i32;
    fn device_online(arg0: *mut device) -> i32;
    fn set_primary_fwnode(arg0: *mut device, arg1: *mut fwnode_handle);
    fn get_device(arg0: *mut device) -> *mut device;
    fn put_device(arg0: *mut device);
    fn async_schedule_dev_nocall(arg0: Option<unsafe extern "C" fn(*mut (), u64)>, arg1: *mut device) -> bool;
    fn auxiliary_device_init(arg0: *mut auxiliary_device) -> i32;
    fn __auxiliary_device_add(arg0: *mut auxiliary_device, arg1: *const u8) -> i32;
    fn fwnode_property_present(arg0: *const fwnode_handle, arg1: *const u8) -> bool;
    fn acpi_format_exception(arg0: u32) -> *const u8;
    fn acpi_get_table(arg0: *mut u8, arg1: u32, arg2: *mut *mut acpi_table_header) -> u32;
    fn acpi_put_table(arg0: *mut acpi_table_header);
    fn acpi_walk_namespace(arg0: u32, arg1: *mut (), arg2: u32, arg3: Option<unsafe extern "C" fn(*mut (), u32, *mut (), *mut *mut ()) -> u32>, arg4: Option<unsafe extern "C" fn(*mut (), u32, *mut (), *mut *mut ()) -> u32>, arg5: *mut (), arg6: *mut *mut ()) -> u32;
    fn acpi_get_name(arg0: *mut (), arg1: u32, arg2: *mut acpi_buffer) -> u32;
    fn acpi_get_handle(arg0: *mut (), arg1: *const u8, arg2: *mut *mut ()) -> u32;
    fn acpi_attach_data(arg0: *mut (), arg1: Option<unsafe extern "C" fn(*mut (), *mut ())>, arg2: *mut ()) -> u32;
    fn acpi_detach_data(arg0: *mut (), arg1: Option<unsafe extern "C" fn(*mut (), *mut ())>) -> u32;
    fn acpi_evaluate_object(arg0: *mut (), arg1: *mut u8, arg2: *mut acpi_object_list, arg3: *mut acpi_buffer) -> u32;
    fn acpi_get_object_info(arg0: *mut (), arg1: *mut *mut acpi_device_info) -> u32;
    fn acpi_get_type(arg0: *mut (), arg1: *mut u32) -> u32;
    fn acpi_get_parent(arg0: *mut (), arg1: *mut *mut ()) -> u32;
    fn acpi_update_all_gpes() -> u32;
    fn acpi_setup_gpe_for_wake(arg0: *mut (), arg1: *mut (), arg2: u32) -> u32;
    fn acpi_walk_resources(arg0: *mut (), arg1: *mut u8, arg2: Option<unsafe extern "C" fn(*mut acpi_resource, *mut ()) -> u32>, arg3: *mut ()) -> u32;
    fn acpi_get_data_full(arg0: *mut (), arg1: Option<unsafe extern "C" fn(*mut (), *mut ())>, arg2: *mut *mut (), arg3: Option<unsafe extern "C" fn(*mut ())>) -> u32;
    fn acpi_evaluate_integer(arg0: *mut (), arg1: *mut u8, arg2: *mut acpi_object_list, arg3: *mut u64) -> u32;
    fn acpi_evaluate_reference(arg0: *mut (), arg1: *mut u8, arg2: *mut acpi_object_list, arg3: *mut acpi_handle_list) -> bool;
    fn acpi_handle_list_free(arg0: *mut acpi_handle_list);
    fn acpi_evaluate_ost(arg0: *mut (), arg1: u32, arg2: u32, arg3: *mut acpi_buffer) -> u32;
    fn acpi_has_method(arg0: *mut (), arg1: *mut u8) -> bool;
    fn acpi_evaluate_ej0(arg0: *mut ()) -> u32;
    fn acpi_evaluate_lck(arg0: *mut (), arg1: i32) -> u32;
    fn acpi_get_physical_device_location(arg0: *mut (), arg1: *mut *mut acpi_pld_info) -> bool;
    fn is_acpi_device_node(arg0: *const fwnode_handle) -> bool;
    fn acpi_dev_for_each_child(arg0: *mut acpi_device, arg1: Option<unsafe extern "C" fn(*mut acpi_device, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn acpi_dev_for_each_child_reverse(arg0: *mut acpi_device, arg1: Option<unsafe extern "C" fn(*mut acpi_device, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn acpi_bus_get_status(arg0: *mut acpi_device) -> i32;
    fn acpi_device_set_power(arg0: *mut acpi_device, arg1: i32) -> i32;
    fn acpi_bus_init_power(arg0: *mut acpi_device) -> i32;
    fn acpi_bus_get_primary_device(arg0: *mut acpi_device) -> *mut device;
    fn acpi_match_device_ids(arg0: *mut acpi_device, arg1: *const acpi_device_id) -> i32;
    fn acpi_table_parse(arg0: *mut u8, arg1: Option<unsafe extern "C" fn(*mut acpi_table_header) -> i32>) -> i32;
    fn acpi_table_parse_madt(arg0: u32, arg1: Option<unsafe extern "C" fn(*mut acpi_subtable_headers, u64) -> i32>, arg2: u32) -> i32;
    fn acpi_dev_is_video_device(arg0: *mut acpi_device) -> bool;
    fn acpi_dev_resource_memory(arg0: *mut acpi_resource, arg1: *mut resource) -> bool;
    fn acpi_dev_free_resource_list(arg0: *mut list_head);
    fn acpi_dev_get_resources(arg0: *mut acpi_device, arg1: *mut list_head, arg2: Option<unsafe extern "C" fn(*mut acpi_resource, *mut ()) -> i32>, arg3: *mut ()) -> i32;
    fn acpi_dev_get_dma_resources(arg0: *mut acpi_device, arg1: *mut list_head) -> i32;
    fn acpi_match_acpi_device(arg0: *const acpi_device_id, arg1: *const acpi_device) -> *const acpi_device_id;
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn acpi_handle_printk(arg0: *const u8, arg1: *mut (), arg2: *const u8, ...);
    fn iommu_fwspec_init(arg0: *mut device, arg1: *mut fwnode_handle) -> i32;
    fn iommu_fwspec_add_ids(arg0: *mut device, arg1: *const u32, arg2: i32) -> i32;
    fn dmi_name_in_vendors(arg0: *const u8) -> i32;
    fn crc32_le(arg0: u32, arg1: *const (), arg2: u64) -> u32;
    fn acpi_pci_root_init();
    fn acpi_pci_link_init();
    fn acpi_processor_init();
    fn acpi_platform_init();
    fn acpi_pnp_init();
    fn acpi_gpe_apply_masked_gpes();
    fn acpi_container_init();
    fn acpi_memory_hotplug_init();
    fn register_dock_dependent_device(arg0: *mut acpi_device, arg1: *mut ());
    fn dock_notify(arg0: *mut acpi_device, arg1: u32) -> i32;
    fn acpi_dock_add(arg0: *mut acpi_device);
    fn acpi_cmos_rtc_init();
    fn acpi_sysfs_add_hotplug_profile(arg0: *mut acpi_hotplug_profile, arg1: *const u8);
    fn acpi_lpss_init();
    fn acpi_apd_init();
    fn acpi_queue_hotplug_work(arg0: *mut work_struct) -> bool;
    fn acpi_device_setup_files(arg0: *mut acpi_device);
    fn acpi_device_remove_files(arg0: *mut acpi_device);
    fn acpi_power_resources_init();
    fn acpi_power_resources_list_free(arg0: *mut list_head);
    fn acpi_extract_power_resources(arg0: *mut acpi_object, arg1: u32, arg2: *mut list_head) -> i32;
    fn acpi_add_power_resource(arg0: *mut ()) -> *mut acpi_device;
    fn acpi_power_add_remove_device(arg0: *mut acpi_device, arg1: bool);
    fn acpi_power_wakeup_list_init(arg0: *mut list_head, arg1: *mut i32) -> i32;
    fn acpi_device_sleep_wake(arg0: *mut acpi_device, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn acpi_power_transition(arg0: *mut acpi_device, arg1: i32) -> i32;
    fn acpi_turn_off_unused_power_resources();
    fn acpi_ec_register_opregions(arg0: *mut acpi_device);
    fn acpi_init_properties(arg0: *mut acpi_device);
    fn acpi_free_properties(arg0: *mut acpi_device);
    fn acpi_init_lpit();
    fn acpi_mipi_check_crs_csi2(arg0: *mut ());
    fn acpi_mipi_scan_crs_csi2();
    fn acpi_mipi_init_crs_csi2_swnodes();
    fn acpi_mipi_crs_csi2_cleanup();
    fn __compiletime_assert_542();
}

unsafe extern "llvm-intrinsic" {
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_0() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
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
core::arch::asm!("31: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 31b - ., 3; .popsection\n# {0}", const 31 as i32, options(nostack, att_syntax));
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
core::arch::asm!("32: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 32b - ., 4; .popsection\n# {0}", const 32 as i32, options(nostack, att_syntax));
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
struct pollfd {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits121U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits122U0,
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
    __slate_bits_0: __slate_bits::__SlateBits123U0,
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
struct futex_private_hash {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct ucounts {
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
struct workqueue_struct {
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
struct rseq {
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
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct seq_file {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits272U0,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits68U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits68U1,
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
    __slate_bits_2: __slate_bits::__SlateBits68U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


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


fn mutex_destroy(lock: *mut mutex) {
}

// mutex_init - initialize the mutex
// @mutex: the mutex to be initialized
//
// Initialize the mutex to unlocked state.
//
// It is not allowed to initialize an already locked mutex.
fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
}

#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


fn __init_work(work__0: *mut work_struct, onstack: i32) {
}

fn queue_work(wq: *mut workqueue_struct, work__0: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work__0) }
}

fn schedule_work(work__0: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work__0)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
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
struct user_namespace {
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
    __slate_bits_0: __slate_bits::__SlateBits401U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
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
struct kernfs_root {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn xa_init_flags(xa: *mut xarray, flags: u32) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*xa).xa_lock) });
unsafe {
*core::ptr::addr_of_mut!((*xa).xa_lock) = *{
__slate_compound_0 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
unsafe {
(*xa).xa_flags = flags;
}
unsafe {
(*xa).xa_head = core::ptr::null_mut::<()>();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn xa_empty(xa: *const xarray) -> bool {
unsafe { (*xa).xa_head }.is_null()
}

// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct ida {
    xa: xarray,
}


fn ida_alloc(ida: *mut ida, gfp: u32) -> i32 {
unsafe { ida_alloc_range(ida, 0u32, 4294967295u32, gfp) }
}

fn ida_init(ida: *mut ida) {
xa_init_flags(unsafe { core::ptr::addr_of_mut!((*ida).xa) }, 1 | (4 | XA_FLAGS_MARK!(0u32)));
}

fn ida_is_empty(ida: *const ida) -> bool {
xa_empty(unsafe { core::ptr::addr_of!((*ida).xa) })
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
    __slate_bits_0: __slate_bits::__SlateBits452U0,
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits454U0,
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
// No address assigned yet
// Driver has marked this resource busy
fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


fn resource_type(res: *const resource) -> u64 {
(unsafe { (*res).flags }) & 7936
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits480U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits480U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits480U2,
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
    __slate_bits_3: __slate_bits::__SlateBits480U3,
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
    __slate_bits_0: __slate_bits::__SlateBits275U0,
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
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits609U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
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
struct msi_device_data {
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
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
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
struct __SlateRecord_auxiliary_bus_h_149 {
    irqs: xarray,
    lock: mutex,
    irq_dir_exists: bool,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_acrestyp_h_145 {
    interrupt: u8,
    __slate_anon_1: __SlateRecord_acrestyp_h_147,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_147 {
    __Empty_interrupts: __SlateRecord_acrestyp_h_147_2,
    interrupts: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_147_2 {
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
union __SlateRecord_acrestyp_h_156 {
    channel: u8,
    __slate_anon_1: __SlateRecord_acrestyp_h_158,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_158 {
    __Empty_channels: __SlateRecord_acrestyp_h_158_2,
    channels: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_158_2 {
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


fn dev_offline(dev: *const device) -> bool {
test_bit__5!(9i32, unsafe { (*dev).flags.as_ptr() as *const u64 }) != 0
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
    __slate_bits_0: __slate_bits::__SlateBits477U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_set_uevent_suppress(dev: *mut device, val: i32) {
unsafe {
(*dev).kobj.__slate_bits_0.__set_uevent_suppress(val as u32);
}
}

fn device_set_pm_not_required(dev: *mut device) {
unsafe {
let __t0 = true;
(*dev).power.__slate_bits_0.__set_no_pm(__t0);
}
unsafe {
let __t1 = true;
(*dev).power.__slate_bits_2.__set_no_callbacks(__t1);
}
}

fn device_supports_offline(dev: *mut device) -> bool {
!unsafe { (*dev).bus }.is_null() && (unsafe { (*(*dev).bus).offline }) != None && (unsafe { (*(*dev).bus).online }) != None
}

#[inline(always)]
fn __free_put_device(p: *mut ()) {
let _T: *mut device = unsafe { *(p as *mut *mut device) };
if !_T.is_null() {
unsafe { put_device(_T) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct auxiliary_device {
    dev: device,
    name: *const u8,
    id: u32,
    sysfs: __SlateRecord_auxiliary_bus_h_149,
    registration_data_rust: *mut (),
}


fn to_auxiliary_dev(dev: *mut device) -> *mut auxiliary_device {
container_of__4!(dev, auxiliary_device)
}

fn auxiliary_device_uninit(auxdev: *mut auxiliary_device) {
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*auxdev).sysfs.lock) });
unsafe { put_device(core::ptr::addr_of_mut!((*auxdev).dev)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct resource_entry {
    node: list_head,
    res: *mut resource,
    offset: u64,
    __res: resource,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
union __SlateRecord_acrestyp_h_341 {
    interrupt: u32,
    __slate_anon_1: __SlateRecord_acrestyp_h_343,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_343 {
    __Empty_interrupts: __SlateRecord_acrestyp_h_343_2,
    interrupts: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acrestyp_h_343_2 {
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


fn fwnode_init(fwnode: *mut fwnode_handle, ops: *const fwnode_operations) {
unsafe {
(*fwnode).secondary = core::ptr::null_mut::<fwnode_handle>();
}
unsafe {
(*fwnode).ops = ops;
}
unsafe {
(*fwnode).dev = core::ptr::null_mut::<device>();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*fwnode).consumers) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*fwnode).suppliers) });
unsafe {
(*fwnode).flags = 0;
}
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


// Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
// across operating systems or even the various UNIX systems. Since ACPICA
// only needs the thread ID as a unique thread identifier, we use a u64
// as the only common data type - it will accommodate any type of pointer or
// any type of integer. It is up to the host-dependent OSL to cast the
// native thread ID type to a u64 (in acpi_os_get_thread_id).
// Fixed by ACPI spec
// 4 chars for name + 1 char for separator
// Constants with special meanings
// Standard notify values
// Byte/Word/Dword/Zero/One/Ones
// byte_const, multiple data_term/Constant/super_name
// Name, multiple Node
// Name,byte_const,word_const,multi Node
// Name,byte_const,Dword_const,byte_const,multi nm_o
// Name, multiple Node
// Arg#, Local#, Name, Debug, ref_of, Index
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
struct acpi_object_list {
    count: u32,
    pointer: *mut acpi_object,
}


// Let ACPICA allocate buffer
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_buffer {
    length: u64,
    pointer: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_device_id {
    length: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_device_id_list {
    count: u32,
    list_size: u32,
    ids: [acpi_pnp_device_id; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_info {
    info_size: u32,
    name: u32,
    r#type: u32,
    param_count: u8,
    valid: u16,
    flags: u8,
    highest_dstates: [u8; 4],
    lowest_dstates: [u8; 5],
    address: u64,
    hardware_id: acpi_pnp_device_id,
    unique_id: acpi_pnp_device_id,
    class_code: acpi_pnp_device_id,
    compatible_id_list: acpi_pnp_device_id_list,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_generic_address {
    space_id: u8,
    bit_width: u8,
    bit_offset: u8,
    access_width: u8,
    address: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_table_fadt {
    header: acpi_table_header,
    facs: u32,
    dsdt: u32,
    model: u8,
    preferred_profile: u8,
    sci_interrupt: u16,
    smi_command: u32,
    acpi_enable: u8,
    acpi_disable: u8,
    s4_bios_request: u8,
    pstate_control: u8,
    pm1a_event_block: u32,
    pm1b_event_block: u32,
    pm1a_control_block: u32,
    pm1b_control_block: u32,
    pm2_control_block: u32,
    pm_timer_block: u32,
    gpe0_block: u32,
    gpe1_block: u32,
    pm1_event_length: u8,
    pm1_control_length: u8,
    pm2_control_length: u8,
    pm_timer_length: u8,
    gpe0_block_length: u8,
    gpe1_block_length: u8,
    gpe1_base: u8,
    cst_control: u8,
    c2_latency: u16,
    c3_latency: u16,
    flush_size: u16,
    flush_stride: u16,
    duty_offset: u8,
    duty_width: u8,
    day_alarm: u8,
    month_alarm: u8,
    century: u8,
    boot_flags: u16,
    reserved: u8,
    flags: u32,
    reset_register: acpi_generic_address,
    reset_value: u8,
    arm_boot_flags: u16,
    minor_revision: u8,
    Xfacs: u64,
    Xdsdt: u64,
    xpm1a_event_block: acpi_generic_address,
    xpm1b_event_block: acpi_generic_address,
    xpm1a_control_block: acpi_generic_address,
    xpm1b_control_block: acpi_generic_address,
    xpm2_control_block: acpi_generic_address,
    xpm_timer_block: acpi_generic_address,
    xgpe0_block: acpi_generic_address,
    xgpe1_block: acpi_generic_address,
    sleep_control: acpi_generic_address,
    sleep_status: acpi_generic_address,
    hypervisor_id: u64,
}


// 04: [V1] Power button is handled as a control method device
// 05: [V1] Sleep button is handled as a control method device
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_subtable_header {
    r#type: u8,
    length: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_cdat_header {
    r#type: u8,
    reserved: u8,
    length: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_cedt_header {
    r#type: u8,
    reserved: u8,
    length: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_hmat_structure {
    r#type: u16,
    reserved: u16,
    length: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_prmt_module_header {
    revision: u16,
    length: u16,
}


// Status Override table
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_table_spcr {
    header: acpi_table_header,
    interface_type: u8,
    reserved: [u8; 3],
    serial_port: acpi_generic_address,
    interrupt_type: u8,
    pc_interrupt: u8,
    interrupt: u32,
    baud_rate: u8,
    parity: u8,
    stop_bits: u8,
    flow_control: u8,
    terminal_type: u8,
    language: u8,
    pci_device_id: u16,
    pci_vendor_id: u16,
    pci_bus: u8,
    pci_device: u8,
    pci_function: u8,
    pci_flags: u32,
    pci_segment: u8,
    uart_clk_freq: u32,
    precise_baudrate: u32,
    name_space_string_length: u16,
    name_space_string_offset: u16,
    name_space_string: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_table_stao {
    header: acpi_table_header,
    ignore_uart: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_irq {
    descriptor_length: u8,
    triggering: u8,
    polarity: u8,
    shareable: u8,
    wake_capable: u8,
    interrupt_count: u8,
    __slate_anon_6: __SlateRecord_acrestyp_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_dma {
    r#type: u8,
    bus_master: u8,
    transfer: u8,
    channel_count: u8,
    __slate_anon_4: __SlateRecord_acrestyp_h_156,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_start_dependent {
    descriptor_length: u8,
    compatibility_priority: u8,
    performance_robustness: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_io {
    io_decode: u8,
    alignment: u8,
    address_length: u8,
    minimum: u16,
    maximum: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_fixed_io {
    address: u16,
    address_length: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_fixed_dma {
    request_lines: u16,
    channels: u16,
    width: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_vendor {
    byte_length: u16,
    byte_data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits831U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_vendor_typed {
    byte_length: u16,
    uuid_subtype: u8,
    uuid: [u8; 16],
    byte_data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_end_tag {
    checksum: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_memory24 {
    write_protect: u8,
    minimum: u16,
    maximum: u16,
    alignment: u16,
    address_length: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_memory32 {
    write_protect: u8,
    minimum: u32,
    maximum: u32,
    alignment: u32,
    address_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits836U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_fixed_memory32 {
    write_protect: u8,
    address: u32,
    address_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_memory_attribute {
    write_protect: u8,
    caching: u8,
    range_type: u8,
    translation: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_io_attribute {
    range_type: u8,
    translation: u8,
    translation_type: u8,
    reserved1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_resource_attribute {
    mem: acpi_memory_attribute,
    io: acpi_io_attribute,
    type_specific: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_label {
    string_length: u16,
    string_ptr: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_source {
    index: u8,
    string_length: u16,
    string_ptr: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_address16_attribute {
    granularity: u16,
    minimum: u16,
    maximum: u16,
    translation_offset: u16,
    address_length: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_address32_attribute {
    granularity: u32,
    minimum: u32,
    maximum: u32,
    translation_offset: u32,
    address_length: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_address64_attribute {
    granularity: u64,
    minimum: u64,
    maximum: u64,
    translation_offset: u64,
    address_length: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_address {
    resource_type: u8,
    producer_consumer: u8,
    decode: u8,
    min_address_fixed: u8,
    max_address_fixed: u8,
    info: acpi_resource_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_acpi_h_1441 {
    probe_table: Option<unsafe extern "C" fn(*mut acpi_table_header) -> i32>,
    probe_subtbl: Option<unsafe extern "C" fn(*mut acpi_subtable_headers, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_address16 {
    resource_type: u8,
    producer_consumer: u8,
    decode: u8,
    min_address_fixed: u8,
    max_address_fixed: u8,
    info: acpi_resource_attribute,
    address: acpi_address16_attribute,
    resource_source: acpi_resource_source,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_address32 {
    resource_type: u8,
    producer_consumer: u8,
    decode: u8,
    min_address_fixed: u8,
    max_address_fixed: u8,
    info: acpi_resource_attribute,
    address: acpi_address32_attribute,
    resource_source: acpi_resource_source,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_address64 {
    resource_type: u8,
    producer_consumer: u8,
    decode: u8,
    min_address_fixed: u8,
    max_address_fixed: u8,
    info: acpi_resource_attribute,
    address: acpi_address64_attribute,
    resource_source: acpi_resource_source,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_extended_address64 {
    resource_type: u8,
    producer_consumer: u8,
    decode: u8,
    min_address_fixed: u8,
    max_address_fixed: u8,
    info: acpi_resource_attribute,
    revision_ID: u8,
    address: acpi_address64_attribute,
    type_specific: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_resource_extended_irq {
    producer_consumer: u8,
    triggering: u8,
    polarity: u8,
    shareable: u8,
    wake_capable: u8,
    interrupt_count: u8,
    resource_source: acpi_resource_source,
    __slate_anon_7: __SlateRecord_acrestyp_h_341,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_generic_register {
    space_id: u8,
    bit_width: u8,
    bit_offset: u8,
    access_size: u8,
    address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iova_bitmap {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_gpio {
    revision_id: u8,
    connection_type: u8,
    producer_consumer: u8,
    pin_config: u8,
    shareable: u8,
    wake_capable: u8,
    io_restriction: u8,
    triggering: u8,
    polarity: u8,
    drive_strength: u16,
    debounce_timeout: u16,
    pin_table_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    pin_table: *mut u16,
    vendor_data: *mut u8,
}


// For values, see Producer/Consumer above
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_common_serialbus {
    revision_id: u8,
    r#type: u8,
    producer_consumer: u8,
    slave_mode: u8,
    connection_sharing: u8,
    type_revision_id: u8,
    type_data_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    vendor_data: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_i2c_serialbus {
    revision_id: u8,
    r#type: u8,
    producer_consumer: u8,
    slave_mode: u8,
    connection_sharing: u8,
    type_revision_id: u8,
    type_data_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    vendor_data: *mut u8,
    access_mode: u8,
    slave_address: u16,
    connection_speed: u32,
    lvr: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_spi_serialbus {
    revision_id: u8,
    r#type: u8,
    producer_consumer: u8,
    slave_mode: u8,
    connection_sharing: u8,
    type_revision_id: u8,
    type_data_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    vendor_data: *mut u8,
    wire_mode: u8,
    device_polarity: u8,
    data_bit_length: u8,
    clock_phase: u8,
    clock_polarity: u8,
    device_selection: u16,
    connection_speed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_viommu {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_uart_serialbus {
    revision_id: u8,
    r#type: u8,
    producer_consumer: u8,
    slave_mode: u8,
    connection_sharing: u8,
    type_revision_id: u8,
    type_data_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    vendor_data: *mut u8,
    endian: u8,
    data_bits: u8,
    stop_bits: u8,
    flow_control: u8,
    parity: u8,
    lines_enabled: u8,
    rx_fifo_size: u16,
    tx_fifo_size: u16,
    default_baud_rate: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_csi2_serialbus {
    revision_id: u8,
    r#type: u8,
    producer_consumer: u8,
    slave_mode: u8,
    connection_sharing: u8,
    type_revision_id: u8,
    type_data_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    vendor_data: *mut u8,
    local_port_instance: u8,
    phy_type: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_pin_function {
    revision_id: u8,
    pin_config: u8,
    shareable: u8,
    function_number: u16,
    pin_table_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    pin_table: *mut u16,
    vendor_data: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_pin_config {
    revision_id: u8,
    producer_consumer: u8,
    shareable: u8,
    pin_config_type: u8,
    pin_config_value: u32,
    pin_table_length: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    pin_table: *mut u16,
    vendor_data: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_clock_input {
    revision_id: u8,
    mode: u8,
    scale: u8,
    frequency_divisor: u16,
    frequency_numerator: u32,
    resource_source: acpi_resource_source,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_pin_group {
    revision_id: u8,
    producer_consumer: u8,
    pin_table_length: u16,
    vendor_length: u16,
    pin_table: *mut u16,
    resource_label: acpi_resource_label,
    vendor_data: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_pin_group_function {
    revision_id: u8,
    producer_consumer: u8,
    shareable: u8,
    function_number: u16,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    resource_source_label: acpi_resource_label,
    vendor_data: *mut u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource_pin_group_config {
    revision_id: u8,
    producer_consumer: u8,
    shareable: u8,
    pin_config_type: u8,
    pin_config_value: u32,
    vendor_length: u16,
    resource_source: acpi_resource_source,
    resource_source_label: acpi_resource_label,
    vendor_data: *mut u8,
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


// ACPI 5.0
#[repr(C)]
#[derive(Clone, Copy)]
union acpi_resource_data {
    irq: acpi_resource_irq,
    dma: acpi_resource_dma,
    start_dpf: acpi_resource_start_dependent,
    io: acpi_resource_io,
    fixed_io: acpi_resource_fixed_io,
    fixed_dma: acpi_resource_fixed_dma,
    vendor: acpi_resource_vendor,
    vendor_typed: acpi_resource_vendor_typed,
    end_tag: acpi_resource_end_tag,
    memory24: acpi_resource_memory24,
    memory32: acpi_resource_memory32,
    fixed_memory32: acpi_resource_fixed_memory32,
    address16: acpi_resource_address16,
    address32: acpi_resource_address32,
    address64: acpi_resource_address64,
    ext_address64: acpi_resource_extended_address64,
    extended_irq: acpi_resource_extended_irq,
    generic_reg: acpi_resource_generic_register,
    gpio: acpi_resource_gpio,
    i2c_serial_bus: acpi_resource_i2c_serialbus,
    spi_serial_bus: acpi_resource_spi_serialbus,
    uart_serial_bus: acpi_resource_uart_serialbus,
    csi2_serial_bus: acpi_resource_csi2_serialbus,
    common_serial_bus: acpi_resource_common_serialbus,
    pin_function: acpi_resource_pin_function,
    pin_config: acpi_resource_pin_config,
    pin_group: acpi_resource_pin_group,
    pin_group_function: acpi_resource_pin_group_function,
    pin_group_config: acpi_resource_pin_group_config,
    clock_input: acpi_resource_clock_input,
    address: acpi_resource_address,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_resource {
    r#type: u32,
    length: u32,
    data: acpi_resource_data,
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


fn acpi_os_free(memory: *mut ()) {
unsafe { kfree(memory as *const ()) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pld_info {
    revision: u8,
    ignore_color: u8,
    red: u8,
    green: u8,
    blue: u8,
    width: u16,
    height: u16,
    user_visible: u8,
    dock: u8,
    lid: u8,
    panel: u8,
    vertical_position: u8,
    horizontal_position: u8,
    shape: u8,
    group_orientation: u8,
    group_token: u8,
    group_position: u8,
    bay: u8,
    ejectable: u8,
    ospm_eject_required: u8,
    cabinet_number: u8,
    card_cage_number: u8,
    reference: u8,
    rotation: u8,
    order: u8,
    reserved: u8,
    vertical_offset: u16,
    horizontal_offset: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_subtable_headers {
    common: acpi_subtable_header,
    hmat: acpi_hmat_structure,
    prmt: acpi_prmt_module_header,
    cedt: acpi_cedt_header,
    cdat: acpi_cdat_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_handle_list {
    count: u32,
    handles: *mut *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits815U0,
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
struct io_tlb_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits821U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1179 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1179_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1179_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1185 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1185_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1185_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1188 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1188_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1188_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1191 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1191_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1191_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1202 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1202_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scan_c_1202_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits822U0,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits272U0 {
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
    pub struct __SlateBits68U0 {
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
    pub struct __SlateBits68U1 {
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
    pub struct __SlateBits275U0 {
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
    pub struct __SlateBits68U2 {
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
    pub struct __SlateBits123U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits609U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits452U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits454U0 {
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
    pub struct __SlateBits122U0 {
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
    pub struct __SlateBits401U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits480U0 {
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
    pub struct __SlateBits477U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits480U1 {
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
    pub struct __SlateBits480U2 {
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
    pub struct __SlateBits480U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits850U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits693U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits821U0 {
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
    pub struct __SlateBits822U0 {
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
    pub struct __SlateBits827U0 {
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
    pub struct __SlateBits829U0 {
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
    pub struct __SlateBits831U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits838U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits834U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits836U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits815U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U0 {
        #[bits(1)]
        pub put_online: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hardware_id {
    list: list_head,
    id: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits827U0,
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_device_hid")]
extern "C" fn acpi_device_hid(device: *mut acpi_device) -> *const u8 {
let mut __v0: *mut acpi_hardware_id;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*device).pnp.ids) };
let mut __v1: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_542() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v1;
let __v3: *mut acpi_hardware_id = if pos__ != head__ {
list_entry__36!(pos__, acpi_hardware_id)
} else {
core::ptr::null_mut::<acpi_hardware_id>()
};
__v0 = __v3;
let hid: *mut acpi_hardware_id = __v0;
if hid.is_null() {
return unsafe { dummy_hid };
}
unsafe { (*hid).id }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_flags {
    __slate_bits_0: __slate_bits::__SlateBits829U0,
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
struct acpi_device_power {
    state: i32,
    flags: acpi_device_power_flags,
    states: [acpi_device_power_state; 5],
    state_for_enumeration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_dep_data {
    node: list_head,
    supplier: *mut (),
    consumer: *mut (),
    honor_dep: bool,
    met: bool,
    free_when_met: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits834U0,
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
struct acpi_device_perf {
    state: i32,
    flags: acpi_device_perf_flags,
    state_count: i32,
    states: *mut acpi_device_perf_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits838U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
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
struct acpi_device_physical_node {
    node: list_head,
    dev: *mut device,
    node_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits841U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
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


fn acpi_fwnode_handle(adev: *mut acpi_device) -> *mut fwnode_handle {
unsafe { core::ptr::addr_of_mut!((*adev).fwnode) }
}

fn acpi_dev_parent(adev: *mut acpi_device) -> *mut acpi_device {
if !unsafe { (*adev).dev.parent }.is_null() {
return to_acpi_device!(unsafe { (*adev).dev.parent });
}
core::ptr::null_mut::<acpi_device>()
}

fn acpi_set_device_status(adev: *mut acpi_device, sta: u32) {
unsafe {
*(core::ptr::addr_of_mut!((*adev).status) as *mut u32) = sta;
}
}

fn acpi_set_hp_context(adev: *mut acpi_device, hp: *mut acpi_hotplug_context) {
unsafe {
(*hp).self_ = adev;
}
unsafe {
(*adev).hp = hp;
}
}

fn acpi_device_enumerated(adev: *mut acpi_device) -> bool {
!adev.is_null() && ((unsafe { (*adev).flags.__slate_bits_0.__get_initialized() }) as i32) != 0 && ((unsafe { (*adev).flags.__slate_bits_0.__get_visited() }) as i32) != 0
}

fn acpi_dev_get(adev: *mut acpi_device) -> *mut acpi_device {
let __v0: *mut acpi_device = if !adev.is_null() {
to_acpi_device!(unsafe { get_device(core::ptr::addr_of_mut!((*adev).dev)) })
} else {
core::ptr::null_mut::<acpi_device>()
};
__v0
}

fn acpi_dev_put(adev: *mut acpi_device) {
if !adev.is_null() {
unsafe { put_device(core::ptr::addr_of_mut!((*adev).dev)) };
}
}

fn acpi_put_acpi_dev(adev: *mut acpi_device) {
acpi_dev_put(adev);
}

fn acpi_arch_dma_setup(dev: *mut device) {
}

// DWORD 3
// DWORD 5
// _OST General Processing Status Code
// _OST Ejection Request (0x3, 0x103) Status Code
fn acpi_device_set_enumerated(adev: *mut acpi_device) {
unsafe {
(*adev).flags.__slate_bits_0.__set_visited(1u32);
}
}

fn acpi_device_clear_enumerated(adev: *mut acpi_device) {
unsafe {
(*adev).flags.__slate_bits_0.__set_visited(0u32);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_probe_entry {
    id: [u8; 5],
    r#type: u8,
    subtable_valid: Option<unsafe extern "C" fn(*mut acpi_subtable_header, *mut acpi_probe_entry) -> bool>,
    __slate_anon_3: __SlateRecord_acpi_h_1441,
    driver_data: u64,
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


fn iort_iommu_configure_id(dev: *mut device, id_in: *const u32) -> i32 {
19i32.wrapping_neg()
}

fn rimt_iommu_configure_id(dev: *mut device, id_in: *const u32) -> i32 {
19i32.wrapping_neg()
}

fn viot_iommu_configure(dev: *mut device) -> i32 {
19i32.wrapping_neg()
}

// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
fn _zero_page(addr: u64) -> *mut page {
unsafe { __zero_page }
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
core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_acpi_scan_lock_acquire: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_scan_lock_release: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_initialize_hp_context: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_fetch_acpi_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_get_acpi_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_bus_get_ejd: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_is_video_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_device_hid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_dma_configure_id: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_dev_clear_dependencies: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_dev_ready_for_enumeration: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_dev_get_next_consumer_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section\t\".initcall5.init\", \"a\"\t\t\n__initcall__kmod_acpi__550_2702_acpi_reserve_motherboard_resources5:\t\t\t\n.long\t{12} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_bus_scan: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_bus_trim: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_bus_register_early_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_reconfig_notifier_register: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_acpi_reconfig_notifier_unregister: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous", sym acpi_scan_lock_acquire, sym acpi_scan_lock_release, sym acpi_initialize_hp_context, sym acpi_fetch_acpi_dev, sym acpi_get_acpi_dev, sym acpi_bus_get_ejd, sym acpi_is_video_device, sym acpi_device_hid, sym acpi_dma_configure_id, sym acpi_dev_clear_dependencies, sym acpi_dev_ready_for_enumeration, sym acpi_dev_get_next_consumer_dev, sym acpi_reserve_motherboard_resources, sym acpi_bus_scan, sym acpi_bus_trim, sym acpi_bus_register_early_device, sym acpi_reconfig_notifier_register, sym acpi_reconfig_notifier_unregister, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits850U0,
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
    __slate_bits_0: __slate_bits::__SlateBits693U0,
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


fn dev_iommu_fwspec_get(dev: *mut device) -> *mut iommu_fwspec {
if !unsafe { (*dev).iommu }.is_null() {
return unsafe { (*(*dev).iommu).fwspec };
} else {
return core::ptr::null_mut::<iommu_fwspec>();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dma_map_ops {
    alloc: Option<unsafe extern "C" fn(*mut device, u64, *mut u64, u32, u64) -> *mut ()>,
    free: Option<unsafe extern "C" fn(*mut device, u64, *mut (), u64, u64)>,
    alloc_pages_op: Option<unsafe extern "C" fn(*mut device, u64, *mut u64, u32, u32) -> *mut page>,
    free_pages: Option<unsafe extern "C" fn(*mut device, u64, *mut page, u64, u32)>,
    mmap: Option<unsafe extern "C" fn(*mut device, *mut vm_area_struct, *mut (), u64, u64, u64) -> i32>,
    get_sgtable: Option<unsafe extern "C" fn(*mut device, *mut sg_table, *mut (), u64, u64, u64) -> i32>,
    map_phys: Option<unsafe extern "C" fn(*mut device, u64, u64, u32, u64) -> u64>,
    unmap_phys: Option<unsafe extern "C" fn(*mut device, u64, u64, u32, u64)>,
    map_sg: Option<unsafe extern "C" fn(*mut device, *mut scatterlist, i32, u32, u64) -> i32>,
    unmap_sg: Option<unsafe extern "C" fn(*mut device, *mut scatterlist, i32, u32, u64)>,
    sync_single_for_cpu: Option<unsafe extern "C" fn(*mut device, u64, u64, u32)>,
    sync_single_for_device: Option<unsafe extern "C" fn(*mut device, u64, u64, u32)>,
    sync_sg_for_cpu: Option<unsafe extern "C" fn(*mut device, *mut scatterlist, i32, u32)>,
    sync_sg_for_device: Option<unsafe extern "C" fn(*mut device, *mut scatterlist, i32, u32)>,
    cache_sync: Option<unsafe extern "C" fn(*mut device, *mut (), u64, u32)>,
    dma_supported: Option<unsafe extern "C" fn(*mut device, u64) -> i32>,
    get_required_mask: Option<unsafe extern "C" fn(*mut device) -> u64>,
    max_mapping_size: Option<unsafe extern "C" fn(*mut device) -> u64>,
    opt_mapping_size: Option<unsafe extern "C" fn() -> u64>,
    get_merge_boundary: Option<unsafe extern "C" fn(*mut device) -> u64>,
}


fn set_dma_ops(dev: *mut device, dma_ops: *const dma_map_ops) {
}

fn arch_setup_dma_ops(dev: *mut device, coherent: bool) {
}

fn crc32(crc: u32, p: *const (), len: u64) -> u32 {
unsafe { crc32_le(crc, p, len) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_pool {
    start: u64,
    end: u64,
    vaddr: *mut (),
    nslabs: u64,
    late_alloc: bool,
    nareas: u32,
    area_nslabs: u32,
    areas: *mut io_tlb_area,
    slots: *mut io_tlb_slot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
    defpool: io_tlb_pool,
    nslabs: u64,
    debugfs: *mut dentry,
    force_bounce: bool,
    for_alloc: bool,
    cc_shared: bool,
    total_used: __SlateRecord_types_h_197,
    used_hiwater: __SlateRecord_types_h_197,
    transient_nslabs: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
    cpu_start: u64,
    dma_start: u64,
    size: u64,
}


// Device Node Initialization / Removal
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_bus_id {
    bus_id: *const u8,
    instance_ida: ida,
    node: list_head,
}


// Device properties
fn acpi_watchdog_init() {
}

static mut dummy_hid: *const u8 = (b"device\0".as_ptr() as *mut u8) as *const u8;

static mut acpi_dep_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_dep_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_dep_list) } };

static mut acpi_dep_list_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

#[unsafe(no_mangle)]
static mut acpi_bus_id_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) } };

static mut acpi_scan_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

static mut acpi_scan_handlers_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_scan_handlers_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_scan_handlers_list) } };

#[unsafe(no_mangle)]
static mut acpi_device_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

#[unsafe(no_mangle)]
static mut acpi_wakeup_device_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_wakeup_device_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_wakeup_device_list) } };

static mut acpi_hp_context_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

static mut acpi_scan_system_dev_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_scan_system_dev_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_scan_system_dev_list) } };

/// The UART device described by the SPCR table is the only object which needs
/// special-casing. Everything else is covered by ACPI namespace paths in STAO
/// table.
static mut spcr_uart_addr: u64 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_lock_acquire")]
extern "C" fn acpi_scan_lock_acquire() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_scan_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_scan_lock_acquire_535: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn()>, *mut ()>(Some(acpi_scan_lock_acquire)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_lock_release")]
extern "C" fn acpi_scan_lock_release() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_scan_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_scan_lock_release_536: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn()>, *mut ()>(Some(acpi_scan_lock_release)) };

#[unsafe(no_mangle)]
extern "C" fn acpi_lock_hp_context() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_hp_context_lock)) };
}

#[unsafe(no_mangle)]
extern "C" fn acpi_unlock_hp_context() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_hp_context_lock)) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_initialize_hp_context")]
extern "C" fn acpi_initialize_hp_context(adev: *mut acpi_device, hp: *mut acpi_hotplug_context, notify: Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>, uevent: Option<unsafe extern "C" fn(*mut acpi_device, u32)>) {
acpi_lock_hp_context();
unsafe {
(*hp).notify = notify;
}
unsafe {
(*hp).uevent = uevent;
}
acpi_set_hp_context(adev, hp);
acpi_unlock_hp_context();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_initialize_hp_context_537: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut acpi_device, *mut acpi_hotplug_context, Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>, Option<unsafe extern "C" fn(*mut acpi_device, u32)>)>, *mut ()>(Some(acpi_initialize_hp_context)) };

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_add_handler(handler: *mut acpi_scan_handler) -> i32 {
if handler.is_null() {
return 22i32.wrapping_neg();
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*handler).list_node) }, unsafe { core::ptr::addr_of_mut!(acpi_scan_handlers_list) });
0
}

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_add_handler_with_hotplug(handler: *mut acpi_scan_handler, hotplug_profile_name: *const u8) -> i32 {
let error: i32 = acpi_scan_add_handler(handler);
if error != 0 {
return error;
}
unsafe { acpi_sysfs_add_hotplug_profile(core::ptr::addr_of_mut!((*handler).hotplug), hotplug_profile_name) };
0
}

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_is_offline(adev: *mut acpi_device, uevent: bool) -> bool {
let mut pn: *mut acpi_device_physical_node = unsafe { core::mem::zeroed() };
let mut offline: bool = true;
let mut envp: __SlateAlign16<[*mut u8; 2]> = __SlateAlign16([c"EVENT=offline".as_ptr() as *mut u8, core::ptr::null_mut::<u8>()]);
// acpi_container_offline() calls this for all of the container's
// children under the container's physical_node_lock lock.
mutex_lock_nested!(unsafe { core::ptr::addr_of_mut!((*adev).physical_node_lock) });
pn = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*adev).physical_node_list) });
while list_entry_is_head!(pn, unsafe { core::ptr::addr_of_mut!((*adev).physical_node_list) }, node) == 0 {
let __v1: bool = device_supports_offline(unsafe { (*pn).dev }) && !dev_offline((unsafe { (*pn).dev }) as *const device);
if __v1 {
if uevent {
unsafe { kobject_uevent_env(core::ptr::addr_of_mut!((*(*pn).dev).kobj), 2u32, envp.0.as_mut_ptr() as *mut *mut u8) };
}
offline = false;
break;
}
pn = list_next_entry__6!(pn, node);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*adev).physical_node_lock)) };
offline
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_offline")]
extern "C" fn acpi_bus_offline(handle: *mut (), lvl: u32, data: *mut (), ret_p: *mut *mut ()) -> u32 {
let device: *mut acpi_device = acpi_fetch_acpi_dev(handle);
let mut pn: *mut acpi_device_physical_node = unsafe { core::mem::zeroed() };
let second_pass: bool = !data.is_null();
let mut status: u32 = 0;
if device.is_null() {
return 0;
}
if !unsafe { (*device).handler }.is_null() && ((unsafe { (*(*device).handler).hotplug.__slate_bits_0.__get_enabled() }) as i32) == 0 {
unsafe {
*ret_p = core::ptr::addr_of_mut!((*device).dev) as *mut ();
}
return EXCEP_ENV!(15i32);
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*device).physical_node_lock)) };
pn = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*device).physical_node_list) });
'__slate_break_1: while list_entry_is_head!(pn, unsafe { core::ptr::addr_of_mut!((*device).physical_node_list) }, node) == 0 {
'__slate_continue_1: {
let mut ret: i32 = 0;
if second_pass {
// Skip devices offlined by the first pass.
if ((unsafe { (*pn).__slate_bits_0.__get_put_online() }) as i32) != 0 {
break '__slate_continue_1;
}
} else {
unsafe {
let __t0 = false;
(*pn).__slate_bits_0.__set_put_online(__t0);
}
}
ret = unsafe { device_offline((*pn).dev) };
if ret >= 0 {
unsafe {
let __t1 = ret == 0;
(*pn).__slate_bits_0.__set_put_online(__t1);
}
} else {
unsafe {
*ret_p = (*pn).dev as *mut ();
}
if second_pass {
status = EXCEP_ENV!(1i32);
break '__slate_break_1;
}
}
}
pn = list_next_entry__6!(pn, node);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*device).physical_node_lock)) };
status
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_online")]
extern "C" fn acpi_bus_online(handle: *mut (), lvl: u32, data: *mut (), ret_p: *mut *mut ()) -> u32 {
let device: *mut acpi_device = acpi_fetch_acpi_dev(handle);
let mut pn: *mut acpi_device_physical_node = unsafe { core::mem::zeroed() };
if device.is_null() {
return 0;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*device).physical_node_lock)) };
pn = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*device).physical_node_list) });
while list_entry_is_head!(pn, unsafe { core::ptr::addr_of_mut!((*device).physical_node_list) }, node) == 0 {
if ((unsafe { (*pn).__slate_bits_0.__get_put_online() }) as i32) != 0 {
unsafe { device_online((*pn).dev) };
unsafe {
let __t0 = false;
(*pn).__slate_bits_0.__set_put_online(__t0);
}
}
pn = list_next_entry__6!(pn, node);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*device).physical_node_lock)) };
0
}

fn acpi_scan_try_to_offline(device: *mut acpi_device) -> i32 {
let handle: *mut () = unsafe { (*device).handle };
let mut errdev: *mut device = core::ptr::null_mut::<device>();
let mut status: u32 = 0;
// Carry out two passes here and ignore errors in the first pass,
// because if the devices in question are memory blocks and
// CONFIG_MEMCG is set, one of the blocks may hold data structures
// that the other blocks depend on, but it is not known in advance which
// block holds them.
//
// If the first pass is successful, the second one isn't needed, though.
status = unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, None, Some(acpi_bus_offline), core::ptr::null_mut::<()>(), core::ptr::addr_of_mut!(errdev) as *mut *mut ()) };
if status == EXCEP_ENV!(15i32) {
while false {
}
unsafe { _dev_warn(errdev as *const device, c"Offline disabled.\n".as_ptr() as *const u8) };
unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, Some(acpi_bus_online), None, core::ptr::null_mut::<()>(), core::ptr::null_mut::<*mut ()>()) };
return 1i32.wrapping_neg();
}
acpi_bus_offline(handle, 0u32, core::ptr::null_mut::<()>(), core::ptr::addr_of_mut!(errdev) as *mut *mut ());
if !errdev.is_null() {
errdev = core::ptr::null_mut::<device>();
unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, None, Some(acpi_bus_offline), 1i32 as *mut (), core::ptr::addr_of_mut!(errdev) as *mut *mut ()) };
if errdev.is_null() {
acpi_bus_offline(handle, 0u32, 1i32 as *mut (), core::ptr::addr_of_mut!(errdev) as *mut *mut ());
}
if !errdev.is_null() {
while false {
}
unsafe { _dev_warn(errdev as *const device, c"Offline failed.\n".as_ptr() as *const u8) };
acpi_bus_online(handle, 0u32, core::ptr::null_mut::<()>(), core::ptr::null_mut::<*mut ()>());
unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, Some(acpi_bus_online), None, core::ptr::null_mut::<()>(), core::ptr::null_mut::<*mut ()>()) };
return 16i32.wrapping_neg();
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_check_and_detach")]
extern "C" fn acpi_scan_check_and_detach(adev: *mut acpi_device, p: *mut ()) -> i32 {
let handler: *mut acpi_scan_handler = unsafe { (*adev).handler };
let flags: u64 = p as u64;
unsafe { acpi_dev_for_each_child_reverse(adev, Some(acpi_scan_check_and_detach), p) };
if flags & BIT!(0i32) != 0 {
unsafe { acpi_bus_get_status(adev) };
// Skip devices that are still there and take the enabled
// flag into account.
if acpi_device_is_enabled(adev as *const acpi_device) {
return 0;
}
// Skip device that have not been enumerated.
if !acpi_device_enumerated(adev) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Still not enumerated\n".as_ptr() as *const u8) };
}
return 0;
}
}
if !handler.is_null() && (unsafe { (*handler).detach }) != None {
unsafe { unsafe { (*handler).detach }.unwrap()(adev) };
}
// Most likely, the device is going away, so put it into D3cold before
// that.
unsafe { acpi_device_set_power(adev, 4i32) };
unsafe {
(*adev).flags.__slate_bits_0.__set_initialized(0u32);
}
// For eject this is deferred to acpi_bus_post_eject()
if flags & BIT!(1i32) == 0 {
unsafe {
(*adev).handler = core::ptr::null_mut::<acpi_scan_handler>();
}
acpi_device_clear_enumerated(adev);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_post_eject")]
extern "C" fn acpi_bus_post_eject(adev: *mut acpi_device, not_used: *mut ()) -> i32 {
let handler: *mut acpi_scan_handler = unsafe { (*adev).handler };
unsafe { acpi_dev_for_each_child_reverse(adev, Some(acpi_bus_post_eject), core::ptr::null_mut::<()>()) };
if !handler.is_null() {
if (unsafe { (*handler).post_eject }) != None {
unsafe { unsafe { (*handler).post_eject }.unwrap()(adev) };
}
unsafe {
(*adev).handler = core::ptr::null_mut::<acpi_scan_handler>();
}
}
acpi_device_clear_enumerated(adev);
0
}

fn acpi_scan_check_subtree(adev: *mut acpi_device) {
let flags: u64 = BIT!(0i32);
acpi_scan_check_and_detach(adev, flags as *mut ());
}

fn acpi_scan_hot_remove(device: *mut acpi_device) -> i32 {
let handle: *mut () = unsafe { (*device).handle };
let mut sta: u64 = 0;
let mut status: u32 = 0;
let flags: u64 = BIT!(1i32);
if !unsafe { (*device).handler }.is_null() && ((unsafe { (*(*device).handler).hotplug.__slate_bits_0.__get_demand_offline() }) as i32) != 0 {
if !acpi_scan_is_offline(device, true) {
return 16i32.wrapping_neg();
}
} else {
let error: i32 = acpi_scan_try_to_offline(device);
if error != 0 {
return error;
}
}
let mut __v0: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"Ejecting\n".as_ptr() as *const u8) };
}
__v0 = 0;
acpi_scan_check_and_detach(device, flags as *mut ());
unsafe { acpi_evaluate_lck(handle, 0i32) };
// TBD: _EJD support.
status = unsafe { acpi_evaluate_ej0(handle) };
if status == EXCEP_ENV!(5i32) {
return 19i32.wrapping_neg();
} else {
if ACPI_FAILURE!(status) != 0 {
return 5i32.wrapping_neg();
}
}
// Verify if eject was indeed successful.  If not, log an error
// message.  No need to call _OST since _EJ0 call was made OK.
status = unsafe { acpi_evaluate_integer(handle, c"_STA".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(sta)) };
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_handle_printk(c"\x014".as_ptr() as *const u8, handle, c"Status check after eject failed (0x%x)\n".as_ptr() as *const u8, status) };
} else {
if sta & 2 != 0 {
unsafe { acpi_handle_printk(c"\x014".as_ptr() as *const u8, handle, c"Eject incomplete - status 0x%llx\n".as_ptr() as *const u8, sta) };
} else {
acpi_bus_post_eject(device, core::ptr::null_mut::<()>());
}
}
0
}

fn acpi_scan_rescan_bus(adev: *mut acpi_device) -> i32 {
let handler: *mut acpi_scan_handler = unsafe { (*adev).handler };
let ret: i32 = if !handler.is_null() && (unsafe { (*handler).hotplug.scan_dependent }) != None {
unsafe { unsafe { (*handler).hotplug.scan_dependent }.unwrap()(adev) }
} else {
acpi_bus_scan(unsafe { (*adev).handle })
};
if ret != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Namespace scan failure\n".as_ptr() as *const u8) };
}
ret
}

fn acpi_scan_device_check(adev: *mut acpi_device) -> i32 {
let mut parent: *mut acpi_device = unsafe { core::mem::zeroed() };
acpi_scan_check_subtree(adev);
if !acpi_device_is_present(adev as *const acpi_device) {
return 0;
}
// This function is only called for device objects for which matching
// scan handlers exist.  The only situation in which the scan handler
// is not attached to this device object yet is when the device has
// just appeared (either it wasn't present at all before or it was
// removed and then added again).
if !unsafe { (*adev).handler }.is_null() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Already enumerated\n".as_ptr() as *const u8) };
}
return 0;
}
parent = acpi_dev_parent(adev);
if parent.is_null() {
parent = adev;
}
acpi_scan_rescan_bus(parent)
}

fn acpi_scan_bus_check(adev: *mut acpi_device) -> i32 {
acpi_scan_check_subtree(adev);
acpi_scan_rescan_bus(adev)
}

fn acpi_generic_hotplug_event(adev: *mut acpi_device, r#type: u32) -> i32 {
match r#type {
0 => {
return acpi_scan_bus_check(adev);
}
1 => {
return acpi_scan_device_check(adev);
}
3 | 259 => {
if !unsafe { (*adev).handler }.is_null() && ((unsafe { (*(*adev).handler).hotplug.__slate_bits_0.__get_enabled() }) as i32) == 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Eject disabled\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
unsafe { acpi_evaluate_ost((*adev).handle, 3u32, 132u32, core::ptr::null_mut::<acpi_buffer>()) };
return acpi_scan_hot_remove(adev);
}
_ => {
}
}
22i32.wrapping_neg()
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_hotplug(adev: *mut acpi_device, src: u32) {
let mut notify: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32> = core::ptr::addr_of_mut!(notify) as *mut Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>;
let mut error: i32 = unsafe { core::mem::zeroed() };
let mut ost_code: u32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
ost_code = 1;
error = 19i32.wrapping_neg();
lock_device_hotplug();
mutex_lock(core::ptr::addr_of_mut!(acpi_scan_lock));
// The device object's ACPI handle cannot become invalid as long as we
// are holding acpi_scan_lock, but it might have become invalid before
// that lock was acquired.
if (*adev).handle != (ZERO_PAGE!(0i32) as *mut ()) {
if ((*adev).flags.__slate_bits_0.__get_is_dock_station() as i32) != 0 {
error = dock_notify(adev, src);
} else {
if ((*adev).flags.__slate_bits_0.__get_hotplug_notify() as i32) != 0 {
error = acpi_generic_hotplug_event(adev, src);
} else {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
acpi_lock_hp_context();
*__slate_slot_0 = {
let __t1: Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32> = if !unsafe { (*adev).hp }.is_null() { (*(*adev).hp).notify } else { None };
__t1
};
acpi_unlock_hp_context();
// There may be additional notify handlers for device objects
// without the .event() callback, so ignore them here.
if *__slate_slot_0 != None {
error = (*__slate_slot_0).unwrap()(adev, src);
} else {
break '__join_0;
}
}
}
let __t0: i32 = error;
if __t0 == 0 {
ost_code = 0;
} else {
if __t0 == -1 {
ost_code = 128;
} else {
if __t0 == -16 {
ost_code = 130;
} else {
ost_code = 1;
}
}
}
}
acpi_evaluate_ost((*adev).handle, src, ost_code, core::ptr::null_mut::<acpi_buffer>());
}
acpi_put_acpi_dev(adev);
mutex_unlock(core::ptr::addr_of_mut!(acpi_scan_lock));
unlock_device_hotplug();
}
}

fn acpi_free_power_resources_lists(device: *mut acpi_device) {
let mut i: i32 = 0;
if ((unsafe { (*device).wakeup.flags.__slate_bits_0.__get_valid() }) as i32) != 0 {
unsafe { acpi_power_resources_list_free(core::ptr::addr_of_mut!((*device).wakeup.resources)) };
}
if ((unsafe { (*device).power.flags.__slate_bits_0.__get_power_resources() }) as i32) == 0 {
return;
}
while i <= 3 {
let ps: *mut acpi_device_power_state = unsafe { (*device).power.states.as_mut_ptr() as *mut acpi_device_power_state }.wrapping_offset(i as isize);
unsafe { acpi_power_resources_list_free(core::ptr::addr_of_mut!((*ps).resources)) };
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_device_release")]
extern "C" fn acpi_device_release(dev: *mut device) {
let acpi_dev: *mut acpi_device = to_acpi_device!(dev);
unsafe { acpi_free_properties(acpi_dev) };
acpi_free_pnp_ids(unsafe { core::ptr::addr_of_mut!((*acpi_dev).pnp) });
acpi_free_power_resources_lists(acpi_dev);
unsafe { kfree(acpi_dev as *const ()) };
}

fn acpi_device_cleanup(device: *mut acpi_device) {
let mut acpi_device_bus_id: *mut acpi_device_bus_id = unsafe { core::mem::zeroed() };
acpi_device_bus_id = list_first_entry__18!(unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) });
while list_entry_is_head!(acpi_device_bus_id, unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) }, node) == 0 {
if (unsafe { strcmp((*acpi_device_bus_id).bus_id, acpi_device_hid(device)) }) == 0 {
unsafe { ida_free(core::ptr::addr_of_mut!((*acpi_device_bus_id).instance_ida), (*device).pnp.instance_no as u32) };
if ida_is_empty((unsafe { core::ptr::addr_of_mut!((*acpi_device_bus_id).instance_ida) }) as *const ida) {
list_del(unsafe { core::ptr::addr_of_mut!((*acpi_device_bus_id).node) });
unsafe { kfree_const((*acpi_device_bus_id).bus_id as *const ()) };
unsafe { kfree(acpi_device_bus_id as *const ()) };
}
break;
}
acpi_device_bus_id = list_next_entry__15!(acpi_device_bus_id, node);
}
list_del(unsafe { core::ptr::addr_of_mut!((*device).wakeup_list) });
}

fn acpi_device_del(device: *mut acpi_device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_device_lock)) };
acpi_device_cleanup(device);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_device_lock)) };
unsafe { acpi_power_add_remove_device(device, false) };
unsafe { acpi_device_remove_files(device) };
if (unsafe { (*device).remove }) != None {
unsafe { unsafe { (*device).remove }.unwrap()(device) };
}
unsafe { device_del(core::ptr::addr_of_mut!((*device).dev)) };
}

static mut acpi_reconfig_chain: blocking_notifier_head = blocking_notifier_head { rwsem: rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() }, head: core::ptr::null_mut::<notifier_block>() };

static mut acpi_device_del_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) }, prev: unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) } };

static mut acpi_device_del_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_device_del_work_fn")]
extern "C" fn acpi_device_del_work_fn(work_not_used: *mut work_struct) {
loop {
let mut adev: *mut acpi_device = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_device_del_lock)) };
if list_empty((unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) }) as *const list_head) != 0 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_device_del_lock)) };
break;
}
adev = list_first_entry__34!(unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) }, acpi_device);
list_del(unsafe { core::ptr::addr_of_mut!((*adev).del_list) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_device_del_lock)) };
unsafe { blocking_notifier_call_chain(core::ptr::addr_of_mut!(acpi_reconfig_chain), 1u64, adev as *mut ()) };
acpi_device_del(adev);
// Drop references to all power resources that might have been
// used by the device.
unsafe { acpi_power_transition(adev, 4i32) };
acpi_dev_put(adev);
}
}

/// acpi_scan_drop_device - Drop an ACPI device object.
/// @handle: Handle of an ACPI namespace node, not used.
/// @context: Address of the ACPI device object to drop.
///
/// This is invoked by acpi_ns_delete_node() during the removal of the ACPI
/// namespace node the device object pointed to by @context is attached to.
///
/// The unregistration is carried out asynchronously to avoid running
/// acpi_device_del() under the ACPICA's namespace mutex and the list is used to
/// ensure the correct ordering (the device objects must be unregistered in the
/// same order in which the corresponding namespace nodes are deleted).
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_drop_device")]
extern "C" fn acpi_scan_drop_device(handle: *mut (), context: *mut ()) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_device_del_lock)) };
// Use the ACPI hotplug workqueue which is ordered, so this work item
// won't run after any hotplug work items submitted subsequently.  That
// prevents attempts to register device objects identical to those being
// deleted from happening concurrently (such attempts result from
// hotplug events handled via the ACPI hotplug workqueue).  It also will
// run after all of the work items submitted previously, which helps
// those work items to ensure that they are not accessing stale device
// objects.
if list_empty((unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) }) as *const list_head) != 0 {
unsafe { acpi_queue_hotplug_work(core::ptr::addr_of_mut!(work)) };
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*(context as *mut acpi_device)).del_list) }, unsafe { core::ptr::addr_of_mut!(acpi_device_del_list) });
// Make acpi_ns_validate_handle() return NULL for this handle.
unsafe {
(*(context as *mut acpi_device)).handle = ZERO_PAGE!(0i32) as *mut ();
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_device_del_lock)) };
}

static mut work: work_struct = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(work.entry) } }, func: Some(acpi_device_del_work_fn) };

fn handle_to_device(handle: *mut (), callback: Option<unsafe extern "C" fn(*mut ())>) -> *mut acpi_device {
let mut adev: *mut acpi_device = core::ptr::null_mut::<acpi_device>();
let status: u32 = unsafe { acpi_get_data_full(handle, Some(acpi_scan_drop_device), core::ptr::addr_of_mut!(adev) as *mut *mut (), callback) };
if ACPI_FAILURE!(status) != 0 || adev.is_null() {
let mut __v0: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"No context!\n".as_ptr() as *const u8) };
}
__v0 = 0;
return core::ptr::null_mut::<acpi_device>();
}
adev
}

/// acpi_fetch_acpi_dev - Retrieve ACPI device object.
/// @handle: ACPI handle associated with the requested ACPI device object.
///
/// Return a pointer to the ACPI device object associated with @handle, if
/// present, or NULL otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_fetch_acpi_dev")]
extern "C" fn acpi_fetch_acpi_dev(handle: *mut ()) -> *mut acpi_device {
handle_to_device(handle, None)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_fetch_acpi_dev_538: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ()) -> *mut acpi_device>, *mut ()>(Some(acpi_fetch_acpi_dev)) };

#[unsafe(link_section = ".text.slate_distinct.scan.get_acpi_device")]
extern "C" fn get_acpi_device(dev: *mut ()) {
acpi_dev_get(dev as *mut acpi_device);
}

/// acpi_get_acpi_dev - Retrieve ACPI device object and reference count it.
/// @handle: ACPI handle associated with the requested ACPI device object.
///
/// Return a pointer to the ACPI device object associated with @handle and bump
/// up that object's reference counter (under the ACPI Namespace lock), if
/// present, or return NULL otherwise.
///
/// The ACPI device object reference acquired by this function needs to be
/// dropped via acpi_dev_put().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_get_acpi_dev")]
extern "C" fn acpi_get_acpi_dev(handle: *mut ()) -> *mut acpi_device {
handle_to_device(handle, Some(get_acpi_device))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_get_acpi_dev_539: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ()) -> *mut acpi_device>, *mut ()>(Some(acpi_get_acpi_dev)) };

fn acpi_device_bus_id_match(dev_id: *const u8) -> *mut acpi_device_bus_id {
let mut acpi_device_bus_id: *mut acpi_device_bus_id = unsafe { core::mem::zeroed() };
// Find suitable bus_id and instance number in acpi_bus_id_list.
acpi_device_bus_id = list_first_entry__18!(unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) });
while list_entry_is_head!(acpi_device_bus_id, unsafe { core::ptr::addr_of_mut!(acpi_bus_id_list) }, node) == 0 {
if (unsafe { strcmp((*acpi_device_bus_id).bus_id, dev_id) }) == 0 {
return acpi_device_bus_id;
}
acpi_device_bus_id = list_next_entry__15!(acpi_device_bus_id, node);
}
core::ptr::null_mut::<acpi_device_bus_id>()
}

fn acpi_device_set_name(device: *mut acpi_device, acpi_device_bus_id: *mut acpi_device_bus_id) -> i32 {
let instance_ida: *mut ida = unsafe { core::ptr::addr_of_mut!((*acpi_device_bus_id).instance_ida) };
let result: i32 = ida_alloc(instance_ida, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if result < 0 {
return result;
}
unsafe {
(*device).pnp.instance_no = result;
}
unsafe { dev_set_name(core::ptr::addr_of_mut!((*device).dev), c"%s:%02x".as_ptr() as *const u8, (*acpi_device_bus_id).bus_id, result) };
0
}

#[unsafe(no_mangle)]
extern "C" fn acpi_tie_acpi_dev(adev: *mut acpi_device) -> i32 {
let handle: *mut () = unsafe { (*adev).handle };
let mut status: u32 = 0;
if handle.is_null() {
return 0;
}
status = unsafe { acpi_attach_data(handle, Some(acpi_scan_drop_device), adev as *mut ()) };
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_handle_printk(c"\x013".as_ptr() as *const u8, handle, c"Unable to attach device data\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
0
}

fn acpi_store_pld_crc(adev: *mut acpi_device) {
let mut pld: *mut acpi_pld_info = unsafe { core::mem::zeroed() };
if !unsafe { acpi_get_physical_device_location((*adev).handle, core::ptr::addr_of_mut!(pld)) } {
return;
}
unsafe {
(*adev).pld_crc = crc32(4294967295u32, pld as *const (), core::mem::size_of::<acpi_pld_info>() as u64);
}
ACPI_FREE!(pld);
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_add(device: *mut acpi_device) -> i32 {
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut acpi_device_bus_id: core::mem::MaybeUninit<*mut acpi_device_bus_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut acpi_device_bus_id = core::ptr::addr_of_mut!(acpi_device_bus_id) as *mut *mut acpi_device_bus_id;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
// Linkage
// Link this device to its parent and siblings.
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*device).wakeup_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*device).physical_node_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*device).del_list));
loop {
__mutex_init(core::ptr::addr_of_mut!((*device).physical_node_lock), c"&device->physical_node_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__750));
if true {
break;
}
}
'__join_0: {
mutex_lock(core::ptr::addr_of_mut!(acpi_device_lock));
*__slate_slot_7 = acpi_device_bus_id_match(acpi_device_hid(device));
if !(*__slate_slot_7).is_null() {
*__slate_slot_6 = acpi_device_set_name(device, *__slate_slot_7);
if *__slate_slot_6 != 0 {
break '__join_0;
}
} else {
__obj_size = size_mul(core::mem::size_of::<acpi_device_bus_id>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
}
*__slate_slot_7 = *__slate_slot_1 as *mut acpi_device_bus_id;
if (*__slate_slot_7).is_null() {
*__slate_slot_6 = 12i32.wrapping_neg();
break '__join_0;
} else {
(*(*__slate_slot_7)).bus_id = kstrdup_const(acpi_device_hid(device), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if unsafe { (*(*__slate_slot_7)).bus_id }.is_null() {
kfree(*__slate_slot_7 as *const ());
*__slate_slot_6 = 12i32.wrapping_neg();
break '__join_0;
} else {
ida_init(core::ptr::addr_of_mut!((*(*__slate_slot_7)).instance_ida));
*__slate_slot_6 = acpi_device_set_name(device, *__slate_slot_7);
if *__slate_slot_6 != 0 {
kfree_const((*(*__slate_slot_7)).bus_id as *const ());
kfree(*__slate_slot_7 as *const ());
break '__join_0;
} else {
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_7)).node), core::ptr::addr_of_mut!(acpi_bus_id_list));
}
}
}
}
if ((*device).wakeup.flags.__slate_bits_0.__get_valid() as i32) != 0 {
list_add_tail(core::ptr::addr_of_mut!((*device).wakeup_list), core::ptr::addr_of_mut!(acpi_wakeup_device_list));
}
acpi_store_pld_crc(device);
mutex_unlock(core::ptr::addr_of_mut!(acpi_device_lock));
*__slate_slot_6 = device_add(core::ptr::addr_of_mut!((*device).dev));
if *__slate_slot_6 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*device).dev) as *const device, c"Error registering device\n".as_ptr() as *const u8);
mutex_lock(core::ptr::addr_of_mut!(acpi_device_lock));
acpi_device_cleanup(device);
} else {
acpi_device_setup_files(device);
return 0;
}
}
mutex_unlock(core::ptr::addr_of_mut!(acpi_device_lock));
acpi_detach_data((*device).handle, Some(acpi_scan_drop_device));
return *__slate_slot_6;
}
}

static mut __key__750: lock_class_key = unsafe { core::mem::zeroed() };

/// Device Enumeration
fn acpi_info_matches_ids(info: *mut acpi_device_info, ids: *const *const u8) -> bool {
let mut cid_list: *mut acpi_pnp_device_id_list = core::ptr::null_mut::<acpi_pnp_device_id_list>();
let mut i: i32 = 0;
let mut index: i32 = 0;
if ((unsafe { (*info).valid }) as i32) & 4 == 0 {
return false;
}
index = unsafe { match_string(ids, 1i32.wrapping_neg() as u64, (*info).hardware_id.string as *const u8) };
if index >= 0 {
return true;
}
if ((unsafe { (*info).valid }) as i32) & 32 != 0 {
cid_list = unsafe { core::ptr::addr_of_mut!((*info).compatible_id_list) };
}
if cid_list.is_null() {
return false;
}
while (i as u32) < unsafe { (*cid_list).count } {
index = unsafe { match_string(ids, 1i32.wrapping_neg() as u64, (*(core::ptr::addr_of_mut!((*cid_list).ids) as *mut acpi_pnp_device_id).wrapping_offset(i as isize)).string as *const u8) };
if index >= 0 {
return true;
}
i = i.wrapping_add(1);
}
false
}

// List of HIDs for which we ignore matching ACPI devices, when checking _DEP lists.
/// Windows-compatible System Power Management Controller
/// Intel Baytrail Mailbox Device
/// Lattice FW Update Client Driver
static acpi_ignore_dep_ids: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"PNP0D80\0".as_ptr() as *mut u8) as *const u8, (b"INT33BD\0".as_ptr() as *mut u8) as *const u8, (b"LATT2021\0".as_ptr() as *mut u8) as *const u8, core::ptr::null::<u8>()]);

// List of HIDs for which we honor deps of matching ACPI devs, when checking _DEP lists.
/// ARM GICv5 IWB
/// Camera sensor PMIC / clk and regulator info
/// IVSC (TGL) driver must be loaded to allow i2c access to camera sensors
/// IVSC (ADL) driver must be loaded to allow i2c access to camera sensors
/// IVSC (RPL) driver must be loaded to allow i2c access to camera sensors
/// IVSC (MTL) driver must be loaded to allow i2c access to camera sensors
/// CVS (LNL) driver must be loaded to allow camera streaming
/// CVS (ARL) driver must be loaded to allow camera streaming
/// CVS (PTL) driver must be loaded to allow camera streaming
/// CVS (NVL) driver must be loaded to allow camera streaming
/// RISC-V PLIC
/// RISC-V APLIC
/// RISC-V SBI MPXY MBOX
/// RISC-V RPMI SYSMSI
/// PCI Link Device
/// CXL/PCIe host bridge: CXL root (ACPI0017) depends on PCI root attach
static acpi_honor_dep_ids: __SlateAlign16<[*const u8; 17]> = __SlateAlign16([(b"ARMH0003\0".as_ptr() as *mut u8) as *const u8, (b"INT3472\0".as_ptr() as *mut u8) as *const u8, (b"INTC1059\0".as_ptr() as *mut u8) as *const u8, (b"INTC1095\0".as_ptr() as *mut u8) as *const u8, (b"INTC100A\0".as_ptr() as *mut u8) as *const u8, (b"INTC10CF\0".as_ptr() as *mut u8) as *const u8, (b"INTC10DE\0".as_ptr() as *mut u8) as *const u8, (b"INTC10E0\0".as_ptr() as *mut u8) as *const u8, (b"INTC10E1\0".as_ptr() as *mut u8) as *const u8, (b"INTC10FA\0".as_ptr() as *mut u8) as *const u8, (b"RSCV0001\0".as_ptr() as *mut u8) as *const u8, (b"RSCV0002\0".as_ptr() as *mut u8) as *const u8, (b"RSCV0005\0".as_ptr() as *mut u8) as *const u8, (b"RSCV0006\0".as_ptr() as *mut u8) as *const u8, (b"PNP0C0F\0".as_ptr() as *mut u8) as *const u8, (b"ACPI0016\0".as_ptr() as *mut u8) as *const u8, core::ptr::null::<u8>()]);

fn acpi_find_parent_acpi_dev(mut handle: *mut ()) -> *mut acpi_device {
let mut adev: *mut acpi_device = unsafe { core::mem::zeroed() };
// Fixed hardware devices do not appear in the namespace and do not
// have handles, but we fabricate acpi_devices for them, so we have
// to deal with them specially.
if handle.is_null() {
return unsafe { acpi_root };
}
loop {
let status: u32 = unsafe { acpi_get_parent(handle, core::ptr::addr_of_mut!(handle)) };
if ACPI_FAILURE!(status) != 0 {
if status != EXCEP_ENV!(10i32) {
return unsafe { acpi_root };
}
return core::ptr::null_mut::<acpi_device>();
}
adev = acpi_fetch_acpi_dev(handle);
if !adev.is_null() {
break;
}
}
adev
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_get_ejd")]
extern "C" fn acpi_bus_get_ejd(handle: *mut (), ejd: *mut *mut ()) -> u32 {
let mut status: u32 = 0;
let mut tmp: *mut () = unsafe { core::mem::zeroed() };
let mut buffer: acpi_buffer = acpi_buffer { length: 1i32.wrapping_neg() as u64, pointer: core::ptr::null_mut::<()>() };
status = unsafe { acpi_get_handle(handle, c"_EJD".as_ptr() as *const u8, core::ptr::addr_of_mut!(tmp)) };
if ACPI_FAILURE!(status) != 0 {
return status;
}
status = unsafe { acpi_evaluate_object(handle, c"_EJD".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(buffer)) };
if ACPI_SUCCESS!(status) {
let obj: *mut acpi_object = buffer.pointer as *mut acpi_object;
status = unsafe { acpi_get_handle(ACPI_TO_POINTER!(18446744073709551615u64), (*obj).string.pointer as *const u8, ejd) };
unsafe { kfree(buffer.pointer as *const ()) };
}
status
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_bus_get_ejd_540: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut *mut ()) -> u32>, *mut ()>(Some(acpi_bus_get_ejd)) };

fn acpi_bus_extract_wakeup_device_power_package(dev: *mut acpi_device) -> i32 {
let mut sleep_state: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(sleep_state) as *mut i32;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut element: *mut acpi_object = unsafe { core::mem::zeroed() };
let mut package: *mut acpi_object = unsafe { core::mem::zeroed() };
let mut buffer: core::mem::MaybeUninit<acpi_buffer> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut acpi_buffer = core::ptr::addr_of_mut!(buffer) as *mut acpi_buffer;
let mut wakeup: *mut acpi_device_wakeup = unsafe { core::mem::zeroed() };
let mut handle: *mut () = unsafe { core::mem::zeroed() };
unsafe {
handle = (*dev).handle;
wakeup = core::ptr::addr_of_mut!((*dev).wakeup);
core::ptr::write(__slate_slot_5, acpi_buffer { length: 1i32.wrapping_neg() as u64, pointer: core::ptr::null_mut::<()>() });
package = core::ptr::null_mut::<acpi_object>();
element = core::ptr::null_mut::<acpi_object>();
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
err = 61i32.wrapping_neg();
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*wakeup).resources));
// _PRW
*__slate_slot_2 = acpi_evaluate_object(handle, c"_PRW".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(*__slate_slot_5));
if ACPI_FAILURE!(*__slate_slot_2) != 0 {
acpi_handle_printk(c"\x016".as_ptr() as *const u8, handle, c"_PRW evaluation failed: %s\n".as_ptr() as *const u8, acpi_format_exception(*__slate_slot_2));
return err;
} else {
'__join_0: {
package = (*__slate_slot_5).pointer as *mut acpi_object;
if !(package.is_null() || (*package).package.count < 2) {
element = unsafe { (*package).package.elements }.wrapping_offset(0 as isize);
if !element.is_null() {
if (*element).r#type == 4 {
if (*element).package.count < 2 || (*unsafe { (*element).package.elements }.wrapping_offset(0 as isize)).r#type != 20 || (*unsafe { (*element).package.elements }.wrapping_offset(1 as isize)).r#type != 1 {
break '__join_0;
} else {
(*wakeup).gpe_device = (*unsafe { (*element).package.elements }.wrapping_offset(0 as isize)).reference.handle;
(*wakeup).gpe_number = ((*unsafe { (*element).package.elements }.wrapping_offset(1 as isize)).integer.value as u32) as u64;
}
} else {
if (*element).r#type == 1 {
(*wakeup).gpe_device = core::ptr::null_mut::<()>();
(*wakeup).gpe_number = (*element).integer.value;
} else {
break '__join_0;
}
}
element = unsafe { (*package).package.elements }.wrapping_offset(1 as isize);
if (*element).r#type == 1 {
(*wakeup).sleep_state = (*element).integer.value;
err = acpi_extract_power_resources(package, 2u32, core::ptr::addr_of_mut!((*wakeup).resources));
if err == 0 {
if list_empty(core::ptr::addr_of_mut!((*wakeup).resources) as *const list_head) == 0 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = acpi_power_wakeup_list_init(core::ptr::addr_of_mut!((*wakeup).resources), core::ptr::addr_of_mut!(*__slate_slot_0));
if err != 0 {
acpi_handle_printk(c"\x014".as_ptr() as *const u8, handle, c"Retrieving current states of wakeup power resources failed\n".as_ptr() as *const u8);
acpi_power_resources_list_free(core::ptr::addr_of_mut!((*wakeup).resources));
} else {
if (*__slate_slot_0 as u64) < (*wakeup).sleep_state {
acpi_handle_printk(c"\x014".as_ptr() as *const u8, handle, c"Overriding _PRW sleep state (S%d) by S%d from power resources\n".as_ptr() as *const u8, (*wakeup).sleep_state as i32, *__slate_slot_0);
(*wakeup).sleep_state = *__slate_slot_0 as u64;
}
}
}
}
}
}
}
}
kfree((*__slate_slot_5).pointer as *const ());
return err;
}
}
}

fn acpi_wakeup_gpe_init(device: *mut acpi_device) -> bool {
// Lid
// Sleep button
let wakeup: *mut acpi_device_wakeup = unsafe { core::ptr::addr_of_mut!((*device).wakeup) };
let mut status: u32 = 0;
unsafe {
(*wakeup).flags.__slate_bits_0.__set_notifier_present(0u8);
}
let r#match: *const acpi_device_id = unsafe { acpi_match_acpi_device(core::ptr::addr_of!(button_device_ids.0) as *const acpi_device_id, device as *const acpi_device) };
if !r#match.is_null() && (unsafe { (*wakeup).sleep_state }) == 5 {
unsafe {
(*wakeup).sleep_state = 4;
}
}
status = unsafe { acpi_setup_gpe_for_wake((*device).handle, (*wakeup).gpe_device, (*wakeup).gpe_number as u32) };
ACPI_SUCCESS!(status)
}

static button_device_ids: __SlateAlign16<[acpi_device_id; 3]> = __SlateAlign16([{
let mut __t0: acpi_device_id = unsafe { core::mem::zeroed() };
__t0.id = [80u8, 78u8, 80u8, 48u8, 67u8, 48u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.driver_data = 0u64;
__t0
}, {
let mut __t1: acpi_device_id = unsafe { core::mem::zeroed() };
__t1.id = [80u8, 78u8, 80u8, 48u8, 67u8, 48u8, 69u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.driver_data = 0u64;
__t1
}, {
let mut __t2: acpi_device_id = unsafe { core::mem::zeroed() };
__t2.id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.driver_data = 0u64;
__t2
}]);

fn acpi_bus_get_wakeup_device_flags(device: *mut acpi_device) {
let mut err: i32 = 0;
// Presence of _PRW indicates wake capable
if !unsafe { acpi_has_method((*device).handle, c"_PRW".as_ptr() as *mut u8) } {
return;
}
err = acpi_bus_extract_wakeup_device_power_package(device);
if err != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*device).dev) as *const device, c"Unable to extract wakeup power resources".as_ptr() as *const u8) };
return;
}
unsafe {
let __t0 = acpi_wakeup_gpe_init(device) as u8;
(*device).wakeup.flags.__slate_bits_0.__set_valid(__t0);
}
unsafe {
(*device).wakeup.prepare_count = 0;
}
// Call _PSW/_DSW object to disable its ability to wake the sleeping
// system for the ACPI device with the _PRW object.
// The _PSW object is deprecated in ACPI 3.0 and is replaced by _DSW.
// So it is necessary to call _DSW object first. Only when it is not
// present will the _PSW object used.
err = unsafe { acpi_device_sleep_wake(device, 0i32, 0i32, 0i32) };
if err != 0 {
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017ACPI: error in _DSW or _PSW evaluation\n".as_ptr() as *const u8) };
}
__v1 = 0;
}
}

fn acpi_bus_init_power_state(device: *mut acpi_device, state: i32) {
let ps: *mut acpi_device_power_state = unsafe { (*device).power.states.as_mut_ptr() as *mut acpi_device_power_state }.wrapping_offset(state as isize);
let mut pathname: [u8; 5] = [95, 80, 82, 48i32.wrapping_add(state) as u8, 0];
let mut buffer: acpi_buffer = acpi_buffer { length: 1i32.wrapping_neg() as u64, pointer: core::ptr::null_mut::<()>() };
let mut status: u32 = 0;
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ps).resources) });
// Evaluate "_PRx" to get referenced power resources
status = unsafe { acpi_evaluate_object((*device).handle, pathname.as_mut_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(buffer)) };
if ACPI_SUCCESS!(status) {
let package: *mut acpi_object = buffer.pointer as *mut acpi_object;
if buffer.length != 0 && !package.is_null() && (unsafe { (*package).r#type }) == 4 && (unsafe { (*package).package.count }) != 0 {
unsafe { acpi_extract_power_resources(package, 0u32, core::ptr::addr_of_mut!((*ps).resources)) };
}
ACPI_FREE__2!(buffer.pointer);
}
// Evaluate "_PSx" to see if we can do explicit sets
pathname[2] = 83;
if unsafe { acpi_has_method((*device).handle, pathname.as_mut_ptr() as *mut u8) } {
unsafe {
(*ps).flags.__slate_bits_0.__set_explicit_set(1u8);
}
}
// State is valid if there are means to put the device into it.
if list_empty((unsafe { core::ptr::addr_of_mut!((*ps).resources) }) as *const list_head) == 0 || ((unsafe { (*ps).flags.__slate_bits_0.__get_explicit_set() }) as i32) != 0 {
unsafe {
(*ps).flags.__slate_bits_0.__set_valid(1u8);
}
}
unsafe {
(*ps).power = 1i32.wrapping_neg();
}
// Unknown - driver assigned
unsafe {
(*ps).latency = 1i32.wrapping_neg();
}
// Unknown - driver assigned
}

fn acpi_bus_get_power_flags(device: *mut acpi_device) {
let mut dsc: u64 = 0;
// Enumerate supported power management states
let mut i: u32 = 0;
// Presence of _PS0|_PR0 indicates 'power manageable'
let __v0: bool = !unsafe { acpi_has_method((*device).handle, c"_PS0".as_ptr() as *mut u8) } && !unsafe { acpi_has_method((*device).handle, c"_PR0".as_ptr() as *mut u8) };
if __v0 {
return;
}
unsafe {
(*device).flags.__slate_bits_0.__set_power_manageable(1u32);
}
// Power Management Flags
if unsafe { acpi_has_method((*device).handle, c"_PSC".as_ptr() as *mut u8) } {
unsafe {
(*device).power.flags.__slate_bits_0.__set_explicit_get(1u32);
}
}
if unsafe { acpi_has_method((*device).handle, c"_IRC".as_ptr() as *mut u8) } {
unsafe {
(*device).power.flags.__slate_bits_0.__set_inrush_current(1u32);
}
}
if unsafe { acpi_has_method((*device).handle, c"_DSW".as_ptr() as *mut u8) } {
unsafe {
(*device).power.flags.__slate_bits_0.__set_dsw_present(1u32);
}
}
unsafe { acpi_evaluate_integer((*device).handle, c"_DSC".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(dsc)) };
unsafe {
(*device).power.state_for_enumeration = dsc as u8;
}
while i <= 3 {
acpi_bus_init_power_state(device, i as i32);
i = i.wrapping_add(1);
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(4 as isize)).resources) });
// Set the defaults for D0 and D3hot (always supported).
unsafe {
(*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(0 as isize)).flags.__slate_bits_0.__set_valid(1u8);
}
unsafe {
(*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(0 as isize)).power = 100;
}
unsafe {
(*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(3 as isize)).flags.__slate_bits_0.__set_valid(1u8);
}
// Use power resources only if the D0 list of them is populated, because
// some platforms may provide _PR3 only to indicate D3cold support and
// in those cases the power resources list returned by it may be bogus.
if list_empty((unsafe { core::ptr::addr_of_mut!((*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(0 as isize)).resources) }) as *const list_head) == 0 {
unsafe {
(*device).power.flags.__slate_bits_0.__set_power_resources(1u32);
}
// D3cold is supported if the D3hot list of power resources is
// not empty.
if list_empty((unsafe { core::ptr::addr_of_mut!((*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(3 as isize)).resources) }) as *const list_head) == 0 {
unsafe {
(*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(4 as isize)).flags.__slate_bits_0.__set_valid(1u8);
}
}
}
if (unsafe { acpi_bus_init_power(device) }) != 0 {
unsafe {
(*device).flags.__slate_bits_0.__set_power_manageable(0u32);
}
}
}

fn acpi_bus_get_flags(device: *mut acpi_device) {
// Presence of _STA indicates 'dynamic_status'
if unsafe { acpi_has_method((*device).handle, c"_STA".as_ptr() as *mut u8) } {
unsafe {
(*device).flags.__slate_bits_0.__set_dynamic_status(1u32);
}
}
// Presence of _RMV indicates 'removable'
if unsafe { acpi_has_method((*device).handle, c"_RMV".as_ptr() as *mut u8) } {
unsafe {
(*device).flags.__slate_bits_0.__set_removable(1u32);
}
}
// Presence of _EJD|_EJ0 indicates 'ejectable'
let __v0: bool = (unsafe { acpi_has_method((*device).handle, c"_EJD".as_ptr() as *mut u8) }) || unsafe { acpi_has_method((*device).handle, c"_EJ0".as_ptr() as *mut u8) };
if __v0 {
unsafe {
(*device).flags.__slate_bits_0.__set_ejectable(1u32);
}
}
}

fn acpi_device_get_busid(device: *mut acpi_device) {
let mut bus_id: [u8; 5] = {
let mut __t0: [u8; 5] = unsafe { core::mem::zeroed() };
__t0[0] = 63u8;
__t0[1] = 0u8;
__t0
};
let mut buffer: acpi_buffer = acpi_buffer { length: core::mem::size_of::<[u8; 5]>() as u64, pointer: bus_id.as_mut_ptr() as *mut () };
let mut i: i32 = 0;
// Bus ID
// The device's Bus ID is simply the object name.
// TBD: Shouldn't this value be unique (within the ACPI namespace)?
if acpi_dev_parent(device).is_null() {
unsafe { sized_strscpy((*device).pnp.bus_id.as_mut_ptr() as *mut u8, c"ACPI".as_ptr() as *const u8, (core::mem::size_of::<[u8; 8]>() as u64).wrapping_add(__must_be_array__19!() as u64).wrapping_add(__must_be_cstr__18!() as u64).wrapping_add(__must_be_cstr__32!() as u64)) };
return;
}
match unsafe { (*device).device_type } {
4 => {
unsafe { sized_strscpy((*device).pnp.bus_id.as_mut_ptr() as *mut u8, c"PWRF".as_ptr() as *const u8, (core::mem::size_of::<[u8; 8]>() as u64).wrapping_add(__must_be_array__33!() as u64).wrapping_add(__must_be_cstr__7!() as u64).wrapping_add(__must_be_cstr__25!() as u64)) };
}
5 => {
unsafe { sized_strscpy((*device).pnp.bus_id.as_mut_ptr() as *mut u8, c"SLPF".as_ptr() as *const u8, (core::mem::size_of::<[u8; 8]>() as u64).wrapping_add(__must_be_array__4!() as u64).wrapping_add(__must_be_cstr__22!() as u64).wrapping_add(__must_be_cstr__45!() as u64)) };
}
6 => {
unsafe { sized_strscpy((*device).pnp.bus_id.as_mut_ptr() as *mut u8, c"ECDT".as_ptr() as *const u8, (core::mem::size_of::<[u8; 8]>() as u64).wrapping_add(__must_be_array__17!() as u64).wrapping_add(__must_be_cstr__23!() as u64).wrapping_add(__must_be_cstr__12!() as u64)) };
}
_ => {
unsafe { acpi_get_name((*device).handle, 1u32, core::ptr::addr_of_mut!(buffer)) };
// Clean up trailing underscores (if any)
i = 3;
while i > 1 {
if (bus_id[i as usize] as i32) == 95 {
bus_id[i as usize] = 0;
} else {
break;
}
i = i.wrapping_sub(1);
}
unsafe { sized_strscpy((*device).pnp.bus_id.as_mut_ptr() as *mut u8, bus_id.as_mut_ptr() as *const u8, (core::mem::size_of::<[u8; 8]>() as u64).wrapping_add(__must_be_array__38!() as u64).wrapping_add(__must_be_cstr__6!() as u64).wrapping_add(__must_be_cstr__13!() as u64)) };
}
}
}

/// acpi_ata_match - see if an acpi object is an ATA device
///
/// If an acpi object has one of the ACPI ATA methods defined,
/// then we can safely call it an ATA device.
#[unsafe(no_mangle)]
extern "C" fn acpi_ata_match(handle: *mut ()) -> bool {
let __v0: bool = (unsafe { acpi_has_method(handle, c"_GTF".as_ptr() as *mut u8) }) || unsafe { acpi_has_method(handle, c"_GTM".as_ptr() as *mut u8) };
let __v1: bool = __v0 || unsafe { acpi_has_method(handle, c"_STM".as_ptr() as *mut u8) };
__v1 || unsafe { acpi_has_method(handle, c"_SDD".as_ptr() as *mut u8) }
}

/// acpi_bay_match - see if an acpi object is an ejectable driver bay
///
/// If an acpi object is ejectable and has one of the ACPI ATA methods defined,
/// then we can safely call it an ejectable drive bay
#[unsafe(no_mangle)]
extern "C" fn acpi_bay_match(handle: *mut ()) -> bool {
let mut phandle: *mut () = unsafe { core::mem::zeroed() };
if !unsafe { acpi_has_method(handle, c"_EJ0".as_ptr() as *mut u8) } {
return false;
}
if acpi_ata_match(handle) {
return true;
}
if ACPI_FAILURE!(unsafe { acpi_get_parent(handle, core::ptr::addr_of_mut!(phandle)) }) != 0 {
return false;
}
acpi_ata_match(phandle)
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_is_battery(adev: *mut acpi_device) -> bool {
let mut hwid: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
hwid = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*adev).pnp.ids) });
while list_entry_is_head!(hwid, unsafe { core::ptr::addr_of_mut!((*adev).pnp.ids) }, list) == 0 {
if (unsafe { strcmp(c"PNP0C0A".as_ptr() as *const u8, (*hwid).id) }) == 0 {
return true;
}
hwid = list_next_entry__4!(hwid, list);
}
false
}

fn is_ejectable_bay(adev: *mut acpi_device) -> bool {
let handle: *mut () = unsafe { (*adev).handle };
let __v0: bool = (unsafe { acpi_has_method(handle, c"_EJ0".as_ptr() as *mut u8) }) && acpi_device_is_battery(adev);
if __v0 {
return true;
}
acpi_bay_match(handle)
}

/// acpi_dock_match - see if an acpi object has a _DCK method
#[unsafe(no_mangle)]
extern "C" fn acpi_dock_match(handle: *mut ()) -> bool {
unsafe { acpi_has_method(handle, c"_DCK".as_ptr() as *mut u8) }
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_backlight_cap_match")]
extern "C" fn acpi_backlight_cap_match(handle: *mut (), level: u32, context: *mut (), return_value: *mut *mut ()) -> u32 {
let __v0: bool = (unsafe { acpi_has_method(handle, c"_BCM".as_ptr() as *mut u8) }) && unsafe { acpi_has_method(handle, c"_BCL".as_ptr() as *mut u8) };
if __v0 {
let mut __v1: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"Found generic backlight support\n".as_ptr() as *const u8) };
}
__v1 = 0;
unsafe {
*(context as *mut i64) = *(context as *mut i64) | 8;
}
// We have backlight support, no need to scan further
return EXCEP_CTL!(3i32);
}
0
}

/// Returns true if the ACPI object is a video device which can be
/// handled by video.ko.
/// The device will get a Linux specific CID added in scan.c to
/// identify the device as an ACPI graphics device
/// Be aware that the graphics device may not be physically present
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_is_video_device")]
extern "C" fn acpi_is_video_device(handle: *mut ()) -> i64 {
let mut video_caps: i64 = 0;
// Is this device able to support video switching ?
let __v0: bool = (unsafe { acpi_has_method(handle, c"_DOD".as_ptr() as *mut u8) }) || unsafe { acpi_has_method(handle, c"_DOS".as_ptr() as *mut u8) };
if __v0 {
video_caps |= 1;
}
// Is this device able to retrieve a video ROM ?
if unsafe { acpi_has_method(handle, c"_ROM".as_ptr() as *mut u8) } {
video_caps |= 4;
}
// Is this device able to configure which video head to be POSTed ?
let __v5: bool = (unsafe { acpi_has_method(handle, c"_VPO".as_ptr() as *mut u8) }) && unsafe { acpi_has_method(handle, c"_GPD".as_ptr() as *mut u8) };
let __v6: bool = __v5 && unsafe { acpi_has_method(handle, c"_SPD".as_ptr() as *mut u8) };
if __v6 {
video_caps |= 2;
}
// Only check for backlight functionality if one of the above hit.
if video_caps != 0 {
unsafe { acpi_walk_namespace(6u32, handle, 4294967295u32, Some(acpi_backlight_cap_match), None, core::ptr::addr_of_mut!(video_caps) as *mut (), core::ptr::null_mut::<*mut ()>()) };
}
video_caps
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_is_video_device_541: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ()) -> i64>, *mut ()>(Some(acpi_is_video_device)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_device_hid_543: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut acpi_device) -> *const u8>, *mut ()>(Some(acpi_device_hid)) };

fn acpi_add_id(pnp: *mut acpi_device_pnp, dev_id: *const u8) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut id: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
let mut __v0: *mut acpi_hardware_id;
let __obj_size: u64 = size_mul(core::mem::size_of::<acpi_hardware_id>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut acpi_hardware_id;
id = __v0;
if id.is_null() {
return;
}
unsafe {
(*id).id = kstrdup_const(dev_id, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
if unsafe { (*id).id }.is_null() {
unsafe { kfree(id as *const ()) };
return;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*id).list) }, unsafe { core::ptr::addr_of_mut!((*pnp).ids) });
unsafe {
(*pnp).r#type.__slate_bits_0.__set_hardware_id(1u32);
}
}

/// Old IBM workstations have a DSDT bug wherein the SMBus object
/// lacks the SMBUS01 HID and the methods do not have the necessary "_"
/// prefix.  Work around this.
fn acpi_ibm_smbus_match(handle: *mut ()) -> bool {
let mut node_name: [u8; 5] = [0u8; 5];
let mut path: acpi_buffer = acpi_buffer { length: core::mem::size_of::<[u8; 5]>() as u64, pointer: node_name.as_mut_ptr() as *mut () };
if (unsafe { dmi_name_in_vendors(c"IBM".as_ptr() as *const u8) }) == 0 {
return false;
}
// Look for SMBS object
let __v0: bool = ACPI_FAILURE!(unsafe { acpi_get_name(handle, 1u32, core::ptr::addr_of_mut!(path)) }) != 0 || (unsafe { strcmp(c"SMBS".as_ptr() as *const u8, path.pointer as *const u8) }) != 0;
if __v0 {
return false;
}
// Does it have the necessary (but misnamed) methods?
let __v1: bool = (unsafe { acpi_has_method(handle, c"SBI".as_ptr() as *mut u8) }) && unsafe { acpi_has_method(handle, c"SBR".as_ptr() as *mut u8) };
let __v2: bool = __v1 && unsafe { acpi_has_method(handle, c"SBW".as_ptr() as *mut u8) };
if __v2 {
return true;
}
false
}

fn acpi_object_is_system_bus(handle: *mut ()) -> bool {
let mut tmp: *mut () = unsafe { core::mem::zeroed() };
if ACPI_SUCCESS!(unsafe { acpi_get_handle(core::ptr::null_mut::<()>(), c"\\_SB".as_ptr() as *const u8, core::ptr::addr_of_mut!(tmp)) }) && tmp == handle {
return true;
}
if ACPI_SUCCESS!(unsafe { acpi_get_handle(core::ptr::null_mut::<()>(), c"\\_TZ".as_ptr() as *const u8, core::ptr::addr_of_mut!(tmp)) }) && tmp == handle {
return true;
}
false
}

fn acpi_set_pnp_ids(handle: *mut (), pnp: *mut acpi_device_pnp, device_type: i32) {
let mut info: *mut acpi_device_info = core::ptr::null_mut::<acpi_device_info>();
let mut cid_list: *mut acpi_pnp_device_id_list = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
match device_type {
0 => {
if handle == ACPI_TO_POINTER!(18446744073709551615u64) {
acpi_add_id(pnp, c"LNXSYSTM".as_ptr() as *const u8);
} else {
unsafe { acpi_get_object_info(handle, core::ptr::addr_of_mut!(info)) };
if info.is_null() {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x013ACPI: %s: Error reading device info\n".as_ptr() as *const u8, c"acpi_set_pnp_ids".as_ptr() as *mut u8) };
return;
}
if ((unsafe { (*info).valid }) as i32) & 4 != 0 {
acpi_add_id(pnp, (unsafe { (*info).hardware_id.string }) as *const u8);
unsafe {
(*pnp).r#type.__slate_bits_0.__set_platform_id(1u32);
}
}
if ((unsafe { (*info).valid }) as i32) & 32 != 0 {
cid_list = unsafe { core::ptr::addr_of_mut!((*info).compatible_id_list) };
i = 0;
while (i as u32) < unsafe { (*cid_list).count } {
acpi_add_id(pnp, (unsafe { (*(core::ptr::addr_of_mut!((*cid_list).ids) as *mut acpi_pnp_device_id).wrapping_offset(i as isize)).string }) as *const u8);
i = i.wrapping_add(1);
}
}
if ((unsafe { (*info).valid }) as i32) & 2 != 0 {
unsafe {
(*pnp).bus_address = (*info).address;
}
unsafe {
(*pnp).r#type.__slate_bits_0.__set_bus_address(1u32);
}
}
if ((unsafe { (*info).valid }) as i32) & 8 != 0 {
unsafe {
(*pnp).unique_id = kstrdup((*info).unique_id.string as *const u8, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
}
if ((unsafe { (*info).valid }) as i32) & 64 != 0 {
acpi_add_id(pnp, (unsafe { (*info).class_code.string }) as *const u8);
}
unsafe { kfree(info as *const ()) };
// Some devices don't reliably have _HIDs & _CIDs, so add
// synthetic HIDs to make sure drivers can find them.
if acpi_is_video_device(handle) != 0 {
acpi_add_id(pnp, c"LNXVIDEO".as_ptr() as *const u8);
unsafe {
(*pnp).r#type.__slate_bits_0.__set_backlight(1u32);
}
} else {
if acpi_bay_match(handle) {
acpi_add_id(pnp, c"LNXIOBAY".as_ptr() as *const u8);
} else {
if acpi_dock_match(handle) {
acpi_add_id(pnp, c"LNXDOCK".as_ptr() as *const u8);
} else {
if acpi_ibm_smbus_match(handle) {
acpi_add_id(pnp, c"SMBUSIBM".as_ptr() as *const u8);
} else {
let __v6: bool = list_empty((unsafe { core::ptr::addr_of_mut!((*pnp).ids) }) as *const list_head) != 0 && acpi_object_is_system_bus(handle);
if __v6 {
// \_SB, \_TZ, LNXSYBUS
acpi_add_id(pnp, c"LNXSYBUS".as_ptr() as *const u8);
}
}
}
}
}
}
}
1 => {
acpi_add_id(pnp, c"LNXPOWER".as_ptr() as *const u8);
}
2 => {
acpi_add_id(pnp, c"LNXCPU".as_ptr() as *const u8);
}
3 => {
acpi_add_id(pnp, c"LNXTHERM".as_ptr() as *const u8);
unsafe {
(*pnp).r#type.__slate_bits_0.__set_platform_id(1u32);
}
}
4 => {
acpi_add_id(pnp, c"LNXPWRBN".as_ptr() as *const u8);
}
5 => {
acpi_add_id(pnp, c"LNXSLPBN".as_ptr() as *const u8);
}
6 => {
acpi_add_id(pnp, c"LNXEC".as_ptr() as *const u8);
}
_ => {
}
}
}

#[unsafe(no_mangle)]
extern "C" fn acpi_free_pnp_ids(pnp: *mut acpi_device_pnp) {
let mut id: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
let mut tmp: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
id = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*pnp).ids) });
tmp = list_next_entry__4!(id, list);
while list_entry_is_head!(id, unsafe { core::ptr::addr_of_mut!((*pnp).ids) }, list) == 0 {
unsafe { kfree_const((*id).id as *const ()) };
unsafe { kfree(id as *const ()) };
id = tmp;
tmp = list_next_entry__4!(tmp, list);
}
unsafe { kfree((*pnp).unique_id as *const ()) };
}

/// acpi_dma_supported - Check DMA support for the specified device.
/// @adev: The pointer to acpi device
///
/// Return false if DMA is not supported. Otherwise, return true
#[unsafe(no_mangle)]
extern "C" fn acpi_dma_supported(adev: *const acpi_device) -> bool {
if adev.is_null() {
return false;
}
if ((unsafe { (*adev).flags.__slate_bits_0.__get_cca_seen() }) as i32) != 0 {
return true;
}
// Per ACPI 6.0 sec 6.2.17, assume devices can do cache-coherent
// DMA on "Intel platforms".  Presumably that includes all x86 and
// ia64, and other arches will set CONFIG_ACPI_CCA_REQUIRED=y.
if IS_ENABLED!() == 0 {
return true;
}
false
}

/// acpi_get_dma_attr - Check the supported DMA attr for the specified device.
/// @adev: The pointer to acpi device
///
/// Return enum dev_dma_attr.
#[unsafe(no_mangle)]
extern "C" fn acpi_get_dma_attr(adev: *mut acpi_device) -> u32 {
if !acpi_dma_supported(adev as *const acpi_device) {
return 0;
}
if ((unsafe { (*adev).flags.__slate_bits_0.__get_coherent_dma() }) as i32) != 0 {
return 2;
} else {
return 1;
}
}

/// acpi_dma_get_range() - Get device DMA parameters.
///
/// @dev: device to configure
/// @map: pointer to DMA ranges result
///
/// Evaluate DMA regions and return pointer to DMA regions on
/// parsing success; it does not update the passed in values on failure.
///
/// Return 0 on success, < 0 on failure.
#[unsafe(no_mangle)]
extern "C" fn acpi_dma_get_range(dev: *mut device, map: *mut *const bus_dma_region) -> i32 {
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<*mut bus_dma_region> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut bus_dma_region = core::ptr::addr_of_mut!(r) as *mut *mut bus_dma_region;
let mut dma_dev: *mut device = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut rentry: core::mem::MaybeUninit<*mut resource_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut resource_entry = core::ptr::addr_of_mut!(rentry) as *mut *mut resource_entry;
let mut list: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut list_head = core::ptr::addr_of_mut!(list) as *mut list_head;
let mut adev: core::mem::MaybeUninit<*mut acpi_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut acpi_device = core::ptr::addr_of_mut!(adev) as *mut *mut acpi_device;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_13, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_13), prev: core::ptr::addr_of_mut!(*__slate_slot_13) });
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
dma_dev = dev;
let __slate_auto_bytes_5 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
// Walk the device tree chasing an ACPI companion with a _DMA
// object while we go. Stop if we find a device with an ACPI
// companion containing a _DMA method.
loop {
*__slate_slot_14 = ACPI_COMPANION!(dma_dev);
if !(*__slate_slot_14).is_null() {
__v8 = acpi_has_method((*(*__slate_slot_14)).handle, c"_DMA".as_ptr() as *mut u8);
} else {
__v8 = false;
}
if __v8 {
break;
} else {
dma_dev = (*dma_dev).parent;
if dma_dev.is_null() {
break;
}
}
}
if dma_dev.is_null() {
return 19i32.wrapping_neg();
} else {
if !acpi_has_method((*(*__slate_slot_14)).handle, c"_CRS".as_ptr() as *mut u8) {
acpi_handle_printk(c"\x014".as_ptr() as *const u8, (*(*__slate_slot_14)).handle, c"_DMA is valid only if _CRS is present\n".as_ptr() as *const u8);
return 22i32.wrapping_neg();
} else {
'__join_0: {
*__slate_slot_11 = acpi_dev_get_dma_resources(*__slate_slot_14, core::ptr::addr_of_mut!(*__slate_slot_13));
if *__slate_slot_11 > 0 {
__obj_size = size_mul(core::mem::size_of::<bus_dma_region>() as u64, (*__slate_slot_11).wrapping_add(1) as u64);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
}
*__slate_slot_9 = *__slate_slot_3 as *mut bus_dma_region;
if (*__slate_slot_9).is_null() {
*__slate_slot_11 = 12i32.wrapping_neg();
} else {
*map = *__slate_slot_9 as *const bus_dma_region;
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_13));
loop {
if list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!(*__slate_slot_13), node) == 0 {
if (*(*(*__slate_slot_12)).res).start >= (*(*(*__slate_slot_12)).res).end {
break;
} else {
(*(*__slate_slot_9)).cpu_start = (*(*(*__slate_slot_12)).res).start;
(*(*__slate_slot_9)).dma_start = unsafe { (*(*(*__slate_slot_12)).res).start }.wrapping_sub((*(*__slate_slot_12)).offset);
(*(*__slate_slot_9)).size = resource_size((*(*__slate_slot_12)).res as *const resource);
let __v1: *mut bus_dma_region = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_offset(1 as isize);
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, node);
}
} else {
break '__join_0;
}
}
kfree(*map as *const ());
*map = core::ptr::null::<bus_dma_region>();
*__slate_slot_11 = 22i32.wrapping_neg();
}
}
}
acpi_dev_free_resource_list(core::ptr::addr_of_mut!(*__slate_slot_13));
return if *__slate_slot_11 >= 0 { 0 } else { *__slate_slot_11 };
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn acpi_iommu_fwspec_init(dev: *mut device, mut id: u32, fwnode: *mut fwnode_handle) -> i32 {
let ret: i32 = unsafe { iommu_fwspec_init(dev, fwnode) };
if ret != 0 {
return ret;
}
unsafe { iommu_fwspec_add_ids(dev, core::ptr::addr_of_mut!(id) as *const u32, 1i32) }
}

fn acpi_iommu_configure_id(dev: *mut device, id_in: *const u32) -> i32 {
let mut err: i32 = 0;
// Serialise to make dev->iommu stable under our potential fwspec
unsafe { mutex_lock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
// If we already translated the fwspec there is nothing left to do
if !dev_iommu_fwspec_get(dev).is_null() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
return 0;
}
err = iort_iommu_configure_id(dev, id_in);
if err != 0 && err != 517i32.wrapping_neg() {
err = rimt_iommu_configure_id(dev, id_in);
}
if err != 0 && err != 517i32.wrapping_neg() {
err = viot_iommu_configure(dev);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(iommu_probe_device_lock)) };
err
}

/// acpi_dma_configure_id - Set-up DMA configuration for the device.
/// @dev: The pointer to the device
/// @attr: device dma attributes
/// @input_id: input device id const value pointer
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_dma_configure_id")]
extern "C" fn acpi_dma_configure_id(dev: *mut device, attr: u32, input_id: *const u32) -> i32 {
let mut ret: i32 = 0;
if (attr as u32) == 0 {
set_dma_ops(dev, unsafe { core::ptr::addr_of!(dma_dummy_ops) });
return 0;
}
acpi_arch_dma_setup(dev);
// Ignore all other errors apart from EPROBE_DEFER
ret = acpi_iommu_configure_id(dev, input_id);
if ret == 517i32.wrapping_neg() {
return 517i32.wrapping_neg();
}
if ret != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, dev as *const device, c"Adding to IOMMU failed: %d\n".as_ptr() as *const u8, ret) };
}
}
arch_setup_dma_ops(dev, (attr as u32) == 2);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_dma_configure_id_544: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, u32, *const u32) -> i32>, *mut ()>(Some(acpi_dma_configure_id)) };

fn acpi_init_coherency(adev: *mut acpi_device) {
let mut cca: u64 = 0;
let mut status: u32 = 0;
let parent: *mut acpi_device = acpi_dev_parent(adev);
if !parent.is_null() && ((unsafe { (*parent).flags.__slate_bits_0.__get_cca_seen() }) as i32) != 0 {
// From ACPI spec, OSPM will ignore _CCA if an ancestor
// already saw one.
unsafe {
(*adev).flags.__slate_bits_0.__set_cca_seen(1u32);
}
cca = ((unsafe { (*parent).flags.__slate_bits_0.__get_coherent_dma() }) as i32) as u64;
} else {
status = unsafe { acpi_evaluate_integer((*adev).handle, c"_CCA".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(cca)) };
if ACPI_SUCCESS!(status) {
unsafe {
(*adev).flags.__slate_bits_0.__set_cca_seen(1u32);
}
} else {
if IS_ENABLED!() == 0 {
cca = 1;
} else {
let mut __v0: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, (*adev).handle, c"ACPI device is missing _CCA.\n".as_ptr() as *const u8) };
}
__v0 = 0;
}
}
// If architecture does not specify that _CCA is
// required for DMA-able devices (e.g. x86),
// we default to _CCA=1.
}
unsafe {
(*adev).flags.__slate_bits_0.__set_coherent_dma(cca as u32);
}
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_check_serial_bus_slave")]
extern "C" fn acpi_check_serial_bus_slave(ares: *mut acpi_resource, data: *mut ()) -> i32 {
if (unsafe { (*ares).r#type }) != 19 {
return 1;
}
unsafe {
*(data as *mut bool) = true;
}
// no need to do more checking
1i32.wrapping_neg()
}

fn acpi_is_indirect_io_slave(device: *mut acpi_device) -> bool {
let parent: *mut acpi_device = acpi_dev_parent(device);
!parent.is_null() && (unsafe { acpi_match_device_ids(parent, core::ptr::addr_of!(indirect_io_hosts.0) as *const acpi_device_id) }) == 0
}

static indirect_io_hosts: __SlateAlign16<[acpi_device_id; 2]> = __SlateAlign16([{
let mut __t0: acpi_device_id = unsafe { core::mem::zeroed() };
__t0.id = [72u8, 73u8, 83u8, 73u8, 48u8, 49u8, 57u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.driver_data = 0u64;
__t0
}, {
let mut __t1: acpi_device_id = unsafe { core::mem::zeroed() };
__t1
}]);

fn acpi_device_enumeration_by_parent(device: *mut acpi_device) -> bool {
let mut resource_list: list_head = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(resource_list) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut is_serial_bus_slave: bool = false;
// These devices have multiple SerialBus resources and a client
// device must be instantiated for each of them, each with
// its own device id.
// Normally we only instantiate one client device for the first
// resource, using the ACPI HID as id. These special cases are handled
// by the drivers/platform/x86/serial-multi-instantiate.c driver, which
// knows which client device id to use for each resource.
// Non-conforming _HID for Cirrus Logic already released
// Some ACPI devs contain SerialBus resources even though they are not
// attached to a serial bus at all.
// HIDs of device with an UartSerialBusV2 resource for which userspace
// expects a regular tty cdev to be created (instead of the in kernel
// serdev) and which have a kernel driver which expects a platform_dev
// such as the rfkill-gpio driver.
if acpi_is_indirect_io_slave(device) {
return true;
}
// Macs use device properties in lieu of _CRS resources
let __v0: bool = if unsafe { x86_apple_machine } {
let __v1: bool = (unsafe { fwnode_property_present(core::ptr::addr_of_mut!((*device).fwnode) as *const fwnode_handle, c"spiSclkPeriod".as_ptr() as *const u8) }) || unsafe { fwnode_property_present(core::ptr::addr_of_mut!((*device).fwnode) as *const fwnode_handle, c"i2cAddress".as_ptr() as *const u8) };
let __v2: bool = __v1 || unsafe { fwnode_property_present(core::ptr::addr_of_mut!((*device).fwnode) as *const fwnode_handle, c"baud".as_ptr() as *const u8) };
__v2
} else {
false
};
if __v0 {
return true;
}
if unsafe { acpi_dev_is_video_device(device) } {
return false;
}
if (unsafe { acpi_match_device_ids(device, core::ptr::addr_of!(ignore_serial_bus_ids.0) as *const acpi_device_id) }) == 0 {
return false;
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!(resource_list));
unsafe { acpi_dev_get_resources(device, core::ptr::addr_of_mut!(resource_list), Some(acpi_check_serial_bus_slave), core::ptr::addr_of_mut!(is_serial_bus_slave) as *mut ()) };
unsafe { acpi_dev_free_resource_list(core::ptr::addr_of_mut!(resource_list)) };
is_serial_bus_slave
}

static ignore_serial_bus_ids: __SlateAlign16<[acpi_device_id; 16]> = __SlateAlign16([{
let mut __t0: acpi_device_id = unsafe { core::mem::zeroed() };
__t0.id = [65u8, 87u8, 68u8, 90u8, 56u8, 51u8, 57u8, 57u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0
}, {
let mut __t1: acpi_device_id = unsafe { core::mem::zeroed() };
__t1.id = [66u8, 83u8, 71u8, 49u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1
}, {
let mut __t2: acpi_device_id = unsafe { core::mem::zeroed() };
__t2.id = [66u8, 83u8, 71u8, 50u8, 49u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
}, {
let mut __t3: acpi_device_id = unsafe { core::mem::zeroed() };
__t3.id = [67u8, 83u8, 67u8, 51u8, 53u8, 53u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
}, {
let mut __t4: acpi_device_id = unsafe { core::mem::zeroed() };
__t4.id = [67u8, 83u8, 67u8, 51u8, 53u8, 53u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4
}, {
let mut __t5: acpi_device_id = unsafe { core::mem::zeroed() };
__t5.id = [67u8, 83u8, 67u8, 51u8, 53u8, 53u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5
}, {
let mut __t6: acpi_device_id = unsafe { core::mem::zeroed() };
__t6.id = [67u8, 83u8, 67u8, 51u8, 53u8, 53u8, 55u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
}, {
let mut __t7: acpi_device_id = unsafe { core::mem::zeroed() };
__t7.id = [73u8, 78u8, 84u8, 51u8, 51u8, 70u8, 69u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
}, {
let mut __t8: acpi_device_id = unsafe { core::mem::zeroed() };
__t8.id = [73u8, 78u8, 84u8, 51u8, 53u8, 49u8, 53u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8
}, {
let mut __t9: acpi_device_id = unsafe { core::mem::zeroed() };
__t9.id = [84u8, 88u8, 78u8, 87u8, 50u8, 55u8, 56u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t9
}, {
let mut __t10: acpi_device_id = unsafe { core::mem::zeroed() };
__t10.id = [67u8, 76u8, 83u8, 65u8, 48u8, 49u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10
}, {
let mut __t11: acpi_device_id = unsafe { core::mem::zeroed() };
__t11.id = [67u8, 76u8, 83u8, 65u8, 48u8, 49u8, 48u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
}, {
let mut __t12: acpi_device_id = unsafe { core::mem::zeroed() };
__t12.id = [77u8, 83u8, 72u8, 87u8, 48u8, 48u8, 50u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12
}, {
let mut __t13: acpi_device_id = unsafe { core::mem::zeroed() };
__t13.id = [66u8, 67u8, 77u8, 52u8, 55u8, 53u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13
}, {
let mut __t14: acpi_device_id = unsafe { core::mem::zeroed() };
__t14.id = [76u8, 78u8, 86u8, 52u8, 55u8, 53u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14
}, {
let mut __t15: acpi_device_id = unsafe { core::mem::zeroed() };
__t15
}]);

#[unsafe(no_mangle)]
extern "C" fn acpi_init_device_object(device: *mut acpi_device, handle: *mut (), r#type: i32, release: Option<unsafe extern "C" fn(*mut device)>) {
let parent: *mut acpi_device = acpi_find_parent_acpi_dev(handle);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*device).pnp.ids) });
unsafe {
(*device).device_type = r#type;
}
unsafe {
(*device).handle = handle;
}
unsafe {
(*device).dev.parent = if !parent.is_null() { core::ptr::addr_of_mut!((*parent).dev) } else { core::ptr::null_mut::<device>() };
}
unsafe {
(*device).dev.release = release;
}
unsafe {
(*device).dev.bus = core::ptr::addr_of!(acpi_bus_type);
}
unsafe {
(*device).dev.groups = core::ptr::addr_of_mut!(acpi_groups) as *mut *const attribute_group;
}
device_set_pm_not_required(unsafe { core::ptr::addr_of_mut!((*device).dev) });
fwnode_init(unsafe { core::ptr::addr_of_mut!((*device).fwnode) }, unsafe { core::ptr::addr_of!(acpi_device_fwnode_ops) });
acpi_set_device_status(device, 15u32);
acpi_device_get_busid(device);
acpi_set_pnp_ids(handle, unsafe { core::ptr::addr_of_mut!((*device).pnp) }, r#type);
unsafe { acpi_init_properties(device) };
acpi_bus_get_flags(device);
unsafe {
(*device).flags.__slate_bits_0.__set_initialized(1u32);
}
unsafe {
let __t0 = acpi_device_enumeration_by_parent(device) as u32;
(*device).flags.__slate_bits_0.__set_enumeration_by_parent(__t0);
}
acpi_device_clear_enumerated(device);
unsafe { device_initialize(core::ptr::addr_of_mut!((*device).dev)) };
dev_set_uevent_suppress(unsafe { core::ptr::addr_of_mut!((*device).dev) }, 1i32);
acpi_init_coherency(device);
}

fn acpi_scan_dep_init(adev: *mut acpi_device) {
let mut dep: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
dep = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(acpi_dep_list) });
while list_entry_is_head!(dep, unsafe { core::ptr::addr_of_mut!(acpi_dep_list) }, node) == 0 {
if (unsafe { (*dep).consumer }) == unsafe { (*adev).handle } {
if unsafe { (*dep).honor_dep } {
unsafe {
(*adev).flags.__slate_bits_0.__set_honor_deps(1u32);
}
}
if !unsafe { (*dep).met } {
let __v2: u32 = unsafe { (*adev).dep_unmet };
unsafe {
(*adev).dep_unmet = __v2.wrapping_add(1);
}
}
}
dep = list_next_entry__10!(dep, node);
}
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_add_finalize(device: *mut acpi_device) {
dev_set_uevent_suppress(unsafe { core::ptr::addr_of_mut!((*device).dev) }, 0i32);
unsafe { kobject_uevent(core::ptr::addr_of_mut!((*device).dev.kobj), 0u32) };
}

fn acpi_scan_init_status(adev: *mut acpi_device) {
if (unsafe { acpi_bus_get_status(adev) }) != 0 {
acpi_set_device_status(adev, 0u32);
}
}

fn acpi_add_single_object(child: &mut *mut acpi_device, handle: *mut (), r#type: i32, dep_init: bool) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut device: *mut acpi_device = unsafe { core::mem::zeroed() };
let mut release_dep_lock: bool = false;
let mut result: i32 = 0;
let mut __v0: *mut acpi_device;
let __obj_size: u64 = size_mul(core::mem::size_of::<acpi_device>() as u64, 1u64);
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
__v0 = __v1 as *mut acpi_device;
device = __v0;
if device.is_null() {
return 12i32.wrapping_neg();
}
acpi_init_device_object(device, handle, r#type, Some(acpi_device_release));
// Getting the status is delayed till here so that we can call
// acpi_bus_get_status() and use its quirk handling.  Note that
// this must be done before the get power-/wakeup_dev-flags calls.
if r#type == 0 || r#type == 2 {
if dep_init {
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
// Hold the lock until the acpi_tie_acpi_dev() call
// below to prevent concurrent acpi_scan_clear_dep()
// from deleting a dependency list entry without
// updating dep_unmet for the device.
release_dep_lock = true;
acpi_scan_dep_init(device);
}
acpi_scan_init_status(device);
}
acpi_bus_get_power_flags(device);
acpi_bus_get_wakeup_device_flags(device);
result = acpi_tie_acpi_dev(device);
if release_dep_lock {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
}
if result == 0 {
result = acpi_device_add(device);
}
if result != 0 {
acpi_dev_put(device);
return result;
}
unsafe { acpi_power_add_remove_device(device, true) };
acpi_device_add_finalize(device);
let mut __v6: i32;
if false {
let __v7: *const u8 = if !unsafe { (*device).dev.parent }.is_null() {
dev_name((unsafe { (*device).dev.parent }) as *const device)
} else {
c"(null)".as_ptr() as *const u8
};
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"Added as %s, parent %s\n".as_ptr() as *const u8, dev_name(core::ptr::addr_of_mut!((*device).dev) as *const device), __v7) };
}
__v6 = 0;
unsafe {
*(child as *mut *mut acpi_device) = device;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_get_resource_memory")]
extern "C" fn acpi_get_resource_memory(ares: *mut acpi_resource, context: *mut ()) -> u32 {
let res: *mut resource = context as *mut resource;
if unsafe { acpi_dev_resource_memory(ares, res) } {
return EXCEP_CTL!(3i32);
}
0
}

fn acpi_device_should_be_hidden(handle: *mut ()) -> bool {
let mut status: u32 = 0;
let mut res: resource = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(res) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
// Check if it should ignore the UART device
let __v0: bool = (unsafe { spcr_uart_addr }) != 0 && unsafe { acpi_has_method(handle, c"_CRS".as_ptr() as *mut u8) };
if !__v0 {
return false;
}
// The UART device described in SPCR table is assumed to have only one
// memory resource present. So we only look for the first one here.
status = unsafe { acpi_walk_resources(handle, c"_CRS".as_ptr() as *mut u8, Some(acpi_get_resource_memory), core::ptr::addr_of_mut!(res) as *mut ()) };
if ACPI_FAILURE!(status) != 0 || res.start != unsafe { spcr_uart_addr } {
return false;
}
unsafe { acpi_handle_printk(c"\x016".as_ptr() as *const u8, handle, c"The UART device @%pa in SPCR table will be hidden\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(res.start)) };
true
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_is_present(adev: *const acpi_device) -> bool {
((unsafe { (*adev).status.__slate_bits_0.__get_present() }) as i32) != 0 || ((unsafe { (*adev).status.__slate_bits_0.__get_functional() }) as i32) != 0
}

#[unsafe(no_mangle)]
extern "C" fn acpi_device_is_enabled(adev: *const acpi_device) -> bool {
((unsafe { (*adev).status.__slate_bits_0.__get_enabled() }) as i32) != 0
}

fn acpi_scan_handler_matching(handler: *mut acpi_scan_handler, idstr: *const u8, matchid: *mut *const acpi_device_id) -> bool {
let mut devid: *const acpi_device_id = unsafe { core::mem::zeroed() };
if (unsafe { (*handler).r#match }) != None {
return unsafe { unsafe { (*handler).r#match }.unwrap()(idstr, matchid) };
}
devid = unsafe { (*handler).ids };
while (unsafe { *((*devid).id.as_ptr() as *const u8).wrapping_offset(0 as isize) }) != 0 {
if (unsafe { strcmp((((*devid).id.as_ptr() as *const u8) as *mut u8) as *const u8, idstr) }) == 0 {
if !matchid.is_null() {
unsafe {
*matchid = devid;
}
}
return true;
}
let __v1: *const acpi_device_id = devid;
devid = __v1.wrapping_offset(1 as isize);
}
false
}

fn acpi_scan_match_handler(idstr: *const u8, matchid: *mut *const acpi_device_id) -> *mut acpi_scan_handler {
let mut handler: *mut acpi_scan_handler = unsafe { core::mem::zeroed() };
handler = list_first_entry__28!(unsafe { core::ptr::addr_of_mut!(acpi_scan_handlers_list) });
while list_entry_is_head!(handler, unsafe { core::ptr::addr_of_mut!(acpi_scan_handlers_list) }, list_node) == 0 {
if acpi_scan_handler_matching(handler, idstr, matchid) {
return handler;
}
handler = list_next_entry__32!(handler, list_node);
}
core::ptr::null_mut::<acpi_scan_handler>()
}

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_hotplug_enabled(hotplug: *mut acpi_hotplug_profile, val: bool) {
if ((((unsafe { (*hotplug).__slate_bits_0.__get_enabled() }) as i32) != 0) as i32) == (val as i32) {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_scan_lock)) };
unsafe {
(*hotplug).__slate_bits_0.__set_enabled(val);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_scan_lock)) };
}

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_add_dep(handle: *mut (), dep_devices: *mut acpi_handle_list) -> i32 {
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut count: u32 = 0;
let mut i: i32 = 0;
'__slate_break_1: while (i as u32) < unsafe { (*dep_devices).count } {
let mut info: *mut acpi_device_info = unsafe { core::mem::zeroed() };
let mut dep: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
let mut skip: bool = false;
let mut honor_dep: bool = false;
let status: u32 = unsafe { acpi_get_object_info(*unsafe { (*dep_devices).handles }.wrapping_offset(i as isize), core::ptr::addr_of_mut!(info)) };
if ACPI_FAILURE!(status) != 0 {
let mut __v0: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"Error reading _DEP device info\n".as_ptr() as *const u8) };
}
__v0 = 0;
} else {
skip = acpi_info_matches_ids(info, unsafe { core::ptr::addr_of!(acpi_ignore_dep_ids.0) as *const *const u8 });
honor_dep = acpi_info_matches_ids(info, unsafe { core::ptr::addr_of!(acpi_honor_dep_ids.0) as *const *const u8 });
unsafe { kfree(info as *const ()) };
if !skip {
let mut __v2: *mut acpi_dep_data;
let __obj_size: u64 = size_mul(core::mem::size_of::<acpi_dep_data>() as u64, 1u64);
let mut __v3: *mut ();
let mut __v4: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
}
__v4 = _res;
__v3 = __v4;
__v2 = __v3 as *mut acpi_dep_data;
dep = __v2;
if !dep.is_null() {
count = count.wrapping_add(1);
unsafe {
(*dep).supplier = *unsafe { (*dep_devices).handles }.wrapping_offset(i as isize);
}
unsafe {
(*dep).consumer = handle;
}
unsafe {
(*dep).honor_dep = honor_dep;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*dep).node) }, unsafe { core::ptr::addr_of_mut!(acpi_dep_list) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
}
}
}
i = i.wrapping_add(1);
}
unsafe { acpi_handle_list_free(dep_devices) };
count as i32
}

fn acpi_scan_init_hotplug(adev: *mut acpi_device) {
let mut hwid: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
let __v0: bool = acpi_dock_match(unsafe { (*adev).handle }) || is_ejectable_bay(adev);
if __v0 {
unsafe { acpi_dock_add(adev) };
return;
}
hwid = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*adev).pnp.ids) });
while list_entry_is_head!(hwid, unsafe { core::ptr::addr_of_mut!((*adev).pnp.ids) }, list) == 0 {
let handler: *mut acpi_scan_handler = acpi_scan_match_handler(unsafe { (*hwid).id }, core::ptr::null_mut::<*const acpi_device_id>());
if !handler.is_null() {
unsafe {
(*adev).flags.__slate_bits_0.__set_hotplug_notify(1u32);
}
break;
}
hwid = list_next_entry__4!(hwid, list);
}
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn arch_acpi_add_auto_dep(handle: *mut ()) -> u32 {
0
}

fn acpi_scan_check_dep(handle: *mut ()) -> u32 {
let mut dep_devices: acpi_handle_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dep_devices) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut count: u32 = 0;
// Some architectures like RISC-V need to add dependencies for
// all devices which use GSI to the interrupt controller so that
// interrupt controller is probed before any of those devices.
// Instead of mandating _DEP on all the devices, detect the
// dependency and add automatically.
let __v0: u32 = arch_acpi_add_auto_dep(handle);
count = count.wrapping_add(__v0);
// Check for _HID here to avoid deferring the enumeration of:
// 1. PCI devices.
// 2. ACPI nodes describing USB ports.
// Still, checking for _HID catches more then just these cases ...
let __v3: bool = !unsafe { acpi_has_method(handle, c"_DEP".as_ptr() as *mut u8) } || !unsafe { acpi_has_method(handle, c"_HID".as_ptr() as *mut u8) };
if __v3 {
return count;
}
if !unsafe { acpi_evaluate_reference(handle, c"_DEP".as_ptr() as *mut u8, core::ptr::null_mut::<acpi_object_list>(), core::ptr::addr_of_mut!(dep_devices)) } {
let mut __v4: i32;
if false {
unsafe { acpi_handle_printk(c"\x017".as_ptr() as *const u8, handle, c"Failed to evaluate _DEP.\n".as_ptr() as *const u8) };
}
__v4 = 0;
return count;
}
let __v5: i32 = acpi_scan_add_dep(handle, core::ptr::addr_of_mut!(dep_devices));
count = count.wrapping_add(__v5 as u32);
count
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_check_crs_csi2_cb")]
extern "C" fn acpi_scan_check_crs_csi2_cb(handle: *mut (), a: u32, b: *mut (), c: *mut *mut ()) -> u32 {
unsafe { acpi_mipi_check_crs_csi2(handle) };
0
}

fn acpi_bus_check_add(handle: *mut (), first_pass: bool, adev_p: *mut *mut acpi_device) -> u32 {
let mut r#type: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(r#type) as *mut i32;
let mut acpi_type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(acpi_type) as *mut u32;
let mut device: *mut acpi_device = unsafe { core::mem::zeroed() };
unsafe {
device = acpi_fetch_acpi_dev(handle);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if device.is_null() {
if ACPI_FAILURE!(acpi_get_type(handle, core::ptr::addr_of_mut!(*__slate_slot_1))) != 0 {
return 0;
} else {
'__join_5: {
'__join_10: {
let __t0: u32 = *__slate_slot_1;
if __t0 == 6 {
if acpi_device_should_be_hidden(handle) {
return 0;
} else {
if first_pass {
acpi_mipi_check_crs_csi2(handle);
// Bail out if there are dependencies.
if acpi_scan_check_dep(handle) > 0 {
// The entire CSI-2 connection graph needs to be
// extracted before any drivers or scan handlers
// are bound to struct device objects, so scan
// _CRS CSI-2 resource descriptors for all
// devices below the current handle.
acpi_walk_namespace(6u32, handle, 4294967295u32, Some(acpi_scan_check_crs_csi2_cb), None, core::ptr::null_mut::<()>(), core::ptr::null_mut::<*mut ()>());
return EXCEP_CTL!(6i32);
}
}
break '__join_10;
}
} else {
if __t0 == 0 {
break '__join_10;
} else {
if __t0 == 12 {
*__slate_slot_0 = 2;
break '__join_5;
} else {
if __t0 == 13 {
*__slate_slot_0 = 3;
break '__join_5;
} else {
if __t0 == 11 {
acpi_add_power_resource(handle);
}
}
}
}
}
return 0;
}
*__slate_slot_0 = 0; // for ACPI_ROOT_OBJECT
}
// If first_pass is true at this point, the device has no dependencies,
// or the creation of the device object would have been postponed above.
acpi_add_single_object(&mut (*core::ptr::addr_of_mut!(device)), handle, *__slate_slot_0, !first_pass);
if device.is_null() {
return EXCEP_CTL!(6i32);
} else {
acpi_scan_init_hotplug(device);
}
}
}
if unsafe { *adev_p }.is_null() {
*adev_p = device;
}
return 0;
}
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_check_add_1")]
extern "C" fn acpi_bus_check_add_1(handle: *mut (), lvl_not_used: u32, not_used: *mut (), ret_p: *mut *mut ()) -> u32 {
acpi_bus_check_add(handle, true, ret_p as *mut *mut acpi_device)
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_check_add_2")]
extern "C" fn acpi_bus_check_add_2(handle: *mut (), lvl_not_used: u32, not_used: *mut (), ret_p: *mut *mut ()) -> u32 {
acpi_bus_check_add(handle, false, ret_p as *mut *mut acpi_device)
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_video_bus_device_release")]
extern "C" fn acpi_video_bus_device_release(dev: *mut device) {
let aux_dev: *mut auxiliary_device = to_auxiliary_dev(dev);
unsafe { kfree(aux_dev as *const ()) };
}

fn acpi_create_video_bus_device(adev: *mut acpi_device, parent: *mut acpi_device) {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut aux_dev: *mut auxiliary_device = unsafe { core::mem::zeroed() };
let mut phys_parent: *mut device = unsafe { acpi_bus_get_primary_device(parent) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(phys_parent) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_put_device(__slate_cleanup_ptr_0 as *mut _));
if phys_parent.is_null() {
return;
}
let mut __v1: *mut auxiliary_device;
let __obj_size: u64 = size_mul(core::mem::size_of::<auxiliary_device>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut auxiliary_device;
aux_dev = __v1;
if aux_dev.is_null() {
return;
}
let __v7: u32 = unsafe { aux_dev_id };
unsafe {
aux_dev_id = __v7.wrapping_add(1);
}
unsafe {
(*aux_dev).id = __v7;
}
unsafe {
(*aux_dev).name = c"video_bus".as_ptr() as *const u8;
}
unsafe {
(*aux_dev).dev.parent = phys_parent;
}
unsafe {
(*aux_dev).dev.release = Some(acpi_video_bus_device_release);
}
if (unsafe { auxiliary_device_init(aux_dev) }) != 0 {
unsafe { kfree(aux_dev as *const ()) };
return;
}
ACPI_COMPANION_SET!(unsafe { core::ptr::addr_of_mut!((*aux_dev).dev) }, adev);
if (unsafe { __auxiliary_device_add(aux_dev, c"acpi".as_ptr() as *const u8) }) != 0 {
auxiliary_device_uninit(aux_dev);
}
}

static mut aux_dev_id: u32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_scan_system_dev {
    node: list_head,
    adev: *mut acpi_device,
}


/// Memory controller
/// Motherboard resource
static acpi_system_dev_ids: __SlateAlign16<[*const u8; 3]> = __SlateAlign16([(b"PNP0C01\0".as_ptr() as *mut u8) as *const u8, (b"PNP0C02\0".as_ptr() as *mut u8) as *const u8, core::ptr::null::<u8>()]);

fn acpi_default_enumeration(device: *mut acpi_device) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
// Do not enumerate devices with enumeration_by_parent flag set as
// they will be enumerated by their respective parents.
if ((unsafe { (*device).flags.__slate_bits_0.__get_enumeration_by_parent() }) as i32) != 0 {
unsafe { blocking_notifier_call_chain(core::ptr::addr_of_mut!(acpi_reconfig_chain), 0u64, device as *mut ()) };
return;
}
if (unsafe { match_string(core::ptr::addr_of!(acpi_system_dev_ids.0) as *const *const u8, 1i32.wrapping_neg() as u64, acpi_device_hid(device)) }) >= 0 {
let mut sd: *mut acpi_scan_system_dev = unsafe { core::mem::zeroed() };
// This is a generic system device, so there is no need to
// create a platform device for it, but its resources need to be
// reserved.  However, that needs to be done after all of the
// other device objects have been processed and PCI has claimed
// BARs in case there are resource conflicts.
let mut __v0: *mut acpi_scan_system_dev;
let __obj_size: u64 = size_mul(core::mem::size_of::<acpi_scan_system_dev>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut acpi_scan_system_dev;
sd = __v0;
if !sd.is_null() {
unsafe {
(*sd).adev = device;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*sd).node) }, unsafe { core::ptr::addr_of_mut!(acpi_scan_system_dev_list) });
}
} else {
if ((unsafe { (*device).pnp.r#type.__slate_bits_0.__get_backlight() }) as i32) != 0 {
let parent: *mut acpi_device = acpi_dev_parent(device);
if !parent.is_null() {
acpi_create_video_bus_device(device, parent);
}
} else {
// For a regular device object, create a platform device.
unsafe { acpi_create_platform_device(device, core::ptr::null::<property_entry>()) };
}
}
acpi_device_set_enumerated(device);
}

static generic_device_ids: __SlateAlign16<[acpi_device_id; 2]> = __SlateAlign16([{
let mut __t0: acpi_device_id = unsafe { core::mem::zeroed() };
__t0.id = [80u8, 82u8, 80u8, 48u8, 48u8, 48u8, 49u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0
}, {
let mut __t1: acpi_device_id = unsafe { core::mem::zeroed() };
__t1.id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1
}]);

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_generic_device_attach")]
extern "C" fn acpi_generic_device_attach(adev: *mut acpi_device, not_used: *const acpi_device_id) -> i32 {
// Since ACPI_DT_NAMESPACE_HID is the only ID handled here, the test
// below can be unconditional.
if !unsafe { (*adev).data.of_compatible }.is_null() {
acpi_default_enumeration(adev);
}
1
}

static mut generic_device_handler: acpi_scan_handler = {
let mut __t0: acpi_scan_handler = unsafe { core::mem::zeroed() };
__t0.ids = unsafe { core::ptr::addr_of!(generic_device_ids.0) as *const acpi_device_id };
__t0.attach = Some(acpi_generic_device_attach);
__t0
};

fn acpi_scan_attach_handler(device: *mut acpi_device) -> i32 {
let mut hwid: *mut acpi_hardware_id = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
hwid = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*device).pnp.ids) });
while list_entry_is_head!(hwid, unsafe { core::ptr::addr_of_mut!((*device).pnp.ids) }, list) == 0 {
let mut devid: *const acpi_device_id = unsafe { core::mem::zeroed() };
let handler: *mut acpi_scan_handler = acpi_scan_match_handler(unsafe { (*hwid).id }, core::ptr::addr_of_mut!(devid));
if !handler.is_null() {
if (unsafe { (*handler).attach }) == None {
unsafe {
(*device).pnp.r#type.__slate_bits_0.__set_platform_id(0u32);
}
} else {
unsafe {
(*device).handler = handler;
}
ret = unsafe { unsafe { (*handler).attach }.unwrap()(device, devid) };
if ret > 0 {
break;
}
unsafe {
(*device).handler = core::ptr::null_mut::<acpi_scan_handler>();
}
if ret < 0 {
break;
}
}
}
hwid = list_next_entry__4!(hwid, list);
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_attach")]
extern "C" fn acpi_bus_attach(device: *mut acpi_device, first_pass: *mut ()) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ejd: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(ejd) as *mut *mut ();
let mut skip: bool = unsafe { core::mem::zeroed() };
unsafe {
'__join_2: {
skip = first_pass.is_null() && ((*device).flags.__slate_bits_0.__get_visited() as i32) != 0;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !skip {
if ACPI_SUCCESS!(acpi_bus_get_ejd((*device).handle, core::ptr::addr_of_mut!(*__slate_slot_1))) {
register_dock_dependent_device(device, *__slate_slot_1);
}
acpi_bus_get_status(device);
// Skip devices that are not ready for enumeration (e.g. not present)
if !acpi_dev_ready_for_enumeration(device as *const acpi_device) {
(*device).flags.__slate_bits_0.__set_initialized(0u32);
acpi_device_clear_enumerated(device);
(*device).flags.__slate_bits_0.__set_power_manageable(0u32);
return 0;
} else {
if unsafe { (*device).handler }.is_null() {
acpi_ec_register_opregions(device);
if ((*device).flags.__slate_bits_0.__get_initialized() as i32) == 0 {
let __t0 = ((*((*device).power.states.as_mut_ptr() as *mut acpi_device_power_state).wrapping_offset(0 as isize)).flags.__slate_bits_0.__get_valid() as i32) as u32;
(*device).flags.__slate_bits_0.__set_power_manageable(__t0);
if acpi_bus_init_power(device) != 0 {
(*device).flags.__slate_bits_0.__set_power_manageable(0u32);
}
(*device).flags.__slate_bits_0.__set_initialized(1u32);
} else {
if ((*device).flags.__slate_bits_0.__get_visited() as i32) != 0 {
break '__join_2;
}
}
*__slate_slot_0 = acpi_scan_attach_handler(device);
if *__slate_slot_0 < 0 {
return 0;
} else {
if *__slate_slot_0 > 0 && ((*device).flags.__slate_bits_0.__get_enumeration_by_parent() as i32) == 0 {
acpi_device_set_enumerated(device);
} else {
if ((*device).pnp.r#type.__slate_bits_0.__get_platform_id() as i32) != 0 || ((*device).pnp.r#type.__slate_bits_0.__get_backlight() as i32) != 0 || ((*device).flags.__slate_bits_0.__get_enumeration_by_parent() as i32) != 0 {
acpi_default_enumeration(device);
} else {
acpi_device_set_enumerated(device);
}
}
}
}
}
}
}
acpi_dev_for_each_child(device, Some(acpi_bus_attach), first_pass);
if !skip && !unsafe { (*device).handler }.is_null() && (*(*device).handler).hotplug.notify_online != None {
unsafe { (*(*device).handler).hotplug.notify_online }.unwrap()(device);
}
return 0;
}
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_dev_get_next_consumer_dev_cb")]
extern "C" fn acpi_dev_get_next_consumer_dev_cb(dep: *mut acpi_dep_data, data: *mut ()) -> i32 {
let mut adev: *mut acpi_device = unsafe { *(data as *mut *mut acpi_device) };
// If we're passed a 'previous' consumer device then we need to skip
// any consumers until we meet the previous one, and then NULL @data
// so the next one can be returned.
if !adev.is_null() {
if (unsafe { (*dep).consumer }) == unsafe { (*adev).handle } {
unsafe {
*(data as *mut *mut acpi_device) = core::ptr::null_mut::<acpi_device>();
}
}
return 0;
}
adev = acpi_get_acpi_dev(unsafe { (*dep).consumer });
if !adev.is_null() {
unsafe {
*(data as *mut *mut acpi_device) = adev;
}
return 1;
}
// Continue parsing if the device object is not present.
0
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_clear_dep_fn")]
extern "C" fn acpi_scan_clear_dep_fn(dev: *mut (), cookie: u64) {
let adev: *mut acpi_device = to_acpi_device__2!(dev);
acpi_scan_lock_acquire();
acpi_bus_attach(adev, 1i32 as *mut ());
acpi_scan_lock_release();
acpi_dev_put(adev);
}

fn acpi_scan_clear_dep_queue(adev: *mut acpi_device) -> bool {
if (unsafe { (*adev).dep_unmet }) != 0 {
return false;
}
// Async schedule the deferred acpi_scan_clear_dep_fn() since:
// - acpi_bus_attach() needs to hold acpi_scan_lock which cannot
//   be acquired under acpi_dep_list_lock (held here)
// - the deferred work at boot stage is ensured to be finished
//   before userspace init task by the async_synchronize_full()
//   barrier
//
// Use _nocall variant since it'll return on failure instead of
// run the function synchronously.
unsafe { async_schedule_dev_nocall(Some(acpi_scan_clear_dep_fn), core::ptr::addr_of_mut!((*adev).dev)) }
}

fn acpi_scan_delete_dep_data(dep: *mut acpi_dep_data) {
list_del(unsafe { core::ptr::addr_of_mut!((*dep).node) });
unsafe { kfree(dep as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_scan_clear_dep")]
extern "C" fn acpi_scan_clear_dep(dep: *mut acpi_dep_data, data: *mut ()) -> i32 {
let adev: *mut acpi_device = acpi_get_acpi_dev(unsafe { (*dep).consumer });
if !adev.is_null() {
let __v1: u32 = unsafe { (*adev).dep_unmet };
unsafe {
(*adev).dep_unmet = __v1.wrapping_sub(1);
}
if !acpi_scan_clear_dep_queue(adev) {
acpi_dev_put(adev);
}
}
if unsafe { (*dep).free_when_met } {
acpi_scan_delete_dep_data(dep);
} else {
unsafe {
(*dep).met = true;
}
}
0
}

/// acpi_walk_dep_device_list - Apply a callback to every entry in acpi_dep_list
/// @handle:	The ACPI handle of the supplier device
/// @callback:	Pointer to the callback function to apply
/// @data:	Pointer to some data to pass to the callback
///
/// The return value of the callback determines this function's behaviour. If 0
/// is returned we continue to iterate over acpi_dep_list. If a positive value
/// is returned then the loop is broken but this function returns 0. If a
/// negative value is returned by the callback then the loop is broken and that
/// value is returned as the final error.
fn acpi_walk_dep_device_list(handle: *mut (), callback: Option<unsafe extern "C" fn(*mut acpi_dep_data, *mut ()) -> i32>, data: *mut ()) -> i32 {
let mut dep: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
let mut tmp: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
dep = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(acpi_dep_list) });
tmp = list_next_entry__10!(dep, node);
while list_entry_is_head!(dep, unsafe { core::ptr::addr_of_mut!(acpi_dep_list) }, node) == 0 {
if (unsafe { (*dep).supplier }) == handle {
ret = unsafe { callback.unwrap()(dep, data) };
if ret != 0 {
break;
}
}
dep = tmp;
tmp = list_next_entry__10!(tmp, node);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
if ret > 0 { 0 } else { ret }
}

/// acpi_dev_clear_dependencies - Inform consumers that the device is now active
/// @supplier: Pointer to the supplier &struct acpi_device
///
/// Clear dependencies on the given device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_dev_clear_dependencies")]
extern "C" fn acpi_dev_clear_dependencies(supplier: *mut acpi_device) {
acpi_walk_dep_device_list(unsafe { (*supplier).handle }, Some(acpi_scan_clear_dep), core::ptr::null_mut::<()>());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_dev_clear_dependencies_545: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut acpi_device)>, *mut ()>(Some(acpi_dev_clear_dependencies)) };

/// acpi_dev_ready_for_enumeration - Check if the ACPI device is ready for enumeration
/// @device: Pointer to the &struct acpi_device to check
///
/// Check if the device is present and has no unmet dependencies.
///
/// Return true if the device is ready for enumeratino. Otherwise, return false.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_dev_ready_for_enumeration")]
extern "C" fn acpi_dev_ready_for_enumeration(device: *const acpi_device) -> bool {
if ((unsafe { (*device).flags.__slate_bits_0.__get_honor_deps() }) as i32) != 0 && (unsafe { (*device).dep_unmet }) != 0 {
return false;
}
acpi_device_is_present(device)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_dev_ready_for_enumeration_546: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const acpi_device) -> bool>, *mut ()>(Some(acpi_dev_ready_for_enumeration)) };

/// acpi_dev_get_next_consumer_dev - Return the next adev dependent on @supplier
/// @supplier: Pointer to the dependee device
/// @start: Pointer to the current dependent device
///
/// Returns the next &struct acpi_device which declares itself dependent on
/// @supplier via the _DEP buffer, parsed from the acpi_dep_list.
///
/// If the returned adev is not passed as @start to this function, the caller is
/// responsible for putting the reference to adev when it is no longer needed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_dev_get_next_consumer_dev")]
extern "C" fn acpi_dev_get_next_consumer_dev(supplier: *mut acpi_device, start: *mut acpi_device) -> *mut acpi_device {
let mut adev: *mut acpi_device = start;
acpi_walk_dep_device_list(unsafe { (*supplier).handle }, Some(acpi_dev_get_next_consumer_dev_cb), core::ptr::addr_of_mut!(adev) as *mut ());
acpi_dev_put(start);
if adev == start {
return core::ptr::null_mut::<acpi_device>();
}
adev
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_dev_get_next_consumer_dev_547: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut acpi_device, *mut acpi_device) -> *mut acpi_device>, *mut ()>(Some(acpi_dev_get_next_consumer_dev)) };

fn acpi_scan_postponed_branch(handle: *mut ()) {
let mut adev: *mut acpi_device = core::ptr::null_mut::<acpi_device>();
if ACPI_FAILURE!(acpi_bus_check_add(handle, false, core::ptr::addr_of_mut!(adev))) != 0 {
return;
}
unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, Some(acpi_bus_check_add_2), None, core::ptr::null_mut::<()>(), core::ptr::addr_of_mut!(adev) as *mut *mut ()) };
// Populate the ACPI _CRS CSI-2 software nodes for the ACPI devices that
// have been added above.
unsafe { acpi_mipi_init_crs_csi2_swnodes() };
acpi_bus_attach(adev, core::ptr::null_mut::<()>());
}

fn acpi_scan_postponed() {
let mut dep: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
let mut tmp: *mut acpi_dep_data = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
dep = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(acpi_dep_list) });
tmp = list_next_entry__10!(dep, node);
while list_entry_is_head!(dep, unsafe { core::ptr::addr_of_mut!(acpi_dep_list) }, node) == 0 {
let handle: *mut () = unsafe { (*dep).consumer };
// In case there are multiple acpi_dep_list entries with the
// same consumer, skip the current entry if the consumer device
// object corresponding to it is present already.
if acpi_fetch_acpi_dev(handle).is_null() {
// Even though the lock is released here, tmp is
// guaranteed to be valid, because none of the list
// entries following dep is marked as "free when met"
// and so they cannot be deleted.
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
acpi_scan_postponed_branch(handle);
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
}
if unsafe { (*dep).met } {
acpi_scan_delete_dep_data(dep);
} else {
unsafe {
(*dep).free_when_met = true;
}
}
dep = tmp;
tmp = list_next_entry__10!(tmp, node);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_dep_list_lock)) };
}

fn acpi_scan_claim_resources(adev: *mut acpi_device) {
let mut r: core::mem::MaybeUninit<*mut resource> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut resource = core::ptr::addr_of_mut!(r) as *mut *mut resource;
let mut res: *mut resource = unsafe { core::mem::zeroed() };
let mut regionid: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *const u8 = core::ptr::addr_of_mut!(regionid) as *mut *const u8;
let mut count: u32 = unsafe { core::mem::zeroed() };
let mut resource_list: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut list_head = core::ptr::addr_of_mut!(resource_list) as *mut list_head;
let mut rentry: core::mem::MaybeUninit<*mut resource_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut resource_entry = core::ptr::addr_of_mut!(rentry) as *mut *mut resource_entry;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_9, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_9), prev: core::ptr::addr_of_mut!(*__slate_slot_9) });
count = 0;
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if acpi_dev_get_resources(adev, core::ptr::addr_of_mut!(*__slate_slot_9), None, core::ptr::null_mut::<()>()) <= 0 {
return;
} else {
*__slate_slot_7 = kstrdup(dev_name(core::ptr::addr_of_mut!((*adev).dev) as *const device), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const u8;
if !(*__slate_slot_7).is_null() {
*__slate_slot_10 = list_first_entry!(core::ptr::addr_of_mut!(*__slate_slot_9));
while list_entry_is_head!(*__slate_slot_10, core::ptr::addr_of_mut!(*__slate_slot_9), node) == 0 {
'__join_4: {
res = (*(*__slate_slot_10)).res;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Skip disabled and invalid resources.
if !((*res).flags & 268435456 != 0 || (*res).end < (*res).start) {
if resource_type(res as *const resource) == 256 {
// Follow the PNP system driver and on x86 skip I/O
// resources that start below 0x100 (the "standard PC
// hardware" boundary).
if IS_ENABLED__2!() != 0 && (*res).start < 256 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Skipped %pR\n".as_ptr() as *const u8, res);
break '__join_4;
} else {
*__slate_slot_5 = request_region__2!(unsafe { (*res).start }, resource_size(res as *const resource), *__slate_slot_7);
}
} else {
if resource_type(res as *const resource) == 512 {
*__slate_slot_5 = request_mem_region__2!(unsafe { (*res).start }, resource_size(res as *const resource), *__slate_slot_7);
} else {
break '__join_4;
}
}
if !(*__slate_slot_5).is_null() {
let __v4: *mut resource = *__slate_slot_5;
let __v2: u64 = (*__v4).flags & 2147483647u64;
(*__v4).flags = __v2;
while false {
}
_dev_info(core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Reserved %pR\n".as_ptr() as *const u8, *__slate_slot_5);
count = count.wrapping_add(1);
} else {
// Failures at this point are usually harmless. PCI
// quirks, for example, reserve resources they know
// about too, so there may well be double reservations.
while false {
}
_dev_info(core::ptr::addr_of_mut!((*adev).dev) as *const device, c"Could not reserve %pR\n".as_ptr() as *const u8, res);
}
}
}
*__slate_slot_10 = list_next_entry!(*__slate_slot_10, node);
}
if count == 0 {
kfree(*__slate_slot_7 as *const ());
}
}
acpi_dev_free_resource_list(core::ptr::addr_of_mut!(*__slate_slot_9));
}
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn acpi_reserve_motherboard_resources() -> i32 {
let mut sd: *mut acpi_scan_system_dev = unsafe { core::mem::zeroed() };
let mut tmp: *mut acpi_scan_system_dev = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_548: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(acpi_scan_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_548) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_549: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(acpi_scan_lock) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_549) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
sd = list_first_entry__27!(unsafe { core::ptr::addr_of_mut!(acpi_scan_system_dev_list) });
tmp = list_next_entry__25!(sd, node);
while list_entry_is_head!(sd, unsafe { core::ptr::addr_of_mut!(acpi_scan_system_dev_list) }, node) == 0 {
acpi_scan_claim_resources(unsafe { (*sd).adev });
list_del(unsafe { core::ptr::addr_of_mut!((*sd).node) });
unsafe { kfree(sd as *const ()) };
sd = tmp;
tmp = list_next_entry__25!(tmp, node);
}
0
}

/// Reserve motherboard resources after PCI claims BARs, but before PCI assigns
/// resources for uninitialized PCI devices.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_reserve_motherboard_resources_551: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(acpi_reserve_motherboard_resources)) };

/// acpi_bus_scan - Add ACPI device node objects in a given namespace scope.
/// @handle: Root of the namespace scope to scan.
///
/// Scan a given ACPI tree (probably recently hot-plugged) and create and add
/// found devices.
///
/// If no devices were found, -ENODEV is returned, but it does not mean that
/// there has been a real error.  There just have been no suitable ACPI objects
/// in the table trunk from which the kernel could create a device and add an
/// appropriate driver.
///
/// Must be called under acpi_scan_lock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_scan")]
extern "C" fn acpi_bus_scan(handle: *mut ()) -> i32 {
let mut device: *mut acpi_device = core::ptr::null_mut::<acpi_device>();
// Pass 1: Avoid enumerating devices with missing dependencies.
if ACPI_SUCCESS!(acpi_bus_check_add(handle, true, core::ptr::addr_of_mut!(device))) {
unsafe { acpi_walk_namespace(0u32, handle, 4294967295u32, Some(acpi_bus_check_add_1), None, core::ptr::null_mut::<()>(), core::ptr::addr_of_mut!(device) as *mut *mut ()) };
}
if device.is_null() {
return 19i32.wrapping_neg();
}
// Set up ACPI _CRS CSI-2 software nodes using information extracted
// from the _CRS CSI-2 resource descriptors during the ACPI namespace
// walk above and MIPI DisCo for Imaging device properties.
unsafe { acpi_mipi_scan_crs_csi2() };
unsafe { acpi_mipi_init_crs_csi2_swnodes() };
acpi_bus_attach(device, 1i32 as *mut ());
// Pass 2: Enumerate all of the remaining devices.
acpi_scan_postponed();
unsafe { acpi_mipi_crs_csi2_cleanup() };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_bus_scan_552: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ()) -> i32>, *mut ()>(Some(acpi_bus_scan)) };

/// acpi_bus_trim - Detach scan handlers and drivers from ACPI device objects.
/// @adev: Root of the ACPI namespace scope to walk.
///
/// Must be called under acpi_scan_lock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_trim")]
extern "C" fn acpi_bus_trim(adev: *mut acpi_device) {
let flags: u64 = 0;
acpi_scan_check_and_detach(adev, flags as *mut ());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_bus_trim_553: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut acpi_device)>, *mut ()>(Some(acpi_bus_trim)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_bus_register_early_device")]
extern "C" fn acpi_bus_register_early_device(r#type: i32) -> i32 {
let mut device: *mut acpi_device = core::ptr::null_mut::<acpi_device>();
let result: i32 = acpi_add_single_object(unsafe { &mut (*core::ptr::addr_of_mut!(device)) }, core::ptr::null_mut::<()>(), r#type, false);
if result != 0 {
return result;
}
acpi_default_enumeration(device);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_bus_register_early_device_554: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> i32>, *mut ()>(Some(acpi_bus_register_early_device)) };

fn acpi_bus_add_fixed_device_object(r#type: u32) {
let mut adev: *mut acpi_device = core::ptr::null_mut::<acpi_device>();
acpi_add_single_object(unsafe { &mut (*core::ptr::addr_of_mut!(adev)) }, core::ptr::null_mut::<()>(), (r#type as u32) as i32, false);
if !adev.is_null() {
acpi_default_enumeration(adev);
}
}

fn acpi_bus_scan_fixed() {
if (unsafe { acpi_gbl_FADT.flags }) & 16u32 == 0 {
acpi_bus_add_fixed_device_object(4u32);
}
if (unsafe { acpi_gbl_FADT.flags }) & 32u32 == 0 {
acpi_bus_add_fixed_device_object(5u32);
}
}

#[unsafe(link_section = ".init.text")]
fn acpi_get_spcr_uart_addr() {
let mut status: u32 = 0;
let mut spcr_ptr: *mut acpi_table_spcr = unsafe { core::mem::zeroed() };
status = unsafe { acpi_get_table(c"SPCR".as_ptr() as *mut u8, 0u32, core::ptr::addr_of_mut!(spcr_ptr) as *mut *mut acpi_table_header) };
if ACPI_FAILURE!(status) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014ACPI: STAO table present, but SPCR is missing\n".as_ptr() as *const u8) };
return;
}
unsafe {
spcr_uart_addr = (*spcr_ptr).serial_port.address;
}
unsafe { acpi_put_table(spcr_ptr as *mut acpi_table_header) };
}

static mut acpi_scan_initialized: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn acpi_scan_init() {
let mut stao_ptr: core::mem::MaybeUninit<*mut acpi_table_stao> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut acpi_table_stao = core::ptr::addr_of_mut!(stao_ptr) as *mut *mut acpi_table_stao;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
acpi_pci_root_init();
acpi_pci_link_init();
acpi_processor_init();
acpi_platform_init();
acpi_lpss_init();
acpi_apd_init();
acpi_cmos_rtc_init();
acpi_container_init();
acpi_memory_hotplug_init();
acpi_watchdog_init();
acpi_pnp_init();
acpi_power_resources_init();
acpi_init_lpit();
acpi_scan_add_handler(core::ptr::addr_of_mut!(generic_device_handler));
// If there is STAO table, check whether it needs to ignore the UART
// device in SPCR table.
*__slate_slot_2 = acpi_get_table(c"STAO".as_ptr() as *mut u8, 0u32, core::ptr::addr_of_mut!(*__slate_slot_1) as *mut *mut acpi_table_header);
if ACPI_SUCCESS!(*__slate_slot_2) {
if ((*(*__slate_slot_1)).header.length as u64) > (core::mem::size_of::<acpi_table_stao>() as u64) {
while false {
}
_printk(c"\x016ACPI: STAO Name List not yet supported.\n".as_ptr() as *const u8);
}
if (*(*__slate_slot_1)).ignore_uart != 0 {
acpi_get_spcr_uart_addr();
}
acpi_put_table(*__slate_slot_1 as *mut acpi_table_header);
}
acpi_gpe_apply_masked_gpes();
acpi_update_all_gpes();
// Although we call __add_memory() that is documented to require the
// device_hotplug_lock, it is not necessary here because this is an
// early code when userspace or any other code path cannot trigger
// hotplug/hotunplug operations.
mutex_lock(core::ptr::addr_of_mut!(acpi_scan_lock));
// Enumerate devices in the ACPI namespace.
if acpi_bus_scan(ACPI_TO_POINTER!(18446744073709551615u64)) == 0 {
acpi_root = acpi_fetch_acpi_dev(ACPI_TO_POINTER!(18446744073709551615u64));
if !unsafe { acpi_root }.is_null() {
// Fixed feature devices do not exist on HW-reduced platform
if acpi_gbl_reduced_hardware == 0 {
acpi_bus_scan_fixed();
}
acpi_turn_off_unused_power_resources();
acpi_scan_initialized = true;
}
}
mutex_unlock(core::ptr::addr_of_mut!(acpi_scan_lock));
}
}

static mut ape: *mut acpi_probe_entry = unsafe { core::mem::zeroed() };

static mut acpi_probe_count: i32 = unsafe { core::mem::zeroed() };

static mut acpi_probe_mutex: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn acpi_match_madt(header: *mut acpi_subtable_headers, end: u64) -> i32 {
let __v0: bool = (unsafe { (*ape).subtable_valid }) == None || unsafe { unsafe { (*ape).subtable_valid }.unwrap()(core::ptr::addr_of_mut!((*header).common), ape) };
if __v0 {
if (unsafe { unsafe { (*ape).__slate_anon_3.probe_subtbl }.unwrap()(header, end) }) == 0 {
let __v1: i32 = unsafe { acpi_probe_count };
unsafe {
acpi_probe_count = __v1.wrapping_add(1);
}
}
}
0
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn arch_sort_irqchip_probe(ap_head: *mut acpi_probe_entry, nr: i32) {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn __acpi_probe_device_table(ap_head: *mut acpi_probe_entry, mut nr: i32) -> i32 {
let mut count: i32 = 0;
if (unsafe { acpi_disabled }) != 0 {
return 0;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(acpi_probe_mutex)) };
arch_sort_irqchip_probe(ap_head, nr);
unsafe {
ape = ap_head;
}
while nr != 0 {
if ACPI_COMPARE_NAMESEG!(c"APIC".as_ptr() as *mut u8, unsafe { (*ape).id.as_mut_ptr() as *mut u8 }) {
unsafe {
acpi_probe_count = 0;
}
unsafe { acpi_table_parse_madt((*ape).r#type as u32, Some(acpi_match_madt), 0u32) };
count = count.wrapping_add(unsafe { acpi_probe_count });
} else {
let res: i32 = unsafe { acpi_table_parse(((*ape).id.as_mut_ptr() as *mut u8) as *mut u8, (*ape).__slate_anon_3.probe_table) };
if res == 0 {
count = count.wrapping_add(1);
}
}
let __v5: *mut acpi_probe_entry = unsafe { ape };
unsafe {
ape = __v5.wrapping_offset(1 as isize);
}
nr = nr.wrapping_sub(1);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(acpi_probe_mutex)) };
count
}

#[unsafe(link_section = ".text.slate_distinct.scan.acpi_table_events_fn")]
extern "C" fn acpi_table_events_fn(work__0: *mut work_struct) {
acpi_scan_lock_acquire();
acpi_bus_scan(ACPI_TO_POINTER!(18446744073709551615u64));
acpi_scan_lock_release();
unsafe { kfree(work__0 as *const ()) };
}

#[unsafe(no_mangle)]
extern "C" fn acpi_scan_table_notify() {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut work__0: *mut work_struct = unsafe { core::mem::zeroed() };
if !unsafe { acpi_scan_initialized } {
return;
}
let mut __v0: *mut work_struct;
let __obj_size: u64 = size_mul(core::mem::size_of::<work_struct>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut work_struct;
work__0 = __v0;
if work__0.is_null() {
return;
}
'__slate_break_8: loop {
'__slate_break_7: loop {
__init_work(work__0, 0i32);
unsafe {
(*work__0).data = *{
__slate_compound_6 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*work__0).entry) });
unsafe {
(*work__0).func = Some(acpi_table_events_fn);
}
if true {
break;
}
}
if true {
break;
}
}
schedule_work(work__0);
}

static mut __key__2947: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_reconfig_notifier_register")]
extern "C" fn acpi_reconfig_notifier_register(nb: *mut notifier_block) -> i32 {
unsafe { blocking_notifier_chain_register(core::ptr::addr_of_mut!(acpi_reconfig_chain), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_reconfig_notifier_register_555: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(acpi_reconfig_notifier_register)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scan.acpi_reconfig_notifier_unregister")]
extern "C" fn acpi_reconfig_notifier_unregister(nb: *mut notifier_block) -> i32 {
unsafe { blocking_notifier_chain_unregister(core::ptr::addr_of_mut!(acpi_reconfig_chain), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_acpi_reconfig_notifier_unregister_556: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(acpi_reconfig_notifier_unregister)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
