
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
//!  libahci.c - Common AHCI SATA low-level routines
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
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
//! # chars in a path name including nul
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! array_index_mask_nospec() - generate a mask that is ~0UL when the
//! 	bounds check succeeds and 0 otherwise
//! @idx: array element index
//! @sz: number of elements in array
//!
//! Returns:
//!     0 - (@idx < @sz)
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! @a is a power of 2 value
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
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
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
//! 64bit FS base
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Used in tsk->__state:
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
//! array_index_nospec - sanitize an array index after a bounds check
//!
//! For a code sequence like:
//!
//!     if (index < size) {
//!         index = array_index_nospec(index, size);
//!         val = array[index];
//!     }
//!
//! ...if the CPU speculates past the bounds check then
//! array_index_nospec() will clamp the index within the range of [0,
//! size).
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Generate SD flag indexes
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! These flags used only by the kernel as part of the
//! irq handling routines.
//!
//! IRQF_SHARED - allow sharing the irq among several devices
//! IRQF_PROBE_SHARED - set by callers when they expect sharing mismatches to occur
//! IRQF_TIMER - Flag to mark this interrupt as timer interrupt
//! IRQF_PERCPU - Interrupt is per cpu
//! IRQF_NOBALANCING - Flag to exclude this interrupt from irq balancing
//! IRQF_IRQPOLL - Interrupt is used for polling (only the interrupt that is
//!                registered first in a shared interrupt is considered for
//!                performance reasons)
//! IRQF_ONESHOT - Interrupt is not reenabled after the hardirq handler finished.
//!                Used by threaded interrupts which need to keep the
//!                irq line disabled until the threaded handler has been run.
//! IRQF_NO_SUSPEND - Do not disable this IRQ during suspend.  Does not guarantee
//!                   that this interrupt will wake the system from a suspended
//!                   state.  See Documentation/power/suspend-and-interrupts.rst
//! IRQF_FORCE_RESUME - Force enable it on resume even if IRQF_NO_SUSPEND is set
//! IRQF_NO_THREAD - Interrupt cannot be threaded
//! IRQF_EARLY_RESUME - Resume IRQ early during syscore instead of at device
//!                resume time.
//! IRQF_COND_SUSPEND - If the IRQ is shared with a NO_SUSPEND user, execute this
//!                interrupt handler after suspending interrupts. For system
//!                wakeup devices users need to implement wakeup detection in
//!                their interrupt handlers.
//! IRQF_NO_AUTOEN - Don't enable IRQ or NMI automatically when users request it.
//!                Users will enable it explicitly by enable_irq() or enable_nmi()
//!                later.
//! IRQF_NO_DEBUG - Exclude from runnaway detection for IPI and similar handlers,
//! 		   depends on IRQF_PERCPU.
//! IRQF_COND_ONESHOT - Agree to do IRQF_ONESHOT if already set for a shared
//!                 interrupt.
//! See the comment in asm-generic/io.h about ioremap_np().
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
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
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
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
//! Enclosure Management Control
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Prevent speculative execution past this barrier.
// Atomic operations are already serializing on x86
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
// Operation is not supported
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
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__100 {
() => {
((core::mem::size_of::<[i32; 5]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__55!() as u64)
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

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__19 {
() => {
core::mem::size_of::<__SlateRecord_libahci_c_2313>() as i32
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
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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
macro_rules! GENMASK_TYPE__34 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__19!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__9 {
($h:expr, $l:expr) => {
GENMASK_TYPE__34!(u64, $h, $l)
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
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

macro_rules! ____is_defined {
() => {
0i32
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire__2 {
($x:expr) => {
__acquire_ctx_lock__2(($x) as *const spinlock);
};
}

macro_rules! __const_hweight16__2 {
($w:expr) => {
__const_hweight8__2!($w).wrapping_add(__const_hweight8__2!(($w) >> 8i32))
};
}

macro_rules! __const_hweight16 {
($w:expr) => {
__const_hweight8!($w).wrapping_add(__const_hweight8!(($w) >> 8i32))
};
}

macro_rules! __const_hweight32__2 {
($w:expr) => {
__const_hweight16__2!($w).wrapping_add(__const_hweight16__2!(($w) >> 16i32))
};
}

macro_rules! __const_hweight32 {
($w:expr) => {
__const_hweight16!($w).wrapping_add(__const_hweight16!(($w) >> 16i32))
};
}

macro_rules! __const_hweight64 {
($w:expr) => {
__const_hweight32!($w).wrapping_add(__const_hweight32!(($w) >> 32i32))
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8__2 {
($w:expr) => {
(((($w) as u64) & 1u64 != 0) as i32).wrapping_add(((($w) as u64) & 2u64 != 0) as i32).wrapping_add(((($w) as u64) & 4u64 != 0) as i32).wrapping_add(((($w) as u64) & 8u64 != 0) as i32).wrapping_add(((($w) as u64) & 16u64 != 0) as i32).wrapping_add(((($w) as u64) & 32u64 != 0) as i32).wrapping_add(((($w) as u64) & 64u64 != 0) as i32).wrapping_add(((($w) as u64) & 128u64 != 0) as i32) as u32
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
((($w) & 1u64 != 0) as i32).wrapping_add((($w) & 2u64 != 0) as i32).wrapping_add((($w) & 4u64 != 0) as i32).wrapping_add((($w) & 8u64 != 0) as i32).wrapping_add((($w) & 16u64 != 0) as i32).wrapping_add((($w) & 32u64 != 0) as i32).wrapping_add((($w) & 64u64 != 0) as i32).wrapping_add((($w) & 128u64 != 0) as i32) as u32
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
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

macro_rules! __must_be_array__55 {
() => {
core::mem::size_of::<__SlateRecord_libahci_c_648>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__22 {
() => {
((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release__2 {
($x:expr) => {
__release_ctx_lock__1(($x) as *const raw_spinlock);
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

macro_rules! class_to_shost {
($d:expr) => {
{
let mut __v: *mut Scsi_Host;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__57!() as isize)) }) as *mut ()) as *mut Scsi_Host;
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
macro_rules! container_of__54 {
($ptr:expr) => {
{
let mut __v: *mut ahci_em_priv;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__88!() as isize)) }) as *mut ()) as *mut ahci_em_priv;
__v
}
};
}

macro_rules! hweight32__2 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe($w) } {
__const_hweight32__2!($w)
} else {
__arch_hweight32($w)
};
__v
}
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32(($w) as u32)
};
__v
}
};
}

macro_rules! hweight64 {
($w:expr) => {
{
let __v: u64 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight64!($w) as u64
} else {
__arch_hweight64($w)
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! offsetof__2 {
() => {
40u64
};
}

macro_rules! offsetof__88 {
() => {
core::mem::offset_of!(ahci_em_priv, timer) as u64
};
}

macro_rules! offsetof__57 {
() => {
1392u64
};
}

macro_rules! printk_ratelimit {
() => {
unsafe { __printk_ratelimit(c"ahci_read_em_buffer".as_ptr() as *const u8) }
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

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release__2!(spinlock_check($lock));
__acquire__2!($lock);
if true {
break;
}
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
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

macro_rules! timer_container_of__2 {
($callback_timer:expr) => {
container_of__54!($callback_timer)
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

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_bool: kernel_param_ops;
    static mut irq_stat: __SlateRecord_hardirq_h_65;
    static mut dev_attr_unload_heads: device_attribute;
    static mut dev_attr_link_power_management_supported: device_attribute;
    static mut dev_attr_link_power_management_policy: device_attribute;
    static mut dev_attr_ncq_prio_supported: device_attribute;
    static mut dev_attr_ncq_prio_enable: device_attribute;
    static mut dev_attr_em_message_type: device_attribute;
    static mut dev_attr_em_message: device_attribute;
    static mut dev_attr_sw_activity: device_attribute;
    static mut ata_dummy_port_ops: ata_port_operations;
    static mut sata_deb_timing_normal: [u32; 0];
    static mut sata_deb_timing_hotplug: [u32; 0];
    static mut sata_pmp_port_ops: ata_port_operations;
    static mut kstat: kernel_stat;
    static mut pm_suspend_global_flags: u32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __printk_ratelimit(arg0: *const u8) -> i32;
    fn kstrtouint(arg0: *const u8, arg1: u32, arg2: *mut u32) -> i32;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_29();
    fn __compiletime_assert_57();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn devm_kmalloc(arg0: *mut device, arg1: u64, arg2: u32) -> *mut ();
    fn devm_kfree(arg0: *mut device, arg1: *const ());
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn devm_request_threaded_irq(arg0: *mut device, arg1: u32, arg2: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg3: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg4: u64, arg5: *const u8, arg6: *mut ()) -> i32;
    fn disable_irq(arg0: u32);
    fn dmam_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn acpi_storage_d3(arg0: *mut device) -> bool;
    fn cdrom_multisession(arg0: *mut cdrom_device_info, arg1: *mut cdrom_multisession) -> i32;
    fn ata_wait_after_reset(arg0: *mut ata_link, arg1: u64, arg2: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn ata_std_postreset(arg0: *mut ata_link, arg1: *mut u32);
    fn ata_host_start(arg0: *mut ata_host) -> i32;
    fn ata_host_register(arg0: *mut ata_host, arg1: *const scsi_host_template) -> i32;
    fn ata_host_activate(arg0: *mut ata_host, arg1: i32, arg2: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg3: u64, arg4: *const scsi_host_template) -> i32;
    fn ata_ratelimit() -> i32;
    fn ata_msleep(arg0: *mut ata_port, arg1: u32);
    fn ata_wait_register(arg0: *mut ata_port, arg1: *mut (), arg2: u32, arg3: u32, arg4: u32, arg5: u32) -> u32;
    fn ata_std_qc_defer(arg0: *mut ata_queued_cmd) -> i32;
    fn ata_port_classify(arg0: *mut ata_port, arg1: *const ata_taskfile) -> u32;
    fn sata_link_hardreset(arg0: *mut ata_link, arg1: *const u32, arg2: u64, arg3: *mut bool, arg4: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn sata_link_scr_lpm(arg0: *mut ata_link, arg1: u32, arg2: bool) -> i32;
    fn ata_tf_to_fis(arg0: *const ata_taskfile, arg1: u8, arg2: i32, arg3: *mut u8);
    fn ata_tf_from_fis(arg0: *const u8, arg1: *mut ata_taskfile);
    fn ata_qc_complete_multiple(arg0: *mut ata_port, arg1: u64) -> i32;
    fn sata_lpm_ignore_phy_events(arg0: *mut ata_link) -> bool;
    fn sata_async_notification(arg0: *mut ata_port) -> i32;
    fn ata_acpi_gtm(arg0: *mut ata_port, arg1: *mut ata_acpi_gtm) -> i32;
    fn ata_link_abort(arg0: *mut ata_link) -> i32;
    fn ata_port_abort(arg0: *mut ata_port) -> i32;
    fn ata_port_freeze(arg0: *mut ata_port) -> i32;
    fn ata_ehi_push_desc(arg0: *mut ata_eh_info, arg1: *const u8, ...);
    fn ata_ehi_clear_desc(arg0: *mut ata_eh_info);
    fn ata_port_desc(arg0: *mut ata_port, arg1: *const u8, ...);
    fn ata_link_next(arg0: *mut ata_link, arg1: *mut ata_port, arg2: u32) -> *mut ata_link;
    fn sata_pmp_qc_defer_cmd_switch(arg0: *mut ata_queued_cmd) -> i32;
    fn sata_pmp_error_handler(arg0: *mut ata_port);
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn ata_dev_set_feature(arg0: *mut ata_device, arg1: u8, arg2: u8) -> u32;
    fn __compiletime_assert_694();
    fn __compiletime_assert_695();
    fn __builtin_unreachable();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
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
fn __acquire_ctx_lock__1(var: &raw_spinlock) {
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


#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
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
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
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
fn __arch_hweight64(mut w: u64) -> u64 {
let mut res: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight64\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntq %rdi, %rax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u64, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
res
}

// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
#[inline(always)]
fn hweight_long(w: u64) -> u64 {
let __v0: u64 = if (core::mem::size_of::<u64>() as u64) == 4 {
hweight32!(w) as u64
} else {
hweight64!(w)
};
__v0
}

fn __ffs64(word: u64) -> u32 {
__ffs!(word) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
    __slate_bits_0: __slate_bits::__SlateBits97U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits98U0,
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


#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset__0: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset__0 >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset__0);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset__0) }
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
fn hlist_unhashed_lockless(h: *const hlist_node) -> i32 {
let mut __v0: *mut *mut hlist_node;
loop {
if !(__native_word__22!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_29() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*h).pprev) as *const *mut *mut hlist_node) };
__v0.is_null() as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
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
    __slate_bits_0: __slate_bits::__SlateBits117U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct pollfd {
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
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
fn __release_ctx_lock__2(var: &spinlock) {
}

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


#[inline(always)]
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_ctx_lock__3(var: &rwlock) {
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
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_ctx_lock__4(var: &mutex) {
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


#[inline(always)]
fn __acquire_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_ctx_lock__5(var: &seqlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
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
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits285U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
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


#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __ctx_lock_RCU {
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
struct vm_struct {
}


#[inline(always)]
fn __acquire_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_ctx_lock__7(var: &rw_semaphore) {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
fn jiffies_to_msecs(j: u64) -> u32 {
1u64.wrapping_mul(j) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


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

fn timer_pending(timer: *const timer_list) -> i32 {
(hlist_unhashed_lockless(unsafe { core::ptr::addr_of!((*timer).entry) }) == 0) as i32
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


#[inline(always)]
fn __acquire_ctx_lock__8(var: &srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[inline(always)]
fn __release_ctx_lock__8(var: &srcu_struct) {
}

// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits403U0,
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
struct irq_domain {
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
#[inline(always)]
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __release_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
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
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[inline(always)]
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_ctx_lock__11(var: &__context_bitlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __context_bitlock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
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
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct fiemap_extent_info {
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
struct poll_table_struct {
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
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
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
struct posix_acl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits581U0,
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
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits647U0,
    vm_ops: *const vm_operations_struct,
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits590U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
struct wake_irq {
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
struct dev_pm_qos {
}


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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_hardirq_h_65 {
    counts: [u32; 18],
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
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits706U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits706U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits706U2,
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
    __slate_bits_3: __slate_bits::__SlateBits706U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
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
struct bio_alloc_cache {
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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


// DEVICE_ATTR - Define a device attribute.
// @_name: Attribute name.
// @_mode: File mode.
// @_show: Show handler. Optional, but mandatory if attribute is readable.
// @_store: Store handler. Optional, but mandatory if attribute is writable.
//
// Convenience macro for defining a struct device_attribute.
//
// For example, ``DEVICE_ATTR(foo, 0644, foo_show, foo_store);`` expands to:
//
// .. code-block:: c
//
// 	struct device_attribute dev_attr_foo = {
// 		.attr	= { .name = "foo", .mode = 0644 },
// 		.show	= foo_show,
// 		.store	= foo_store,
// 	};
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
struct timer_rand_state {
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
    __slate_bits_0: __slate_bits::__SlateBits703U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
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
struct scsi_transport_template {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bsg_device {
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
struct scsi_device_handler {
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
struct io_comp_batch {
    req_list: rq_list,
    need_ts: bool,
    complete: Option<unsafe extern "C" fn(*mut io_comp_batch)>,
    poll_ctx: *mut (),
}


// Maximum contiguous integrity buffer allocation.
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

fn devm_request_irq(dev: *mut device, irq: u32, handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, irqflags: u64, devname: *const u8, dev_id: *mut ()) -> i32 {
unsafe { devm_request_threaded_irq(dev, irq, handler, None, irqflags | 2097152, devname, dev_id) }
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
fn dmam_alloc_coherent(dev: *mut device, size: u64, dma_handle: *mut u64, gfp: u32) -> *mut () {
unsafe { dmam_alloc_attrs(dev, size, dma_handle, gfp, if gfp & (BIT!(13i32) as u32) != 0 { 256u64 } else { 0u64 }) }
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_771 {
    id: [u16; 256],
    gscr: [u32; 128],
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
struct clk {
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
    __slate_bits_0: __slate_bits::__SlateBits823U0,
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
    __slate_bits_0: __slate_bits::__SlateBits820U0,
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
    __slate_bits_0: __slate_bits::__SlateBits818U0,
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
struct scsi_target {
    starget_sdev_user: *mut scsi_device,
    siblings: list_head,
    devices: list_head,
    dev: device,
    reap_ref: kref,
    channel: u32,
    id: u32,
    __slate_bits_0: __slate_bits::__SlateBits819U0,
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
struct cgroup_taskset {
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
struct cdrom_device_info {
    ops: *const cdrom_device_ops,
    list: list_head,
    disk: *mut gendisk,
    handle: *mut (),
    mask: i32,
    speed: i32,
    capacity: i32,
    __slate_bits_0: __slate_bits::__SlateBits782U0,
    vfs_events: u32,
    ioctl_events: u32,
    use_count: i32,
    name: [u8; 20],
    __slate_bits_1: __slate_bits::__SlateBits782U1,
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
struct swap_cluster_info {
}


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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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
struct ata_cpr {
    num: u8,
    num_storage_elements: u8,
    start_lba: u64,
    num_lbas: u64,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ata_acpi_gtm {
    drive: [ata_acpi_drive; 2],
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reset_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_40_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_43_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_105_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_106_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_107_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_108_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_109_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_111_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_201_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_208_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_648 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libahci_c_2313 {
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
    pub struct __SlateBits549U0 {
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
    pub struct __SlateBits581U0 {
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
    pub struct __SlateBits647U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits918U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U0 {
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
    pub struct __SlateBits703U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U1 {
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
    pub struct __SlateBits706U2 {
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
    pub struct __SlateBits706U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U0 {
        #[bits(30)]
        pub options: u32,
        #[bits(2)]
        pub mc_flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U1 {
        #[bits(2)]
        pub sanyo_slot: u8,
        #[bits(1)]
        pub keeplocked: u8,
        #[bits(5)]
        pub reserved: u8,
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
    pub struct __SlateBits98U0 {
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
    pub struct __SlateBits117U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits590U0 {
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
    pub struct __SlateBits97U0 {
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
    pub struct __SlateBits403U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits819U0 {
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
    pub struct __SlateBits818U0 {
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
    pub struct __SlateBits823U0 {
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
    pub struct __SlateBits820U0 {
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
    pub struct __SlateBits941U0 {
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
    pub struct __SlateBits959U0 {
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
struct ata_port_info {
    flags: u64,
    link_flags: u64,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    port_ops: *mut ata_port_operations,
    private_data: *mut (),
}


fn ata_is_atapi(prot: u8) -> bool {
(prot as i32) & 8 != 0
}

fn ata_is_ncq(prot: u8) -> bool {
(prot as i32) & 4 != 0
}

fn ata_port_is_dummy(ap: *mut ata_port) -> i32 {
((unsafe { (*ap).ops }) == unsafe { core::ptr::addr_of_mut!(ata_dummy_port_ops) }) as i32
}

fn ata_port_is_frozen(ap: *const ata_port) -> bool {
(unsafe { (*ap).pflags }) & 4 != 0
}

fn sata_ehc_deb_timing(ehc: *mut ata_eh_context) -> *const u32 {
if (unsafe { (*ehc).i.flags }) & 1 != 0 {
return unsafe { core::ptr::addr_of!(sata_deb_timing_hotplug) as *const u32 };
} else {
return unsafe { core::ptr::addr_of!(sata_deb_timing_normal) as *const u32 };
}
}

fn sata_pmp_supported(ap: *mut ata_port) -> bool {
(unsafe { (*ap).flags }) & 524288 != 0
}

fn sata_pmp_attached(ap: *mut ata_port) -> bool {
(unsafe { (*ap).nr_pmp_links }) != 0
}

fn ata_is_host_link(link: *const ata_link) -> bool {
link == ((unsafe { core::ptr::addr_of_mut!((*(*link).ap).link) }) as *const ata_link) || link == ((unsafe { (*(*link).ap).slave_link }) as *const ata_link)
}

fn sata_srst_pmp(link: *mut ata_link) -> i32 {
let __v0: bool = sata_pmp_supported(unsafe { (*link).ap }) && ata_is_host_link(link as *const ata_link);
if __v0 {
return 15;
}
unsafe { (*link).pmp }
}

fn ata_ehi_hotplugged(ehi: *mut ata_eh_info) {
unsafe {
(*ehi).probe_mask = (*ehi).probe_mask | 3u32;
}
unsafe {
(*ehi).flags = (*ehi).flags | 1;
}
unsafe {
(*ehi).action = (*ehi).action | 14u32;
}
unsafe {
(*ehi).err_mask = (*ehi).err_mask | 16;
}
}

fn ata_port_desc_misc(ap: *mut ata_port, irq: i32) {
unsafe { ata_port_desc(ap, c"irq %d".as_ptr() as *const u8, irq) };
unsafe { ata_port_desc(ap, c"lpm-pol %d".as_ptr() as *const u8, (*ap).target_lpm_policy as u32) };
if (unsafe { (*ap).pflags }) & 4194304 != 0 {
unsafe { ata_port_desc(ap, c"ext".as_ptr() as *const u8) };
}
}

fn ata_tag_internal(tag: u32) -> bool {
tag == 32
}

fn ata_tag_valid(tag: u32) -> bool {
tag < 32 || ata_tag_internal(tag)
}

fn ata_class_enabled(class: u32) -> u32 {
(class == 1 || class == 3 || class == 5 || class == 7 || class == 9) as u32
}

fn ata_dev_enabled(dev: *const ata_device) -> u32 {
ata_class_enabled(unsafe { (*dev).class })
}

fn ata_link_active(link: *mut ata_link) -> i32 {
(ata_tag_valid(unsafe { (*link).active_tag }) || (unsafe { (*link).sactive }) != 0) as i32
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
fn __ata_qc_from_tag(ap: *mut ata_port, tag: u32) -> *mut ata_queued_cmd {
if ata_tag_valid(tag) {
return unsafe { (*ap).qcmd.as_mut_ptr() as *mut ata_queued_cmd }.wrapping_offset(tag as isize);
}
core::ptr::null_mut::<ata_queued_cmd>()
}

fn ata_qc_from_tag(ap: *mut ata_port, tag: u32) -> *mut ata_queued_cmd {
let qc: *mut ata_queued_cmd = __ata_qc_from_tag(ap, tag);
if unlikely!(qc.is_null()) != 0 {
return qc;
}
if (unsafe { (*qc).flags }) & 65537u64 == 1 {
return qc;
}
core::ptr::null_mut::<ata_queued_cmd>()
}

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

fn ata_shost_to_port(host: *mut Scsi_Host) -> *mut ata_port {
unsafe { *((core::ptr::addr_of_mut!((*host).hostdata) as *mut u64).wrapping_offset(0 as isize) as *mut *mut ata_port) }
}

fn ata_check_ready(status: u8) -> i32 {
if (status as i32) & 128 == 0 {
return 1;
}
// 0xff indicates either no device or device not ready
if (status as i32) == 255 {
return 19i32.wrapping_neg();
}
0
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
fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn pm_runtime_put(dev: *mut device) {
unsafe { __pm_runtime_idle(dev, 5i32) };
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
    __slate_bits_0: __slate_bits::__SlateBits918U0,
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
    __slate_bits_0: __slate_bits::__SlateBits549U0,
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ahci_ignore_sss: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_shost_groups: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_sdev_groups: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_ops: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_pmp_retry_srst_ops: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_save_initial_config: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_start_engine: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_stop_engine: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_start_fis_rx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_reset_controller: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_reset_em: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_init_controller: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_dev_classify: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_fill_cmd_slot: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_kick_engine: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_do_softreset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_check_ready: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_do_hardreset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_handle_port_intr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_qc_issue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_error_handler: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_port_resume: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_print_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_set_em_messages: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ahci_host_activate: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous", sym ahci_ignore_sss, sym ahci_shost_groups, sym ahci_sdev_groups, sym ahci_ops, sym ahci_pmp_retry_srst_ops, sym ahci_save_initial_config, sym ahci_start_engine, sym ahci_stop_engine, sym ahci_start_fis_rx, sym ahci_reset_controller, sym ahci_reset_em, sym ahci_init_controller, sym ahci_dev_classify, sym ahci_fill_cmd_slot, sym ahci_kick_engine, sym ahci_do_softreset, sym ahci_check_ready, sym ahci_do_hardreset, sym ahci_handle_port_intr, sym ahci_qc_issue, sym ahci_error_handler, sym ahci_port_resume, sym ahci_print_info, sym ahci_set_em_messages, sym ahci_host_activate, options(att_syntax));

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
struct phy_configure_opts_dp {
    link_rate: u32,
    lanes: u32,
    voltage: [u32; 4],
    pre: [u32; 4],
    __slate_bits_0: __slate_bits::__SlateBits941U0,
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


// Enclosure Management LED Message Type
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
struct ahci_sg {
    addr: u32,
    addr_hi: u32,
    reserved: u32,
    flags_size: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits959U0,
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


#[unsafe(no_mangle)]
static mut ahci_ignore_sss: i32 = unsafe { core::mem::zeroed() };

fn __ahci_port_base(hpriv: *mut ahci_host_priv, port_no: u32) -> *mut () {
let mmio: *mut () = unsafe { (*hpriv).mmio };
(unsafe { (((mmio as *const u8).wrapping_offset(256 as isize) as *mut ()) as *const u8).wrapping_offset(port_no.wrapping_mul(128) as isize) }) as *mut ()
}

fn ahci_port_base(ap: *mut ata_port) -> *mut () {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
__ahci_port_base(hpriv, unsafe { (*ap).port_no })
}

fn ahci_nr_ports(cap: u32) -> i32 {
(cap & 31).wrapping_add(1) as i32
}

static mut ahci_skip_host_reset: i32 = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_ignore_sss_673: *mut () = (unsafe { core::ptr::addr_of_mut!(ahci_ignore_sss) }) as *mut ();

fn __check_skip_host_reset() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(ahci_skip_host_reset) }
}

static __param_str_skip_host_reset: __SlateAlign16<[u8; 24]> = __SlateAlign16([108u8, 105u8, 98u8, 97u8, 104u8, 99u8, 105u8, 46u8, 115u8, 107u8, 105u8, 112u8, 95u8, 104u8, 111u8, 115u8, 116u8, 95u8, 114u8, 101u8, 115u8, 101u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_skip_host_reset: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_skip_host_reset.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_libahci_c_40>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_40_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_40_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_40_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_40_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_40_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ahci_skip_host_reset) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_674: [u8; 37] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 107, 105, 112, 95, 104, 111, 115, 116, 95, 114, 101, 115, 101, 116, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_675: [u8; 75] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 115, 107, 105, 112, 95, 104, 111, 115, 116, 95, 114, 101, 115, 101, 116, 58, 115, 107, 105, 112, 32, 103, 108, 111, 98, 97, 108, 32, 104, 111, 115, 116, 32, 114, 101, 115, 101, 116, 32, 40, 48, 61, 100, 111, 110, 39, 116, 32, 115, 107, 105, 112, 44, 32, 49, 61, 115, 107, 105, 112, 41, 0];

fn __check_ignore_sss() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(ahci_ignore_sss) }
}

static __param_str_ignore_sss: __SlateAlign16<[u8; 19]> = __SlateAlign16([108u8, 105u8, 98u8, 97u8, 104u8, 99u8, 105u8, 46u8, 105u8, 103u8, 110u8, 111u8, 114u8, 101u8, 95u8, 115u8, 115u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_ignore_sss: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_ignore_sss.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_libahci_c_43>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_43_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_43_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_43_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_43_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_43_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ahci_ignore_sss) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_676: [u8; 32] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 105, 103, 110, 111, 114, 101, 95, 115, 115, 115, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_677: [u8; 80] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 105, 103, 110, 111, 114, 101, 95, 115, 115, 115, 58, 73, 103, 110, 111, 114, 101, 32, 115, 116, 97, 103, 103, 101, 114, 101, 100, 32, 115, 112, 105, 110, 117, 112, 32, 102, 108, 97, 103, 32, 40, 48, 61, 100, 111, 110, 39, 116, 32, 105, 103, 110, 111, 114, 101, 44, 32, 49, 61, 105, 103, 110, 111, 114, 101, 41, 0];

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_led_store")]
extern "C" fn ahci_led_store(ap: *mut ata_port, buf: *const u8, size: u64) -> i64 {
let mut state: u32 = 0;
let mut pmp: i32 = 0;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut emp: *mut ahci_em_priv = unsafe { core::mem::zeroed() };
if (unsafe { kstrtouint(buf, 0u32, core::ptr::addr_of_mut!(state)) }) < 0 {
return 22i32.wrapping_neg() as i64;
}
// get the slot number from the message
pmp = ((state & 65280) >> 8i32) as i32;
if (pmp as u32) < 15 {
let mut __v0: i32;
let _i: i32 = pmp;
let _s: u32 = 15;
let mut _mask: u64 = 0;
let mut __v1: u64;
let mut __idx: u32 = _i as u32;
let mut __sz: u32 = _s;
let mut __mask: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("cmp {1:e},{2:e}; sbb {0:r},{0:r}", lateout(reg) __t0, in(reg) __sz as u32, in(reg) __idx as u32, options(nostack, att_syntax));
__mask = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v1 = __mask;
_mask = __v1;
loop {
if ((const { core::mem::size_of::<i32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_694() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<u32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_695() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = ((_i as u64) & _mask) as i32;
pmp = __v0;
emp = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset(pmp as isize);
} else {
return 22i32.wrapping_neg() as i64;
}
// mask off the activity bits if we are in sw_activity
// mode, user should turn off sw_activity before setting
// activity led through em_message
if ((unsafe { (*emp).blink_policy }) as u32) != 0 {
let __v5: u32 = state & 4294508543u32;
state = __v5;
}
unsafe { unsafe { (*(*ap).ops).transmit_led_message }.unwrap()(ap, state, size as i64) }
}

static mut dev_attr_ahci_host_caps: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"ahci_host_caps\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_105>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_105_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_105_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_105_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_105_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_105_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_show_host_caps) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut dev_attr_ahci_host_cap2: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"ahci_host_cap2\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_106>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_106_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_106_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_106_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_106_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_106_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_show_host_cap2) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut dev_attr_ahci_host_version: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"ahci_host_version\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_107>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_107_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_107_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_107_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_107_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_107_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_show_host_version) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut dev_attr_ahci_port_cmd: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"ahci_port_cmd\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_108>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_108_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_108_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_108_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_108_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_108_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_show_port_cmd) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut dev_attr_em_buffer: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"em_buffer\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_109>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_109_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_109_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_109_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_109_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_109_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_read_em_buffer) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(ahci_store_em_buffer) } };

static mut dev_attr_em_message_supported: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"em_message_supported\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libahci_c_111>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_111_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_111_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_111_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_111_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_111_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ahci_show_em_supported) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

static mut ahci_shost_attrs: __SlateAlign16<[*mut attribute; 11]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_link_power_management_supported.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_link_power_management_policy.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_em_message_type.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_em_message.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ahci_host_caps.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ahci_host_cap2.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ahci_host_version.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ahci_port_cmd.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_em_buffer.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_em_message_supported.attr) }, core::ptr::null_mut::<attribute>()]);

static ahci_shost_attr_group: __SlateConst<attribute_group> = __SlateConst({
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(ahci_shost_attrs.0) as *mut *mut attribute } };
__t0
});

#[unsafe(no_mangle)]
static mut ahci_shost_groups: __SlateAlign16<[*const attribute_group; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of!(ahci_shost_attr_group.0) }, core::ptr::null::<attribute_group>()]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_shost_groups_678: *mut () = (unsafe { core::ptr::addr_of_mut!(ahci_shost_groups.0) }) as *mut ();

static mut ahci_sdev_attrs: __SlateAlign16<[*mut attribute; 5]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_sw_activity.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_unload_heads.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ncq_prio_supported.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_ncq_prio_enable.attr) }, core::ptr::null_mut::<attribute>()]);

static ahci_sdev_attr_group: __SlateConst<attribute_group> = __SlateConst({
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(ahci_sdev_attrs.0) as *mut *mut attribute } };
__t0
});

#[unsafe(no_mangle)]
static mut ahci_sdev_groups: __SlateAlign16<[*const attribute_group; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of!(ahci_sdev_attr_group.0) }, core::ptr::null::<attribute_group>()]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_sdev_groups_679: *mut () = (unsafe { core::ptr::addr_of_mut!(ahci_sdev_groups.0) }) as *mut ();

#[unsafe(no_mangle)]
static mut ahci_ops: ata_port_operations = {
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.qc_defer = Some(ahci_pmp_qc_defer);
__t0.qc_prep = Some(ahci_qc_prep);
__t0.qc_issue = Some(ahci_qc_issue);
__t0.qc_fill_rtf = Some(ahci_qc_fill_rtf);
__t0.qc_ncq_fill_rtf = Some(ahci_qc_ncq_fill_rtf);
__t0.dev_config = Some(ahci_dev_config);
__t0.freeze = Some(ahci_freeze);
__t0.thaw = Some(ahci_thaw);
__t0.reset = {
let mut __t1: ata_reset_operations = unsafe { core::mem::zeroed() };
__t1.softreset = Some(ahci_softreset);
__t1.hardreset = Some(ahci_hardreset);
__t1.postreset = Some(ahci_postreset);
__t1
};
__t0.pmp_reset = {
let mut __t2: ata_reset_operations = unsafe { core::mem::zeroed() };
__t2.softreset = Some(ahci_softreset);
__t2
};
__t0.error_handler = Some(ahci_error_handler);
__t0.post_internal_cmd = Some(ahci_post_internal_cmd);
__t0.scr_read = Some(ahci_scr_read);
__t0.scr_write = Some(ahci_scr_write);
__t0.pmp_attach = Some(ahci_pmp_attach);
__t0.pmp_detach = Some(ahci_pmp_detach);
__t0.set_lpm = Some(ahci_set_lpm);
__t0.port_suspend = Some(ahci_port_suspend);
__t0.port_resume = Some(ahci_port_resume);
__t0.port_start = Some(ahci_port_start);
__t0.port_stop = Some(ahci_port_stop);
__t0.em_show = Some(ahci_led_show);
__t0.em_store = Some(ahci_led_store);
__t0.sw_activity_show = Some(ahci_activity_show);
__t0.sw_activity_store = Some(ahci_activity_store);
__t0.transmit_led_message = Some(ahci_transmit_led_message);
__t0.inherits = unsafe { core::ptr::addr_of!(sata_pmp_port_ops) };
__t0
};

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_ops_680: *mut () = (unsafe { core::ptr::addr_of_mut!(ahci_ops) }) as *mut ();

#[unsafe(no_mangle)]
static mut ahci_pmp_retry_srst_ops: ata_port_operations = {
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.reset = {
let mut __t1: ata_reset_operations = unsafe { core::mem::zeroed() };
__t1.softreset = Some(ahci_pmp_retry_softreset);
__t1
};
__t0.inherits = (unsafe { core::ptr::addr_of_mut!(ahci_ops) }) as *const ata_port_operations;
__t0
};

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_pmp_retry_srst_ops_681: *mut () = (unsafe { core::ptr::addr_of_mut!(ahci_pmp_retry_srst_ops) }) as *mut ();

#[unsafe(link_section = ".data..read_mostly")]
static mut ahci_em_messages: bool = 1i32 != 0i32;

fn __check_ahci_em_messages() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(ahci_em_messages) }
}

static __param_str_ahci_em_messages: __SlateAlign16<[u8; 25]> = __SlateAlign16([108u8, 105u8, 98u8, 97u8, 104u8, 99u8, 105u8, 46u8, 97u8, 104u8, 99u8, 105u8, 95u8, 101u8, 109u8, 95u8, 109u8, 101u8, 115u8, 115u8, 97u8, 103u8, 101u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_ahci_em_messages: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_ahci_em_messages.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_libahci_c_201>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_201_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_201_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_201_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_201_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_201_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ahci_em_messages) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_682: [u8; 39] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 97, 104, 99, 105, 95, 101, 109, 95, 109, 101, 115, 115, 97, 103, 101, 115, 58, 98, 111, 111, 108, 0];

/// add other LED protocol types when they become supported
#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_683: [u8; 90] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 97, 104, 99, 105, 95, 101, 109, 95, 109, 101, 115, 115, 97, 103, 101, 115, 58, 65, 72, 67, 73, 32, 69, 110, 99, 108, 111, 115, 117, 114, 101, 32, 77, 97, 110, 97, 103, 101, 109, 101, 110, 116, 32, 77, 101, 115, 115, 97, 103, 101, 32, 99, 111, 110, 116, 114, 111, 108, 32, 40, 48, 32, 61, 32, 111, 102, 102, 44, 32, 49, 32, 61, 32, 111, 110, 41, 0];

/// device sleep idle timeout in ms
#[unsafe(link_section = ".data..read_mostly")]
static mut devslp_idle_timeout: i32 = 1000;

fn __check_devslp_idle_timeout() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(devslp_idle_timeout) }
}

static __param_str_devslp_idle_timeout: __SlateAlign16<[u8; 28]> = __SlateAlign16([108u8, 105u8, 98u8, 97u8, 104u8, 99u8, 105u8, 46u8, 100u8, 101u8, 118u8, 115u8, 108u8, 112u8, 95u8, 105u8, 100u8, 108u8, 101u8, 95u8, 116u8, 105u8, 109u8, 101u8, 111u8, 117u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_devslp_idle_timeout: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_devslp_idle_timeout.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_libahci_c_208>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_208_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_208_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_208_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_208_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libahci_c_208_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(devslp_idle_timeout) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_684: [u8; 41] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 100, 101, 118, 115, 108, 112, 95, 105, 100, 108, 101, 95, 116, 105, 109, 101, 111, 117, 116, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_685: [u8; 59] = [108, 105, 98, 97, 104, 99, 105, 46, 112, 97, 114, 109, 61, 100, 101, 118, 115, 108, 112, 95, 105, 100, 108, 101, 95, 116, 105, 109, 101, 111, 117, 116, 58, 100, 101, 118, 105, 99, 101, 32, 115, 108, 101, 101, 112, 32, 105, 100, 108, 101, 32, 116, 105, 109, 101, 111, 117, 116, 0];

fn ahci_enable_ahci(mmio: *mut ()) {
let mut tmp: u32 = 0;
// turn on AHCI_EN
tmp = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ());
if tmp & 2147483648 != 0 {
return;
}
// Some controllers need AHCI_EN to be written multiple times.
// Try a few times before giving up.
for i in 0i32..5i32 {
tmp |= 2147483648;
writel(tmp, (unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
tmp = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ()); // flush && sanity check
if tmp & 2147483648 != 0 {
return;
}
unsafe { msleep(10u32) };
}
let mut __v5: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("686: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 686b - ., 3; .popsection\n# {0}", const 686 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ahci_enable_ahci_6, sym __slate_asm_string_ahci_enable_ahci_7, const 233 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("687: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 687b - ., 4; .popsection\n# {0}", const 687 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}

/// ahci_rpm_get_port - Make sure the port is powered on
/// @ap: Port to power on
///
/// Whenever there is need to access the AHCI host registers outside of
/// normal execution paths, call this function to make sure the host is
/// actually powered on.
fn ahci_rpm_get_port(ap: *mut ata_port) -> i32 {
pm_runtime_get_sync(unsafe { (*ap).dev })
}

/// ahci_rpm_put_port - Undoes ahci_rpm_get_port()
/// @ap: Port to power down
///
/// Undoes ahci_rpm_get_port() and possibly powers down the AHCI host
/// if it has no more active users.
fn ahci_rpm_put_port(ap: *mut ata_port) {
pm_runtime_put(unsafe { (*ap).dev });
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_show_host_caps")]
extern "C" fn ahci_show_host_caps(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
(unsafe { sprintf(buf, c"%x\n".as_ptr() as *const u8, (*hpriv).cap) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_show_host_cap2")]
extern "C" fn ahci_show_host_cap2(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
(unsafe { sprintf(buf, c"%x\n".as_ptr() as *const u8, (*hpriv).cap2) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_show_host_version")]
extern "C" fn ahci_show_host_version(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
(unsafe { sprintf(buf, c"%x\n".as_ptr() as *const u8, (*hpriv).version) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_show_port_cmd")]
extern "C" fn ahci_show_port_cmd(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let port_mmio: *mut () = ahci_port_base(ap);
let mut ret: i64 = 0;
ahci_rpm_get_port(ap);
ret = (unsafe { sprintf(buf, c"%x\n".as_ptr() as *const u8, readl(((port_mmio as *const u8).wrapping_offset(24 as isize) as *mut ()) as *const ())) }) as i64;
ahci_rpm_put_port(ap);
ret
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_read_em_buffer")]
extern "C" fn ahci_read_em_buffer(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut em_mmio: *mut () = (unsafe { (mmio as *const u8).wrapping_offset((*hpriv).em_loc as isize) }) as *mut ();
let mut em_ctl: u32 = 0;
let mut msg: u32 = 0;
let mut flags: u64 = 0;
let mut count: u64 = 0;
let mut i: i32 = 0;
ahci_rpm_get_port(ap);
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
em_ctl = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
if (unsafe { (*ap).flags }) & 2097152 == 0 || em_ctl & 33554432 != 0 || (unsafe { (*hpriv).em_msg_type }) & 8 == 0 {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
return 22i32.wrapping_neg() as i64;
}
if em_ctl & 1 == 0 {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
return 11i32.wrapping_neg() as i64;
}
if em_ctl & 16777216 == 0 {
em_mmio = (unsafe { (em_mmio as *const u8).wrapping_offset((*hpriv).em_buf_sz as isize) }) as *mut ();
}
count = (unsafe { (*hpriv).em_buf_sz }) as u64;
// the count should not be larger than PAGE_SIZE
if count > _AC!() << 12i32 {
if printk_ratelimit!() != 0 {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x014ata%u: EM read buffer size too large: buffer size %u, page size %lu\n".as_ptr() as *const u8, (*ap).print_id, (*hpriv).em_buf_sz, _AC!() << 12i32) };
}
count = _AC!() << 12i32;
}
while (i as u64) < count {
msg = readl(((unsafe { (em_mmio as *const u8).wrapping_offset(i as isize) }) as *mut ()) as *const ());
unsafe {
*buf.wrapping_offset(i as isize) = (msg & 255) as u8;
}
unsafe {
*buf.wrapping_offset(i.wrapping_add(1) as isize) = (msg >> 8i32 & 255) as u8;
}
unsafe {
*buf.wrapping_offset(i.wrapping_add(2) as isize) = (msg >> 16i32 & 255) as u8;
}
unsafe {
*buf.wrapping_offset(i.wrapping_add(3) as isize) = (msg >> 24i32 & 255) as u8;
}
i = i.wrapping_add(4);
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
i as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_store_em_buffer")]
extern "C" fn ahci_store_em_buffer(dev: *mut device, attr: *mut device_attribute, buf: *const u8, size: u64) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let em_mmio: *mut () = (unsafe { (mmio as *const u8).wrapping_offset((*hpriv).em_loc as isize) }) as *mut ();
let msg_buf: *const u8 = buf as *const u8;
let mut em_ctl: u32 = 0;
let mut msg: u32 = 0;
let mut flags: u64 = 0;
let mut i: i32 = 0;
// check size validity
if (unsafe { (*ap).flags }) & 2097152 == 0 || (unsafe { (*hpriv).em_msg_type }) & 8 == 0 || size % 4 != 0 || size > ((unsafe { (*hpriv).em_buf_sz }) as u64) {
return 22i32.wrapping_neg() as i64;
}
ahci_rpm_get_port(ap);
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
em_ctl = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
if em_ctl & 256 != 0 {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
return 16i32.wrapping_neg() as i64;
}
while (i as u64) < size {
msg = (((unsafe { *msg_buf.wrapping_offset(i as isize) }) as i32) | ((unsafe { *msg_buf.wrapping_offset(i.wrapping_add(1) as isize) }) as i32) << 8i32 | ((unsafe { *msg_buf.wrapping_offset(i.wrapping_add(2) as isize) }) as i32) << 16i32 | ((unsafe { *msg_buf.wrapping_offset(i.wrapping_add(3) as isize) }) as i32) << 24i32) as u32;
writel(msg, (unsafe { (em_mmio as *const u8).wrapping_offset(i as isize) }) as *mut ());
i = i.wrapping_add(4);
}
writel(em_ctl | 256, (unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ());
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
size as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_show_em_supported")]
extern "C" fn ahci_show_em_supported(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let shost: *mut Scsi_Host = class_to_shost!(dev);
let ap: *mut ata_port = ata_shost_to_port(shost);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut em_ctl: u32 = 0;
ahci_rpm_get_port(ap);
em_ctl = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
ahci_rpm_put_port(ap);
(unsafe { sprintf(buf, c"%s%s%s%s\n".as_ptr() as *const u8, if em_ctl & 65536 != 0 { c"led ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if em_ctl & 131072 != 0 { c"saf-te ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if em_ctl & 262144 != 0 { c"ses-2 ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if em_ctl & 524288 != 0 { c"sgpio ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) }) as i64
}

/// ahci_save_initial_config - Save and fixup initial config values
/// @dev: target AHCI device
/// @hpriv: host private area to store config values
///
/// Some registers containing configuration info might be setup by
/// BIOS and might be cleared on reset.  This function saves the
/// initial values of those registers into @hpriv such that they
/// can be restored after controller reset.
///
/// If inconsistent, config values are fixed up by this function.
///
/// If it is not set already this function sets hpriv->start_engine to
/// ahci_start_engine.
///
/// LOCKING:
/// None.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_save_initial_config")]
extern "C" fn ahci_save_initial_config(dev: *mut device, hpriv: *mut ahci_host_priv) {
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut port_map: u64 = 0;
let mut cap: u32 = 0;
let mut cap2: u32 = 0;
let mut vers: u32 = 0;
// Preserve the ports capabilities defined by the platform. Note there
// is no need in storing the rest of the P#.CMD fields since they are
// volatile.
let mut i: i32 = 0;
// make sure AHCI mode is enabled before accessing CAP
ahci_enable_ahci(mmio);
// Values prefixed with saved_ are written back to the HBA and ports
// registers after reset. Values without are used for driver operation.
// Override HW-init HBA capability fields with the platform-specific
// values. The rest of the HBA capabilities are defined as Read-only
// and can't be modified in CSR anyway.
cap = readl(((unsafe { (mmio as *const u8).wrapping_offset(0 as isize) }) as *mut ()) as *const ());
if (unsafe { (*hpriv).saved_cap }) != 0 {
cap = cap & !(134217728 | 268435456) | unsafe { (*hpriv).saved_cap };
}
unsafe {
(*hpriv).saved_cap = cap;
}
// CAP2 register is only defined for AHCI 1.2 and later
vers = readl(((unsafe { (mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if vers >> 16i32 > 1 || vers >> 16i32 == 1 && vers & 65535 >= 512 {
let __v0: u32 = readl(((unsafe { (mmio as *const u8).wrapping_offset(36 as isize) }) as *mut ()) as *const ());
cap2 = __v0;
unsafe {
(*hpriv).saved_cap2 = __v0;
}
} else {
cap2 = 0;
unsafe {
(*hpriv).saved_cap2 = 0;
}
}
// some chips have errata preventing 64bit use
if cap & 2147483648 != 0 && (unsafe { (*hpriv).flags }) & 8 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do 64bit DMA, forcing 32bit\n".as_ptr() as *const u8) };
let __v3: u32 = cap & !2147483648;
cap = __v3;
}
if cap & 1073741824 != 0 && (unsafe { (*hpriv).flags }) & 1 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do NCQ, turning off CAP_NCQ\n".as_ptr() as *const u8) };
let __v6: u32 = cap & !1073741824;
cap = __v6;
}
if cap & 1073741824 == 0 && (unsafe { (*hpriv).flags }) & 512 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can do NCQ, turning on CAP_NCQ\n".as_ptr() as *const u8) };
cap |= 1073741824;
}
if cap & 131072 != 0 && (unsafe { (*hpriv).flags }) & 64 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do PMP, turning off CAP_PMP\n".as_ptr() as *const u8) };
let __v12: u32 = cap & !131072;
cap = __v12;
}
if cap & 536870912 != 0 && (unsafe { (*hpriv).flags }) & 4096 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do SNTF, turning off CAP_SNTF\n".as_ptr() as *const u8) };
let __v15: u32 = cap & !536870912;
cap = __v15;
}
if cap2 & 8 != 0 && (unsafe { (*hpriv).flags }) & 131072 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do DEVSLP, turning off\n".as_ptr() as *const u8) };
let __v18: u32 = cap2 & !8;
cap2 = __v18;
let __v20: u32 = cap2 & !16;
cap2 = __v20;
}
if cap & 65536 == 0 && (unsafe { (*hpriv).flags }) & 16384 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can do FBS, turning on CAP_FBS\n".as_ptr() as *const u8) };
cap |= 65536;
}
if cap & 65536 != 0 && (unsafe { (*hpriv).flags }) & 262144 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can't do FBS, turning off CAP_FBS\n".as_ptr() as *const u8) };
let __v26: u32 = cap & !65536;
cap = __v26;
}
if cap & 67108864 == 0 && (unsafe { (*hpriv).flags }) & 8388608 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller can do ALPM, turning on CAP_ALPM\n".as_ptr() as *const u8) };
cap |= 67108864;
}
if cap & 32 != 0 && (unsafe { (*hpriv).flags }) & 67108864 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"controller does not support SXS, disabling CAP_SXS\n".as_ptr() as *const u8) };
let __v32: u32 = cap & !32;
cap = __v32;
}
// Override the HBA ports mapping if the platform needs it
port_map = readl(((unsafe { (mmio as *const u8).wrapping_offset(12 as isize) }) as *mut ()) as *const ()) as u64;
if (unsafe { (*hpriv).saved_port_map }) != 0 && port_map != ((unsafe { (*hpriv).saved_port_map }) as u64) {
while false {
}
unsafe { _dev_info(dev as *const device, c"forcing port_map 0x%lx -> 0x%x\n".as_ptr() as *const u8, port_map, (*hpriv).saved_port_map) };
port_map = (unsafe { (*hpriv).saved_port_map }) as u64;
} else {
unsafe {
(*hpriv).saved_port_map = port_map as u32;
}
}
// mask_port_map not set means that all ports are available
if (unsafe { (*hpriv).mask_port_map }) != 0 {
while false {
}
unsafe { _dev_warn(dev as *const device, c"masking port_map 0x%lx -> 0x%lx\n".as_ptr() as *const u8, port_map, port_map & ((*hpriv).mask_port_map as u64)) };
port_map &= (unsafe { (*hpriv).mask_port_map }) as u64;
}
// cross check port_map and cap.n_ports
if port_map != 0 {
let map_ports: i32 = hweight_long(port_map) as i32;
// If PI has more ports than n_ports, whine, clear
// port_map and let it be generated from n_ports.
if map_ports > ahci_nr_ports(cap) {
while false {
}
unsafe { _dev_warn(dev as *const device, c"implemented port map (0x%lx) contains more ports than nr_ports (%u), using nr_ports\n".as_ptr() as *const u8, port_map, ahci_nr_ports(cap)) };
port_map = 0;
}
}
// fabricate port_map from cap.nr_ports for < AHCI 1.3
if port_map == 0 && vers < 66304 {
port_map = (1i32 << ahci_nr_ports(cap)).wrapping_sub(1) as u64;
while false {
}
unsafe { _dev_warn(dev as *const device, c"forcing PORTS_IMPL to 0x%lx\n".as_ptr() as *const u8, port_map) };
// write the fixed up value to the PI register
unsafe {
(*hpriv).saved_port_map = port_map as u32;
}
}
loop {
i = find_next_bit(core::ptr::addr_of_mut!(port_map) as *const u64, 32u64, i as u64) as i32;
if (i as u32) >= 32 {
break;
}
if (unsafe { *((*hpriv).saved_port_cap.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }) == 0 {
let port_mmio: *mut () = __ahci_port_base(hpriv, i as u32);
unsafe {
*((*hpriv).saved_port_cap.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = readl(((port_mmio as *const u8).wrapping_offset(24 as isize) as *mut ()) as *const ()) & 8126464;
}
}
i = i.wrapping_add(1);
}
// record values to use during operation
unsafe {
(*hpriv).cap = cap;
}
unsafe {
(*hpriv).cap2 = cap2;
}
unsafe {
(*hpriv).version = vers;
}
unsafe {
(*hpriv).port_map = port_map as u32;
}
if (unsafe { (*hpriv).start_engine }) == None {
unsafe {
(*hpriv).start_engine = Some(ahci_start_engine);
}
}
if (unsafe { (*hpriv).stop_engine }) == None {
unsafe {
(*hpriv).stop_engine = Some(ahci_stop_engine);
}
}
if (unsafe { (*hpriv).irq_handler }) == None {
unsafe {
(*hpriv).irq_handler = Some(ahci_single_level_irq_intr);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_save_initial_config_688: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, *mut ahci_host_priv)>, *mut ()>(Some(ahci_save_initial_config)) };

/// ahci_restore_initial_config - Restore initial config
/// @host: target ATA host
///
/// Restore initial config stored by ahci_save_initial_config().
///
/// LOCKING:
/// None.
fn ahci_restore_initial_config(host: *mut ata_host) {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mut port_map: u64 = (unsafe { (*hpriv).port_map }) as u64;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut i: i32 = 0;
writel(unsafe { (*hpriv).saved_cap }, (unsafe { (mmio as *const u8).wrapping_offset(0 as isize) }) as *mut ());
if (unsafe { (*hpriv).saved_cap2 }) != 0 {
writel(unsafe { (*hpriv).saved_cap2 }, (unsafe { (mmio as *const u8).wrapping_offset(36 as isize) }) as *mut ());
}
writel(unsafe { (*hpriv).saved_port_map }, (unsafe { (mmio as *const u8).wrapping_offset(12 as isize) }) as *mut ());
readl(((unsafe { (mmio as *const u8).wrapping_offset(12 as isize) }) as *mut ()) as *const ()); // flush
loop {
i = find_next_bit(core::ptr::addr_of_mut!(port_map) as *const u64, 32u64, i as u64) as i32;
if (i as u32) >= 32 {
break;
}
let port_mmio: *mut () = __ahci_port_base(hpriv, i as u32);
writel(unsafe { *((*hpriv).saved_port_cap.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
i = i.wrapping_add(1);
}
}

fn ahci_scr_offset(ap: *mut ata_port, sc_reg: u32) -> u32 {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
if (sc_reg as u64) < ARRAY_SIZE__100!() && (sc_reg != 4 || (unsafe { (*hpriv).cap }) & 536870912 != 0) {
return (unsafe { offset.0[sc_reg as usize] }) as u32;
}
0
}

static offset: __SlateAlign16<[i32; 5]> = __SlateAlign16([40i32, 48i32, 44i32, 52i32, 60i32]);

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_scr_read")]
extern "C" fn ahci_scr_read(link: *mut ata_link, sc_reg: u32, val: *mut u32) -> i32 {
let port_mmio: *mut () = ahci_port_base(unsafe { (*link).ap });
let offset__0: i32 = ahci_scr_offset(unsafe { (*link).ap }, sc_reg) as i32;
if offset__0 != 0 {
unsafe {
*val = readl(((port_mmio as *const u8).wrapping_offset(offset__0 as isize) as *mut ()) as *const ());
}
return 0;
}
22i32.wrapping_neg()
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_scr_write")]
extern "C" fn ahci_scr_write(link: *mut ata_link, sc_reg: u32, val: u32) -> i32 {
let port_mmio: *mut () = ahci_port_base(unsafe { (*link).ap });
let offset__0: i32 = ahci_scr_offset(unsafe { (*link).ap }, sc_reg) as i32;
if offset__0 != 0 {
writel(val, (unsafe { (port_mmio as *const u8).wrapping_offset(offset__0 as isize) }) as *mut ());
return 0;
}
22i32.wrapping_neg()
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_start_engine")]
extern "C" fn ahci_start_engine(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let mut tmp: u32 = 0;
// start DMA
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
tmp |= 1;
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ()); // flush
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_start_engine_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ahci_start_engine)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_stop_engine")]
extern "C" fn ahci_stop_engine(ap: *mut ata_port) -> i32 {
let port_mmio: *mut () = ahci_port_base(ap);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mut tmp: u32 = 0;
// On some controllers, stopping a port's DMA engine while the port
// is in ALPM state (partial or slumber) results in failures on
// subsequent DMA engine starts.  For those controllers, put the
// port back in active state before stopping its DMA engine.
let __v0: bool = (unsafe { (*hpriv).flags }) & 4194304 != 0 && ((unsafe { (*ap).link.lpm_policy }) as u32) > 1 && ahci_set_lpm(unsafe { core::ptr::addr_of_mut!((*ap).link) }, 1u32, 4u32) != 0;
if __v0 {
while false {
}
unsafe { _dev_err((*(*ap).host).dev as *const device, c"Failed to wake up port before engine stop\n".as_ptr() as *const u8) };
return 5i32.wrapping_neg();
}
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
// check if the HBA is idle
if tmp & (1 | 32768) == 0 {
return 0;
}
// Don't try to issue commands but return with ENODEV if the
// AHCI controller not available anymore (e.g. due to PCIe hot
// unplugging). Otherwise a 500ms delay for each port is added.
if tmp == 4294967295 {
while false {
}
unsafe { _dev_err((*(*ap).host).dev as *const device, c"AHCI controller unavailable!\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
// setting HBA to idle
let __v4: u32 = tmp & !1;
tmp = __v4;
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
// wait for engine to stop. This could be as long as 500 msec
tmp = unsafe { ata_wait_register(ap, (port_mmio as *const u8).wrapping_offset(24 as isize) as *mut (), 32768u32, 32768u32, 1u32, 500u32) };
if tmp & 32768 != 0 {
return 5i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_stop_engine_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> i32>, *mut ()>(Some(ahci_stop_engine)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_start_fis_rx")]
extern "C" fn ahci_start_fis_rx(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut tmp: u32 = 0;
// On HBAs that only support 32-bit addressing PxCLBU is read only '0'.
// When applying the AHCI_HFLAG_32BIT_ONLY quirk, PxCLBU is RW, and the
// reset value is Implementation Specific, so we need to clear it to 0.
if (unsafe { (*hpriv).cap }) & 2147483648 != 0 {
writel(((unsafe { (*pp).cmd_slot_dma }) >> 16i32 >> 16i32) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
} else {
if (unsafe { (*hpriv).flags }) & 8 != 0 {
writel(0u32, (unsafe { (port_mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
}
}
writel(((unsafe { (*pp).cmd_slot_dma }) & 4294967295) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(0 as isize) }) as *mut ());
// On HBAs that only support 32-bit addressing PxFBU is read only '0'.
// When applying the AHCI_HFLAG_32BIT_ONLY quirk, PxFBU is RW, and the
// reset value is Implementation Specific, so we need to clear it to 0.
if (unsafe { (*hpriv).cap }) & 2147483648 != 0 {
writel(((unsafe { (*pp).rx_fis_dma }) >> 16i32 >> 16i32) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(12 as isize) }) as *mut ());
} else {
if (unsafe { (*hpriv).flags }) & 8 != 0 {
writel(0u32, (unsafe { (port_mmio as *const u8).wrapping_offset(12 as isize) }) as *mut ());
}
}
writel(((unsafe { (*pp).rx_fis_dma }) & 4294967295) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
// enable FIS reception
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
tmp |= 16;
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
// flush
readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_start_fis_rx_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ahci_start_fis_rx)) };

fn ahci_stop_fis_rx(ap: *mut ata_port) -> i32 {
let port_mmio: *mut () = ahci_port_base(ap);
let mut tmp: u32 = 0;
// disable FIS reception
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
let __v1: u32 = tmp & !16;
tmp = __v1;
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
// wait for completion, spec says 500ms, give it 1000
tmp = unsafe { ata_wait_register(ap, (port_mmio as *const u8).wrapping_offset(24 as isize) as *mut (), 16384u32, 16384u32, 10u32, 1000u32) };
if tmp & 16384 != 0 {
return 16i32.wrapping_neg();
}
0
}

fn ahci_power_up(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut cmd: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ()) & !4026531840;
// spin up device
if (unsafe { (*hpriv).cap }) & 134217728 != 0 {
cmd |= 2;
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
}
// wake up link
writel(cmd | 268435456, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_set_lpm")]
extern "C" fn ahci_set_lpm(link: *mut ata_link, policy: u32, mut hints: u32) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let port_mmio: *mut () = ahci_port_base(ap);
if (policy as u32) != 1 {
// wakeup flag only applies to the max power policy
let __v1: u32 = hints & 4294967291u32;
hints = __v1;
// Disable interrupts on Phy Ready. This keeps us from
// getting woken up due to spurious phy ready
// interrupts.
let __v4: u32 = (unsafe { (*pp).intr_mask }) & !4194304;
unsafe {
(*pp).intr_mask = __v4;
}
writel(unsafe { (*pp).intr_mask }, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
unsafe { sata_link_scr_lpm(link, policy, false) };
}
if (unsafe { (*hpriv).cap }) & 67108864 != 0 {
let mut cmd: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
if (policy as u32) == 1 || hints & 2 == 0 {
if hints & 4 == 0 {
let __v6: u32 = cmd & !(134217728 | 67108864);
cmd = __v6;
}
cmd |= 268435456;
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
// wait 10ms to be sure we've come out of LPM state
unsafe { ata_msleep(ap, 10u32) };
if hints & 4 != 0 {
return 0;
}
} else {
cmd |= 67108864;
if (policy as u32) == 5 {
cmd |= 134217728;
} else {
if (policy as u32) == 4 {
let __v14: u32 = cmd & !134217728;
cmd = __v14;
}
}
// write out new cmd value
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
}
}
// set aggressive device sleep
if (unsafe { (*hpriv).cap2 }) & 8 != 0 && (unsafe { (*hpriv).cap2 }) & 16 != 0 && (unsafe { (*((*link).device.as_mut_ptr() as *mut ata_device)).flags }) & 536870912 != 0 {
if (policy as u32) == 5 || (policy as u32) == 4 {
ahci_set_aggressive_devslp(ap, true);
} else {
ahci_set_aggressive_devslp(ap, false);
}
}
if (policy as u32) == 1 {
unsafe { sata_link_scr_lpm(link, policy, false) };
// turn PHYRDY IRQ back on
unsafe {
(*pp).intr_mask = (*pp).intr_mask | 4194304;
}
writel(unsafe { (*pp).intr_mask }, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
}
0
}

fn ahci_power_down(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut cmd: u32 = 0;
let mut scontrol: u32 = 0;
if (unsafe { (*hpriv).cap }) & 134217728 == 0 {
return;
}
// put device into listen mode, first set PxSCTL.DET to 0
scontrol = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(44 as isize) }) as *mut ()) as *const ());
let __v1: u32 = scontrol & 4294967280u32;
scontrol = __v1;
writel(scontrol, (unsafe { (port_mmio as *const u8).wrapping_offset(44 as isize) }) as *mut ());
// then set PxCMD.SUD to 0
cmd = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ()) & !4026531840;
let __v3: u32 = cmd & !2;
cmd = __v3;
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
}

fn ahci_start_port(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
let mut rc: i64 = 0;
let mut i: i32 = 0;
// enable FIS reception
ahci_start_fis_rx(ap);
// enable DMA
if (unsafe { (*hpriv).flags }) & 32768 == 0 {
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
}
// turn on LEDs
if (unsafe { (*ap).flags }) & 2097152 != 0 {
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
// EM Transmit bit maybe busy during init
i = 0;
while (i as u32) < 5 {
rc = unsafe { unsafe { (*(*ap).ops).transmit_led_message }.unwrap()(ap, (*emp).led_state as u32, 4i64) };
// If busy, give a breather but do not
// release EH ownership by using msleep()
// instead of ata_msleep().  EM Transmit
// bit is busy for the whole host and
// releasing ownership will cause other
// ports to fail the same way.
if rc == (16i32.wrapping_neg() as i64) {
unsafe { msleep(1u32) };
} else {
break;
}
i = i.wrapping_add(1);
}
link = unsafe { ata_link_next(link, ap, 0u32) };
}
}
if (unsafe { (*ap).flags }) & 4194304 != 0 {
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
ahci_init_sw_activity(link);
link = unsafe { ata_link_next(link, ap, 0u32) };
}
}
}

fn ahci_deinit_port(ap: *mut ata_port, emsg: &mut *const u8) -> i32 {
let mut rc: i32 = 0;
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
// disable DMA
rc = unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
if rc != 0 {
unsafe {
*(emsg as *mut *const u8) = c"failed to stop engine".as_ptr() as *const u8;
}
return rc;
}
// disable FIS reception
rc = ahci_stop_fis_rx(ap);
if rc != 0 {
unsafe {
*(emsg as *mut *const u8) = c"failed stop FIS RX".as_ptr() as *const u8;
}
return rc;
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_reset_controller")]
extern "C" fn ahci_reset_controller(host: *mut ata_host) -> i32 {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut tmp: u32 = 0;
// We must be in AHCI mode, before using anything AHCI-specific, such
// as HOST_RESET.
ahci_enable_ahci(mmio);
// Global controller reset
if (unsafe { ahci_skip_host_reset }) != 0 {
while false {
}
unsafe { _dev_info((*host).dev as *const device, c"Skipping global host reset\n".as_ptr() as *const u8) };
return 0;
}
tmp = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ());
if tmp & 1 == 0 {
writel(tmp | 1, (unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ()); // flush
}
// To perform host reset, OS should set HOST_RESET and poll until this
// bit is read to be "0". Reset must complete within 1 second, or the
// hardware should be considered fried.
tmp = unsafe { ata_wait_register(core::ptr::null_mut::<ata_port>(), (mmio as *const u8).wrapping_offset(4 as isize) as *mut (), 1u32, 1u32, 10u32, 1000u32) };
if tmp & 1 != 0 {
while false {
}
unsafe { _dev_err((*host).dev as *const device, c"Controller reset failed (0x%x)\n".as_ptr() as *const u8, tmp) };
return 5i32.wrapping_neg();
}
// Turn on AHCI mode
ahci_enable_ahci(mmio);
// Some registers might be cleared on reset. Restore initial values.
if (unsafe { (*hpriv).flags }) & 16777216 == 0 {
ahci_restore_initial_config(host);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_reset_controller_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host) -> i32>, *mut ()>(Some(ahci_reset_controller)) };

fn ahci_sw_activity(link: *mut ata_link) {
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
if (unsafe { (*link).flags }) & 128 == 0 {
return;
}
let __v1: u64 = unsafe { (*emp).activity };
unsafe {
(*emp).activity = __v1.wrapping_add(1);
}
if timer_pending((unsafe { core::ptr::addr_of_mut!((*emp).timer) }) as *const timer_list) == 0 {
unsafe { mod_timer(core::ptr::addr_of_mut!((*emp).timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(10u32))) };
}
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_sw_activity_blink")]
extern "C" fn ahci_sw_activity_blink(t: *mut timer_list) {
let emp: *mut ahci_em_priv = timer_container_of__2!(t);
let link: *mut ata_link = unsafe { (*emp).link };
let ap: *mut ata_port = unsafe { (*link).ap };
let mut led_message: u64 = unsafe { (*emp).led_state };
let mut activity_led_state: u32 = 0;
let mut flags: u64 = 0;
led_message &= 4294901760;
led_message |= ((unsafe { (*ap).port_no }) | (((unsafe { (*link).pmp }) << 8i32) as u32)) as u64;
// check to see if we've had activity.  If so,
// toggle state of LED and reset timer.  If not,
// turn LED to desired idle state.
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
if (unsafe { (*emp).saved_activity }) != unsafe { (*emp).activity } {
unsafe {
(*emp).saved_activity = (*emp).activity;
}
// get the current LED state
activity_led_state = (led_message & 65536) as u32;
if activity_led_state != 0 {
activity_led_state = 0;
} else {
activity_led_state = 1;
}
// clear old state
let __v9: u64 = led_message & 18446744073709092863u64;
led_message = __v9;
// toggle state
led_message |= (activity_led_state << 16i32) as u64;
unsafe { mod_timer(core::ptr::addr_of_mut!((*emp).timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(100u32))) };
} else {
// switch to idle
let __v13: u64 = led_message & 18446744073709092863u64;
led_message = __v13;
if ((unsafe { (*emp).blink_policy }) as u32) == 2 {
led_message |= 65536u64;
}
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { unsafe { (*(*ap).ops).transmit_led_message }.unwrap()(ap, led_message as u32, 4i64) };
}

fn ahci_init_sw_activity(link: *mut ata_link) {
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
// init activity stats, setup timer
unsafe {
(*emp).activity = 0;
}
unsafe {
(*emp).saved_activity = 0;
}
unsafe {
(*emp).link = link;
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*emp).timer) }, Some(ahci_sw_activity_blink), 0i32);
// check our blink policy and set flag for link if it's enabled
if ((unsafe { (*emp).blink_policy }) as u32) != 0 {
unsafe {
(*link).flags = (*link).flags | 128;
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_reset_em")]
extern "C" fn ahci_reset_em(host: *mut ata_host) -> i32 {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let em_ctl: u32 = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
if em_ctl & 256 != 0 || em_ctl & 512 != 0 {
return 22i32.wrapping_neg();
}
writel(em_ctl | 512, (unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ());
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_reset_em_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host) -> i32>, *mut ()>(Some(ahci_reset_em)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_transmit_led_message")]
extern "C" fn ahci_transmit_led_message(ap: *mut ata_port, state: u32, size: i64) -> i64 {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut em_ctl: u32 = 0;
let mut message: [u32; 2] = [0, 0];
let mut flags: u64 = 0;
let mut pmp: i32 = 0;
let mut emp: *mut ahci_em_priv = unsafe { core::mem::zeroed() };
// get the slot number from the message
pmp = ((state & 65280) >> 8i32) as i32;
if (pmp as u32) < 15 {
emp = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset(pmp as isize);
} else {
return 22i32.wrapping_neg() as i64;
}
ahci_rpm_get_port(ap);
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
// if we are still busy transmitting a previous message,
// do not allow
em_ctl = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
if em_ctl & 256 != 0 {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
return 16i32.wrapping_neg() as i64;
}
if (unsafe { (*hpriv).em_msg_type }) & 1 != 0 {
// create message header - this is all zero except for
// the message size, which is 4 bytes.
let __v3: *mut u32 = (message.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize);
unsafe {
*__v3 = *__v3 | 1024u32;
}
// ignore 0:4 of byte zero, fill in port info yourself
message[1] = state & 4294967280u32 | unsafe { (*ap).port_no };
// write message to EM_LOC
writel(message[0], (unsafe { (mmio as *const u8).wrapping_offset((*hpriv).em_loc as isize) }) as *mut ());
writel(message[1], (unsafe { (((mmio as *const u8).wrapping_offset((*hpriv).em_loc as isize) as *mut ()) as *const u8).wrapping_offset(4 as isize) }) as *mut ());
// tell hardware to transmit the message
writel(em_ctl | 256, (unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ());
}
// save off new led state for port/slot
unsafe {
(*emp).led_state = state as u64;
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ahci_rpm_put_port(ap);
size
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_led_show")]
extern "C" fn ahci_led_show(ap: *mut ata_port, buf: *mut u8) -> i64 {
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
let mut rc: i32 = 0;
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
let __v0: i32 = unsafe { sprintf(buf, c"%lx\n".as_ptr() as *const u8, (*emp).led_state) };
rc = rc.wrapping_add(__v0);
link = unsafe { ata_link_next(link, ap, 0u32) };
}
rc as i64
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_activity_store")]
extern "C" fn ahci_activity_store(dev: *mut ata_device, val: u32) -> i64 {
let link: *mut ata_link = unsafe { (*dev).link };
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
let mut port_led_state: u32 = (unsafe { (*emp).led_state }) as u32;
// save the desired Activity LED behavior
if (val as u32) == 0 {
// clear LFLAG
let __v2: u32 = (unsafe { (*link).flags }) & 4294967167u32;
unsafe {
(*link).flags = __v2;
}
// set the LED to OFF
port_led_state &= 4294443008;
port_led_state |= (unsafe { (*ap).port_no }) | (((unsafe { (*link).pmp }) << 8i32) as u32);
unsafe { unsafe { (*(*ap).ops).transmit_led_message }.unwrap()(ap, port_led_state, 4i64) };
} else {
unsafe {
(*link).flags = (*link).flags | 128;
}
if (val as u32) == 2 {
// set LED to ON for idle
port_led_state &= 4294443008;
port_led_state |= (unsafe { (*ap).port_no }) | (((unsafe { (*link).pmp }) << 8i32) as u32);
port_led_state |= 65536; // check this
unsafe { unsafe { (*(*ap).ops).transmit_led_message }.unwrap()(ap, port_led_state, 4i64) };
}
}
unsafe {
(*emp).blink_policy = val;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_activity_show")]
extern "C" fn ahci_activity_show(dev: *mut ata_device, buf: *mut u8) -> i64 {
let link: *mut ata_link = unsafe { (*dev).link };
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let emp: *mut ahci_em_priv = unsafe { (*pp).em_priv.as_mut_ptr() as *mut ahci_em_priv }.wrapping_offset((unsafe { (*link).pmp }) as isize);
// display the saved value of activity behavior for this
// disk.
(unsafe { sprintf(buf, c"%d\n".as_ptr() as *const u8, (*emp).blink_policy as u32) }) as i64
}

fn ahci_port_clear_pending_irq(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut tmp: u32 = 0;
// clear SError
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(48 as isize) }) as *mut ()) as *const ());
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*(*ap).host).dev as *const device, c"PORT_SCR_ERR 0x%x\n".as_ptr() as *const u8, tmp) };
}
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(48 as isize) }) as *mut ());
// clear port IRQ
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*(*ap).host).dev as *const device, c"PORT_IRQ_STAT 0x%x\n".as_ptr() as *const u8, tmp) };
}
if tmp != 0 {
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ());
}
writel((1i32 << unsafe { (*ap).port_no }) as u32, (unsafe { ((*hpriv).mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
}

fn ahci_port_init(dev: *mut device, ap: *mut ata_port, port_no: i32, mmio: *mut (), port_mmio: *mut ()) {
let mut emsg: *const u8 = core::ptr::null::<u8>();
let mut rc: i32 = 0;
// make sure port is not active
rc = ahci_deinit_port(ap, unsafe { &mut (*core::ptr::addr_of_mut!(emsg)) });
if rc != 0 {
while false {
}
unsafe { _dev_warn(dev as *const device, c"%s (%d)\n".as_ptr() as *const u8, emsg, rc) };
}
ahci_port_clear_pending_irq(ap);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_init_controller")]
extern "C" fn ahci_init_controller(host: *mut ata_host) {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let mut i: i32 = 0;
let mut tmp: u32 = 0;
while (i as u32) < unsafe { (*host).n_ports } {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) };
let port_mmio: *mut () = ahci_port_base(ap);
if ata_port_is_dummy(ap) == 0 {
ahci_port_init(unsafe { (*host).dev }, ap, i, mmio, port_mmio);
}
i = i.wrapping_add(1);
}
tmp = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ());
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*host).dev as *const device, c"HOST_CTL 0x%x\n".as_ptr() as *const u8, tmp) };
}
writel(tmp | 2, (unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ());
tmp = readl(((unsafe { (mmio as *const u8).wrapping_offset(4 as isize) }) as *mut ()) as *const ());
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*host).dev as *const device, c"HOST_CTL 0x%x\n".as_ptr() as *const u8, tmp) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_init_controller_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host)>, *mut ()>(Some(ahci_init_controller)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_dev_config")]
extern "C" fn ahci_dev_config(dev: *mut ata_device) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*(*(*dev).link).ap).host).private_data }) as *mut ahci_host_priv;
if (unsafe { (*dev).class }) == 3 && (unsafe { (*hpriv).flags }) & 536870912 != 0 {
unsafe {
(*dev).quirks = (*dev).quirks | 512;
}
}
if (unsafe { (*hpriv).flags }) & 256 != 0 {
unsafe {
(*dev).max_sectors = 255;
}
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016ata%u.%02u: SB600 AHCI: limiting to 255 sectors per cmd\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_dev_classify")]
extern "C" fn ahci_dev_classify(ap: *mut ata_port) -> u32 {
let port_mmio: *mut () = ahci_port_base(ap);
let mut tf: ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let tmp: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(36 as isize) }) as *mut ()) as *const ());
tf.lbah = (tmp >> 24i32 & 255) as u8;
tf.lbam = (tmp >> 16i32 & 255) as u8;
tf.lbal = (tmp >> 8i32 & 255) as u8;
tf.nsect = (tmp & 255) as u8;
unsafe { ata_port_classify(ap, core::ptr::addr_of_mut!(tf) as *const ata_taskfile) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_dev_classify_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> u32>, *mut ()>(Some(ahci_dev_classify)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_fill_cmd_slot")]
extern "C" fn ahci_fill_cmd_slot(pp: *mut ahci_port_priv, tag: u32, opts: u32) {
let cmd_tbl_dma: u64 = unsafe { (*pp).cmd_tbl_dma }.wrapping_add(tag.wrapping_mul(2816) as u64);
unsafe {
(*unsafe { (*pp).cmd_slot }.wrapping_offset(tag as isize)).opts = opts;
}
unsafe {
(*unsafe { (*pp).cmd_slot }.wrapping_offset(tag as isize)).status = 0;
}
unsafe {
(*unsafe { (*pp).cmd_slot }.wrapping_offset(tag as isize)).tbl_addr = (cmd_tbl_dma & 4294967295) as u32;
}
unsafe {
(*unsafe { (*pp).cmd_slot }.wrapping_offset(tag as isize)).tbl_addr_hi = (cmd_tbl_dma >> 16i32 >> 16i32) as u32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_fill_cmd_slot_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ahci_port_priv, u32, u32)>, *mut ()>(Some(ahci_fill_cmd_slot)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_kick_engine")]
extern "C" fn ahci_kick_engine(ap: *mut ata_port) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut busy: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(busy) as *mut i32;
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut status: u8 = unsafe { core::mem::zeroed() };
let mut hpriv: *mut ahci_host_priv = unsafe { core::mem::zeroed() };
let mut port_mmio: *mut () = unsafe { core::mem::zeroed() };
unsafe {
port_mmio = ahci_port_base(ap);
hpriv = (*(*ap).host).private_data as *mut ahci_host_priv;
status = (readl(((port_mmio as *const u8).wrapping_offset(32 as isize) as *mut ()) as *const ()) & 255) as u8;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// stop engine
*__slate_slot_3 = unsafe { (*hpriv).stop_engine }.unwrap()(ap);
if *__slate_slot_3 == 0 {
// need to do CLO?
// always do CLO if PMP is attached (AHCI-1.3 9.2)
*__slate_slot_4 = (status as i32) & (128 | 8);
if *__slate_slot_4 == 0 {
__v2 = !sata_pmp_attached(ap);
} else {
__v2 = false;
}
if __v2 {
*__slate_slot_3 = 0;
} else {
if (*hpriv).cap & 16777216 == 0 {
*__slate_slot_3 = 95i32.wrapping_neg();
} else {
// perform CLO
*__slate_slot_5 = readl(((port_mmio as *const u8).wrapping_offset(24 as isize) as *mut ()) as *const ());
*__slate_slot_5 |= 8;
writel(*__slate_slot_5, (port_mmio as *const u8).wrapping_offset(24 as isize) as *mut ());
*__slate_slot_3 = 0;
*__slate_slot_5 = ata_wait_register(ap, (port_mmio as *const u8).wrapping_offset(24 as isize) as *mut (), 8u32, 8u32, 1u32, 500u32);
if *__slate_slot_5 & 8 != 0 {
*__slate_slot_3 = 5i32.wrapping_neg();
}
// restart engine
}
}
}
unsafe { (*hpriv).start_engine }.unwrap()(ap);
return *__slate_slot_3;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_kick_engine_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> i32>, *mut ()>(Some(ahci_kick_engine)) };

fn ahci_exec_polled_cmd(ap: *mut ata_port, pmp: i32, tf: *mut ata_taskfile, is_cmd: i32, flags: u16, timeout_msec: u32) -> i32 {
let cmd_fis_len: u32 = 5; // five dwords
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let fis: *mut u8 = (unsafe { (*pp).cmd_tbl }) as *mut u8;
let mut tmp: u32 = 0;
// prep the command
unsafe { ata_tf_to_fis(tf as *const ata_taskfile, pmp as u8, is_cmd, fis) };
ahci_fill_cmd_slot(pp, 0u32, cmd_fis_len | ((flags as i32) as u32) | ((pmp << 12i32) as u32));
// set port value for softreset of Port Multiplier
if (unsafe { (*pp).fbs_enabled }) && (unsafe { (*pp).fbs_last_dev }) != pmp {
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
let __v1: u32 = tmp & !(3840 | 2);
tmp = __v1;
tmp |= (pmp << 8u32) as u32;
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ());
unsafe {
(*pp).fbs_last_dev = pmp;
}
}
// issue & wait
writel(1u32, (unsafe { (port_mmio as *const u8).wrapping_offset(56 as isize) }) as *mut ());
if timeout_msec != 0 {
tmp = unsafe { ata_wait_register(ap, (port_mmio as *const u8).wrapping_offset(56 as isize) as *mut (), 1u32, 1u32, 1u32, timeout_msec) };
if tmp & 1 != 0 {
ahci_kick_engine(ap);
return 16i32.wrapping_neg();
}
} else {
readl(((unsafe { (port_mmio as *const u8).wrapping_offset(56 as isize) }) as *mut ()) as *const ());
}
// flush
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_do_softreset")]
extern "C" fn ahci_do_softreset(link: *mut ata_link, class: *mut u32, pmp: i32, deadline: u64, check_ready: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut fbs_disabled: bool = unsafe { core::mem::zeroed() };
let mut tf: core::mem::MaybeUninit<ata_taskfile> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut ata_taskfile = core::ptr::addr_of_mut!(tf) as *mut ata_taskfile;
let mut msecs: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(msecs) as *mut u32;
let mut now: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(now) as *mut u64;
let mut reason: *const u8 = unsafe { core::mem::zeroed() };
let mut pp: *mut ahci_port_priv = unsafe { core::mem::zeroed() };
let mut hpriv: *mut ahci_host_priv = unsafe { core::mem::zeroed() };
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
unsafe {
ap = (*link).ap;
hpriv = (*(*ap).host).private_data as *mut ahci_host_priv;
pp = (*ap).private_data as *mut ahci_port_priv;
reason = core::ptr::null::<u8>();
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 32usize);
fbs_disabled = false;
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// prepare for SRST (AHCI-1.1 10.4.1)
*__slate_slot_10 = ahci_kick_engine(ap);
if *__slate_slot_10 != 0 && *__slate_slot_10 != 95i32.wrapping_neg() {
loop {
if sata_pmp_attached((*link).ap) || !unsafe { (*(*link).ap).slave_link }.is_null() {
while false {
}
_printk(c"\x014ata%u.%02u: failed to reset engine (errno=%d)\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp, *__slate_slot_10);
} else {
while false {
}
_printk(c"\x014ata%u: failed to reset engine (errno=%d)\n".as_ptr() as *const u8, (*(*link).ap).print_id, *__slate_slot_10);
}
if true {
break;
}
}
}
// According to AHCI-1.2 9.3.9: if FBS is enable, software shall
// clear PxFBS.EN to '0' prior to issuing software reset to devices
// that is attached to port multiplier.
if !ata_is_host_link(link as *const ata_link) && (*pp).fbs_enabled {
ahci_disable_fbs(ap);
fbs_disabled = true;
}
ata_tf_init((*link).device.as_mut_ptr() as *mut ata_device, core::ptr::addr_of_mut!(*__slate_slot_12));
// issue the first H2D Register FIS
*__slate_slot_13 = 0;
*__slate_slot_14 = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
if time_after__2!(deadline, *__slate_slot_14) {
*__slate_slot_13 = jiffies_to_msecs(deadline.wrapping_sub(*__slate_slot_14));
}
'__join_6: {
(*__slate_slot_12).ctl = (((*__slate_slot_12).ctl as i32) | 4) as u8;
if ahci_exec_polled_cmd(ap, pmp, core::ptr::addr_of_mut!(*__slate_slot_12), 0i32, 1280u16, *__slate_slot_13) != 0 {
*__slate_slot_10 = 5i32.wrapping_neg();
reason = c"1st FIS failed".as_ptr() as *const u8;
} else {
'__join_9: {
// spec says at least 5us, but be generous and sleep for 1ms
ata_msleep(ap, 1u32);
// issue the second H2D Register FIS
let __v4: u8 = (((*__slate_slot_12).ctl as i32) & !4) as u8;
(*__slate_slot_12).ctl = __v4;
ahci_exec_polled_cmd(ap, pmp, core::ptr::addr_of_mut!(*__slate_slot_12), 0i32, 0u16, 0u32);
// wait for link to become ready
*__slate_slot_10 = ata_wait_after_reset(link, deadline, check_ready);
if *__slate_slot_10 == 16i32.wrapping_neg() && (*hpriv).flags & 2048 != 0 {
// Workaround for cases where link online status can't
// be trusted.  Treat device readiness timeout as link
// offline.
loop {
if sata_pmp_attached((*link).ap) || !unsafe { (*(*link).ap).slave_link }.is_null() {
while false {
}
_printk(c"\x016ata%u.%02u: device not ready, treating as offline\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp);
} else {
while false {
}
_printk(c"\x016ata%u: device not ready, treating as offline\n".as_ptr() as *const u8, (*(*link).ap).print_id);
}
if true {
break;
}
}
*class = 11;
} else {
if *__slate_slot_10 != 0 {
// link occupied, -ENODEV too is an error
reason = c"device not ready".as_ptr() as *const u8;
break '__join_6;
} else {
*class = ahci_dev_classify(ap);
break '__join_9;
}
}
}
// re-enable FBS if disabled before
if fbs_disabled {
ahci_enable_fbs(ap);
}
return 0;
}
}
loop {
if sata_pmp_attached((*link).ap) || !unsafe { (*(*link).ap).slave_link }.is_null() {
while false {
}
_printk(c"\x013ata%u.%02u: softreset failed (%s)\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp, reason);
} else {
while false {
}
_printk(c"\x013ata%u: softreset failed (%s)\n".as_ptr() as *const u8, (*(*link).ap).print_id, reason);
}
if true {
break;
}
}
return *__slate_slot_10;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_do_softreset_700: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, *mut u32, i32, u64, Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32>, *mut ()>(Some(ahci_do_softreset)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_check_ready")]
extern "C" fn ahci_check_ready(link: *mut ata_link) -> i32 {
let port_mmio: *mut () = ahci_port_base(unsafe { (*link).ap });
let status: u8 = (readl(((unsafe { (port_mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ()) & 255) as u8;
ata_check_ready(status)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_check_ready_701: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link) -> i32>, *mut ()>(Some(ahci_check_ready)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_softreset")]
extern "C" fn ahci_softreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let pmp: i32 = sata_srst_pmp(link);
ahci_do_softreset(link, class, pmp, deadline, Some(ahci_check_ready))
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_bad_pmp_check_ready")]
extern "C" fn ahci_bad_pmp_check_ready(link: *mut ata_link) -> i32 {
let port_mmio: *mut () = ahci_port_base(unsafe { (*link).ap });
let status: u8 = (readl(((unsafe { (port_mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ()) & 255) as u8;
let irq_status: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
// There is no need to check TFDATA if BAD PMP is found due to HW bug,
// which can save timeout delay.
if irq_status & 8388608 != 0 {
return 5i32.wrapping_neg();
}
ata_check_ready(status)
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_pmp_retry_softreset")]
extern "C" fn ahci_pmp_retry_softreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let port_mmio: *mut () = ahci_port_base(ap);
let pmp: i32 = sata_srst_pmp(link);
let mut rc: i32 = ahci_do_softreset(link, class, pmp, deadline, Some(ahci_bad_pmp_check_ready));
let mut irq_sts: u32 = 0;
// Soft reset fails with IPMS set when PMP is enabled but
// SATA HDD/ODD is connected to SATA port, do soft reset
// again to port 0.
if rc == 5i32.wrapping_neg() {
irq_sts = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
if irq_sts & 8388608 != 0 {
loop {
if sata_pmp_attached(unsafe { (*link).ap }) || !unsafe { (*(*link).ap).slave_link }.is_null() {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014ata%u.%02u: applying PMP SRST workaround and retrying\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp) };
} else {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014ata%u: applying PMP SRST workaround and retrying\n".as_ptr() as *const u8, (*(*link).ap).print_id) };
}
if true {
break;
}
}
rc = ahci_do_softreset(link, class, 0i32, deadline, Some(ahci_check_ready));
}
}
rc
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_do_hardreset")]
extern "C" fn ahci_do_hardreset(link: *mut ata_link, class: *mut u32, deadline: u64, online: *mut bool) -> i32 {
let timing: *const u32 = sata_ehc_deb_timing(unsafe { core::ptr::addr_of_mut!((*link).eh_context) });
let ap: *mut ata_port = unsafe { (*link).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let d2h_fis: *mut u8 = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *mut u8;
let mut tf: ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut rc: i32 = 0;
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
// clear D2H reception area to properly wait for D2H FIS
ata_tf_init(unsafe { (*link).device.as_mut_ptr() as *mut ata_device }, core::ptr::addr_of_mut!(tf));
unsafe {
tf.__slate_anon_14.status = 128;
}
unsafe { ata_tf_to_fis(core::ptr::addr_of_mut!(tf) as *const ata_taskfile, 0u8, 0i32, d2h_fis) };
ahci_port_clear_pending_irq(ap);
rc = unsafe { sata_link_hardreset(link, timing, deadline, online, Some(ahci_check_ready)) };
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
if unsafe { *online } {
unsafe {
*class = ahci_dev_classify(ap);
}
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_do_hardreset_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64, *mut bool) -> i32>, *mut ()>(Some(ahci_do_hardreset)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_hardreset")]
extern "C" fn ahci_hardreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let mut online: bool = false;
ahci_do_hardreset(link, class, deadline, core::ptr::addr_of_mut!(online))
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_postreset")]
extern "C" fn ahci_postreset(link: *mut ata_link, class: *mut u32) {
let ap: *mut ata_port = unsafe { (*link).ap };
let port_mmio: *mut () = ahci_port_base(ap);
let mut new_tmp: u32 = 0;
let mut tmp: u32 = 0;
unsafe { ata_std_postreset(link, class) };
// Make sure port's ATAPI bit is set appropriately
let __v0: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
tmp = __v0;
new_tmp = __v0;
if (unsafe { *class }) == 3 {
new_tmp |= 16777216;
} else {
let __v4: u32 = new_tmp & !16777216;
new_tmp = __v4;
}
if new_tmp != tmp {
writel(new_tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ()); // flush
}
}

fn ahci_fill_sg(qc: *mut ata_queued_cmd, cmd_tbl: *mut ()) -> u32 {
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let ahci_sg: *mut ahci_sg = ((unsafe { (cmd_tbl as *const u8).wrapping_offset(128 as isize) }) as *mut ()) as *mut ahci_sg;
// Next, the S/G list.
let mut si: u32 = 0;
sg = unsafe { (*qc).sg };
while si < unsafe { (*qc).n_elem } {
let addr: u64 = sg_dma_address!(sg);
let sg_len: u32 = sg_dma_len!(sg);
unsafe {
(*ahci_sg.wrapping_offset(si as isize)).addr = (addr & 4294967295) as u32;
}
unsafe {
(*ahci_sg.wrapping_offset(si as isize)).addr_hi = (addr >> 16i32 >> 16i32) as u32;
}
unsafe {
(*ahci_sg.wrapping_offset(si as isize)).flags_size = sg_len.wrapping_sub(1);
}
si = si.wrapping_add(1);
sg = sg_next(sg);
}
si
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_pmp_qc_defer")]
extern "C" fn ahci_pmp_qc_defer(qc: *mut ata_queued_cmd) -> i32 {
let ap: *mut ata_port = unsafe { (*qc).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
if !sata_pmp_attached(ap) || unsafe { (*pp).fbs_enabled } {
return unsafe { ata_std_qc_defer(qc) };
} else {
return unsafe { sata_pmp_qc_defer_cmd_switch(qc) };
}
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_qc_prep")]
extern "C" fn ahci_qc_prep(qc: *mut ata_queued_cmd) -> u32 {
let ap: *mut ata_port = unsafe { (*qc).ap };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let is_atapi: i32 = ata_is_atapi(unsafe { (*qc).tf.protocol }) as i32;
let mut opts: u32 = 0;
let cmd_fis_len: u32 = 5; // five dwords
let mut n_elem: u32 = 0;
// Fill in command table information.  First, the header,
// a SATA Register - Host to Device command FIS.
let cmd_tbl: *mut () = (unsafe { ((*pp).cmd_tbl as *const u8).wrapping_offset(unsafe { (*qc).hw_tag }.wrapping_mul(2816) as isize) }) as *mut ();
unsafe { ata_tf_to_fis(core::ptr::addr_of_mut!((*qc).tf) as *const ata_taskfile, (*(*(*qc).dev).link).pmp as u8, 1i32, cmd_tbl as *mut u8) };
if is_atapi != 0 {
unsafe { memset((cmd_tbl as *const u8).wrapping_offset(64 as isize) as *mut (), 0i32, 32u64) };
unsafe { memcpy((cmd_tbl as *const u8).wrapping_offset(64 as isize) as *mut (), ((*qc).cdb.as_mut_ptr() as *mut u8) as *const (), (*(*qc).dev).cdb_len as u64) };
}
if (unsafe { (*qc).flags }) & 2 != 0 {
n_elem = ahci_fill_sg(qc, cmd_tbl);
}
// Fill in command slot information.
opts = cmd_fis_len | n_elem << 16i32 | (((unsafe { (*(*(*qc).dev).link).pmp }) << 12i32) as u32);
if (unsafe { (*qc).tf.flags }) & 8 != 0 {
opts |= 64;
}
if is_atapi != 0 {
opts |= 32 | 128;
}
ahci_fill_cmd_slot(pp, unsafe { (*qc).hw_tag }, opts);
0
}

fn ahci_fbs_dec_intr(ap: *mut ata_port) {
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut fbs: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
let mut retries: i32 = 3;
loop {
if unlikely!(!unsafe { (*pp).fbs_enabled }) != 0 {
loop {
unsafe {
core::arch::asm!("703: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 703b - ., 3; .popsection\n# {0}", const 703 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ahci_fbs_dec_intr_0, sym __slate_asm_string_ahci_fbs_dec_intr_1, const 1744 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// time to wait for DEC is not specified by AHCI spec,
// add a retry loop for safety.
writel(fbs | 2, (unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ());
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
loop {
let __v5: bool = if fbs & 2 != 0 {
let __v6: i32 = retries;
retries = __v6.wrapping_sub(1);
__v6 != 0
} else {
false
};
if !__v5 {
break;
}
udelay(1u64);
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
}
if fbs & 2 != 0 {
while false {
}
unsafe { _dev_err((*(*ap).host).dev as *const device, c"failed to clear device error\n".as_ptr() as *const u8) };
}
}

fn ahci_error_intr(ap: *mut ata_port, mut irq_stat__0: u32) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let host_ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*ap).link.eh_info) };
let mut link: *mut ata_link = core::ptr::null_mut::<ata_link>();
let mut fbs_need_dec: bool = false;
let mut serror: u32 = 0;
// determine active link with error
if unsafe { (*pp).fbs_enabled } {
let port_mmio: *mut () = ahci_port_base(ap);
let fbs: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
let pmp: i32 = (fbs >> 16u32) as i32;
if fbs & 4 != 0 && pmp < unsafe { (*ap).nr_pmp_links } {
link = unsafe { (*ap).pmp_link }.wrapping_offset(pmp as isize);
fbs_need_dec = true;
}
} else {
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
if ata_link_active(link) != 0 {
break;
}
link = unsafe { ata_link_next(link, ap, 0u32) };
}
}
if link.is_null() {
link = unsafe { core::ptr::addr_of_mut!((*ap).link) };
}
let active_qc: *mut ata_queued_cmd = ata_qc_from_tag(ap, unsafe { (*link).active_tag });
let active_ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*link).eh_info) };
// record irq stat
unsafe { ata_ehi_clear_desc(host_ehi) };
unsafe { ata_ehi_push_desc(host_ehi, c"irq_stat 0x%08x".as_ptr() as *const u8, irq_stat__0) };
// AHCI needs SError cleared; otherwise, it might lock up
ahci_scr_read(unsafe { core::ptr::addr_of_mut!((*ap).link) }, 1u32, core::ptr::addr_of_mut!(serror));
ahci_scr_write(unsafe { core::ptr::addr_of_mut!((*ap).link) }, 1u32, serror);
unsafe {
(*host_ehi).serror = (*host_ehi).serror | serror;
}
// some controllers set IRQ_IF_ERR on device errors, ignore it
if (unsafe { (*hpriv).flags }) & 2 != 0 {
let __v5: u32 = irq_stat__0 & !134217728;
irq_stat__0 = __v5;
}
if irq_stat__0 & 1073741824 != 0 {
// If qc is active, charge it; otherwise, the active
// link.  There's no active qc on NCQ errors.  It will
// be determined by EH by reading log page 10h.
if !active_qc.is_null() {
unsafe {
(*active_qc).err_mask = (*active_qc).err_mask | 1;
}
} else {
unsafe {
(*active_ehi).err_mask = (*active_ehi).err_mask | 1;
}
}
if (unsafe { (*hpriv).flags }) & 4 != 0 {
let __v14: u32 = (unsafe { (*host_ehi).serror }) & 4294965247u32;
unsafe {
(*host_ehi).serror = __v14;
}
}
}
if irq_stat__0 & 16 != 0 {
let unk: *mut u32 = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(96 as isize) }) as *mut ()) as *mut u32;
unsafe {
(*active_ehi).err_mask = (*active_ehi).err_mask | 2;
}
unsafe {
(*active_ehi).action = (*active_ehi).action | 6;
}
unsafe { ata_ehi_push_desc(active_ehi, c"unknown FIS %08x %08x %08x %08x".as_ptr() as *const u8, *unk.wrapping_offset(0 as isize), *unk.wrapping_offset(1 as isize), *unk.wrapping_offset(2 as isize), *unk.wrapping_offset(3 as isize)) };
}
if sata_pmp_attached(ap) && irq_stat__0 & 8388608 != 0 {
unsafe {
(*active_ehi).err_mask = (*active_ehi).err_mask | 2;
}
unsafe {
(*active_ehi).action = (*active_ehi).action | 6;
}
unsafe { ata_ehi_push_desc(active_ehi, c"incorrect PMP".as_ptr() as *const u8) };
}
if irq_stat__0 & (536870912 | 268435456) != 0 {
unsafe {
(*host_ehi).err_mask = (*host_ehi).err_mask | 32;
}
unsafe {
(*host_ehi).action = (*host_ehi).action | 6;
}
unsafe { ata_ehi_push_desc(host_ehi, c"host bus error".as_ptr() as *const u8) };
}
if irq_stat__0 & 134217728 != 0 {
if fbs_need_dec {
unsafe {
(*active_ehi).err_mask = (*active_ehi).err_mask | 1;
}
} else {
unsafe {
(*host_ehi).err_mask = (*host_ehi).err_mask | 16;
}
unsafe {
(*host_ehi).action = (*host_ehi).action | 6;
}
}
unsafe { ata_ehi_push_desc(host_ehi, c"interface fatal error".as_ptr() as *const u8) };
}
if irq_stat__0 & (64 | 4194304) != 0 {
ata_ehi_hotplugged(host_ehi);
unsafe { ata_ehi_push_desc(host_ehi, c"%s".as_ptr() as *const u8, if irq_stat__0 & 64 != 0 { c"connection status changed".as_ptr() as *mut u8 } else { c"PHY RDY changed".as_ptr() as *mut u8 }) };
}
// okay, let's hand over to EH
if irq_stat__0 & 683671632 != 0 {
unsafe { ata_port_freeze(ap) };
} else {
if fbs_need_dec {
unsafe { ata_link_abort(link) };
ahci_fbs_dec_intr(ap);
} else {
unsafe { ata_port_abort(ap) };
}
}
}

fn ahci_qc_complete(ap: *mut ata_port, port_mmio: *mut ()) {
let ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*ap).link.eh_info) };
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut qc_active: u32 = 0;
let mut rc: i32 = 0;
// pp->active_link is not reliable once FBS is enabled, both
// PORT_SCR_ACT and PORT_CMD_ISSUE should be checked because
// NCQ and non-NCQ commands may be in flight at the same time.
if unsafe { (*pp).fbs_enabled } {
if (unsafe { (*ap).qc_active }) != 0 {
qc_active = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(52 as isize) }) as *mut ()) as *const ());
let __v0: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(56 as isize) }) as *mut ()) as *const ());
qc_active |= __v0;
}
} else {
// pp->active_link is valid iff any command is in flight
if (unsafe { (*ap).qc_active }) != 0 && (unsafe { (*(*pp).active_link).sactive }) != 0 {
qc_active = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(52 as isize) }) as *mut ()) as *const ());
} else {
qc_active = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(56 as isize) }) as *mut ()) as *const ());
}
}
rc = unsafe { ata_qc_complete_multiple(ap, qc_active as u64) };
if unlikely!(rc < 0 && (unsafe { (*ap).pflags }) & 256 == 0) != 0 {
unsafe {
(*ehi).err_mask = (*ehi).err_mask | 2;
}
unsafe {
(*ehi).action = (*ehi).action | 6;
}
unsafe { ata_port_freeze(ap) };
}
}

fn ahci_handle_port_interrupt(ap: *mut ata_port, port_mmio: *mut (), mut status: u32) {
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
// ignore BAD_PMP while resetting
if unlikely__3!((unsafe { (*ap).pflags }) & 256) != 0 {
let __v1: u32 = status & !8388608;
status = __v1;
}
if unsafe { sata_lpm_ignore_phy_events(core::ptr::addr_of_mut!((*ap).link)) } {
let __v3: u32 = status & !4194304;
status = __v3;
ahci_scr_write(unsafe { core::ptr::addr_of_mut!((*ap).link) }, 1u32, 65536u32);
}
if unlikely__3!(status & 2025848912) != 0 {
// Before getting the error notification, we may have
// received SDB FISes notifying successful completions.
// Handle these first and then handle the error.
ahci_qc_complete(ap, port_mmio);
ahci_error_intr(ap, status);
return;
}
if status & 8 != 0 {
// If SNotification is available, leave notification
// handling to sata_async_notification().  If not,
// emulate it by snooping SDB FIS RX area.
//
// Snooping FIS RX area is probably cheaper than
// poking SNotification but some constrollers which
// implement SNotification, ICH9 for example, don't
// store AN SDB FIS into receive area.
if (unsafe { (*hpriv).cap }) & 536870912 != 0 {
unsafe { sata_async_notification(ap) };
} else {
// If the 'N' bit in word 0 of the FIS is set,
// we just received asynchronous notification.
// Tell libata about it.
//
// Lack of SNotification should not appear in
// ahci 1.2, so the workaround is unnecessary
// when FBS is enabled.
if unsafe { (*pp).fbs_enabled } {
let mut __v4: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("704: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 704b - ., 3; .popsection\n# {0}", const 704 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ahci_handle_port_interrupt_5, sym __slate_asm_string_ahci_handle_port_interrupt_6, const 1952 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("705: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 705b - ., 4; .popsection\n# {0}", const 705 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
} else {
let f: *const u32 = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(88 as isize) }) as *mut ()) as *const u32;
let f0: u32 = unsafe { *f.wrapping_offset(0 as isize) };
if f0 & 32768u32 != 0 {
unsafe { sata_async_notification(ap) };
}
}
}
}
// Handle completed commands
ahci_qc_complete(ap, port_mmio);
}

fn ahci_port_intr(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let status: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
writel(status, (unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ());
ahci_handle_port_interrupt(ap, port_mmio, status);
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_multi_irqs_intr_hard")]
extern "C" fn ahci_multi_irqs_intr_hard(irq: i32, dev_instance: *mut ()) -> u32 {
let ap: *mut ata_port = dev_instance as *mut ata_port;
let port_mmio: *mut () = ahci_port_base(ap);
let status: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
writel(status, (unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ());
spin_lock(unsafe { (*ap).lock });
ahci_handle_port_interrupt(ap, port_mmio, status);
spin_unlock(unsafe { (*ap).lock });
1
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_handle_port_intr")]
extern "C" fn ahci_handle_port_intr(host: *mut ata_host, irq_masked: u32) -> u32 {
let mut i: u32 = 0;
let mut handled: u32 = 0;
while i < unsafe { (*host).n_ports } {
if irq_masked & ((1i32 << i) as u32) != 0 {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) };
if !ap.is_null() {
ahci_port_intr(ap);
} else {
if (unsafe { ata_ratelimit() }) != 0 {
while false {
}
unsafe { _dev_warn((*host).dev as *const device, c"interrupt on disabled port %u\n".as_ptr() as *const u8, i) };
}
}
handled = 1;
}
i = i.wrapping_add(1);
}
handled
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_handle_port_intr_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host, u32) -> u32>, *mut ()>(Some(ahci_handle_port_intr)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_single_level_irq_intr")]
extern "C" fn ahci_single_level_irq_intr(irq: i32, dev_instance: *mut ()) -> u32 {
let host: *mut ata_host = dev_instance as *mut ata_host;
let mut rc: u32 = 0;
let mut irq_stat__0: u32 = 0;
let mut irq_masked: u32 = 0;
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
// sigh.  0xffffffff is a valid return from h/w
irq_stat__0 = readl(((unsafe { (mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ()) as *const ());
if irq_stat__0 == 0 {
return 0;
}
irq_masked = irq_stat__0 & unsafe { (*hpriv).port_map };
spin_lock(unsafe { core::ptr::addr_of_mut!((*host).lock) });
rc = ahci_handle_port_intr(host, irq_masked);
// HOST_IRQ_STAT behaves as level triggered latch meaning that
// it should be cleared after all the port events are cleared;
// otherwise, it will raise a spurious interrupt after each
// valid one.  Please read section 10.6.2 of ahci 1.1 for more
// information.
//
// Also, use the unmasked value to clear interrupt as spurious
// pending event on a dummy port might cause screaming IRQ.
writel(irq_stat__0, (unsafe { (mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
spin_unlock(unsafe { core::ptr::addr_of_mut!((*host).lock) });
IRQ_RETVAL!(rc != 0) as u32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_qc_issue")]
extern "C" fn ahci_qc_issue(qc: *mut ata_queued_cmd) -> u32 {
let ap: *mut ata_port = unsafe { (*qc).ap };
let port_mmio: *mut () = ahci_port_base(ap);
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
// Keep track of the currently active link.  It will be used
// in completion path to determine whether NCQ phase is in
// progress.
unsafe {
(*pp).active_link = (*(*qc).dev).link;
}
if ata_is_ncq(unsafe { (*qc).tf.protocol }) {
writel((1i32 << unsafe { (*qc).hw_tag }) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(52 as isize) }) as *mut ());
}
if (unsafe { (*pp).fbs_enabled }) && (unsafe { (*pp).fbs_last_dev }) != unsafe { (*(*(*qc).dev).link).pmp } {
let mut fbs: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
let __v1: u32 = fbs & !(3840 | 2);
fbs = __v1;
fbs |= ((unsafe { (*(*(*qc).dev).link).pmp }) << 8u32) as u32;
writel(fbs, (unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ());
unsafe {
(*pp).fbs_last_dev = (*(*(*qc).dev).link).pmp;
}
}
writel((1i32 << unsafe { (*qc).hw_tag }) as u32, (unsafe { (port_mmio as *const u8).wrapping_offset(56 as isize) }) as *mut ());
ahci_sw_activity(unsafe { (*(*qc).dev).link });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_qc_issue_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, *mut ()>(Some(ahci_qc_issue)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_qc_fill_rtf")]
extern "C" fn ahci_qc_fill_rtf(qc: *mut ata_queued_cmd) {
let pp: *mut ahci_port_priv = (unsafe { (*(*qc).ap).private_data }) as *mut ahci_port_priv;
let mut rx_fis: *mut u8 = (unsafe { (*pp).rx_fis }) as *mut u8;
if unsafe { (*pp).fbs_enabled } {
let __v0: *mut u8 = rx_fis;
rx_fis = __v0.wrapping_offset(((unsafe { (*(*(*qc).dev).link).pmp }) as u32).wrapping_mul(256) as isize);
}
// After a successful execution of an ATA PIO data-in command,
// the device doesn't send D2H Reg FIS to update the TF and
// the host should take TF and E_Status from the preceding PIO
// Setup FIS.
if ((unsafe { (*qc).tf.protocol }) as i32) == 1 && (unsafe { (*qc).dma_dir }) == 2 && (unsafe { (*qc).flags }) & 65536 == 0 {
unsafe { ata_tf_from_fis(rx_fis.wrapping_offset(32 as isize) as *const u8, core::ptr::addr_of_mut!((*qc).result_tf)) };
unsafe {
(*qc).result_tf.__slate_anon_14.status = *rx_fis.wrapping_offset(32 as isize).wrapping_offset(15 as isize);
}
return;
}
// For NCQ commands, we never get a D2H FIS, so reading the D2H Register
// FIS area of the Received FIS Structure (which contains a copy of the
// last D2H FIS received) will contain an outdated status code.
// For NCQ commands, we instead get a SDB FIS, so read the SDB FIS area
// instead. However, the SDB FIS does not contain the LBA, so we can't
// use the ata_tf_from_fis() helper.
if ata_is_ncq(unsafe { (*qc).tf.protocol }) {
let fis: *const u8 = rx_fis.wrapping_offset(88 as isize) as *const u8;
// Successful NCQ commands have been filled already.
// A failed NCQ command will read the status here.
// (Note that a failed NCQ command will get a more specific
// error when reading the NCQ Command Error log.)
unsafe {
(*qc).result_tf.__slate_anon_14.status = *fis.wrapping_offset(2 as isize);
}
unsafe {
(*qc).result_tf.__slate_anon_8.error = *fis.wrapping_offset(3 as isize);
}
return;
}
unsafe { ata_tf_from_fis(rx_fis.wrapping_offset(64 as isize) as *const u8, core::ptr::addr_of_mut!((*qc).result_tf)) };
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_qc_ncq_fill_rtf")]
extern "C" fn ahci_qc_ncq_fill_rtf(ap: *mut ata_port, mut done_mask: u64) {
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut fis: *const u8 = unsafe { core::mem::zeroed() };
// No outstanding commands.
if (unsafe { (*ap).qc_active }) == 0 {
return;
}
// FBS not enabled, so read status and error once, since they are shared
// for all QCs.
if !unsafe { (*pp).fbs_enabled } {
let mut status: u8 = 0;
let mut error: u8 = 0;
// No outstanding NCQ commands.
if (unsafe { (*(*pp).active_link).sactive }) == 0 {
return;
}
fis = ((unsafe { ((*pp).rx_fis as *const u8).wrapping_offset(88 as isize) }) as *mut ()) as *const u8;
status = unsafe { *fis.wrapping_offset(2 as isize) };
error = unsafe { *fis.wrapping_offset(3 as isize) };
while done_mask != 0 {
let tag: u32 = __ffs64(done_mask);
let qc: *mut ata_queued_cmd = ata_qc_from_tag(ap, tag);
let __v0: bool = !qc.is_null() && ata_is_ncq(unsafe { (*qc).tf.protocol });
if __v0 {
unsafe {
(*qc).result_tf.__slate_anon_14.status = status;
}
unsafe {
(*qc).result_tf.__slate_anon_8.error = error;
}
unsafe {
(*qc).result_tf.flags = (*qc).tf.flags;
}
unsafe {
(*qc).flags = (*qc).flags | 4;
}
}
let __v5: u64 = done_mask & !(1u64 << tag);
done_mask = __v5;
}
return;
}
// FBS enabled, so read the status and error for each QC, since the QCs
// can belong to different PMP links. (Each PMP link has its own FIS
// Receive Area.)
while done_mask != 0 {
let tag: u32 = __ffs64(done_mask);
let qc: *mut ata_queued_cmd = ata_qc_from_tag(ap, tag);
let __v7: bool = !qc.is_null() && ata_is_ncq(unsafe { (*qc).tf.protocol });
if __v7 {
fis = (unsafe { (*pp).rx_fis }) as *const u8;
let __v8: *const u8 = fis;
fis = __v8.wrapping_offset(((unsafe { (*(*(*qc).dev).link).pmp }) as u32).wrapping_mul(256) as isize);
let __v10: *const u8 = fis;
fis = __v10.wrapping_offset(88 as isize);
unsafe {
(*qc).result_tf.__slate_anon_14.status = *fis.wrapping_offset(2 as isize);
}
unsafe {
(*qc).result_tf.__slate_anon_8.error = *fis.wrapping_offset(3 as isize);
}
unsafe {
(*qc).result_tf.flags = (*qc).tf.flags;
}
unsafe {
(*qc).flags = (*qc).flags | 4;
}
}
let __v16: u64 = done_mask & !(1u64 << tag);
done_mask = __v16;
}
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_freeze")]
extern "C" fn ahci_freeze(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
// turn IRQ off
writel(0u32, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_thaw")]
extern "C" fn ahci_thaw(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let port_mmio: *mut () = ahci_port_base(ap);
let mut tmp: u32 = 0;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
// clear IRQ
tmp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const ());
writel(tmp, (unsafe { (port_mmio as *const u8).wrapping_offset(16 as isize) }) as *mut ());
writel((1i32 << unsafe { (*ap).port_no }) as u32, (unsafe { (mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
// turn IRQ back on
writel(unsafe { (*pp).intr_mask }, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_error_handler")]
extern "C" fn ahci_error_handler(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
if !ata_port_is_frozen(ap as *const ata_port) {
// restart engine
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
}
unsafe { sata_pmp_error_handler(ap) };
if ata_dev_enabled((unsafe { (*ap).link.device.as_mut_ptr() as *mut ata_device }) as *const ata_device) == 0 {
unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_error_handler_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ahci_error_handler)) };

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_post_internal_cmd")]
extern "C" fn ahci_post_internal_cmd(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
// make DMA engine forget about the failed command
if (unsafe { (*qc).flags }) & 65536 != 0 {
ahci_kick_engine(ap);
}
}

fn ahci_set_aggressive_devslp(ap: *mut ata_port, sleep: bool) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let dev: *mut ata_device = unsafe { (*ap).link.device.as_mut_ptr() as *mut ata_device };
let mut devslp: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(68 as isize) }) as *mut ()) as *const ());
let mut dm: u32 = 0;
let mut dito: u32 = 0;
let mut mdat: u32 = 0;
let mut deto: u32 = 0;
let mut dito_conf: u32 = 0;
let mut rc: i32 = 0;
let mut err_mask: u32 = 0;
if devslp & 2 == 0 {
while false {
}
unsafe { _dev_info((*(*ap).host).dev as *const device, c"port does not support device sleep\n".as_ptr() as *const u8) };
return;
}
// disable device sleep
if !sleep {
if devslp & 1 != 0 {
writel(devslp & !1, (unsafe { (port_mmio as *const u8).wrapping_offset(68 as isize) }) as *mut ());
err_mask = unsafe { ata_dev_set_feature(dev, 144u8, 9u8) };
if err_mask != 0 && err_mask != 1 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014ata%u.%02u: failed to disable DEVSLP\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
}
return;
}
dm = (devslp & 503316480) >> 25u32;
dito = ((unsafe { devslp_idle_timeout }) as u32) / dm.wrapping_add(1);
if dito > 1023 {
dito = 1023;
}
dito_conf = devslp >> 15u32 & 1023;
// device sleep was already enabled and same dito
if devslp & 1 != 0 && dito_conf == dito {
return;
}
// set DITO, MDAT, DETO and enable DevSlp, need to stop engine first
rc = unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
if rc != 0 {
return;
}
// Use the nominal value 10 ms if the read MDAT is zero,
// the nominal value of DETO is 20 ms.
if ((unsafe { *((*dev).devslp_timing.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) }) as i32) & 128 != 0 {
mdat = (((unsafe { *((*dev).devslp_timing.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 31) as u32;
if mdat == 0 {
mdat = 10;
}
deto = (unsafe { *((*dev).devslp_timing.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u32;
if deto == 0 {
deto = 20;
}
} else {
mdat = 10;
deto = 20;
}
// Make dito, mdat, deto bits to 0s
let __v3: u32 = devslp;
devslp = ((__v3 as u64) & !GENMASK_ULL__9!(24i32, 2i32)) as u32;
devslp |= dito << 15u32 | mdat << 10u32 | deto << 2u32 | 1;
writel(devslp, (unsafe { (port_mmio as *const u8).wrapping_offset(68 as isize) }) as *mut ());
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
// enable device sleep feature for the drive
err_mask = unsafe { ata_dev_set_feature(dev, 16u8, 9u8) };
if err_mask != 0 && err_mask != 1 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x014ata%u.%02u: failed to enable DEVSLP\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
}

fn ahci_enable_fbs(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut fbs: u32 = 0;
let mut rc: i32 = 0;
if !unsafe { (*pp).fbs_supported } {
return;
}
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
if fbs & 1 != 0 {
unsafe {
(*pp).fbs_enabled = true;
}
unsafe {
(*pp).fbs_last_dev = 1i32.wrapping_neg();
}
// initialization
return;
}
rc = unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
if rc != 0 {
return;
}
writel(fbs | 1, (unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ());
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
if fbs & 1 != 0 {
while false {
}
unsafe { _dev_info((*(*ap).host).dev as *const device, c"FBS is enabled\n".as_ptr() as *const u8) };
unsafe {
(*pp).fbs_enabled = true;
}
unsafe {
(*pp).fbs_last_dev = 1i32.wrapping_neg();
}
// initialization
} else {
while false {
}
unsafe { _dev_err((*(*ap).host).dev as *const device, c"Failed to enable FBS\n".as_ptr() as *const u8) };
}
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
}

fn ahci_disable_fbs(ap: *mut ata_port) {
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let port_mmio: *mut () = ahci_port_base(ap);
let mut fbs: u32 = 0;
let mut rc: i32 = 0;
if !unsafe { (*pp).fbs_supported } {
return;
}
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
if fbs & 1 == 0 {
unsafe {
(*pp).fbs_enabled = false;
}
return;
}
rc = unsafe { unsafe { (*hpriv).stop_engine }.unwrap()(ap) };
if rc != 0 {
return;
}
writel(fbs & !1, (unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ());
fbs = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(64 as isize) }) as *mut ()) as *const ());
if fbs & 1 != 0 {
while false {
}
unsafe { _dev_err((*(*ap).host).dev as *const device, c"Failed to disable FBS\n".as_ptr() as *const u8) };
} else {
while false {
}
unsafe { _dev_info((*(*ap).host).dev as *const device, c"FBS is disabled\n".as_ptr() as *const u8) };
unsafe {
(*pp).fbs_enabled = false;
}
}
unsafe { unsafe { (*hpriv).start_engine }.unwrap()(ap) };
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_pmp_attach")]
extern "C" fn ahci_pmp_attach(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut cmd: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
cmd |= 131072;
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
ahci_enable_fbs(ap);
unsafe {
(*pp).intr_mask = (*pp).intr_mask | 8388608;
}
// We must not change the port interrupt mask register if the
// port is marked frozen, the value in pp->intr_mask will be
// restored later when the port is thawed.
//
// Note that during initialization, the port is marked as
// frozen since the irq handler is not yet registered.
if !ata_port_is_frozen(ap as *const ata_port) {
writel(unsafe { (*pp).intr_mask }, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
}
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_pmp_detach")]
extern "C" fn ahci_pmp_detach(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let pp: *mut ahci_port_priv = (unsafe { (*ap).private_data }) as *mut ahci_port_priv;
let mut cmd: u32 = 0;
ahci_disable_fbs(ap);
cmd = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
let __v1: u32 = cmd & !131072;
cmd = __v1;
writel(cmd, (unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ());
let __v4: u32 = (unsafe { (*pp).intr_mask }) & !8388608;
unsafe {
(*pp).intr_mask = __v4;
}
// see comment above in ahci_pmp_attach()
if !ata_port_is_frozen(ap as *const ata_port) {
writel(unsafe { (*pp).intr_mask }, (unsafe { (port_mmio as *const u8).wrapping_offset(20 as isize) }) as *mut ());
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_port_resume")]
extern "C" fn ahci_port_resume(ap: *mut ata_port) -> i32 {
ahci_rpm_get_port(ap);
ahci_power_up(ap);
ahci_start_port(ap);
if sata_pmp_attached(ap) {
ahci_pmp_attach(ap);
} else {
ahci_pmp_detach(ap);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_port_resume_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> i32>, *mut ()>(Some(ahci_port_resume)) };

fn ahci_handle_s2idle(ap: *mut ata_port) {
let port_mmio: *mut () = ahci_port_base(ap);
let mut devslp: u32 = 0;
if pm_suspend_via_firmware() {
return;
}
devslp = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(68 as isize) }) as *mut ()) as *const ());
if devslp & 1 != 0 {
unsafe { ata_msleep(ap, devslp_idle_timeout as u32) };
}
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_port_suspend")]
extern "C" fn ahci_port_suspend(ap: *mut ata_port, mesg: pm_message) -> i32 {
let mut emsg: *const u8 = core::ptr::null::<u8>();
let rc: i32 = ahci_deinit_port(ap, unsafe { &mut (*core::ptr::addr_of_mut!(emsg)) });
if rc == 0 {
ahci_power_down(ap);
} else {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013ata%u: %s (%d)\n".as_ptr() as *const u8, (*ap).print_id, emsg, rc) };
unsafe { ata_port_freeze(ap) };
}
if unsafe { acpi_storage_d3((*(*ap).host).dev) } {
ahci_handle_s2idle(ap);
}
ahci_rpm_put_port(ap);
rc
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_port_start")]
extern "C" fn ahci_port_start(ap: *mut ata_port) -> i32 {
let mut __slate_compound_11: spinlock = unsafe { core::mem::zeroed() };
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let dev: *mut device = unsafe { (*(*ap).host).dev };
let mut pp: *mut ahci_port_priv = unsafe { core::mem::zeroed() };
let mut mem: *mut () = unsafe { core::mem::zeroed() };
let mut mem_dma: u64 = 0;
let mut dma_sz: u64 = 0;
let mut rx_fis_sz: u64 = 0;
pp = devm_kzalloc(dev, core::mem::size_of::<ahci_port_priv>() as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut ahci_port_priv;
if pp.is_null() {
return 12i32.wrapping_neg();
}
if (unsafe { (*(*ap).host).n_ports }) > 1 {
unsafe {
(*pp).irq_desc = devm_kzalloc(dev, 8u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut u8;
}
if unsafe { (*pp).irq_desc }.is_null() {
unsafe { devm_kfree(dev, pp as *const ()) };
return 12i32.wrapping_neg();
}
unsafe { snprintf((*pp).irq_desc, 8u64, c"%s%d".as_ptr() as *const u8, dev_driver_string(dev as *const device), (*ap).port_no) };
}
// check FBS capability
let __v0: bool = (unsafe { (*hpriv).cap }) & 65536 != 0 && sata_pmp_supported(ap);
if __v0 {
let port_mmio: *mut () = ahci_port_base(ap);
let cmd: u32 = readl(((unsafe { (port_mmio as *const u8).wrapping_offset(24 as isize) }) as *mut ()) as *const ());
if cmd & 4194304 != 0 {
unsafe {
(*pp).fbs_supported = true;
}
} else {
if (unsafe { (*hpriv).flags }) & 16384 != 0 {
while false {
}
unsafe { _dev_info(dev as *const device, c"port %d can do FBS, forcing FBSCP\n".as_ptr() as *const u8, (*ap).port_no) };
unsafe {
(*pp).fbs_supported = true;
}
} else {
while false {
}
unsafe { _dev_warn(dev as *const device, c"port %d is not capable of FBS\n".as_ptr() as *const u8, (*ap).port_no) };
}
}
}
if unsafe { (*pp).fbs_supported } {
dma_sz = 95232;
rx_fis_sz = 4096u64;
} else {
dma_sz = 91392;
rx_fis_sz = 256;
}
mem = dmam_alloc_coherent(dev, dma_sz, core::ptr::addr_of_mut!(mem_dma), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if mem.is_null() {
return 12i32.wrapping_neg();
}
// First item in chunk of DMA memory: 32-slot command table,
// 32 bytes each in size
unsafe {
(*pp).cmd_slot = mem as *mut ahci_cmd_hdr;
}
unsafe {
(*pp).cmd_slot_dma = mem_dma;
}
mem = (unsafe { (mem as *const u8).wrapping_offset(1024 as isize) }) as *mut ();
mem_dma = mem_dma.wrapping_add(1024);
// Second item: Received-FIS area
unsafe {
(*pp).rx_fis = mem;
}
unsafe {
(*pp).rx_fis_dma = mem_dma;
}
mem = (unsafe { (mem as *const u8).wrapping_offset(rx_fis_sz as isize) }) as *mut ();
mem_dma = mem_dma.wrapping_add(rx_fis_sz);
// Third item: data area for storing a single command
// and its scatter-gather table
unsafe {
(*pp).cmd_tbl = mem;
}
unsafe {
(*pp).cmd_tbl_dma = mem_dma;
}
// Save off initial list of interrupts to be enabled.
// This could be changed later
unsafe {
(*pp).intr_mask = 2025848959;
}
// Switch to per-port locking in case each port has its own MSI vector.
if (unsafe { (*hpriv).flags }) & 1048576 != 0 {
'__slate_break_12: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*pp).lock) });
unsafe {
*core::ptr::addr_of_mut!((*pp).lock) = *{
__slate_compound_11 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_11)
};
}
if true {
break;
}
}
unsafe {
(*ap).lock = core::ptr::addr_of_mut!((*pp).lock);
}
}
unsafe {
(*ap).private_data = pp as *mut ();
}
// engage engines, captain
ahci_port_resume(ap)
}

#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_port_stop")]
extern "C" fn ahci_port_stop(ap: *mut ata_port) {
let mut emsg: *const u8 = core::ptr::null::<u8>();
let hpriv: *mut ahci_host_priv = (unsafe { (*(*ap).host).private_data }) as *mut ahci_host_priv;
let host_mmio: *mut () = unsafe { (*hpriv).mmio };
let mut rc: i32 = 0;
// de-initialize port
rc = ahci_deinit_port(ap, unsafe { &mut (*core::ptr::addr_of_mut!(emsg)) });
if rc != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014ata%u: %s (%d)\n".as_ptr() as *const u8, (*ap).print_id, emsg, rc) };
}
// Clear GHC.IS to prevent stuck INTx after disabling MSI and
// re-enabling INTx.
writel((1i32 << unsafe { (*ap).port_no }) as u32, (unsafe { (host_mmio as *const u8).wrapping_offset(8 as isize) }) as *mut ());
ahci_rpm_put_port(ap);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_print_info")]
extern "C" fn ahci_print_info(host: *mut ata_host, scc_s: *const u8) {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let mut vers: u32 = 0;
let mut cap: u32 = 0;
let mut cap2: u32 = 0;
let mut r#impl: u32 = 0;
let mut speed: u32 = 0;
let mut speed_s: *const u8 = unsafe { core::mem::zeroed() };
vers = unsafe { (*hpriv).version };
cap = unsafe { (*hpriv).cap };
cap2 = unsafe { (*hpriv).cap2 };
r#impl = unsafe { (*hpriv).port_map };
speed = cap >> 20i32 & 15;
if speed == 1 {
speed_s = c"1.5".as_ptr() as *const u8;
} else {
if speed == 2 {
speed_s = c"3".as_ptr() as *const u8;
} else {
if speed == 3 {
speed_s = c"6".as_ptr() as *const u8;
} else {
speed_s = c"?".as_ptr() as *const u8;
}
}
}
while false {
}
unsafe { _dev_info((*host).dev as *const device, c"AHCI vers %02x%02x.%02x%02x, %u command slots, %s Gbps, %s mode\n".as_ptr() as *const u8, vers >> 24i32 & 255, vers >> 16i32 & 255, vers >> 8i32 & 255, vers & 255, (cap >> 8i32 & 31).wrapping_add(1), speed_s, scc_s) };
while false {
}
unsafe { _dev_info((*host).dev as *const device, c"%u/%u ports implemented (port mask 0x%x)\n".as_ptr() as *const u8, hweight32__2!(r#impl), ahci_nr_ports(cap), r#impl) };
while false {
}
unsafe { _dev_info((*host).dev as *const device, c"flags: %s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n".as_ptr() as *const u8, if cap & 2147483648 != 0 { c"64bit ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 1073741824 != 0 { c"ncq ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 536870912 != 0 { c"sntf ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 268435456 != 0 { c"ilck ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 134217728 != 0 { c"stag ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 67108864 != 0 { c"pm ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 33554432 != 0 { c"led ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 16777216 != 0 { c"clo ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 262144 != 0 { c"only ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 131072 != 0 { c"pmp ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 65536 != 0 { c"fbs ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 32768 != 0 { c"pio ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 16384 != 0 { c"slum ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 8192 != 0 { c"part ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 128 != 0 { c"ccc ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 64 != 0 { c"ems ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap & 32 != 0 { c"sxs ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 32 != 0 { c"deso ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 16 != 0 { c"sadm ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 8 != 0 { c"sds ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 4 != 0 { c"apst ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 2 != 0 { c"nvmp ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if cap2 & 1 != 0 { c"boh ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_print_info_710: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host, *const u8)>, *mut ()>(Some(ahci_print_info)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_set_em_messages")]
extern "C" fn ahci_set_em_messages(hpriv: *mut ahci_host_priv, pi: *mut ata_port_info) {
let mut messages: u8 = 0;
let mmio: *mut () = unsafe { (*hpriv).mmio };
let em_loc: u32 = readl(((unsafe { (mmio as *const u8).wrapping_offset(28 as isize) }) as *mut ()) as *const ());
let em_ctl: u32 = readl(((unsafe { (mmio as *const u8).wrapping_offset(32 as isize) }) as *mut ()) as *const ());
if !unsafe { ahci_em_messages } || (unsafe { (*hpriv).cap }) & 64 == 0 {
return;
}
messages = ((em_ctl & 983040) >> 16i32) as u8;
if messages != 0 {
// store em_loc
unsafe {
(*hpriv).em_loc = (em_loc >> 16i32).wrapping_mul(4);
}
unsafe {
(*hpriv).em_buf_sz = (em_loc & 255).wrapping_mul(4);
}
unsafe {
(*hpriv).em_msg_type = messages as u32;
}
unsafe {
(*pi).flags = (*pi).flags | 2097152;
}
if em_ctl & 67108864 == 0 {
unsafe {
(*pi).flags = (*pi).flags | 4194304;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_set_em_messages_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ahci_host_priv, *mut ata_port_info)>, *mut ()>(Some(ahci_set_em_messages)) };

fn ahci_host_activate_multi_irqs(host: *mut ata_host, sht: *const scsi_host_template) -> i32 {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
// Requests IRQs according to AHCI-1.1 when multiple MSIs were
// allocated. That is one MSI per port, starting from @irq.
let mut i: i32 = 0;
let mut rc: i32 = unsafe { ata_host_start(host) };
if rc != 0 {
return rc;
}
while (i as u32) < unsafe { (*host).n_ports } {
let pp: *mut ahci_port_priv = (unsafe { (*(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize))).private_data }) as *mut ahci_port_priv;
let irq: i32 = unsafe { unsafe { (*hpriv).get_irq_vector }.unwrap()(host, i) };
// Do not receive interrupts sent by dummy ports
if pp.is_null() {
unsafe { disable_irq(irq as u32) };
} else {
rc = devm_request_irq(unsafe { (*host).dev }, irq as u32, Some(ahci_multi_irqs_intr_hard), 0u64, (unsafe { (*pp).irq_desc }) as *const u8, (unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) }) as *mut ());
if rc != 0 {
return rc;
}
ata_port_desc_misc(unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) }, irq);
}
i = i.wrapping_add(1);
}
unsafe { ata_host_register(host, sht) }
}

/// ahci_host_activate - start AHCI host, request IRQs and register it
/// @host: target ATA host
/// @sht: scsi_host_template to use when registering the host
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libahci.ahci_host_activate")]
extern "C" fn ahci_host_activate(host: *mut ata_host, sht: *const scsi_host_template) -> i32 {
let hpriv: *mut ahci_host_priv = (unsafe { (*host).private_data }) as *mut ahci_host_priv;
let irq: i32 = (unsafe { (*hpriv).irq }) as i32;
let rc: i32 = if (unsafe { (*hpriv).flags }) & 1048576 != 0 {
if (unsafe { (*hpriv).irq_handler }) != None && (unsafe { (*hpriv).irq_handler }) != Some(ahci_single_level_irq_intr) {
while false {
}
unsafe { _dev_warn((*host).dev as *const device, c"both AHCI_HFLAG_MULTI_MSI flag set and custom irq handler implemented\n".as_ptr() as *const u8) };
}
if (unsafe { (*hpriv).get_irq_vector }) == None {
while false {
}
unsafe { _dev_err((*host).dev as *const device, c"AHCI_HFLAG_MULTI_MSI requires ->get_irq_vector!\n".as_ptr() as *const u8) };
return 5i32.wrapping_neg();
}
ahci_host_activate_multi_irqs(host, sht)
} else {
unsafe { ata_host_activate(host, irq, (*hpriv).irq_handler, 128u64, sht) }
};
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ahci_host_activate_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host, *const scsi_host_template) -> i32>, *mut ()>(Some(ahci_host_activate)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_713: [u8; 27] = [108, 105, 98, 97, 104, 99, 105, 46, 97, 117, 116, 104, 111, 114, 61, 74, 101, 102, 102, 32, 71, 97, 114, 122, 105, 107, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_714: [u8; 56] = [108, 105, 98, 97, 104, 99, 105, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 67, 111, 109, 109, 111, 110, 32, 65, 72, 67, 73, 32, 83, 65, 84, 65, 32, 108, 111, 119, 45, 108, 101, 118, 101, 108, 32, 114, 111, 117, 116, 105, 110, 101, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_715: [u8; 33] = [108, 105, 98, 97, 104, 99, 105, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 97, 116, 97, 47, 108, 105, 98, 97, 104, 99, 105, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_716: [u8; 20] = [108, 105, 98, 97, 104, 99, 105, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string_ahci_enable_ahci_6: [u8; 1] = *b"\0";

static __slate_asm_string_ahci_enable_ahci_7: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/ata/libahci.c\0";

static __slate_asm_string_ahci_fbs_dec_intr_0: [u8; 1] = *b"\0";

static __slate_asm_string_ahci_fbs_dec_intr_1: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/ata/libahci.c\0";

static __slate_asm_string_ahci_handle_port_interrupt_5: [u8; 1] = *b"\0";

static __slate_asm_string_ahci_handle_port_interrupt_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/drivers/ata/libahci.c\0";
