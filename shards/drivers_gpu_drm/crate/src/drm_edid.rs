
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
//! Copyright (c) 2006 Luc Verhaegen (quirks list)
//! Copyright (c) 2007-2008 Intel Corporation
//!   Jesse Barnes <jesse.barnes@intel.com>
//! Copyright 2010 Red Hat, Inc.
//!
//! DDC probing routines (drm_ddc_read & drm_do_probe_ddc_edid) originally from
//! FB layer.
//!   Copyright (C) 2006 Dennis Munsie <dmunsie@cecropia.com>
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sub license,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice (including the
//! next paragraph) shall be included in all copies or substantial portions
//! of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
//! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//! DEALINGS IN THE SOFTWARE.
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! This counts to 15. Any more, it will return 16th argument.
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
//! # chars in a path name including nul
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! cec_msg tx/rx_status field
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Internal kernel timer frequency
//! issue num suppressed message on exit
//! New user namespace
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
//! Sign Flag
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
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! @a is a power of 2 value
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
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
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
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
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! See the comment in asm-generic/io.h about ioremap_np().
//! DWORD 3
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! 	Enough for the VT console needs, see its max_font_width/height
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//! ELD Header Block
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Legacy last prefixes
//! REX2 M0
//! Kprobe status flags
//! Video mode flags
//!
//! bit compatible with the xrandr RR_ definitions (bits 0-13)
//!
//! ABI warning: Existing userspace really expects
//! the mode flags to match the xrandr definitions. Any
//! changes that don't match the xrandr definitions will
//! likely need a new client cap or some other mechanism
//! to avoid breaking existing userspace. This includes
//! allocating new flags in the previously unused bits!
//! DisplayID Structure versions
//! These are specified by iBCS2
//! Epoll event masks
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
// The maximum number of logical addresses one device can be assigned to.
// The CEC 2.0 spec allows for only 2 logical addresses at the moment. The
// Analog Devices CEC hardware supports 3. So let's go wild and go for 4.
// as initiator address
// CDC-Only device: supports only CDC messages
// Primary Device Type Operand (prim_devtype)
// SCHED_ISO: reserved but not implemented yet
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
// Overflow Flag
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
// normal but significant condition
// debug-level messages
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
// Same as above but for u64 dividends. divisor must be a 32-bit
// number.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// for_each_cpu_wrap - iterate over every cpu in a mask, starting at a specified location
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
// @start: the start location
//
// The implementation does not assume any bit in @mask is set (including @start).
//
// After the loop, cpu is >= nr_cpu_ids.
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
// PT_TRACE_* event enable flags
// 0x66
//
// LPFX1
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// XOP Byte2
// XOP Byte3
// breakpoint has already gone
// probe is temporarily disabled
// probe is really optimized.
// NOTE:
// this flag is only for optimized_kprobe.
// probe is using ftrace
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
macro_rules! ARRAY_SIZE__4 {
() => {
((core::mem::size_of::<[edid_quirk; 62]>() as u64) / (core::mem::size_of::<edid_quirk>() as u64)).wrapping_add(__must_be_array__63!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[drm_display_mode; 27]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__60!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[drm_display_mode; 88]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__42!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__32 {
() => {
((core::mem::size_of::<[minimode; 7]>() as u64) / (core::mem::size_of::<minimode>() as u64)).wrapping_add(__must_be_array__8!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__38 {
() => {
((core::mem::size_of::<[minimode; 7]>() as u64) / (core::mem::size_of::<minimode>() as u64)).wrapping_add(__must_be_array__77!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__41 {
() => {
((core::mem::size_of::<[drm_display_mode; 5]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__50!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__45 {
() => {
((core::mem::size_of::<[drm_display_mode; 5]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__70!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__54 {
() => {
((core::mem::size_of::<[drm_display_mode; 27]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__13!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__56 {
() => {
((core::mem::size_of::<[drm_display_mode; 127]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__30!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__69 {
() => {
((core::mem::size_of::<[minimode; 7]>() as u64) / (core::mem::size_of::<minimode>() as u64)).wrapping_add(__must_be_array__66!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__70 {
() => {
(((const { core::mem::size_of::<[drm_display_mode; 27]>() }) as u64) / ((const { core::mem::size_of::<drm_display_mode>() }) as u64)).wrapping_add(__must_be_array!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__73 {
() => {
((core::mem::size_of::<[minimode; 44]>() as u64) / (core::mem::size_of::<minimode>() as u64)).wrapping_add(__must_be_array__83!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__77 {
() => {
((core::mem::size_of::<[drm_display_mode; 88]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__69!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__79 {
() => {
((core::mem::size_of::<[drm_display_mode; 5]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__78!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__80 {
() => {
((core::mem::size_of::<[stereo_mandatory_mode; 8]>() as u64) / (core::mem::size_of::<stereo_mandatory_mode>() as u64)).wrapping_add(__must_be_array__88!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__84 {
() => {
((core::mem::size_of::<[drm_display_mode; 127]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__90!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__85 {
() => {
((core::mem::size_of::<[drm_display_mode; 88]>() as u64) / (core::mem::size_of::<drm_display_mode>() as u64)).wrapping_add(__must_be_array__57!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__88 {
() => {
((core::mem::size_of::<[__SlateRecord_drm_edid_c_3489; 7]>() as u64) / (core::mem::size_of::<__SlateRecord_drm_edid_c_3489>() as u64)).wrapping_add(__must_be_array__3!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__89 {
() => {
(((const { core::mem::size_of::<[drm_display_mode; 127]>() }) as u64) / ((const { core::mem::size_of::<drm_display_mode>() }) as u64)).wrapping_add(__must_be_array__39!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__90 {
() => {
((core::mem::size_of::<[u8; 13]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__84!() as u64)
};
}

macro_rules! BITS_PER_TYPE__2 {
() => {
(core::mem::size_of::<u64>() as u64).wrapping_mul(8)
};
}

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

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BIT__2 {
($nr:expr) => {
UL!() << (($nr) as u32)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! DRM_ELD_CEA_SAD {
($mnl:expr, $sad:expr) => {
20i32.wrapping_add($mnl).wrapping_add(3i32.wrapping_mul($sad))
};
}

macro_rules! EDID_PRODUCT_ID {
($e:expr) => {
((unsafe { (core::ptr::addr_of!((*$e).__slate_anon_1.__slate_anon_1.prod_code) as *const u8).wrapping_offset(0 as isize).read_unaligned() }) as i32) | ((unsafe { (core::ptr::addr_of!((*$e).__slate_anon_1.__slate_anon_1.prod_code) as *const u8).wrapping_offset(1 as isize).read_unaligned() }) as i32) << 8i32
};
}

macro_rules! GENMASK_INPUT_CHECK__6 {
() => {
core::mem::size_of::<__SlateRecord_bitmap_h_539>() as i32
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
(GENMASK_INPUT_CHECK__6!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED__2 {
($x:expr, $a:expr) => {
($x) & (($a) as u32).wrapping_sub(1) == 0
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
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! KHZ2PICOS {
($a:expr) => {
1000000000 / (($a) as u64)
};
}

macro_rules! MODE_REFRESH_DIFF {
($c:expr, $t:expr) => {
abs__2!($c.wrapping_sub($t))
};
}

macro_rules! MODE_SIZE {
($m:expr) => {
((unsafe { (*$m).hdisplay }) as i32).wrapping_mul((unsafe { (*$m).vdisplay }) as i32)
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

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u64
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u8
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
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

macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
};
}

macro_rules! __must_be_array {
() => {
((const { core::mem::size_of::<__SlateRecord_drm_edid_c_4265>() }) as u64) as i32
};
}

macro_rules! __must_be_array__3 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3502>() as i32
};
}

macro_rules! __must_be_array__8 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3799>() as i32
};
}

macro_rules! __must_be_array__13 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4269>() as i32
};
}

macro_rules! __must_be_array__30 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4267>() as i32
};
}

macro_rules! __must_be_array__39 {
() => {
((const { core::mem::size_of::<__SlateRecord_drm_edid_c_4264>() }) as u64) as i32
};
}

macro_rules! __must_be_array__42 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3077>() as i32
};
}

macro_rules! __must_be_array__50 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4503>() as i32
};
}

macro_rules! __must_be_array__57 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3735>() as i32
};
}

macro_rules! __must_be_array__60 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4276>() as i32
};
}

macro_rules! __must_be_array__63 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_2982>() as i32
};
}

macro_rules! __must_be_array__66 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3770>() as i32
};
}

macro_rules! __must_be_array__69 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_7315>() as i32
};
}

macro_rules! __must_be_array__70 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4521>() as i32
};
}

macro_rules! __must_be_array__77 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3829>() as i32
};
}

macro_rules! __must_be_array__78 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4465>() as i32
};
}

macro_rules! __must_be_array__83 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_3916>() as i32
};
}

macro_rules! __must_be_array__84 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_5550>() as i32
};
}

macro_rules! __must_be_array__88 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4744>() as i32
};
}

macro_rules! __must_be_array__90 {
() => {
core::mem::size_of::<__SlateRecord_drm_edid_c_4281>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit__2 {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
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
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __swab16 {
($x:expr) => {
unsafe { __slate_intrinsic_1a0c079c71348b00(($x) as u16) }
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

macro_rules! __types_ok3__2 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok {
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

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = ($x) as i32;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
}
};
}

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs__2 {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = $x;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
}
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks {
($_do_alloc:expr) => {
{
let mut __v: *mut ();
let mut __v2: *mut ();
let mut __v3: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
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
macro_rules! container_of__15 {
($ptr:expr) => {
{
let mut __v: *const drm_display_mode;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *const drm_display_mode;
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
macro_rules! container_of__8 {
($ptr:expr) => {
{
let mut __v: *mut drm_display_mode;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut drm_display_mode;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__10!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! dev_is_pci {
($d:expr) => {
(unsafe { (*$d).bus }) == unsafe { core::ptr::addr_of!(pci_bus_type) }
};
}

macro_rules! drm_debug_enabled {
($category:expr) => {
drm_debug_enabled_raw(($category) as u32)
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

/// Note: isspace() must return false for %NUL-terminator
macro_rules! isspace {
($c:expr) => {
(__ismask!($c) as i32) & 32 != 0
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

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

macro_rules! krealloc_node {
($_o:expr, $_s:expr, $_f:expr, $_n:expr) => {
krealloc_node_align!($_o, $_s, 1i32, $_f, $_n)
};
}

macro_rules! krealloc_node__2 {
($_o:expr, $_s:expr, $_f:expr, $_n:expr) => {
krealloc_node_align__2!($_o, $_s, 1i32, $_f, $_n)
};
}

macro_rules! krealloc_node_align {
($p:expr, $new_size:expr, $align:expr, $flags:expr, $nid:expr) => {
alloc_hooks!(unsafe { krealloc_node_align_noprof(($p) as *const (), $new_size, ($align) as u64, $flags, $nid) })
};
}

macro_rules! krealloc_node_align__2 {
($p:expr, $new_size:expr, $align:expr, $flags:expr, $nid:expr) => {
alloc_hooks!(unsafe { krealloc_node_align_noprof(($p) as *const (), ($new_size) as u64, ($align) as u64, $flags, $nid) })
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
macro_rules! list_entry__11 {
($ptr:expr) => {
container_of__15!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__8!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__6 {
($ptr:expr, $type:ty) => {
container_of__9!($ptr, $type)
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

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__2 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head(unsafe { core::ptr::addr_of!((*$pos).$member) }, $head)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__3 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head(unsafe { core::ptr::addr_of!((*$pos).$member) }, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__9 {
($ptr:expr, $type:ty) => {
list_entry__6!(unsafe { (*$ptr).next }, $type)
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
list_entry__4!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__12 {
($ptr:expr) => {
list_entry__11!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__17 {
($pos:expr, $member:ident) => {
list_entry__11!(unsafe { (*$pos).$member.next })
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

macro_rules! offsetof__12 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.flags) }) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__40 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.gtf2.c) }) as u64
};
}

macro_rules! offsetof__41 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.gtf2.k) }) as u64
};
}

macro_rules! offsetof__10 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, head) as u64
};
}

macro_rules! offsetof__44 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.gtf2.j) }) as u64
};
}

macro_rules! offsetof__50 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.cvt.flags) }) as u64
};
}

macro_rules! offsetof__53 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.r#type) }) as u64
};
}

macro_rules! offsetof__22 {
() => {
(const { core::mem::offset_of!(detailed_timing, pixel_clock) }) as u64
};
}

macro_rules! offsetof__63 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.pad1) }) as u64
};
}

macro_rules! offsetof__65 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.gtf2.hfreq_start_khz) }) as u64
};
}

macro_rules! offsetof__4 {
() => {
core::mem::offset_of!(drm_display_mode, head) as u64
};
}

macro_rules! offsetof__83 {
() => {
(const { core::mem::offset_of!(detailed_timing, data.other_data.data.range.formula.gtf2.m) }) as u64
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
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
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
}
};
}

macro_rules! type_max {
() => {
__type_max!()
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut cpuinfo_op: seq_operations;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut _ctype: [u8; 0];
    static mut kstat: kernel_stat;
    static mut pci_bus_type: bus_type;
    static mut __drm_debug: u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strlen(arg0: *const u8) -> u64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn memchr_inv(arg0: *const (), arg1: i32, arg2: u64) -> *mut ();
    fn kmemdup_noprof(arg0: *const (), arg1: u64, arg2: u32) -> *mut ();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn print_hex_dump(arg0: *const u8, arg1: *const u8, arg2: i32, arg3: i32, arg4: i32, arg5: *const (), arg6: u64, arg7: bool);
    fn __bitmap_set(arg0: *mut u64, arg1: u32, arg2: i32);
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn krealloc_node_align_noprof(arg0: *const (), arg1: u64, arg2: u64, arg3: u32, arg4: i32) -> *mut ();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn hdmi_avi_infoframe_init(arg0: *mut hdmi_avi_infoframe);
    fn hdmi_vendor_infoframe_init(arg0: *mut hdmi_vendor_infoframe) -> i32;
    fn acpi_dev_get_pci_dev(arg0: *mut acpi_device) -> *mut pci_dev;
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn i2c_transfer(arg0: *mut i2c_adapter, arg1: *mut i2c_msg, arg2: i32) -> i32;
    fn seq_buf_printf(arg0: *mut seq_buf, arg1: *const u8, ...) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_object_property_set_value(arg0: *mut drm_mode_object, arg1: *mut drm_property, arg2: u64) -> i32;
    fn drm_property_replace_global_blob(arg0: *mut drm_device, arg1: *mut *mut drm_property_blob, arg2: u64, arg3: *const (), arg4: *mut drm_mode_object, arg5: *mut drm_property) -> i32;
    fn drm_connector_set_tile_property(arg0: *mut drm_connector) -> i32;
    fn drm_mode_create_tile_group(arg0: *mut drm_device, arg1: *const u8) -> *mut drm_tile_group;
    fn drm_mode_get_tile_group(arg0: *mut drm_device, arg1: *const u8) -> *mut drm_tile_group;
    fn drm_mode_put_tile_group(arg0: *mut drm_device, arg1: *mut drm_tile_group);
    fn drm_mode_create(arg0: *mut drm_device) -> *mut drm_display_mode;
    fn drm_mode_destroy(arg0: *mut drm_device, arg1: *mut drm_display_mode);
    fn drm_mode_probed_add(arg0: *mut drm_connector, arg1: *mut drm_display_mode);
    fn drm_cvt_mode(arg0: *mut drm_device, arg1: i32, arg2: i32, arg3: i32, arg4: bool, arg5: bool, arg6: bool) -> *mut drm_display_mode;
    fn drm_gtf_mode(arg0: *mut drm_device, arg1: i32, arg2: i32, arg3: i32, arg4: bool, arg5: i32) -> *mut drm_display_mode;
    fn drm_gtf_mode_complex(arg0: *mut drm_device, arg1: i32, arg2: i32, arg3: i32, arg4: bool, arg5: i32, arg6: i32, arg7: i32, arg8: i32, arg9: i32) -> *mut drm_display_mode;
    fn drm_mode_set_name(arg0: *mut drm_display_mode);
    fn drm_mode_vrefresh(arg0: *const drm_display_mode) -> i32;
    fn drm_mode_init(arg0: *mut drm_display_mode, arg1: *const drm_display_mode);
    fn drm_mode_duplicate(arg0: *mut drm_device, arg1: *const drm_display_mode) -> *mut drm_display_mode;
    fn drm_mode_match(arg0: *const drm_display_mode, arg1: *const drm_display_mode, arg2: u32) -> bool;
    fn drm_printf(arg0: *mut drm_printer, arg1: *const u8, ...);
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_mode_create_dumb(arg0: *mut drm_device, arg1: *mut drm_mode_create_dumb, arg2: *mut drm_file) -> i32;
    fn displayid_iter_edid_begin(arg0: *const drm_edid, arg1: *mut displayid_iter);
    fn __displayid_iter_next(arg0: *mut displayid_iter) -> *const displayid_block;
    fn displayid_iter_end(arg0: *mut displayid_iter);
    fn displayid_version(arg0: *const displayid_iter) -> u8;
    fn displayid_primary_use(arg0: *const displayid_iter) -> u8;
    fn drm_ioctl_flags(arg0: u32, arg1: *mut u32) -> bool;
    fn __compiletime_assert_677();
    fn __compiletime_assert_680();
    fn __compiletime_assert_687();
    fn __compiletime_assert_688();
    fn __compiletime_assert_723();
    fn __compiletime_assert_724();
    fn __compiletime_assert_725();
    fn __compiletime_assert_726();
    fn __compiletime_assert_729();
    fn __compiletime_assert_730();
    fn __compiletime_assert_731();
    fn __compiletime_assert_732();
    fn __compiletime_assert_733();
    fn __compiletime_assert_734();
    fn __compiletime_assert_735();
    fn __compiletime_assert_736();
    fn __compiletime_assert_737();
    fn __compiletime_assert_738();
    fn __compiletime_assert_739();
    fn __compiletime_assert_740();
    fn __compiletime_assert_741();
    fn __compiletime_assert_751();
    fn __compiletime_assert_755();
    fn __compiletime_assert_760();
    fn __compiletime_assert_763();
    fn __compiletime_assert_766();
    fn __compiletime_assert_773();
    fn __compiletime_assert_774();
    fn __compiletime_assert_775();
    fn __compiletime_assert_776();
    fn __compiletime_assert_777();
    fn __compiletime_assert_778();
    fn __compiletime_assert_779();
    fn __compiletime_assert_780();
    fn __compiletime_assert_781();
    fn __compiletime_assert_782();
    fn __compiletime_assert_783();
    fn __compiletime_assert_784();
    fn __compiletime_assert_785();
    fn __compiletime_assert_786();
    fn __compiletime_assert_787();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i16"]
    fn __slate_intrinsic_1a0c079c71348b00(arg0: u16) -> u16;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
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
struct static_key_true {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


fn mem_is_zero(s: *const (), n: u64) -> bool {
unsafe { memchr_inv(s, 0i32, n) }.is_null()
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
struct io_bitmap {
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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
// pr_notice - Print a notice-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_NOTICE loglevel. It uses pr_fmt() to
// generate the format string.
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
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
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
struct __SlateRecord_bitmap_h_539 {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits131U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits132U0,
}


#[inline(always)]
fn bitmap_set(map: *mut u64, start: u32, nbits: u32) {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits) }) && nbits == 1 {
__set_bit__2!(start, map);
} else {
if small_const_nbits__2!(start.wrapping_add(nbits)) {
let __v1: u64 = unsafe { *map };
let __v2: u64 = __v1 | GENMASK__4!(start.wrapping_add(nbits).wrapping_sub(1), start);
unsafe {
*map = __v2;
}
} else {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(start & 7u32) }) && IS_ALIGNED__2!(start, 8i32) && unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits & 7u32) } && IS_ALIGNED__2!(nbits, 8i32) {
unsafe { memset((map as *mut u8).wrapping_offset((start / 8) as isize) as *mut (), 255i32, (nbits / 8) as u64) };
} else {
unsafe { __bitmap_set(map, start, nbits as i32) };
}
}
}
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
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


fn __list_splice(list: *const list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*list).next };
let last: *mut list_head = unsafe { (*list).prev };
unsafe {
(*first).prev = prev;
}
unsafe {
(*prev).next = first;
}
unsafe {
(*last).next = next;
}
unsafe {
(*next).prev = last;
}
}

fn list_splice_tail(list: &list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
}
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
    __slate_bits_0: __slate_bits::__SlateBits133U0,
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
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
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct seqcount {
    sequence: u32,
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
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
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
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


// Suits me
// Don't call further
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
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
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
    __slate_bits_0: __slate_bits::__SlateBits420U0,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
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
struct ida {
    xa: xarray,
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
    __slate_bits_0: __slate_bits::__SlateBits472U0,
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
    __slate_bits_0: __slate_bits::__SlateBits474U0,
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


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits499U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits499U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits499U2,
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
    __slate_bits_3: __slate_bits::__SlateBits499U3,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
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
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits645U0,
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
struct __SlateRecord_slab_h_532 {
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits706U0,
    key: __SlateRecord_dynamic_debug_h_55,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
struct __SlateRecord_hdmi_h_214 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_217 {
    x: u16,
    y: u16,
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
    __slate_bits_0: __slate_bits::__SlateBits496U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_any_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_avi_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    itc: bool,
    pixel_repeat: u8,
    colorspace: u32,
    scan_mode: u32,
    colorimetry: u32,
    picture_aspect: u32,
    active_aspect: u32,
    extended_colorimetry: u32,
    quantization_range: u32,
    nups: u32,
    video_code: u8,
    ycc_quantization_range: u32,
    content_type: u32,
    top_bar: u16,
    bottom_bar: u16,
    left_bar: u16,
    right_bar: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_drm_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    eotf: u32,
    metadata_type: u32,
    display_primaries: [__SlateRecord_hdmi_h_214; 3],
    white_point: __SlateRecord_hdmi_h_217,
    max_display_mastering_luminance: u16,
    min_display_mastering_luminance: u16,
    max_cll: u16,
    max_fall: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_spd_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    vendor: [u8; 8],
    product: [u8; 16],
    sdi: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_audio_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    channels: u8,
    coding_type: u32,
    sample_size: u32,
    sample_frequency: u32,
    coding_type_ext: u32,
    channel_allocation: u8,
    level_shift_value: u8,
    downmix_inhibit: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_vendor_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
    vic: u8,
    s3d_struct: i32,
    s3d_ext_data: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_static_metadata {
    eotf: u8,
    metadata_type: u8,
    max_cll: u16,
    max_fall: u16,
    min_cll: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_sink_metadata {
    metadata_type: u32,
    __slate_anon_1: __SlateRecord_hdmi_h_406,
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


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_vendor_any_infoframe {
    any: __SlateRecord_hdmi_h_419,
    hdmi: hdmi_vendor_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_infoframe {
    any: hdmi_any_infoframe,
    avi: hdmi_avi_infoframe,
    spd: hdmi_spd_infoframe,
    vendor: hdmi_vendor_any_infoframe,
    audio: hdmi_audio_infoframe,
    drm: hdmi_drm_infoframe,
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
struct proc_dir_entry {
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
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits806U0,
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits821U0,
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
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits812U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits813U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits825U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits817U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_flags {
    __slate_bits_0: __slate_bits::__SlateBits819U0,
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
struct acpi_gpio_mapping {
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
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits823U0,
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
struct platform_device {
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
    __slate_bits_0: __slate_bits::__SlateBits827U0,
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
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
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


fn str_yes_no(v: bool) -> *const u8 {
(if v { c"yes".as_ptr() as *mut u8 } else { c"no".as_ptr() as *mut u8 }) as *const u8
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
struct cgroup_taskset {
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
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
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
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
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
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kthread_worker {
    flags: u32,
    lock: raw_spinlock,
    work_list: list_head,
    delayed_work_list: list_head,
    task: *mut task_struct,
    current_work: *mut kthread_work,
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
struct kthread_work {
    node: list_head,
    func: Option<unsafe extern "C" fn(*mut kthread_work)>,
    worker: *mut kthread_worker,
    canceling: i32,
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
struct irq_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
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
struct irq_domain_chip_generic {
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
union __SlateRecord_i2c_h_551 {
    xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_i2c_h_557 {
    xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
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
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl_state {
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
    __slate_bits_0: __slate_bits::__SlateBits867U0,
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
struct hotplug_slot {
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
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
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
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits614U0,
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
struct drm_atomic_commit {
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
struct drm_format_info {
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
struct drm_framebuffer {
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
struct rt_mutex_base {
    wait_lock: raw_spinlock,
    waiters: rb_root_cached,
    owner: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex {
    rtmutex: rt_mutex_base,
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
struct drm_master {
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
struct drm_vma_offset_manager {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vram_mm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_device_h_113 {
    resources: list_head,
    final_kfree: *mut (),
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper_surface_size {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf_attachment {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_msg {
    addr: u16,
    flags: u16,
    len: u16,
    buf: *mut u8,
}


// Data for SMBus Messages
// As specified in SMBus standard
#[repr(C)]
#[derive(Clone, Copy)]
union i2c_smbus_data {
    byte: u8,
    word: u16,
    block: [u8; 34],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_algorithm {
    __slate_anon_0: __SlateRecord_i2c_h_551,
    __slate_anon_1: __SlateRecord_i2c_h_557,
    smbus_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    smbus_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    functionality: Option<unsafe extern "C" fn(*mut i2c_adapter) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_lock_operations {
    lock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
    trylock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32) -> i32>,
    unlock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_bus_recovery_info {
    recover_bus: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    get_scl: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_scl: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_sda: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_sda: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_bus_free: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    prepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    unprepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    scl_gpiod: *mut gpio_desc,
    sda_gpiod: *mut gpio_desc,
    pinctrl: *mut pinctrl,
    pins_default: *mut pinctrl_state,
    pins_gpio: *mut pinctrl_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter_quirks {
    flags: u64,
    max_num_msgs: i32,
    max_write_len: u16,
    max_read_len: u16,
    max_comb_1st_msg_len: u16,
    max_comb_2nd_msg_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_126 {
    gtf2: __SlateRecord_drm_edid_h_127,
    cvt: __SlateRecord_drm_edid_h_135,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_127 {
    reserved: u8,
    hfreq_start_khz: u8,
    c: u8,
    m: u16,
    k: u8,
    j: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_135 {
    version: u8,
    data1: u8,
    data2: u8,
    supported_aspects: u8,
    flags: u8,
    supported_scalings: u8,
    preferred_refresh: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter {
    owner: *mut module,
    class: u32,
    algo: *const i2c_algorithm,
    algo_data: *mut (),
    lock_ops: *const i2c_lock_operations,
    bus_lock: rt_mutex,
    mux_lock: rt_mutex,
    timeout: i32,
    retries: i32,
    dev: device,
    locked_flags: u64,
    nr: i32,
    name: [u8; 48],
    dev_released: completion,
    userspace_clients_lock: mutex,
    userspace_clients: list_head,
    bus_recovery_info: *mut i2c_bus_recovery_info,
    quirks: *const i2c_adapter_quirks,
    host_notify_domain: *mut irq_domain,
    bus_regulator: *mut regulator,
    debugfs: *mut dentry,
    addrs_in_instantiation: [u64; 2],
}


// Adapter locking functions, exported for shared pin cases
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
    __slate_bits_0: __slate_bits::__SlateBits926U0,
    kobj: kobject,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_171 {
    str: detailed_data_string,
    range: detailed_data_monitor_range,
    color: detailed_data_wpindex,
    timings: [std_timing; 6],
    cvt: [cvt_timing; 4],
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
union __SlateRecord_drm_edid_h_192 {
    pixel_data: detailed_pixel_timing,
    other_data: detailed_non_pixel,
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
    __slate_bits_0: __slate_bits::__SlateBits837U0,
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
    __slate_bits_1: __slate_bits::__SlateBits837U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits837U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits837U3,
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
    __slate_bits_4: __slate_bits::__SlateBits837U4,
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
    __slate_bits_0: __slate_bits::__SlateBits450U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_294 {
    product_id: drm_edid_product_id,
    __slate_anon_1: __SlateRecord_drm_edid_h_296,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_296 {
    mfg_id: [u8; 2],
    prod_code: [u8; 2],
    serial: u32,
    mfg_week: u8,
    mfg_year: u8,
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
struct fs_pin {
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
struct seq_buf {
    buffer: *mut u8,
    size: u64,
    len: u64,
}


fn seq_buf_has_overflowed(s: &seq_buf) -> bool {
(unsafe { (*(s as *const seq_buf)).len }) > unsafe { (*(s as *const seq_buf)).size }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_helper_funcs {
}


fn seq_buf_buffer_left(s: &seq_buf) -> u32 {
if seq_buf_has_overflowed(s) {
return 0;
}
unsafe { (*(s as *const seq_buf)).size }.wrapping_sub(unsafe { (*(s as *const seq_buf)).len }) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_daifmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_params {
}


fn seq_buf_str(s: &seq_buf) -> *const u8 {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*(s as *const seq_buf)).size }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("642: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 642b - ., 3; .popsection\n# {0}", const 642 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_seq_buf_str_1, sym __slate_asm_string_seq_buf_str_2, const 100 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("643: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 643b - ., 4; .popsection\n# {0}", const 643 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return c"".as_ptr() as *const u8;
}
if seq_buf_buffer_left(s) != 0 {
unsafe {
*unsafe { (*(s as *const seq_buf)).buffer }.wrapping_offset((*(s as *const seq_buf)).len as isize) = 0;
}
} else {
unsafe {
*unsafe { (*(s as *const seq_buf)).buffer }.wrapping_offset(unsafe { (*(s as *const seq_buf)).size }.wrapping_sub(1) as isize) = 0;
}
}
(unsafe { (*(s as *const seq_buf)).buffer }) as *const u8
}

fn vga_switcheroo_lock_ddc(pdev: *mut pci_dev) -> i32 {
19i32.wrapping_neg()
}

fn vga_switcheroo_unlock_ddc(pdev: *mut pci_dev) -> i32 {
19i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_acquire_ctx {
    task: *mut task_struct,
    stamp: u64,
    acquired: u32,
    wounded: u16,
    is_wait_die: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_acquire_ctx {
    ww_ctx: ww_acquire_ctx,
    contended: *mut drm_modeset_lock,
    stack_depot: u32,
    locked: list_head,
    trylock_only: bool,
    interruptible: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_lock {
    mutex: ww_mutex,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_funcs {
    fb_create: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, *const drm_format_info, *const drm_mode_fb_cmd2) -> *mut drm_framebuffer>,
    get_format_info: Option<unsafe extern "C" fn(u32, u64) -> *const drm_format_info>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> i32>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>,
    atomic_state_alloc: Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_atomic_commit>,
    atomic_state_clear: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_state_free: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config {
    mutex: mutex,
    connection_mutex: drm_modeset_lock,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    idr_mutex: mutex,
    object_idr: idr,
    tile_idr: idr,
    fb_lock: mutex,
    num_fb: i32,
    fb_list: list_head,
    connector_list_lock: spinlock,
    num_connector: i32,
    connector_ida: ida,
    connector_list: list_head,
    connector_free_list: llist_head,
    connector_free_work: work_struct,
    num_encoder: i32,
    encoder_list: list_head,
    num_total_plane: i32,
    plane_list: list_head,
    panic_lock: raw_spinlock,
    num_colorop: i32,
    colorop_list: list_head,
    num_crtc: i32,
    crtc_list: list_head,
    property_list: list_head,
    privobj_list: list_head,
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
    funcs: *const drm_mode_config_funcs,
    poll_enabled: bool,
    poll_running: bool,
    delayed_event: bool,
    output_poll_work: delayed_work,
    blob_lock: mutex,
    property_blob_list: list_head,
    edid_property: *mut drm_property,
    dpms_property: *mut drm_property,
    path_property: *mut drm_property,
    tile_property: *mut drm_property,
    panel_type_property: *mut drm_property,
    link_status_property: *mut drm_property,
    plane_type_property: *mut drm_property,
    prop_src_x: *mut drm_property,
    prop_src_y: *mut drm_property,
    prop_src_w: *mut drm_property,
    prop_src_h: *mut drm_property,
    prop_crtc_x: *mut drm_property,
    prop_crtc_y: *mut drm_property,
    prop_crtc_w: *mut drm_property,
    prop_crtc_h: *mut drm_property,
    prop_fb_id: *mut drm_property,
    prop_in_fence_fd: *mut drm_property,
    prop_out_fence_ptr: *mut drm_property,
    prop_crtc_id: *mut drm_property,
    prop_fb_damage_clips: *mut drm_property,
    prop_active: *mut drm_property,
    prop_mode_id: *mut drm_property,
    prop_vrr_enabled: *mut drm_property,
    dvi_i_subconnector_property: *mut drm_property,
    dvi_i_select_subconnector_property: *mut drm_property,
    dp_subconnector_property: *mut drm_property,
    tv_subconnector_property: *mut drm_property,
    tv_select_subconnector_property: *mut drm_property,
    legacy_tv_mode_property: *mut drm_property,
    tv_mode_property: *mut drm_property,
    tv_left_margin_property: *mut drm_property,
    tv_right_margin_property: *mut drm_property,
    tv_top_margin_property: *mut drm_property,
    tv_bottom_margin_property: *mut drm_property,
    tv_brightness_property: *mut drm_property,
    tv_contrast_property: *mut drm_property,
    tv_flicker_reduction_property: *mut drm_property,
    tv_overscan_property: *mut drm_property,
    tv_saturation_property: *mut drm_property,
    tv_hue_property: *mut drm_property,
    scaling_mode_property: *mut drm_property,
    aspect_ratio_property: *mut drm_property,
    content_type_property: *mut drm_property,
    degamma_lut_property: *mut drm_property,
    degamma_lut_size_property: *mut drm_property,
    ctm_property: *mut drm_property,
    gamma_lut_property: *mut drm_property,
    gamma_lut_size_property: *mut drm_property,
    background_color_property: *mut drm_property,
    suggested_x_property: *mut drm_property,
    suggested_y_property: *mut drm_property,
    non_desktop_property: *mut drm_property,
    panel_orientation_property: *mut drm_property,
    writeback_fb_id_property: *mut drm_property,
    writeback_pixel_formats_property: *mut drm_property,
    writeback_out_fence_ptr_property: *mut drm_property,
    hdr_output_metadata_property: *mut drm_property,
    content_protection_property: *mut drm_property,
    hdcp_content_type_property: *mut drm_property,
    preferred_depth: u32,
    prefer_shadow: u32,
    quirk_addfb_prefer_xbgr_30bpp: bool,
    quirk_addfb_prefer_host_byte_order: bool,
    async_page_flip: bool,
    fb_modifiers_not_supported: bool,
    normalize_zpos: bool,
    modifiers_property: *mut drm_property,
    async_modifiers_property: *mut drm_property,
    size_hints_property: *mut drm_property,
    cursor_width: u32,
    cursor_height: u32,
    suspend_state: *mut drm_atomic_commit,
    helper_private: *const drm_mode_config_helper_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_device {
    if_version: i32,
    r#ref: kref,
    dev: *mut device,
    dma_dev: *mut device,
    managed: __SlateRecord_drm_device_h_113,
    driver: *const drm_driver,
    dev_private: *mut (),
    primary: *mut drm_minor,
    render: *mut drm_minor,
    accel: *mut drm_minor,
    registered: bool,
    master: *mut drm_master,
    driver_features: u32,
    unplugged: bool,
    anon_inode: *mut inode,
    unique: *mut u8,
    master_mutex: mutex,
    open_count: __SlateRecord_types_h_190,
    filelist_mutex: mutex,
    filelist: list_head,
    filelist_internal: list_head,
    clientlist_mutex: mutex,
    clientlist: list_head,
    client_sysrq_list: list_head,
    vblank_disable_immediate: bool,
    vblank: *mut drm_vblank_crtc,
    vblank_time_lock: spinlock,
    vbl_lock: spinlock,
    max_vblank_count: u32,
    vblank_event_list: list_head,
    event_lock: spinlock,
    num_crtcs: u32,
    mode_config: drm_mode_config,
    object_name_lock: mutex,
    object_name_idr: idr,
    vma_offset_manager: *mut drm_vma_offset_manager,
    vram_mm: *mut drm_vram_mm,
    switch_power_state: u32,
    fb_helper: *mut drm_fb_helper,
    debugfs_root: *mut dentry,
    gem_lru_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
    load: Option<unsafe extern "C" fn(*mut drm_device, u64) -> i32>,
    open: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file) -> i32>,
    postclose: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    unload: Option<unsafe extern "C" fn(*mut drm_device)>,
    release: Option<unsafe extern "C" fn(*mut drm_device)>,
    master_set: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, bool)>,
    master_drop: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_minor)>,
    gem_create_object: Option<unsafe extern "C" fn(*mut drm_device, u64) -> *mut drm_gem_object>,
    prime_handle_to_fd: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, u32, u32, *mut i32) -> i32>,
    prime_fd_to_handle: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, i32, *mut u32) -> i32>,
    gem_prime_import: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf) -> *mut drm_gem_object>,
    gem_prime_import_sg_table: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf_attachment, *mut sg_table) -> *mut drm_gem_object>,
    dumb_create: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, *mut drm_mode_create_dumb) -> i32>,
    dumb_map_offset: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, u32, *mut u64) -> i32>,
    fbdev_probe: Option<unsafe extern "C" fn(*mut drm_fb_helper, *mut drm_fb_helper_surface_size) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut drm_printer, *mut drm_file)>,
    major: i32,
    minor: i32,
    patchlevel: i32,
    name: *mut u8,
    desc: *mut u8,
    driver_features: u32,
    ioctls: *const drm_ioctl_desc,
    num_ioctls: i32,
    fops: *const file_operations,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct est_timings {
    t1: u8,
    t2: u8,
    mfg_rsvd: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
}


// 00=16:10, 01=4:3, 10=5:4, 11=16:9
// need to add 60
#[repr(C)]
#[derive(Clone, Copy)]
struct std_timing {
    hsize: u8,
    vfreq_aspect: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_commit {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_job {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_pixel_timing {
    hactive_lo: u8,
    hblank_lo: u8,
    hactive_hblank_hi: u8,
    vactive_lo: u8,
    vblank_lo: u8,
    vactive_vblank_hi: u8,
    hsync_offset_lo: u8,
    hsync_pulse_width_lo: u8,
    vsync_offset_pulse_width_lo: u8,
    hsync_vsync_offset_pulse_width_hi: u8,
    width_mm_lo: u8,
    height_mm_lo: u8,
    width_height_mm_hi: u8,
    hborder: u8,
    vborder: u8,
    misc: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_string {
    str: [u8; 13],
}


// 1.4
// 1.4
// 1.3
// 1.4
// 1.3
// 1.4
#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_monitor_range {
    min_vfreq: u8,
    max_vfreq: u8,
    min_hfreq_khz: u8,
    max_hfreq_khz: u8,
    pixel_clock_mhz: u8,
    flags: u8,
    formula: __SlateRecord_drm_edid_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_wpindex {
    white_yx_lo: u8,
    white_x_hi: u8,
    white_y_hi: u8,
    gamma: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cvt_timing {
    code: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_non_pixel {
    pad1: u8,
    r#type: u8,
    pad2: u8,
    data: __SlateRecord_drm_edid_h_171,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct detailed_timing {
    pixel_clock: u16,
    data: __SlateRecord_drm_edid_h_192,
}


// 1.4
// 1.4
// 1.4
// 1.4
// 1.4
// 1.4
// 1.4
// 1.4
// 1.3
// 1.4
// HDMI 2.1 additional fields
// DSC specific
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct drm_edid_product_id {
    manufacturer_name: u16,
    product_code: u16,
    serial_number: u32,
    week_of_manufacture: u8,
    year_of_manufacture: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid {
    header: [u8; 8],
    __slate_anon_1: __SlateRecord_drm_edid_h_294,
    version: u8,
    revision: u8,
    input: u8,
    width_cm: u8,
    height_cm: u8,
    gamma: u8,
    features: u8,
    red_green_lo: u8,
    blue_white_lo: u8,
    red_x: u8,
    red_y: u8,
    green_x: u8,
    green_y: u8,
    blue_x: u8,
    blue_y: u8,
    white_x: u8,
    white_y: u8,
    established_timings: est_timings,
    standard_timings: [std_timing; 8],
    detailed_timings: [detailed_timing; 4],
    extensions: u8,
    checksum: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid_ident {
    panel_id: u32,
    name: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cea_sad {
    format: u8,
    channels: u8,
    freq: u8,
    byte2: u8,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_av_sync_delay")]
extern "C" fn drm_av_sync_delay(connector: *mut drm_connector, mode: *const drm_display_mode) -> i32 {
let mut i: i32 = ((unsafe { (*mode).flags }) & 16u32 != 0) as i32;
let mut a: i32 = 0;
let mut v: i32 = 0;
if !unsafe { *((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(0 as isize) } {
return 0;
}
if !unsafe { *((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(1 as isize) } {
i = 0;
}
a = unsafe { *((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) };
v = unsafe { *((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) };
// HDMI/DP sink doesn't support audio or video?
if a == 255 || v == 255 {
return 0;
}
// Convert raw EDID values to millisecond.
// Treat unknown latency as 0ms.
if a != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__758: i32 = 2i32.wrapping_mul(a.wrapping_sub(1));
let __UNIQUE_ID_y__759: i32 = 500;
loop {
if __types_ok!(__UNIQUE_ID_x__758, __UNIQUE_ID_y__759) == 0 {
unsafe { __compiletime_assert_760() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__758 < __UNIQUE_ID_y__759 { __UNIQUE_ID_x__758 } else { __UNIQUE_ID_y__759 };
a = __v0;
}
if v != 0 {
let mut __v2: i32;
let __UNIQUE_ID_x__761: i32 = 2i32.wrapping_mul(v.wrapping_sub(1));
let __UNIQUE_ID_y__762: i32 = 500;
loop {
if __types_ok!(__UNIQUE_ID_x__761, __UNIQUE_ID_y__762) == 0 {
unsafe { __compiletime_assert_763() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__761 < __UNIQUE_ID_y__762 { __UNIQUE_ID_x__761 } else { __UNIQUE_ID_y__762 };
v = __v2;
}
let mut __v4: i32;
let __UNIQUE_ID_x__764: i32 = v.wrapping_sub(a);
let __UNIQUE_ID_y__765: i32 = 0;
loop {
if __types_ok!(__UNIQUE_ID_x__764, __UNIQUE_ID_y__765) == 0 {
unsafe { __compiletime_assert_766() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__764 > __UNIQUE_ID_y__765 { __UNIQUE_ID_x__764 } else { __UNIQUE_ID_y__765 };
__v4
}

fn drm_edid_decode_mfg_id(mfg_id: u16, vend: *mut u8) -> *const u8 {
unsafe {
*vend.wrapping_offset(0 as isize) = 64i32.wrapping_add((mfg_id as i32) >> 10i32 & 31) as u8;
}
unsafe {
*vend.wrapping_offset(1 as isize) = 64i32.wrapping_add((mfg_id as i32) >> 5i32 & 31) as u8;
}
unsafe {
*vend.wrapping_offset(2 as isize) = 64i32.wrapping_add((mfg_id as i32) >> 0i32 & 31) as u8;
}
unsafe {
*vend.wrapping_offset(3 as isize) = 0;
}
vend as *const u8
}

// drm_edid_encode_panel_id - Encode an ID for matching against drm_edid_get_panel_id()
// @vend_chr_0: First character of the vendor string.
// @vend_chr_1: Second character of the vendor string.
// @vend_chr_2: Third character of the vendor string.
// @product_id: The 16-bit product ID.
//
// This is a macro so that it can be calculated at compile time and used
// as an initializer.
//
// For instance:
//   drm_edid_encode_panel_id('B', 'O', 'E', 0x2d08) => 0x09e52d08
//
// Return: a 32-bit ID per panel.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_get_monitor_name")]
extern "C" fn drm_edid_get_monitor_name(edid: *const edid, name: *mut u8, bufsize: i32) {
let mut name_length: i32 = 0;
if bufsize <= 0 {
return;
}
if !edid.is_null() {
let mut buf: [u8; 13] = [0u8; 13];
let mut drm_edid: drm_edid = drm_edid { size: edid_size(edid) as u64, edid: edid };
let mut __v0: i32;
let __UNIQUE_ID_x__749: i32 = get_monitor_name(core::ptr::addr_of_mut!(drm_edid) as *const drm_edid, buf.as_mut_ptr() as *mut u8);
let __UNIQUE_ID_y__750: i32 = bufsize.wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__749, __UNIQUE_ID_y__750) == 0 {
unsafe { __compiletime_assert_751() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__749 < __UNIQUE_ID_y__750 { __UNIQUE_ID_x__749 } else { __UNIQUE_ID_y__750 };
name_length = __v0;
unsafe { memcpy(name as *mut (), buf.as_mut_ptr() as *const (), name_length as u64) };
}
unsafe {
*name.wrapping_offset(name_length as isize) = 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dma_fence {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_print_product_id")]
extern "C" fn drm_edid_print_product_id(p: *mut drm_printer, id: *const drm_edid_product_id, raw: bool) {
let mut __slate_compound_0: [u8; 40] = unsafe { core::mem::zeroed() };
let mut date: seq_buf = {
let mut __t0: seq_buf = unsafe { core::mem::zeroed() };
__t0.buffer = unsafe { (*{
__slate_compound_0 = {
let mut __t1: [u8; 40] = core::mem::zeroed();
__t1[0] = 0u8;
__t1
};
core::ptr::addr_of_mut!(__slate_compound_0)
}).as_mut_ptr() as *mut u8 };
__t0.size = 40u64;
__t0
};
let mut vend: [u8; 4] = [0u8; 4];
drm_edid_decode_mfg_id(__swab16!(unsafe { (*id).manufacturer_name }), vend.as_mut_ptr() as *mut u8);
decode_date(core::ptr::addr_of_mut!(date), id);
unsafe { drm_printf(p, c"manufacturer name: %s, product code: %u, serial number: %u, %s\n".as_ptr() as *const u8, vend.as_mut_ptr() as *mut u8, (*id).product_code as i32, (*id).serial_number, seq_buf_str(&(*core::ptr::addr_of_mut!(date)))) };
if raw {
unsafe { drm_printf(p, c"raw product id: %*ph\n".as_ptr() as *const u8, core::mem::size_of::<drm_edid_product_id>() as i32, id) };
}
let mut __v1: i64;
let __ret_warn_on: i32 = !(!seq_buf_has_overflowed(unsafe { &(*core::ptr::addr_of_mut!(date)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("711: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 711b - ., 3; .popsection\n# {0}", const 711 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_edid_print_product_id_2, sym __slate_asm_string_drm_edid_print_product_id_3, const 2803 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("712: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 712b - ., 4; .popsection\n# {0}", const 712 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
}


// supports 861D or below
// in dwords!
// ELD Baseline Block for ELD_Ver == 2
// offsets 20..(20+mnl-1) inclusive
fn drm_eld_mnl(eld: *const u8) -> i32 {
(((unsafe { *eld.wrapping_offset(4 as isize) }) as i32) & 31i32) >> 0i32
}

fn drm_eld_sad_count(eld: *const u8) -> i32 {
(((unsafe { *eld.wrapping_offset(5 as isize) }) as i32) & 240i32) >> 4i32
}

fn drm_eld_calc_baseline_block_size(eld: *const u8) -> i32 {
16i32.wrapping_add(drm_eld_mnl(eld)).wrapping_add(drm_eld_sad_count(eld).wrapping_mul(3))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_self_refresh_data {
}


fn drm_eld_size(eld: *const u8) -> i32 {
4i32.wrapping_add(((unsafe { *eld.wrapping_offset(2 as isize) }) as i32).wrapping_mul(4))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_object {
    id: u32,
    r#type: u32,
    properties: *mut drm_object_properties,
    refcount: kref,
    free_cb: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_object_properties {
    count: i32,
    properties: [*mut drm_property; 64],
    values: [u64; 64],
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
struct drm_encoder_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event {
    r#type: u32,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_vblank {
    base: drm_event,
    user_data: u64,
    tv_sec: u32,
    tv_usec: u32,
    sequence: u32,
    crtc_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_crtc_sequence {
    base: drm_event,
    user_data: u64,
    time_ns: i64,
    sequence: u64,
}


// When adding a new stereo mode don't forget to adjust DRM_MODE_FLAGS_3D_MAX
// (define not exposed to user space).
// Link Status options
// Panel type property
// extended-types: rather than continue to consume a bit per type,
// grab a chunk of the bits to use as integer type id.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_fb_cmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_create_dumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property {
    head: list_head,
    base: drm_mode_object,
    flags: u32,
    name: [u8; 32],
    num_values: u32,
    values: *mut u64,
    dev: *mut drm_device,
    enum_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property_blob {
    base: drm_mode_object,
    dev: *mut drm_device,
    head_global: list_head,
    head_file: list_head,
    length: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scrambling {
    supported: bool,
    low_rates: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scdc {
    supported: bool,
    read_request: bool,
    scrambling: drm_scrambling,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_dsc_cap {
    v_1p2: bool,
    native_420: bool,
    all_bpp: bool,
    bpc_supported: u8,
    max_slices: u8,
    clk_per_slice: i32,
    max_lanes: u8,
    max_frl_rate_per_lane: u8,
    total_chunk_kbytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_info {
    scdc: drm_scdc,
    y420_vdb_modes: [u64; 4],
    y420_cmdb_modes: [u64; 4],
    y420_dc_modes: u8,
    max_frl_rate_per_lane: u8,
    max_lanes: u8,
    dsc_cap: drm_hdmi_dsc_cap,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_vblank_h_59 {
    base: drm_event,
    vbl: drm_event_vblank,
    seq: drm_event_crtc_sequence,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_monitor_range_info {
    min_vfreq: u16,
    max_vfreq: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_luminance_range_info {
    min_luminance: u32,
    max_luminance: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_amd_vsdb_info {
    version: u8,
    replay_mode: bool,
    panel_type: u8,
    luminance_range1: drm_luminance_range_info,
    luminance_range2: drm_luminance_range_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_info {
    width_mm: u32,
    height_mm: u32,
    bpc: u32,
    subpixel_order: u32,
    panel_orientation: i32,
    color_formats: u32,
    bus_formats: *const u32,
    num_bus_formats: u32,
    bus_flags: u32,
    max_tmds_clock: i32,
    dvi_dual: bool,
    is_hdmi: bool,
    has_audio: bool,
    has_hdmi_infoframe: bool,
    rgb_quant_range_selectable: bool,
    edid_hdmi_rgb444_dc_modes: u8,
    edid_hdmi_ycbcr444_dc_modes: u8,
    cea_rev: u8,
    hdmi: drm_hdmi_info,
    hdr_sink_metadata: hdr_sink_metadata,
    non_desktop: bool,
    monitor_range: drm_monitor_range_info,
    luminance_range: drm_luminance_range_info,
    mso_stream_count: u8,
    mso_pixel_overlap: u8,
    max_dsc_bpp: u32,
    vics: *mut u8,
    vics_len: i32,
    quirks: u32,
    source_physical_address: u16,
    amd_vsdb: drm_amd_vsdb_info,
    panel_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_tv_margins {
    bottom: u32,
    left: u32,
    right: u32,
    top: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tv_connector_state {
    select_subconnector: u32,
    subconnector: u32,
    margins: drm_connector_tv_margins,
    legacy_mode: u32,
    mode: u32,
    brightness: u32,
    contrast: u32,
    flicker_reduction: u32,
    overscan: u32,
    saturation: u32,
    hue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_infoframe {
    data: hdmi_infoframe,
    set: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_state {
    broadcast_rgb: u32,
    infoframes: __SlateRecord_drm_connector_h_1088,
    is_limited_range: bool,
    output_bpc: u32,
    output_format: u32,
    tmds_char_rate: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_1823_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_2982 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3077 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3489 {
    w: i32,
    h: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3502 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3735 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3770 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3799 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3829 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_3916 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4264 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4265 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4267 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4269 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4276 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4281 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4465 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4503 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4521 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_state {
    connector: *mut drm_connector,
    crtc: *mut drm_crtc,
    best_encoder: *mut drm_encoder,
    link_status: u32,
    state: *mut drm_atomic_commit,
    commit: *mut drm_crtc_commit,
    tv: drm_tv_connector_state,
    self_refresh_aware: bool,
    picture_aspect_ratio: u32,
    content_type: u32,
    hdcp_content_type: u32,
    scaling_mode: u32,
    content_protection: u32,
    colorspace: u32,
    color_format: u32,
    writeback_job: *mut drm_writeback_job,
    max_requested_bpc: u8,
    max_bpc: u8,
    privacy_screen_sw_state: u32,
    hdr_output_metadata: *mut drm_property_blob,
    hdmi: drm_connector_hdmi_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_4744 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio_funcs {
    startup: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut drm_connector)>,
    mute_stream: Option<unsafe extern "C" fn(*mut drm_connector, bool, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec_funcs {
    phys_addr_invalidate: Option<unsafe extern "C" fn(*mut drm_connector)>,
    phys_addr_set: Option<unsafe extern "C" fn(*mut drm_connector, u16)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_5550 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6625_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6646_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_6754_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_c_7315 {
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
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits614U0 {
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
    pub struct __SlateBits645U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits499U0 {
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
    pub struct __SlateBits496U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits499U1 {
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
    pub struct __SlateBits499U2 {
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
    pub struct __SlateBits499U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits474U0 {
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
    pub struct __SlateBits420U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits867U0 {
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
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits131U0 {
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
    pub struct __SlateBits1063U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1075U0 {
        #[bits(1)]
        pub planes_changed: bool,
        #[bits(1)]
        pub mode_changed: bool,
        #[bits(1)]
        pub active_changed: bool,
        #[bits(1)]
        pub connectors_changed: bool,
        #[bits(1)]
        pub zpos_changed: bool,
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits450U0 {
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
    pub struct __SlateBits926U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits837U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits837U1 {
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
    pub struct __SlateBits837U2 {
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
    pub struct __SlateBits837U3 {
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
    pub struct __SlateBits837U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits812U0 {
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
    pub struct __SlateBits813U0 {
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
    pub struct __SlateBits817U0 {
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
    pub struct __SlateBits819U0 {
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
    pub struct __SlateBits821U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits823U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits825U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits806U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_infoframe_funcs {
    clear_infoframe: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    write_infoframe: Option<unsafe extern "C" fn(*mut drm_connector, *const u8, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_funcs {
    tmds_char_rate_valid: Option<unsafe extern "C" fn(*const drm_connector, *const drm_display_mode, u64) -> i32>,
    read_edid: Option<unsafe extern "C" fn(*mut drm_connector) -> *const drm_edid>,
    avi: drm_connector_infoframe_funcs,
    hdmi: drm_connector_infoframe_funcs,
    audio: drm_connector_infoframe_funcs,
    hdr_drm: drm_connector_infoframe_funcs,
    spd: drm_connector_infoframe_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_connector, i32) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut drm_connector)>,
    detect: Option<unsafe extern "C" fn(*mut drm_connector, bool) -> u32>,
    force: Option<unsafe extern "C" fn(*mut drm_connector)>,
    fill_modes: Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_property, u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_connector)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_connector)>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_connector_state)>,
    oob_hotplug_event: Option<unsafe extern "C" fn(*mut drm_connector, u32)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_connector, *mut dentry)>,
    color_format: Option<unsafe extern "C" fn(*const drm_connector_state) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_cmdline_mode {
    name: [u8; 32],
    specified: bool,
    refresh_specified: bool,
    bpp_specified: bool,
    pixel_clock: u32,
    xres: i32,
    yres: i32,
    bpp: i32,
    refresh: i32,
    rb: bool,
    interlace: bool,
    cvt: bool,
    margins: bool,
    force: u32,
    rotation_reflection: u32,
    panel_orientation: i32,
    tv_margins: drm_connector_tv_margins,
    tv_mode: u32,
    tv_mode_specified: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio {
    funcs: *const drm_connector_hdmi_audio_funcs,
    codec_pdev: *mut platform_device,
    lock: mutex,
    plugged_cb: Option<unsafe extern "C" fn(*mut device, bool)>,
    plugged_cb_dev: *mut device,
    last_state: bool,
    dai_port: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi {
    vendor: [u8; 8],
    product: [u8; 16],
    supported_formats: u64,
    funcs: *const drm_connector_hdmi_funcs,
    infoframes: __SlateRecord_drm_connector_h_2062,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec {
    mutex: mutex,
    funcs: *const drm_connector_cec_funcs,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector {
    dev: *mut drm_device,
    kdev: *mut device,
    attr: *mut device_attribute,
    fwnode: *mut fwnode_handle,
    head: list_head,
    global_connector_list_entry: list_head,
    base: drm_mode_object,
    name: *mut u8,
    mutex: mutex,
    index: u32,
    connector_type: i32,
    connector_type_id: i32,
    interlace_allowed: bool,
    doublescan_allowed: bool,
    stereo_allowed: bool,
    ycbcr_420_allowed: bool,
    registration_state: u32,
    modes: list_head,
    status: u32,
    probed_modes: list_head,
    display_info: drm_display_info,
    funcs: *const drm_connector_funcs,
    edid_blob_ptr: *mut drm_property_blob,
    properties: drm_object_properties,
    scaling_mode_property: *mut drm_property,
    vrr_capable_property: *mut drm_property,
    colorspace_property: *mut drm_property,
    color_format_property: *mut drm_property,
    path_blob_ptr: *mut drm_property_blob,
    max_bpc: u32,
    max_bpc_property: *mut drm_property,
    privacy_screen: *mut drm_privacy_screen,
    privacy_screen_notifier: notifier_block,
    privacy_screen_sw_state_property: *mut drm_property,
    privacy_screen_hw_state_property: *mut drm_property,
    broadcast_rgb_property: *mut drm_property,
    polled: u8,
    dpms: i32,
    helper_private: *const drm_connector_helper_funcs,
    cmdline_mode: drm_cmdline_mode,
    force: u32,
    edid_override: *const drm_edid,
    edid_override_mutex: mutex,
    epoch_counter: u64,
    possible_encoders: u32,
    encoder: *mut drm_encoder,
    eld: [u8; 128],
    eld_mutex: mutex,
    latency_present: [bool; 2],
    video_latency: [i32; 2],
    audio_latency: [i32; 2],
    ddc: *mut i2c_adapter,
    null_edid_counter: i32,
    bad_edid_counter: u32,
    edid_corrupt: bool,
    real_edid_checksum: u8,
    debugfs_entry: *mut dentry,
    state: *mut drm_connector_state,
    tile_blob_ptr: *mut drm_property_blob,
    has_tile: bool,
    tile_group: *mut drm_tile_group,
    tile_is_single_monitor: bool,
    num_h_tile: u8,
    num_v_tile: u8,
    tile_h_loc: u8,
    tile_v_loc: u8,
    tile_h_size: u16,
    tile_v_size: u16,
    free_node: llist_node,
    hdmi: drm_connector_hdmi,
    hdmi_audio: drm_connector_hdmi_audio,
    cec: drm_connector_cec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_mode {
    clock: i32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    flags: u32,
    crtc_clock: i32,
    crtc_hdisplay: u16,
    crtc_hblank_start: u16,
    crtc_hblank_end: u16,
    crtc_hsync_start: u16,
    crtc_hsync_end: u16,
    crtc_htotal: u16,
    crtc_hskew: u16,
    crtc_vdisplay: u16,
    crtc_vblank_start: u16,
    crtc_vblank_end: u16,
    crtc_vsync_start: u16,
    crtc_vsync_end: u16,
    crtc_vtotal: u16,
    width_mm: u16,
    height_mm: u16,
    r#type: u8,
    expose_to_userspace: bool,
    head: list_head,
    name: [u8; 32],
    status: i32,
    picture_aspect_ratio: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dump_detail {
    reason: u32,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dumper {
    list: list_head,
    dump: Option<unsafe extern "C" fn(*mut kmsg_dumper, *mut kmsg_dump_detail)>,
    max_reason: u32,
    registered: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_rect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_state {
    plane: *mut drm_plane,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    fence: *mut dma_fence,
    crtc_x: i32,
    crtc_y: i32,
    crtc_w: u32,
    crtc_h: u32,
    src_x: u32,
    src_y: u32,
    src_h: u32,
    src_w: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    alpha: u16,
    pixel_blend_mode: u16,
    rotation: u32,
    zpos: u32,
    normalized_zpos: u32,
    color_encoding: u32,
    color_range: u32,
    fb_damage_clips: *mut drm_property_blob,
    ignore_damage_clips: bool,
    src: drm_rect,
    dst: drm_rect,
    visible: bool,
    scaling_filter: u32,
    color_pipeline: *mut drm_colorop,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
    __slate_bits_0: __slate_bits::__SlateBits1063U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_funcs {
    update_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_crtc, *mut drm_framebuffer, i32, i32, u32, u32, u32, u32, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    disable_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_plane)>,
    reset: Option<unsafe extern "C" fn(*mut drm_plane)>,
    set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_plane, *const drm_plane_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_plane) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_plane)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_plane_state)>,
    format_mod_supported: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
    format_mod_supported_async: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane {
    dev: *mut drm_device,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    possible_crtcs: u32,
    format_types: *mut u32,
    format_count: u32,
    format_default: bool,
    modifiers: *mut u64,
    modifier_count: u32,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    old_fb: *mut drm_framebuffer,
    funcs: *const drm_plane_funcs,
    properties: drm_object_properties,
    r#type: u32,
    index: u32,
    helper_private: *const drm_plane_helper_funcs,
    state: *mut drm_plane_state,
    alpha_property: *mut drm_property,
    zpos_property: *mut drm_property,
    rotation_property: *mut drm_property,
    blend_mode_property: *mut drm_property,
    color_encoding_property: *mut drm_property,
    color_range_property: *mut drm_property,
    color_pipeline_property: *mut drm_property,
    scaling_filter_property: *mut drm_property,
    hotspot_x_property: *mut drm_property,
    hotspot_y_property: *mut drm_property,
    kmsg_panic: kmsg_dumper,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc_entry {
    has_frame_counter: bool,
    frame: u32,
    crcs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc {
    lock: spinlock,
    source: *const u8,
    opened: bool,
    overflow: bool,
    entries: *mut drm_crtc_crc_entry,
    head: i32,
    tail: i32,
    values_cnt: u64,
    wq: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_state {
    crtc: *mut drm_crtc,
    enable: bool,
    active: bool,
    __slate_bits_0: __slate_bits::__SlateBits1075U0,
    no_vblank: bool,
    plane_mask: u32,
    connector_mask: u32,
    encoder_mask: u32,
    adjusted_mode: drm_display_mode,
    mode: drm_display_mode,
    mode_blob: *mut drm_property_blob,
    degamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    background_color: u64,
    target_vblank: u32,
    async_flip: bool,
    vrr_enabled: bool,
    self_refresh_active: bool,
    scaling_filter: u32,
    sharpness_strength: u8,
    event: *mut drm_pending_vblank_event,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    cursor_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32) -> i32>,
    cursor_set2: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32, i32, i32) -> i32>,
    cursor_move: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32) -> i32>,
    gamma_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u16, *mut u16, *mut u16, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_config: Option<unsafe extern "C" fn(*mut drm_mode_set, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip_target: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_crtc_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8) -> i32>,
    verify_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8, *mut u64) -> i32>,
    get_crc_sources: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u64) -> *const *const u8>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_crtc_state)>,
    get_vblank_counter: Option<unsafe extern "C" fn(*mut drm_crtc) -> u32>,
    enable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    disable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    get_vblank_timestamp: Option<unsafe extern "C" fn(*mut drm_crtc, *mut i32, *mut i64, bool) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc {
    dev: *mut drm_device,
    port: *mut device_node,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    primary: *mut drm_plane,
    cursor: *mut drm_plane,
    index: u32,
    cursor_x: i32,
    cursor_y: i32,
    enabled: bool,
    mode: drm_display_mode,
    hwmode: drm_display_mode,
    x: i32,
    y: i32,
    funcs: *const drm_crtc_funcs,
    gamma_size: u32,
    gamma_store: *mut u16,
    helper_private: *const drm_crtc_helper_funcs,
    properties: drm_object_properties,
    scaling_filter_property: *mut drm_property,
    sharpness_strength_property: *mut drm_property,
    state: *mut drm_crtc_state,
    commit_list: list_head,
    commit_lock: spinlock,
    debugfs_entry: *mut dentry,
    crc: drm_crtc_crc,
    fence_context: u32,
    fence_lock: spinlock,
    fence_seqno: u64,
    timeline_name: [u8; 32],
    self_refresh_data: *mut drm_self_refresh_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_set {
    fb: *mut drm_framebuffer,
    crtc: *mut drm_crtc,
    mode: *mut drm_display_mode,
    x: u32,
    y: u32,
    connectors: *mut *mut drm_connector,
    num_connectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    late_register: Option<unsafe extern "C" fn(*mut drm_encoder) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_encoder, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    name: *mut u8,
    encoder_type: i32,
    index: u32,
    possible_crtcs: u32,
    possible_clones: u32,
    crtc: *mut drm_crtc,
    bridge_chain: list_head,
    bridge_chain_mutex: mutex,
    funcs: *const drm_encoder_funcs,
    helper_private: *const drm_encoder_helper_funcs,
    debugfs_entry: *mut dentry,
}


fn drm_debug_enabled_raw(category: u32) -> bool {
unlikely__3!((unsafe { __drm_debug }) & BIT__2!(category)) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_printer {
    printfn: Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>,
    puts: Option<unsafe extern "C" fn(*mut drm_printer, *const u8)>,
    arg: *mut (),
    origin: *const (),
    prefix: *const u8,
    line: __SlateRecord_drm_print_h_181,
    category: u32,
}


fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// NOTE: this is deprecated in favor of drm_dbg_kms(NULL, ...).
// Helper for struct drm_device based WARNs
fn drm_edid_load_firmware(connector: *mut drm_connector) -> *const drm_edid {
ERR_PTR(2i32.wrapping_neg() as i64) as *const drm_edid
}

#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_block {
    tag: u8,
    rev: u8,
    num_bytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_tiled_block {
    base: displayid_block,
    tile_cap: u8,
    topo: [u8; 3],
    tile_size: [u8; 4],
    tile_pixel_bezel: [u8; 5],
    topology_id: [u8; 9],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct displayid_detailed_timings_1 {
    pixel_clock: [u8; 3],
    flags: u8,
    hactive: u16,
    hblank: u16,
    hsync: u16,
    hsw: u16,
    vactive: u16,
    vblank: u16,
    vsync: u16,
    vsw: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_detailed_timing_block {
    base: displayid_block,
    timings: [displayid_detailed_timings_1; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct displayid_formula_timings_9 {
    flags: u8,
    hactive: u16,
    vactive: u16,
    vrefresh: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_formula_timing_block {
    base: displayid_block,
    timings: [displayid_formula_timings_9; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct displayid_display_params_block {
    base: displayid_block,
    horiz_image_size: u16,
    vert_image_size: u16,
    horiz_pixel_count: u16,
    vert_pixel_count: u16,
    features: u8,
    primary_color1: [u8; 3],
    primary_color2: [u8; 3],
    primary_color3: [u8; 3],
    white_point: [u8; 3],
    max_luminance_full: u16,
    max_luminance_10: u16,
    min_luminance: u16,
    color_depth_and_tech: u8,
    gamma_eotf: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_vesa_vendor_specific_block {
    base: displayid_block,
    oui: [u8; 3],
    data_structure_type: u8,
    mso: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct displayid_iter {
    drm_edid: *const drm_edid,
    section: *const u8,
    length: i32,
    idx: i32,
    ext_index: i32,
    version: u8,
    primary_use: u8,
    quirks: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_ioctl_desc {
    cmd: u32,
    flags: u32,
    func: Option<unsafe extern "C" fn(*mut drm_device, *mut (), *mut drm_file) -> i32>,
    name: *const u8,
}


// The rest seem to be more-or-less nonstandard. Check them!
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_prime_file_private {
    lock: mutex,
    dmabufs: rb_root,
    handles: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
    index: i32,
    r#type: i32,
    kdev: *mut device,
    dev: *mut drm_device,
    debugfs_symlink: *mut dentry,
    debugfs_root: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_event {
    completion: *mut completion,
    completion_release: Option<unsafe extern "C" fn(*mut completion)>,
    event: *mut drm_event,
    fence: *mut dma_fence,
    file_priv: *mut drm_file,
    link: list_head,
    pending_link: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_file {
    authenticated: bool,
    stereo_allowed: bool,
    universal_planes: bool,
    atomic: bool,
    aspect_ratio_allowed: bool,
    writeback_connectors: bool,
    plane_color_pipeline: bool,
    was_master: bool,
    is_master: bool,
    supports_virtualized_cursor_plane: bool,
    master: *mut drm_master,
    master_lookup_lock: spinlock,
    pid: *mut pid,
    client_id: u64,
    magic: u32,
    lhead: list_head,
    minor: *mut drm_minor,
    object_idr: idr,
    table_lock: spinlock,
    syncobj_xa: xarray,
    filp: *mut file,
    driver_priv: *mut (),
    fbs: list_head,
    fbs_lock: mutex,
    blobs: list_head,
    event_wait: wait_queue_head,
    pending_event_list: list_head,
    event_list: list_head,
    event_space: i32,
    event_read_lock: mutex,
    prime: drm_prime_file_private,
    client_name: *const u8,
    client_name_lock: mutex,
    debugfs_client: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
    base: drm_pending_event,
    pipe: u32,
    sequence: u64,
    event: __SlateRecord_drm_vblank_h_59,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_config {
    offdelay_ms: i32,
    disable_immediate: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_timer {
    timer: hrtimer,
    interval_lock: spinlock,
    interval: i64,
    crtc: *mut drm_crtc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
    dev: *mut drm_device,
    queue: wait_queue_head,
    disable_timer: timer_list,
    seqlock: seqlock,
    count: __SlateRecord_types_h_197,
    time: i64,
    refcount: __SlateRecord_types_h_190,
    last: u32,
    max_vblank_count: u32,
    inmodeset: u32,
    pipe: u32,
    framedur_ns: i32,
    linedur_ns: i32,
    hwmode: drm_display_mode,
    config: drm_vblank_crtc_config,
    enabled: bool,
    worker: *mut kthread_worker,
    pending_work: list_head,
    work_wait_queue: wait_queue_head,
    vblank_timer: drm_vblank_crtc_timer,
}


fn oui(first: u8, second: u8, third: u8) -> i32 {
(first as i32) << 16i32 | (second as i32) << 8i32 | (third as i32)
}

// EDID blocks out in the wild have a variety of bugs, try to collect
// them here (note that userspace may work around broken monitors first,
// but fixes should make their way here so that the kernel "just works"
// on as many displays as possible).
#[repr(C)]
#[derive(Clone, Copy)]
struct amd_vsdb_v3_payload {
    oui: [u8; 3],
    version: u8,
    feature_caps: u8,
    rsvd0: [u8; 3],
    cs_eotf_support: u8,
    lum1_max: u8,
    lum1_min: u8,
    lum2_max: u8,
    lum2_min: u8,
    rsvd1: [u8; 2],
    /// Bytes beyond AMD_VSDB_V3_PAYLOAD_MIN_LEN are optional; a
    /// monitor may provide a payload as short as 15 bytes.  Always
    /// check cea_db_payload_len() before accessing extra[].
    extra: [u8; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_mode_closure {
    connector: *mut drm_connector,
    drm_edid: *const drm_edid,
    preferred: bool,
    modes: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid_match_closure {
    ident: *const drm_edid_ident,
    matched: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid_quirk {
    ident: drm_edid_ident,
    quirks: u32,
}


/// Acer AL1706
/// Acer F51
/// AEO model 0 reports 8 bpc, but is a 6 bpc panel
/// BenQ GW2765
/// BOE model on HP Pavilion 15-n233sl reports 8 bpc, but is a 6 bpc panel
/// CPT panel of Asus UX303LA reports 8 bpc, but is a 6 bpc panel
/// SDC panel of Lenovo B50-80 reports 8 bpc, but is a 6 bpc panel
/// BOE model 0x0771 reports 8 bpc, but is a 6 bpc panel
/// Belinea 10 15 55
/// Envision Peripherals, Inc. EN-7100e
/// Envision EN2028
/// Funai Electronics PM36B
/// LG 27GP950
/// LG 27GN950
/// LGD panel of HP zBook 17 G2, eDP 10 bpc, but reports unknown bpc
/// LG Philips LCD LP154W01-A5
/// Samsung SyncMaster 205BW.  Note: irony
/// Samsung SyncMaster 22[5-6]BW
/// Sony PVM-2541A does up to 12 bpc, but only reports max 8 bpc
/// ViewSonic VA2026w
/// Medion MD 30217 PG
/// Lenovo G50
/// Panel in Samsung NP700G7A-S01PL notebook reports 6bpc
/// Rotel RSX-1058 forwards sink's EDID but only does HDMI 1.1
/// Valve Index Headset
/// HTC Vive and Vive Pro VR Headsets
/// Oculus Rift DK1, DK2, CV1 and Rift S VR Headsets
/// Windows Mixed Reality Headsets
/// Sony PlayStation VR Headset
/// Sensics VR Headsets
/// OSVR HDK and HDK2 VR Headsets
/// LQ116M1JW10 displays noise when 8 bpc, but display fine as 6 bpc
/// @drm_edid_internal_quirk entries end here, following with the
/// @drm_edid_quirk entries.
/// HP ZR24w DP AUX DPCD access requires probing to prevent corruption.
static edid_quirk_list: __SlateAlign16<[edid_quirk; 62]> = __SlateAlign16([{
let mut __t0: edid_quirk = unsafe { core::mem::zeroed() };
__t0.ident = {
let mut __t1: drm_edid_ident = unsafe { core::mem::zeroed() };
__t1.panel_id = 74624326u32;
__t1
};
__t0.quirks = 2u32;
__t0
}, {
let mut __t2: edid_quirk = unsafe { core::mem::zeroed() };
__t2.ident = {
let mut __t3: drm_edid_ident = unsafe { core::mem::zeroed() };
__t3.panel_id = 101283330u32;
__t3
};
__t2.quirks = 2u32;
__t2
}, {
let mut __t4: edid_quirk = unsafe { core::mem::zeroed() };
__t4.ident = {
let mut __t5: drm_edid_ident = unsafe { core::mem::zeroed() };
__t5.panel_id = 78577664u32;
__t5
};
__t4.quirks = 1024u32;
__t4
}, {
let mut __t6: edid_quirk = unsafe { core::mem::zeroed() };
__t6.ident = {
let mut __t7: drm_edid_ident = unsafe { core::mem::zeroed() };
__t7.panel_id = 164722902u32;
__t7
};
__t6.quirks = 256u32;
__t6
}, {
let mut __t8: edid_quirk = unsafe { core::mem::zeroed() };
__t8.ident = {
let mut __t9: drm_edid_ident = unsafe { core::mem::zeroed() };
__t9.panel_id = 166004619u32;
__t9
};
__t8.quirks = 1024u32;
__t8
}, {
let mut __t10: edid_quirk = unsafe { core::mem::zeroed() };
__t10.ident = {
let mut __t11: drm_edid_ident = unsafe { core::mem::zeroed() };
__t11.panel_id = 236197855u32;
__t11
};
__t10.quirks = 1024u32;
__t10
}, {
let mut __t12: edid_quirk = unsafe { core::mem::zeroed() };
__t12.ident = {
let mut __t13: drm_edid_ident = unsafe { core::mem::zeroed() };
__t13.panel_id = 1283667538u32;
__t13
};
__t12.quirks = 1024u32;
__t12
}, {
let mut __t14: edid_quirk = unsafe { core::mem::zeroed() };
__t14.ident = {
let mut __t15: drm_edid_ident = unsafe { core::mem::zeroed() };
__t15.panel_id = 166004593u32;
__t15
};
__t14.quirks = 1024u32;
__t14
}, {
let mut __t16: edid_quirk = unsafe { core::mem::zeroed() };
__t16.ident = {
let mut __t17: drm_edid_ident = unsafe { core::mem::zeroed() };
__t17.panel_id = 876086764u32;
__t17
};
__t16.quirks = 2u32;
__t16
}, {
let mut __t18: edid_quirk = unsafe { core::mem::zeroed() };
__t18.ident = {
let mut __t19: drm_edid_ident = unsafe { core::mem::zeroed() };
__t19.panel_id = 876087166u32;
__t19
};
__t18.quirks = 2u32;
__t18
}, {
let mut __t20: edid_quirk = unsafe { core::mem::zeroed() };
__t20.ident = {
let mut __t21: drm_edid_ident = unsafe { core::mem::zeroed() };
__t21.panel_id = 369747840u32;
__t21
};
__t20.quirks = 4u32;
__t20
}, {
let mut __t22: edid_quirk = unsafe { core::mem::zeroed() };
__t22.ident = {
let mut __t23: drm_edid_ident = unsafe { core::mem::zeroed() };
__t23.panel_id = 369696808u32;
__t23
};
__t22.quirks = 2u32;
__t22
}, {
let mut __t24: edid_quirk = unsafe { core::mem::zeroed() };
__t24.ident = {
let mut __t25: drm_edid_ident = unsafe { core::mem::zeroed() };
__t25.panel_id = 409810208u32;
__t25
};
__t24.quirks = 24u32;
__t24
}, {
let mut __t26: edid_quirk = unsafe { core::mem::zeroed() };
__t26.ident = {
let mut __t27: drm_edid_ident = unsafe { core::mem::zeroed() };
__t27.panel_id = 510483391u32;
__t27
};
__t26.quirks = 8192u32;
__t26
}, {
let mut __t28: edid_quirk = unsafe { core::mem::zeroed() };
__t28.ident = {
let mut __t29: drm_edid_ident = unsafe { core::mem::zeroed() };
__t29.panel_id = 510483354u32;
__t29
};
__t28.quirks = 8192u32;
__t28
}, {
let mut __t30: edid_quirk = unsafe { core::mem::zeroed() };
__t30.ident = {
let mut __t31: drm_edid_ident = unsafe { core::mem::zeroed() };
__t31.panel_id = 820249340u32;
__t31
};
__t30.quirks = 2048u32;
__t30
}, {
let mut __t32: edid_quirk = unsafe { core::mem::zeroed() };
__t32.ident = {
let mut __t33: drm_edid_ident = unsafe { core::mem::zeroed() };
__t33.panel_id = 839647232u32;
__t33
};
__t32.quirks = 32u32;
__t32
}, {
let mut __t34: edid_quirk = unsafe { core::mem::zeroed() };
__t34.ident = {
let mut __t35: drm_edid_ident = unsafe { core::mem::zeroed() };
__t35.panel_id = 839657984u32;
__t35
};
__t34.quirks = 32u32;
__t34
}, {
let mut __t36: edid_quirk = unsafe { core::mem::zeroed() };
__t36.ident = {
let mut __t37: drm_edid_ident = unsafe { core::mem::zeroed() };
__t37.panel_id = 1278018077u32;
__t37
};
__t36.quirks = 64u32;
__t36
}, {
let mut __t38: edid_quirk = unsafe { core::mem::zeroed() };
__t38.ident = {
let mut __t39: drm_edid_ident = unsafe { core::mem::zeroed() };
__t39.panel_id = 1278018132u32;
__t39
};
__t38.quirks = 2u32;
__t38
}, {
let mut __t40: edid_quirk = unsafe { core::mem::zeroed() };
__t40.ident = {
let mut __t41: drm_edid_ident = unsafe { core::mem::zeroed() };
__t41.panel_id = 1278018174u32;
__t41
};
__t40.quirks = 2u32;
__t40
}, {
let mut __t42: edid_quirk = unsafe { core::mem::zeroed() };
__t42.ident = {
let mut __t43: drm_edid_ident = unsafe { core::mem::zeroed() };
__t43.panel_id = 1306076481u32;
__t43
};
__t42.quirks = 512u32;
__t42
}, {
let mut __t44: edid_quirk = unsafe { core::mem::zeroed() };
__t44.ident = {
let mut __t45: drm_edid_ident = unsafe { core::mem::zeroed() };
__t45.panel_id = 1516442524u32;
__t45
};
__t44.quirks = 128u32;
__t44
}, {
let mut __t46: edid_quirk = unsafe { core::mem::zeroed() };
__t46.ident = {
let mut __t47: drm_edid_ident = unsafe { core::mem::zeroed() };
__t47.panel_id = 883165112u32;
__t47
};
__t46.quirks = 8u32;
__t46
}, {
let mut __t48: edid_quirk = unsafe { core::mem::zeroed() };
__t48.ident = {
let mut __t49: drm_edid_ident = unsafe { core::mem::zeroed() };
__t49.panel_id = 1283672146u32;
__t49
};
__t48.quirks = 1024u32;
__t48
}, {
let mut __t50: edid_quirk = unsafe { core::mem::zeroed() };
__t50.ident = {
let mut __t51: drm_edid_ident = unsafe { core::mem::zeroed() };
__t51.panel_id = 1285804083u32;
__t51
};
__t50.quirks = 256u32;
__t50
}, {
let mut __t52: edid_quirk = unsafe { core::mem::zeroed() };
__t52.ident = {
let mut __t53: drm_edid_ident = unsafe { core::mem::zeroed() };
__t53.panel_id = 378680904u32;
__t53
};
__t52.quirks = 256u32;
__t52
}, {
let mut __t54: edid_quirk = unsafe { core::mem::zeroed() };
__t54.ident = {
let mut __t55: drm_edid_ident = unsafe { core::mem::zeroed() };
__t55.panel_id = 1503039912u32;
__t55
};
__t54.quirks = 4096u32;
__t54
}, {
let mut __t56: edid_quirk = unsafe { core::mem::zeroed() };
__t56.ident = {
let mut __t57: drm_edid_ident = unsafe { core::mem::zeroed() };
__t57.panel_id = 1503039920u32;
__t57
};
__t56.quirks = 4096u32;
__t56
}, {
let mut __t58: edid_quirk = unsafe { core::mem::zeroed() };
__t58.ident = {
let mut __t59: drm_edid_ident = unsafe { core::mem::zeroed() };
__t59.panel_id = 1503039921u32;
__t59
};
__t58.quirks = 4096u32;
__t58
}, {
let mut __t60: edid_quirk = unsafe { core::mem::zeroed() };
__t60.ident = {
let mut __t61: drm_edid_ident = unsafe { core::mem::zeroed() };
__t61.panel_id = 1503039922u32;
__t61
};
__t60.quirks = 4096u32;
__t60
}, {
let mut __t62: edid_quirk = unsafe { core::mem::zeroed() };
__t62.ident = {
let mut __t63: drm_edid_ident = unsafe { core::mem::zeroed() };
__t63.panel_id = 1503039923u32;
__t63
};
__t62.quirks = 4096u32;
__t62
}, {
let mut __t64: edid_quirk = unsafe { core::mem::zeroed() };
__t64.ident = {
let mut __t65: drm_edid_ident = unsafe { core::mem::zeroed() };
__t65.panel_id = 1503039924u32;
__t65
};
__t64.quirks = 4096u32;
__t64
}, {
let mut __t66: edid_quirk = unsafe { core::mem::zeroed() };
__t66.ident = {
let mut __t67: drm_edid_ident = unsafe { core::mem::zeroed() };
__t67.panel_id = 1503039925u32;
__t67
};
__t66.quirks = 4096u32;
__t66
}, {
let mut __t68: edid_quirk = unsafe { core::mem::zeroed() };
__t68.ident = {
let mut __t69: drm_edid_ident = unsafe { core::mem::zeroed() };
__t69.panel_id = 1503039926u32;
__t69
};
__t68.quirks = 4096u32;
__t68
}, {
let mut __t70: edid_quirk = unsafe { core::mem::zeroed() };
__t70.ident = {
let mut __t71: drm_edid_ident = unsafe { core::mem::zeroed() };
__t71.panel_id = 1503039927u32;
__t71
};
__t70.quirks = 4096u32;
__t70
}, {
let mut __t72: edid_quirk = unsafe { core::mem::zeroed() };
__t72.ident = {
let mut __t73: drm_edid_ident = unsafe { core::mem::zeroed() };
__t73.panel_id = 1503039928u32;
__t73
};
__t72.quirks = 4096u32;
__t72
}, {
let mut __t74: edid_quirk = unsafe { core::mem::zeroed() };
__t74.ident = {
let mut __t75: drm_edid_ident = unsafe { core::mem::zeroed() };
__t75.panel_id = 1503039929u32;
__t75
};
__t74.quirks = 4096u32;
__t74
}, {
let mut __t76: edid_quirk = unsafe { core::mem::zeroed() };
__t76.ident = {
let mut __t77: drm_edid_ident = unsafe { core::mem::zeroed() };
__t77.panel_id = 1503039930u32;
__t77
};
__t76.quirks = 4096u32;
__t76
}, {
let mut __t78: edid_quirk = unsafe { core::mem::zeroed() };
__t78.ident = {
let mut __t79: drm_edid_ident = unsafe { core::mem::zeroed() };
__t79.panel_id = 1503039931u32;
__t79
};
__t78.quirks = 4096u32;
__t78
}, {
let mut __t80: edid_quirk = unsafe { core::mem::zeroed() };
__t80.ident = {
let mut __t81: drm_edid_ident = unsafe { core::mem::zeroed() };
__t81.panel_id = 1503039932u32;
__t81
};
__t80.quirks = 4096u32;
__t80
}, {
let mut __t82: edid_quirk = unsafe { core::mem::zeroed() };
__t82.ident = {
let mut __t83: drm_edid_ident = unsafe { core::mem::zeroed() };
__t83.panel_id = 1503039933u32;
__t83
};
__t82.quirks = 4096u32;
__t82
}, {
let mut __t84: edid_quirk = unsafe { core::mem::zeroed() };
__t84.ident = {
let mut __t85: drm_edid_ident = unsafe { core::mem::zeroed() };
__t85.panel_id = 1503039934u32;
__t85
};
__t84.quirks = 4096u32;
__t84
}, {
let mut __t86: edid_quirk = unsafe { core::mem::zeroed() };
__t86.ident = {
let mut __t87: drm_edid_ident = unsafe { core::mem::zeroed() };
__t87.panel_id = 1503039935u32;
__t87
};
__t86.quirks = 4096u32;
__t86
}, {
let mut __t88: edid_quirk = unsafe { core::mem::zeroed() };
__t88.ident = {
let mut __t89: drm_edid_ident = unsafe { core::mem::zeroed() };
__t89.panel_id = 584231425u32;
__t89
};
__t88.quirks = 4096u32;
__t88
}, {
let mut __t90: edid_quirk = unsafe { core::mem::zeroed() };
__t90.ident = {
let mut __t91: drm_edid_ident = unsafe { core::mem::zeroed() };
__t91.panel_id = 584231426u32;
__t91
};
__t90.quirks = 4096u32;
__t90
}, {
let mut __t92: edid_quirk = unsafe { core::mem::zeroed() };
__t92.ident = {
let mut __t93: drm_edid_ident = unsafe { core::mem::zeroed() };
__t93.panel_id = 1053949953u32;
__t93
};
__t92.quirks = 4096u32;
__t92
}, {
let mut __t94: edid_quirk = unsafe { core::mem::zeroed() };
__t94.ident = {
let mut __t95: drm_edid_ident = unsafe { core::mem::zeroed() };
__t95.panel_id = 1053949955u32;
__t95
};
__t94.quirks = 4096u32;
__t94
}, {
let mut __t96: edid_quirk = unsafe { core::mem::zeroed() };
__t96.ident = {
let mut __t97: drm_edid_ident = unsafe { core::mem::zeroed() };
__t97.panel_id = 1053949956u32;
__t97
};
__t96.quirks = 4096u32;
__t96
}, {
let mut __t98: edid_quirk = unsafe { core::mem::zeroed() };
__t98.ident = {
let mut __t99: drm_edid_ident = unsafe { core::mem::zeroed() };
__t99.panel_id = 1053949970u32;
__t99
};
__t98.quirks = 4096u32;
__t98
}, {
let mut __t100: edid_quirk = unsafe { core::mem::zeroed() };
__t100.ident = {
let mut __t101: drm_edid_ident = unsafe { core::mem::zeroed() };
__t101.panel_id = 74612686u32;
__t101
};
__t100.quirks = 4096u32;
__t100
}, {
let mut __t102: edid_quirk = unsafe { core::mem::zeroed() };
__t102.ident = {
let mut __t103: drm_edid_ident = unsafe { core::mem::zeroed() };
__t103.panel_id = 816710664u32;
__t103
};
__t102.quirks = 4096u32;
__t102
}, {
let mut __t104: edid_quirk = unsafe { core::mem::zeroed() };
__t104.ident = {
let mut __t105: drm_edid_ident = unsafe { core::mem::zeroed() };
__t105.panel_id = 447355248u32;
__t105
};
__t104.quirks = 4096u32;
__t104
}, {
let mut __t106: edid_quirk = unsafe { core::mem::zeroed() };
__t106.ident = {
let mut __t107: drm_edid_ident = unsafe { core::mem::zeroed() };
__t107.panel_id = 279740366u32;
__t107
};
__t106.quirks = 4096u32;
__t106
}, {
let mut __t108: edid_quirk = unsafe { core::mem::zeroed() };
__t108.ident = {
let mut __t109: drm_edid_ident = unsafe { core::mem::zeroed() };
__t109.panel_id = 1285755978u32;
__t109
};
__t108.quirks = 4096u32;
__t108
}, {
let mut __t110: edid_quirk = unsafe { core::mem::zeroed() };
__t110.ident = {
let mut __t111: drm_edid_ident = unsafe { core::mem::zeroed() };
__t111.panel_id = 112443650u32;
__t111
};
__t110.quirks = 4096u32;
__t110
}, {
let mut __t112: edid_quirk = unsafe { core::mem::zeroed() };
__t112.ident = {
let mut __t113: drm_edid_ident = unsafe { core::mem::zeroed() };
__t113.panel_id = 1306068740u32;
__t113
};
__t112.quirks = 4096u32;
__t112
}, {
let mut __t114: edid_quirk = unsafe { core::mem::zeroed() };
__t114.ident = {
let mut __t115: drm_edid_ident = unsafe { core::mem::zeroed() };
__t115.panel_id = 1286475801u32;
__t115
};
__t114.quirks = 4096u32;
__t114
}, {
let mut __t116: edid_quirk = unsafe { core::mem::zeroed() };
__t116.ident = {
let mut __t117: drm_edid_ident = unsafe { core::mem::zeroed() };
__t117.panel_id = 1322389529u32;
__t117
};
__t116.quirks = 4096u32;
__t116
}, {
let mut __t118: edid_quirk = unsafe { core::mem::zeroed() };
__t118.ident = {
let mut __t119: drm_edid_ident = unsafe { core::mem::zeroed() };
__t119.panel_id = 112136465u32;
__t119
};
__t118.quirks = 4096u32;
__t118
}, {
let mut __t120: edid_quirk = unsafe { core::mem::zeroed() };
__t120.ident = {
let mut __t121: drm_edid_ident = unsafe { core::mem::zeroed() };
__t121.panel_id = 1292899660u32;
__t121
};
__t120.quirks = 1024u32;
__t120
}, {
let mut __t122: edid_quirk = unsafe { core::mem::zeroed() };
__t122.ident = {
let mut __t123: drm_edid_ident = unsafe { core::mem::zeroed() };
__t123.panel_id = 586164329u32;
__t123
};
__t122.quirks = 1u32;
__t122
}]);

/// Autogenerated from the DMT spec.
/// This table is copied from xfree86/modes/xf86EdidModes.c.
/// 0x01 - 640x350@85Hz
/// 0x02 - 640x400@85Hz
/// 0x03 - 720x400@85Hz
/// 0x04 - 640x480@60Hz
/// 0x05 - 640x480@72Hz
/// 0x06 - 640x480@75Hz
/// 0x07 - 640x480@85Hz
/// 0x08 - 800x600@56Hz
/// 0x09 - 800x600@60Hz
/// 0x0a - 800x600@72Hz
/// 0x0b - 800x600@75Hz
/// 0x0c - 800x600@85Hz
/// 0x0d - 800x600@120Hz RB
/// 0x0e - 848x480@60Hz
/// 0x0f - 1024x768@43Hz, interlace
/// 0x10 - 1024x768@60Hz
/// 0x11 - 1024x768@70Hz
/// 0x12 - 1024x768@75Hz
/// 0x13 - 1024x768@85Hz
/// 0x14 - 1024x768@120Hz RB
/// 0x15 - 1152x864@75Hz
/// 0x55 - 1280x720@60Hz
/// 0x16 - 1280x768@60Hz RB
/// 0x17 - 1280x768@60Hz
/// 0x18 - 1280x768@75Hz
/// 0x19 - 1280x768@85Hz
/// 0x1a - 1280x768@120Hz RB
/// 0x1b - 1280x800@60Hz RB
/// 0x1c - 1280x800@60Hz
/// 0x1d - 1280x800@75Hz
/// 0x1e - 1280x800@85Hz
/// 0x1f - 1280x800@120Hz RB
/// 0x20 - 1280x960@60Hz
/// 0x21 - 1280x960@85Hz
/// 0x22 - 1280x960@120Hz RB
/// 0x23 - 1280x1024@60Hz
/// 0x24 - 1280x1024@75Hz
/// 0x25 - 1280x1024@85Hz
/// 0x26 - 1280x1024@120Hz RB
/// 0x27 - 1360x768@60Hz
/// 0x28 - 1360x768@120Hz RB
/// 0x51 - 1366x768@60Hz
/// 0x56 - 1366x768@60Hz
/// 0x29 - 1400x1050@60Hz RB
/// 0x2a - 1400x1050@60Hz
/// 0x2b - 1400x1050@75Hz
/// 0x2c - 1400x1050@85Hz
/// 0x2d - 1400x1050@120Hz RB
/// 0x2e - 1440x900@60Hz RB
/// 0x2f - 1440x900@60Hz
/// 0x30 - 1440x900@75Hz
/// 0x31 - 1440x900@85Hz
/// 0x32 - 1440x900@120Hz RB
/// 0x53 - 1600x900@60Hz
/// 0x33 - 1600x1200@60Hz
/// 0x34 - 1600x1200@65Hz
/// 0x35 - 1600x1200@70Hz
/// 0x36 - 1600x1200@75Hz
/// 0x37 - 1600x1200@85Hz
/// 0x38 - 1600x1200@120Hz RB
/// 0x39 - 1680x1050@60Hz RB
/// 0x3a - 1680x1050@60Hz
/// 0x3b - 1680x1050@75Hz
/// 0x3c - 1680x1050@85Hz
/// 0x3d - 1680x1050@120Hz RB
/// 0x3e - 1792x1344@60Hz
/// 0x3f - 1792x1344@75Hz
/// 0x40 - 1792x1344@120Hz RB
/// 0x41 - 1856x1392@60Hz
/// 0x42 - 1856x1392@75Hz
/// 0x43 - 1856x1392@120Hz RB
/// 0x52 - 1920x1080@60Hz
/// 0x44 - 1920x1200@60Hz RB
/// 0x45 - 1920x1200@60Hz
/// 0x46 - 1920x1200@75Hz
/// 0x47 - 1920x1200@85Hz
/// 0x48 - 1920x1200@120Hz RB
/// 0x49 - 1920x1440@60Hz
/// 0x4a - 1920x1440@75Hz
/// 0x4b - 1920x1440@120Hz RB
/// 0x54 - 2048x1152@60Hz
/// 0x4c - 2560x1600@60Hz RB
/// 0x4d - 2560x1600@60Hz
/// 0x4e - 2560x1600@75Hz
/// 0x4f - 2560x1600@85Hz
/// 0x50 - 2560x1600@120Hz RB
/// 0x57 - 4096x2160@60Hz RB
/// 0x58 - 4096x2160@59.94Hz RB
static drm_dmt_modes: __SlateAlign16<[drm_display_mode; 88]> = __SlateAlign16([{
let mut __t0: drm_display_mode = unsafe { core::mem::zeroed() };
__t0.clock = 31500i32;
__t0.hdisplay = 640u16;
__t0.hsync_start = 672u16;
__t0.hsync_end = 736u16;
__t0.htotal = 832u16;
__t0.hskew = 0u16;
__t0.vdisplay = 350u16;
__t0.vsync_start = 382u16;
__t0.vsync_end = 385u16;
__t0.vtotal = 445u16;
__t0.vscan = 0u16;
__t0.flags = 9u32;
__t0.r#type = 64u8;
__t0.name = [54u8, 52u8, 48u8, 120u8, 51u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.status = 0i32;
__t0
}, {
let mut __t1: drm_display_mode = unsafe { core::mem::zeroed() };
__t1.clock = 31500i32;
__t1.hdisplay = 640u16;
__t1.hsync_start = 672u16;
__t1.hsync_end = 736u16;
__t1.htotal = 832u16;
__t1.hskew = 0u16;
__t1.vdisplay = 400u16;
__t1.vsync_start = 401u16;
__t1.vsync_end = 404u16;
__t1.vtotal = 445u16;
__t1.vscan = 0u16;
__t1.flags = 6u32;
__t1.r#type = 64u8;
__t1.name = [54u8, 52u8, 48u8, 120u8, 52u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.status = 0i32;
__t1
}, {
let mut __t2: drm_display_mode = unsafe { core::mem::zeroed() };
__t2.clock = 35500i32;
__t2.hdisplay = 720u16;
__t2.hsync_start = 756u16;
__t2.hsync_end = 828u16;
__t2.htotal = 936u16;
__t2.hskew = 0u16;
__t2.vdisplay = 400u16;
__t2.vsync_start = 401u16;
__t2.vsync_end = 404u16;
__t2.vtotal = 446u16;
__t2.vscan = 0u16;
__t2.flags = 6u32;
__t2.r#type = 64u8;
__t2.name = [55u8, 50u8, 48u8, 120u8, 52u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.status = 0i32;
__t2
}, {
let mut __t3: drm_display_mode = unsafe { core::mem::zeroed() };
__t3.clock = 25175i32;
__t3.hdisplay = 640u16;
__t3.hsync_start = 656u16;
__t3.hsync_end = 752u16;
__t3.htotal = 800u16;
__t3.hskew = 0u16;
__t3.vdisplay = 480u16;
__t3.vsync_start = 490u16;
__t3.vsync_end = 492u16;
__t3.vtotal = 525u16;
__t3.vscan = 0u16;
__t3.flags = 10u32;
__t3.r#type = 64u8;
__t3.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.status = 0i32;
__t3
}, {
let mut __t4: drm_display_mode = unsafe { core::mem::zeroed() };
__t4.clock = 31500i32;
__t4.hdisplay = 640u16;
__t4.hsync_start = 664u16;
__t4.hsync_end = 704u16;
__t4.htotal = 832u16;
__t4.hskew = 0u16;
__t4.vdisplay = 480u16;
__t4.vsync_start = 489u16;
__t4.vsync_end = 492u16;
__t4.vtotal = 520u16;
__t4.vscan = 0u16;
__t4.flags = 10u32;
__t4.r#type = 64u8;
__t4.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.status = 0i32;
__t4
}, {
let mut __t5: drm_display_mode = unsafe { core::mem::zeroed() };
__t5.clock = 31500i32;
__t5.hdisplay = 640u16;
__t5.hsync_start = 656u16;
__t5.hsync_end = 720u16;
__t5.htotal = 840u16;
__t5.hskew = 0u16;
__t5.vdisplay = 480u16;
__t5.vsync_start = 481u16;
__t5.vsync_end = 484u16;
__t5.vtotal = 500u16;
__t5.vscan = 0u16;
__t5.flags = 10u32;
__t5.r#type = 64u8;
__t5.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5.status = 0i32;
__t5
}, {
let mut __t6: drm_display_mode = unsafe { core::mem::zeroed() };
__t6.clock = 36000i32;
__t6.hdisplay = 640u16;
__t6.hsync_start = 696u16;
__t6.hsync_end = 752u16;
__t6.htotal = 832u16;
__t6.hskew = 0u16;
__t6.vdisplay = 480u16;
__t6.vsync_start = 481u16;
__t6.vsync_end = 484u16;
__t6.vtotal = 509u16;
__t6.vscan = 0u16;
__t6.flags = 10u32;
__t6.r#type = 64u8;
__t6.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.status = 0i32;
__t6
}, {
let mut __t7: drm_display_mode = unsafe { core::mem::zeroed() };
__t7.clock = 36000i32;
__t7.hdisplay = 800u16;
__t7.hsync_start = 824u16;
__t7.hsync_end = 896u16;
__t7.htotal = 1024u16;
__t7.hskew = 0u16;
__t7.vdisplay = 600u16;
__t7.vsync_start = 601u16;
__t7.vsync_end = 603u16;
__t7.vtotal = 625u16;
__t7.vscan = 0u16;
__t7.flags = 5u32;
__t7.r#type = 64u8;
__t7.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.status = 0i32;
__t7
}, {
let mut __t8: drm_display_mode = unsafe { core::mem::zeroed() };
__t8.clock = 40000i32;
__t8.hdisplay = 800u16;
__t8.hsync_start = 840u16;
__t8.hsync_end = 968u16;
__t8.htotal = 1056u16;
__t8.hskew = 0u16;
__t8.vdisplay = 600u16;
__t8.vsync_start = 601u16;
__t8.vsync_end = 605u16;
__t8.vtotal = 628u16;
__t8.vscan = 0u16;
__t8.flags = 5u32;
__t8.r#type = 64u8;
__t8.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8.status = 0i32;
__t8
}, {
let mut __t9: drm_display_mode = unsafe { core::mem::zeroed() };
__t9.clock = 50000i32;
__t9.hdisplay = 800u16;
__t9.hsync_start = 856u16;
__t9.hsync_end = 976u16;
__t9.htotal = 1040u16;
__t9.hskew = 0u16;
__t9.vdisplay = 600u16;
__t9.vsync_start = 637u16;
__t9.vsync_end = 643u16;
__t9.vtotal = 666u16;
__t9.vscan = 0u16;
__t9.flags = 5u32;
__t9.r#type = 64u8;
__t9.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t9.status = 0i32;
__t9
}, {
let mut __t10: drm_display_mode = unsafe { core::mem::zeroed() };
__t10.clock = 49500i32;
__t10.hdisplay = 800u16;
__t10.hsync_start = 816u16;
__t10.hsync_end = 896u16;
__t10.htotal = 1056u16;
__t10.hskew = 0u16;
__t10.vdisplay = 600u16;
__t10.vsync_start = 601u16;
__t10.vsync_end = 604u16;
__t10.vtotal = 625u16;
__t10.vscan = 0u16;
__t10.flags = 5u32;
__t10.r#type = 64u8;
__t10.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10.status = 0i32;
__t10
}, {
let mut __t11: drm_display_mode = unsafe { core::mem::zeroed() };
__t11.clock = 56250i32;
__t11.hdisplay = 800u16;
__t11.hsync_start = 832u16;
__t11.hsync_end = 896u16;
__t11.htotal = 1048u16;
__t11.hskew = 0u16;
__t11.vdisplay = 600u16;
__t11.vsync_start = 601u16;
__t11.vsync_end = 604u16;
__t11.vtotal = 631u16;
__t11.vscan = 0u16;
__t11.flags = 5u32;
__t11.r#type = 64u8;
__t11.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11.status = 0i32;
__t11
}, {
let mut __t12: drm_display_mode = unsafe { core::mem::zeroed() };
__t12.clock = 73250i32;
__t12.hdisplay = 800u16;
__t12.hsync_start = 848u16;
__t12.hsync_end = 880u16;
__t12.htotal = 960u16;
__t12.hskew = 0u16;
__t12.vdisplay = 600u16;
__t12.vsync_start = 603u16;
__t12.vsync_end = 607u16;
__t12.vtotal = 636u16;
__t12.vscan = 0u16;
__t12.flags = 9u32;
__t12.r#type = 64u8;
__t12.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12.status = 0i32;
__t12
}, {
let mut __t13: drm_display_mode = unsafe { core::mem::zeroed() };
__t13.clock = 33750i32;
__t13.hdisplay = 848u16;
__t13.hsync_start = 864u16;
__t13.hsync_end = 976u16;
__t13.htotal = 1088u16;
__t13.hskew = 0u16;
__t13.vdisplay = 480u16;
__t13.vsync_start = 486u16;
__t13.vsync_end = 494u16;
__t13.vtotal = 517u16;
__t13.vscan = 0u16;
__t13.flags = 5u32;
__t13.r#type = 64u8;
__t13.name = [56u8, 52u8, 56u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13.status = 0i32;
__t13
}, {
let mut __t14: drm_display_mode = unsafe { core::mem::zeroed() };
__t14.clock = 44900i32;
__t14.hdisplay = 1024u16;
__t14.hsync_start = 1032u16;
__t14.hsync_end = 1208u16;
__t14.htotal = 1264u16;
__t14.hskew = 0u16;
__t14.vdisplay = 768u16;
__t14.vsync_start = 768u16;
__t14.vsync_end = 776u16;
__t14.vtotal = 817u16;
__t14.vscan = 0u16;
__t14.flags = 21u32;
__t14.r#type = 64u8;
__t14.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14.status = 0i32;
__t14
}, {
let mut __t15: drm_display_mode = unsafe { core::mem::zeroed() };
__t15.clock = 65000i32;
__t15.hdisplay = 1024u16;
__t15.hsync_start = 1048u16;
__t15.hsync_end = 1184u16;
__t15.htotal = 1344u16;
__t15.hskew = 0u16;
__t15.vdisplay = 768u16;
__t15.vsync_start = 771u16;
__t15.vsync_end = 777u16;
__t15.vtotal = 806u16;
__t15.vscan = 0u16;
__t15.flags = 10u32;
__t15.r#type = 64u8;
__t15.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15.status = 0i32;
__t15
}, {
let mut __t16: drm_display_mode = unsafe { core::mem::zeroed() };
__t16.clock = 75000i32;
__t16.hdisplay = 1024u16;
__t16.hsync_start = 1048u16;
__t16.hsync_end = 1184u16;
__t16.htotal = 1328u16;
__t16.hskew = 0u16;
__t16.vdisplay = 768u16;
__t16.vsync_start = 771u16;
__t16.vsync_end = 777u16;
__t16.vtotal = 806u16;
__t16.vscan = 0u16;
__t16.flags = 10u32;
__t16.r#type = 64u8;
__t16.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t16.status = 0i32;
__t16
}, {
let mut __t17: drm_display_mode = unsafe { core::mem::zeroed() };
__t17.clock = 78750i32;
__t17.hdisplay = 1024u16;
__t17.hsync_start = 1040u16;
__t17.hsync_end = 1136u16;
__t17.htotal = 1312u16;
__t17.hskew = 0u16;
__t17.vdisplay = 768u16;
__t17.vsync_start = 769u16;
__t17.vsync_end = 772u16;
__t17.vtotal = 800u16;
__t17.vscan = 0u16;
__t17.flags = 5u32;
__t17.r#type = 64u8;
__t17.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t17.status = 0i32;
__t17
}, {
let mut __t18: drm_display_mode = unsafe { core::mem::zeroed() };
__t18.clock = 94500i32;
__t18.hdisplay = 1024u16;
__t18.hsync_start = 1072u16;
__t18.hsync_end = 1168u16;
__t18.htotal = 1376u16;
__t18.hskew = 0u16;
__t18.vdisplay = 768u16;
__t18.vsync_start = 769u16;
__t18.vsync_end = 772u16;
__t18.vtotal = 808u16;
__t18.vscan = 0u16;
__t18.flags = 5u32;
__t18.r#type = 64u8;
__t18.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t18.status = 0i32;
__t18
}, {
let mut __t19: drm_display_mode = unsafe { core::mem::zeroed() };
__t19.clock = 115500i32;
__t19.hdisplay = 1024u16;
__t19.hsync_start = 1072u16;
__t19.hsync_end = 1104u16;
__t19.htotal = 1184u16;
__t19.hskew = 0u16;
__t19.vdisplay = 768u16;
__t19.vsync_start = 771u16;
__t19.vsync_end = 775u16;
__t19.vtotal = 813u16;
__t19.vscan = 0u16;
__t19.flags = 9u32;
__t19.r#type = 64u8;
__t19.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t19.status = 0i32;
__t19
}, {
let mut __t20: drm_display_mode = unsafe { core::mem::zeroed() };
__t20.clock = 108000i32;
__t20.hdisplay = 1152u16;
__t20.hsync_start = 1216u16;
__t20.hsync_end = 1344u16;
__t20.htotal = 1600u16;
__t20.hskew = 0u16;
__t20.vdisplay = 864u16;
__t20.vsync_start = 865u16;
__t20.vsync_end = 868u16;
__t20.vtotal = 900u16;
__t20.vscan = 0u16;
__t20.flags = 5u32;
__t20.r#type = 64u8;
__t20.name = [49u8, 49u8, 53u8, 50u8, 120u8, 56u8, 54u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t20.status = 0i32;
__t20
}, {
let mut __t21: drm_display_mode = unsafe { core::mem::zeroed() };
__t21.clock = 74250i32;
__t21.hdisplay = 1280u16;
__t21.hsync_start = 1390u16;
__t21.hsync_end = 1430u16;
__t21.htotal = 1650u16;
__t21.hskew = 0u16;
__t21.vdisplay = 720u16;
__t21.vsync_start = 725u16;
__t21.vsync_end = 730u16;
__t21.vtotal = 750u16;
__t21.vscan = 0u16;
__t21.flags = 5u32;
__t21.r#type = 64u8;
__t21.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t21.status = 0i32;
__t21
}, {
let mut __t22: drm_display_mode = unsafe { core::mem::zeroed() };
__t22.clock = 68250i32;
__t22.hdisplay = 1280u16;
__t22.hsync_start = 1328u16;
__t22.hsync_end = 1360u16;
__t22.htotal = 1440u16;
__t22.hskew = 0u16;
__t22.vdisplay = 768u16;
__t22.vsync_start = 771u16;
__t22.vsync_end = 778u16;
__t22.vtotal = 790u16;
__t22.vscan = 0u16;
__t22.flags = 9u32;
__t22.r#type = 64u8;
__t22.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t22.status = 0i32;
__t22
}, {
let mut __t23: drm_display_mode = unsafe { core::mem::zeroed() };
__t23.clock = 79500i32;
__t23.hdisplay = 1280u16;
__t23.hsync_start = 1344u16;
__t23.hsync_end = 1472u16;
__t23.htotal = 1664u16;
__t23.hskew = 0u16;
__t23.vdisplay = 768u16;
__t23.vsync_start = 771u16;
__t23.vsync_end = 778u16;
__t23.vtotal = 798u16;
__t23.vscan = 0u16;
__t23.flags = 6u32;
__t23.r#type = 64u8;
__t23.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t23.status = 0i32;
__t23
}, {
let mut __t24: drm_display_mode = unsafe { core::mem::zeroed() };
__t24.clock = 102250i32;
__t24.hdisplay = 1280u16;
__t24.hsync_start = 1360u16;
__t24.hsync_end = 1488u16;
__t24.htotal = 1696u16;
__t24.hskew = 0u16;
__t24.vdisplay = 768u16;
__t24.vsync_start = 771u16;
__t24.vsync_end = 778u16;
__t24.vtotal = 805u16;
__t24.vscan = 0u16;
__t24.flags = 6u32;
__t24.r#type = 64u8;
__t24.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t24.status = 0i32;
__t24
}, {
let mut __t25: drm_display_mode = unsafe { core::mem::zeroed() };
__t25.clock = 117500i32;
__t25.hdisplay = 1280u16;
__t25.hsync_start = 1360u16;
__t25.hsync_end = 1496u16;
__t25.htotal = 1712u16;
__t25.hskew = 0u16;
__t25.vdisplay = 768u16;
__t25.vsync_start = 771u16;
__t25.vsync_end = 778u16;
__t25.vtotal = 809u16;
__t25.vscan = 0u16;
__t25.flags = 6u32;
__t25.r#type = 64u8;
__t25.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t25.status = 0i32;
__t25
}, {
let mut __t26: drm_display_mode = unsafe { core::mem::zeroed() };
__t26.clock = 140250i32;
__t26.hdisplay = 1280u16;
__t26.hsync_start = 1328u16;
__t26.hsync_end = 1360u16;
__t26.htotal = 1440u16;
__t26.hskew = 0u16;
__t26.vdisplay = 768u16;
__t26.vsync_start = 771u16;
__t26.vsync_end = 778u16;
__t26.vtotal = 813u16;
__t26.vscan = 0u16;
__t26.flags = 9u32;
__t26.r#type = 64u8;
__t26.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t26.status = 0i32;
__t26
}, {
let mut __t27: drm_display_mode = unsafe { core::mem::zeroed() };
__t27.clock = 71000i32;
__t27.hdisplay = 1280u16;
__t27.hsync_start = 1328u16;
__t27.hsync_end = 1360u16;
__t27.htotal = 1440u16;
__t27.hskew = 0u16;
__t27.vdisplay = 800u16;
__t27.vsync_start = 803u16;
__t27.vsync_end = 809u16;
__t27.vtotal = 823u16;
__t27.vscan = 0u16;
__t27.flags = 9u32;
__t27.r#type = 64u8;
__t27.name = [49u8, 50u8, 56u8, 48u8, 120u8, 56u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t27.status = 0i32;
__t27
}, {
let mut __t28: drm_display_mode = unsafe { core::mem::zeroed() };
__t28.clock = 83500i32;
__t28.hdisplay = 1280u16;
__t28.hsync_start = 1352u16;
__t28.hsync_end = 1480u16;
__t28.htotal = 1680u16;
__t28.hskew = 0u16;
__t28.vdisplay = 800u16;
__t28.vsync_start = 803u16;
__t28.vsync_end = 809u16;
__t28.vtotal = 831u16;
__t28.vscan = 0u16;
__t28.flags = 6u32;
__t28.r#type = 64u8;
__t28.name = [49u8, 50u8, 56u8, 48u8, 120u8, 56u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t28.status = 0i32;
__t28
}, {
let mut __t29: drm_display_mode = unsafe { core::mem::zeroed() };
__t29.clock = 106500i32;
__t29.hdisplay = 1280u16;
__t29.hsync_start = 1360u16;
__t29.hsync_end = 1488u16;
__t29.htotal = 1696u16;
__t29.hskew = 0u16;
__t29.vdisplay = 800u16;
__t29.vsync_start = 803u16;
__t29.vsync_end = 809u16;
__t29.vtotal = 838u16;
__t29.vscan = 0u16;
__t29.flags = 6u32;
__t29.r#type = 64u8;
__t29.name = [49u8, 50u8, 56u8, 48u8, 120u8, 56u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t29.status = 0i32;
__t29
}, {
let mut __t30: drm_display_mode = unsafe { core::mem::zeroed() };
__t30.clock = 122500i32;
__t30.hdisplay = 1280u16;
__t30.hsync_start = 1360u16;
__t30.hsync_end = 1496u16;
__t30.htotal = 1712u16;
__t30.hskew = 0u16;
__t30.vdisplay = 800u16;
__t30.vsync_start = 803u16;
__t30.vsync_end = 809u16;
__t30.vtotal = 843u16;
__t30.vscan = 0u16;
__t30.flags = 6u32;
__t30.r#type = 64u8;
__t30.name = [49u8, 50u8, 56u8, 48u8, 120u8, 56u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t30.status = 0i32;
__t30
}, {
let mut __t31: drm_display_mode = unsafe { core::mem::zeroed() };
__t31.clock = 146250i32;
__t31.hdisplay = 1280u16;
__t31.hsync_start = 1328u16;
__t31.hsync_end = 1360u16;
__t31.htotal = 1440u16;
__t31.hskew = 0u16;
__t31.vdisplay = 800u16;
__t31.vsync_start = 803u16;
__t31.vsync_end = 809u16;
__t31.vtotal = 847u16;
__t31.vscan = 0u16;
__t31.flags = 9u32;
__t31.r#type = 64u8;
__t31.name = [49u8, 50u8, 56u8, 48u8, 120u8, 56u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t31.status = 0i32;
__t31
}, {
let mut __t32: drm_display_mode = unsafe { core::mem::zeroed() };
__t32.clock = 108000i32;
__t32.hdisplay = 1280u16;
__t32.hsync_start = 1376u16;
__t32.hsync_end = 1488u16;
__t32.htotal = 1800u16;
__t32.hskew = 0u16;
__t32.vdisplay = 960u16;
__t32.vsync_start = 961u16;
__t32.vsync_end = 964u16;
__t32.vtotal = 1000u16;
__t32.vscan = 0u16;
__t32.flags = 5u32;
__t32.r#type = 64u8;
__t32.name = [49u8, 50u8, 56u8, 48u8, 120u8, 57u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t32.status = 0i32;
__t32
}, {
let mut __t33: drm_display_mode = unsafe { core::mem::zeroed() };
__t33.clock = 148500i32;
__t33.hdisplay = 1280u16;
__t33.hsync_start = 1344u16;
__t33.hsync_end = 1504u16;
__t33.htotal = 1728u16;
__t33.hskew = 0u16;
__t33.vdisplay = 960u16;
__t33.vsync_start = 961u16;
__t33.vsync_end = 964u16;
__t33.vtotal = 1011u16;
__t33.vscan = 0u16;
__t33.flags = 5u32;
__t33.r#type = 64u8;
__t33.name = [49u8, 50u8, 56u8, 48u8, 120u8, 57u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t33.status = 0i32;
__t33
}, {
let mut __t34: drm_display_mode = unsafe { core::mem::zeroed() };
__t34.clock = 175500i32;
__t34.hdisplay = 1280u16;
__t34.hsync_start = 1328u16;
__t34.hsync_end = 1360u16;
__t34.htotal = 1440u16;
__t34.hskew = 0u16;
__t34.vdisplay = 960u16;
__t34.vsync_start = 963u16;
__t34.vsync_end = 967u16;
__t34.vtotal = 1017u16;
__t34.vscan = 0u16;
__t34.flags = 9u32;
__t34.r#type = 64u8;
__t34.name = [49u8, 50u8, 56u8, 48u8, 120u8, 57u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t34.status = 0i32;
__t34
}, {
let mut __t35: drm_display_mode = unsafe { core::mem::zeroed() };
__t35.clock = 108000i32;
__t35.hdisplay = 1280u16;
__t35.hsync_start = 1328u16;
__t35.hsync_end = 1440u16;
__t35.htotal = 1688u16;
__t35.hskew = 0u16;
__t35.vdisplay = 1024u16;
__t35.vsync_start = 1025u16;
__t35.vsync_end = 1028u16;
__t35.vtotal = 1066u16;
__t35.vscan = 0u16;
__t35.flags = 5u32;
__t35.r#type = 64u8;
__t35.name = [49u8, 50u8, 56u8, 48u8, 120u8, 49u8, 48u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t35.status = 0i32;
__t35
}, {
let mut __t36: drm_display_mode = unsafe { core::mem::zeroed() };
__t36.clock = 135000i32;
__t36.hdisplay = 1280u16;
__t36.hsync_start = 1296u16;
__t36.hsync_end = 1440u16;
__t36.htotal = 1688u16;
__t36.hskew = 0u16;
__t36.vdisplay = 1024u16;
__t36.vsync_start = 1025u16;
__t36.vsync_end = 1028u16;
__t36.vtotal = 1066u16;
__t36.vscan = 0u16;
__t36.flags = 5u32;
__t36.r#type = 64u8;
__t36.name = [49u8, 50u8, 56u8, 48u8, 120u8, 49u8, 48u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t36.status = 0i32;
__t36
}, {
let mut __t37: drm_display_mode = unsafe { core::mem::zeroed() };
__t37.clock = 157500i32;
__t37.hdisplay = 1280u16;
__t37.hsync_start = 1344u16;
__t37.hsync_end = 1504u16;
__t37.htotal = 1728u16;
__t37.hskew = 0u16;
__t37.vdisplay = 1024u16;
__t37.vsync_start = 1025u16;
__t37.vsync_end = 1028u16;
__t37.vtotal = 1072u16;
__t37.vscan = 0u16;
__t37.flags = 5u32;
__t37.r#type = 64u8;
__t37.name = [49u8, 50u8, 56u8, 48u8, 120u8, 49u8, 48u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t37.status = 0i32;
__t37
}, {
let mut __t38: drm_display_mode = unsafe { core::mem::zeroed() };
__t38.clock = 187250i32;
__t38.hdisplay = 1280u16;
__t38.hsync_start = 1328u16;
__t38.hsync_end = 1360u16;
__t38.htotal = 1440u16;
__t38.hskew = 0u16;
__t38.vdisplay = 1024u16;
__t38.vsync_start = 1027u16;
__t38.vsync_end = 1034u16;
__t38.vtotal = 1084u16;
__t38.vscan = 0u16;
__t38.flags = 9u32;
__t38.r#type = 64u8;
__t38.name = [49u8, 50u8, 56u8, 48u8, 120u8, 49u8, 48u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t38.status = 0i32;
__t38
}, {
let mut __t39: drm_display_mode = unsafe { core::mem::zeroed() };
__t39.clock = 85500i32;
__t39.hdisplay = 1360u16;
__t39.hsync_start = 1424u16;
__t39.hsync_end = 1536u16;
__t39.htotal = 1792u16;
__t39.hskew = 0u16;
__t39.vdisplay = 768u16;
__t39.vsync_start = 771u16;
__t39.vsync_end = 777u16;
__t39.vtotal = 795u16;
__t39.vscan = 0u16;
__t39.flags = 5u32;
__t39.r#type = 64u8;
__t39.name = [49u8, 51u8, 54u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t39.status = 0i32;
__t39
}, {
let mut __t40: drm_display_mode = unsafe { core::mem::zeroed() };
__t40.clock = 148250i32;
__t40.hdisplay = 1360u16;
__t40.hsync_start = 1408u16;
__t40.hsync_end = 1440u16;
__t40.htotal = 1520u16;
__t40.hskew = 0u16;
__t40.vdisplay = 768u16;
__t40.vsync_start = 771u16;
__t40.vsync_end = 776u16;
__t40.vtotal = 813u16;
__t40.vscan = 0u16;
__t40.flags = 9u32;
__t40.r#type = 64u8;
__t40.name = [49u8, 51u8, 54u8, 48u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t40.status = 0i32;
__t40
}, {
let mut __t41: drm_display_mode = unsafe { core::mem::zeroed() };
__t41.clock = 85500i32;
__t41.hdisplay = 1366u16;
__t41.hsync_start = 1436u16;
__t41.hsync_end = 1579u16;
__t41.htotal = 1792u16;
__t41.hskew = 0u16;
__t41.vdisplay = 768u16;
__t41.vsync_start = 771u16;
__t41.vsync_end = 774u16;
__t41.vtotal = 798u16;
__t41.vscan = 0u16;
__t41.flags = 5u32;
__t41.r#type = 64u8;
__t41.name = [49u8, 51u8, 54u8, 54u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t41.status = 0i32;
__t41
}, {
let mut __t42: drm_display_mode = unsafe { core::mem::zeroed() };
__t42.clock = 72000i32;
__t42.hdisplay = 1366u16;
__t42.hsync_start = 1380u16;
__t42.hsync_end = 1436u16;
__t42.htotal = 1500u16;
__t42.hskew = 0u16;
__t42.vdisplay = 768u16;
__t42.vsync_start = 769u16;
__t42.vsync_end = 772u16;
__t42.vtotal = 800u16;
__t42.vscan = 0u16;
__t42.flags = 5u32;
__t42.r#type = 64u8;
__t42.name = [49u8, 51u8, 54u8, 54u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t42.status = 0i32;
__t42
}, {
let mut __t43: drm_display_mode = unsafe { core::mem::zeroed() };
__t43.clock = 101000i32;
__t43.hdisplay = 1400u16;
__t43.hsync_start = 1448u16;
__t43.hsync_end = 1480u16;
__t43.htotal = 1560u16;
__t43.hskew = 0u16;
__t43.vdisplay = 1050u16;
__t43.vsync_start = 1053u16;
__t43.vsync_end = 1057u16;
__t43.vtotal = 1080u16;
__t43.vscan = 0u16;
__t43.flags = 9u32;
__t43.r#type = 64u8;
__t43.name = [49u8, 52u8, 48u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t43.status = 0i32;
__t43
}, {
let mut __t44: drm_display_mode = unsafe { core::mem::zeroed() };
__t44.clock = 121750i32;
__t44.hdisplay = 1400u16;
__t44.hsync_start = 1488u16;
__t44.hsync_end = 1632u16;
__t44.htotal = 1864u16;
__t44.hskew = 0u16;
__t44.vdisplay = 1050u16;
__t44.vsync_start = 1053u16;
__t44.vsync_end = 1057u16;
__t44.vtotal = 1089u16;
__t44.vscan = 0u16;
__t44.flags = 6u32;
__t44.r#type = 64u8;
__t44.name = [49u8, 52u8, 48u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t44.status = 0i32;
__t44
}, {
let mut __t45: drm_display_mode = unsafe { core::mem::zeroed() };
__t45.clock = 156000i32;
__t45.hdisplay = 1400u16;
__t45.hsync_start = 1504u16;
__t45.hsync_end = 1648u16;
__t45.htotal = 1896u16;
__t45.hskew = 0u16;
__t45.vdisplay = 1050u16;
__t45.vsync_start = 1053u16;
__t45.vsync_end = 1057u16;
__t45.vtotal = 1099u16;
__t45.vscan = 0u16;
__t45.flags = 6u32;
__t45.r#type = 64u8;
__t45.name = [49u8, 52u8, 48u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t45.status = 0i32;
__t45
}, {
let mut __t46: drm_display_mode = unsafe { core::mem::zeroed() };
__t46.clock = 179500i32;
__t46.hdisplay = 1400u16;
__t46.hsync_start = 1504u16;
__t46.hsync_end = 1656u16;
__t46.htotal = 1912u16;
__t46.hskew = 0u16;
__t46.vdisplay = 1050u16;
__t46.vsync_start = 1053u16;
__t46.vsync_end = 1057u16;
__t46.vtotal = 1105u16;
__t46.vscan = 0u16;
__t46.flags = 6u32;
__t46.r#type = 64u8;
__t46.name = [49u8, 52u8, 48u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t46.status = 0i32;
__t46
}, {
let mut __t47: drm_display_mode = unsafe { core::mem::zeroed() };
__t47.clock = 208000i32;
__t47.hdisplay = 1400u16;
__t47.hsync_start = 1448u16;
__t47.hsync_end = 1480u16;
__t47.htotal = 1560u16;
__t47.hskew = 0u16;
__t47.vdisplay = 1050u16;
__t47.vsync_start = 1053u16;
__t47.vsync_end = 1057u16;
__t47.vtotal = 1112u16;
__t47.vscan = 0u16;
__t47.flags = 9u32;
__t47.r#type = 64u8;
__t47.name = [49u8, 52u8, 48u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t47.status = 0i32;
__t47
}, {
let mut __t48: drm_display_mode = unsafe { core::mem::zeroed() };
__t48.clock = 88750i32;
__t48.hdisplay = 1440u16;
__t48.hsync_start = 1488u16;
__t48.hsync_end = 1520u16;
__t48.htotal = 1600u16;
__t48.hskew = 0u16;
__t48.vdisplay = 900u16;
__t48.vsync_start = 903u16;
__t48.vsync_end = 909u16;
__t48.vtotal = 926u16;
__t48.vscan = 0u16;
__t48.flags = 9u32;
__t48.r#type = 64u8;
__t48.name = [49u8, 52u8, 52u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t48.status = 0i32;
__t48
}, {
let mut __t49: drm_display_mode = unsafe { core::mem::zeroed() };
__t49.clock = 106500i32;
__t49.hdisplay = 1440u16;
__t49.hsync_start = 1520u16;
__t49.hsync_end = 1672u16;
__t49.htotal = 1904u16;
__t49.hskew = 0u16;
__t49.vdisplay = 900u16;
__t49.vsync_start = 903u16;
__t49.vsync_end = 909u16;
__t49.vtotal = 934u16;
__t49.vscan = 0u16;
__t49.flags = 6u32;
__t49.r#type = 64u8;
__t49.name = [49u8, 52u8, 52u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t49.status = 0i32;
__t49
}, {
let mut __t50: drm_display_mode = unsafe { core::mem::zeroed() };
__t50.clock = 136750i32;
__t50.hdisplay = 1440u16;
__t50.hsync_start = 1536u16;
__t50.hsync_end = 1688u16;
__t50.htotal = 1936u16;
__t50.hskew = 0u16;
__t50.vdisplay = 900u16;
__t50.vsync_start = 903u16;
__t50.vsync_end = 909u16;
__t50.vtotal = 942u16;
__t50.vscan = 0u16;
__t50.flags = 6u32;
__t50.r#type = 64u8;
__t50.name = [49u8, 52u8, 52u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t50.status = 0i32;
__t50
}, {
let mut __t51: drm_display_mode = unsafe { core::mem::zeroed() };
__t51.clock = 157000i32;
__t51.hdisplay = 1440u16;
__t51.hsync_start = 1544u16;
__t51.hsync_end = 1696u16;
__t51.htotal = 1952u16;
__t51.hskew = 0u16;
__t51.vdisplay = 900u16;
__t51.vsync_start = 903u16;
__t51.vsync_end = 909u16;
__t51.vtotal = 948u16;
__t51.vscan = 0u16;
__t51.flags = 6u32;
__t51.r#type = 64u8;
__t51.name = [49u8, 52u8, 52u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t51.status = 0i32;
__t51
}, {
let mut __t52: drm_display_mode = unsafe { core::mem::zeroed() };
__t52.clock = 182750i32;
__t52.hdisplay = 1440u16;
__t52.hsync_start = 1488u16;
__t52.hsync_end = 1520u16;
__t52.htotal = 1600u16;
__t52.hskew = 0u16;
__t52.vdisplay = 900u16;
__t52.vsync_start = 903u16;
__t52.vsync_end = 909u16;
__t52.vtotal = 953u16;
__t52.vscan = 0u16;
__t52.flags = 9u32;
__t52.r#type = 64u8;
__t52.name = [49u8, 52u8, 52u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t52.status = 0i32;
__t52
}, {
let mut __t53: drm_display_mode = unsafe { core::mem::zeroed() };
__t53.clock = 108000i32;
__t53.hdisplay = 1600u16;
__t53.hsync_start = 1624u16;
__t53.hsync_end = 1704u16;
__t53.htotal = 1800u16;
__t53.hskew = 0u16;
__t53.vdisplay = 900u16;
__t53.vsync_start = 901u16;
__t53.vsync_end = 904u16;
__t53.vtotal = 1000u16;
__t53.vscan = 0u16;
__t53.flags = 5u32;
__t53.r#type = 64u8;
__t53.name = [49u8, 54u8, 48u8, 48u8, 120u8, 57u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t53.status = 0i32;
__t53
}, {
let mut __t54: drm_display_mode = unsafe { core::mem::zeroed() };
__t54.clock = 162000i32;
__t54.hdisplay = 1600u16;
__t54.hsync_start = 1664u16;
__t54.hsync_end = 1856u16;
__t54.htotal = 2160u16;
__t54.hskew = 0u16;
__t54.vdisplay = 1200u16;
__t54.vsync_start = 1201u16;
__t54.vsync_end = 1204u16;
__t54.vtotal = 1250u16;
__t54.vscan = 0u16;
__t54.flags = 5u32;
__t54.r#type = 64u8;
__t54.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t54.status = 0i32;
__t54
}, {
let mut __t55: drm_display_mode = unsafe { core::mem::zeroed() };
__t55.clock = 175500i32;
__t55.hdisplay = 1600u16;
__t55.hsync_start = 1664u16;
__t55.hsync_end = 1856u16;
__t55.htotal = 2160u16;
__t55.hskew = 0u16;
__t55.vdisplay = 1200u16;
__t55.vsync_start = 1201u16;
__t55.vsync_end = 1204u16;
__t55.vtotal = 1250u16;
__t55.vscan = 0u16;
__t55.flags = 5u32;
__t55.r#type = 64u8;
__t55.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t55.status = 0i32;
__t55
}, {
let mut __t56: drm_display_mode = unsafe { core::mem::zeroed() };
__t56.clock = 189000i32;
__t56.hdisplay = 1600u16;
__t56.hsync_start = 1664u16;
__t56.hsync_end = 1856u16;
__t56.htotal = 2160u16;
__t56.hskew = 0u16;
__t56.vdisplay = 1200u16;
__t56.vsync_start = 1201u16;
__t56.vsync_end = 1204u16;
__t56.vtotal = 1250u16;
__t56.vscan = 0u16;
__t56.flags = 5u32;
__t56.r#type = 64u8;
__t56.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t56.status = 0i32;
__t56
}, {
let mut __t57: drm_display_mode = unsafe { core::mem::zeroed() };
__t57.clock = 202500i32;
__t57.hdisplay = 1600u16;
__t57.hsync_start = 1664u16;
__t57.hsync_end = 1856u16;
__t57.htotal = 2160u16;
__t57.hskew = 0u16;
__t57.vdisplay = 1200u16;
__t57.vsync_start = 1201u16;
__t57.vsync_end = 1204u16;
__t57.vtotal = 1250u16;
__t57.vscan = 0u16;
__t57.flags = 5u32;
__t57.r#type = 64u8;
__t57.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t57.status = 0i32;
__t57
}, {
let mut __t58: drm_display_mode = unsafe { core::mem::zeroed() };
__t58.clock = 229500i32;
__t58.hdisplay = 1600u16;
__t58.hsync_start = 1664u16;
__t58.hsync_end = 1856u16;
__t58.htotal = 2160u16;
__t58.hskew = 0u16;
__t58.vdisplay = 1200u16;
__t58.vsync_start = 1201u16;
__t58.vsync_end = 1204u16;
__t58.vtotal = 1250u16;
__t58.vscan = 0u16;
__t58.flags = 5u32;
__t58.r#type = 64u8;
__t58.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t58.status = 0i32;
__t58
}, {
let mut __t59: drm_display_mode = unsafe { core::mem::zeroed() };
__t59.clock = 268250i32;
__t59.hdisplay = 1600u16;
__t59.hsync_start = 1648u16;
__t59.hsync_end = 1680u16;
__t59.htotal = 1760u16;
__t59.hskew = 0u16;
__t59.vdisplay = 1200u16;
__t59.vsync_start = 1203u16;
__t59.vsync_end = 1207u16;
__t59.vtotal = 1271u16;
__t59.vscan = 0u16;
__t59.flags = 9u32;
__t59.r#type = 64u8;
__t59.name = [49u8, 54u8, 48u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t59.status = 0i32;
__t59
}, {
let mut __t60: drm_display_mode = unsafe { core::mem::zeroed() };
__t60.clock = 119000i32;
__t60.hdisplay = 1680u16;
__t60.hsync_start = 1728u16;
__t60.hsync_end = 1760u16;
__t60.htotal = 1840u16;
__t60.hskew = 0u16;
__t60.vdisplay = 1050u16;
__t60.vsync_start = 1053u16;
__t60.vsync_end = 1059u16;
__t60.vtotal = 1080u16;
__t60.vscan = 0u16;
__t60.flags = 9u32;
__t60.r#type = 64u8;
__t60.name = [49u8, 54u8, 56u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t60.status = 0i32;
__t60
}, {
let mut __t61: drm_display_mode = unsafe { core::mem::zeroed() };
__t61.clock = 146250i32;
__t61.hdisplay = 1680u16;
__t61.hsync_start = 1784u16;
__t61.hsync_end = 1960u16;
__t61.htotal = 2240u16;
__t61.hskew = 0u16;
__t61.vdisplay = 1050u16;
__t61.vsync_start = 1053u16;
__t61.vsync_end = 1059u16;
__t61.vtotal = 1089u16;
__t61.vscan = 0u16;
__t61.flags = 6u32;
__t61.r#type = 64u8;
__t61.name = [49u8, 54u8, 56u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t61.status = 0i32;
__t61
}, {
let mut __t62: drm_display_mode = unsafe { core::mem::zeroed() };
__t62.clock = 187000i32;
__t62.hdisplay = 1680u16;
__t62.hsync_start = 1800u16;
__t62.hsync_end = 1976u16;
__t62.htotal = 2272u16;
__t62.hskew = 0u16;
__t62.vdisplay = 1050u16;
__t62.vsync_start = 1053u16;
__t62.vsync_end = 1059u16;
__t62.vtotal = 1099u16;
__t62.vscan = 0u16;
__t62.flags = 6u32;
__t62.r#type = 64u8;
__t62.name = [49u8, 54u8, 56u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t62.status = 0i32;
__t62
}, {
let mut __t63: drm_display_mode = unsafe { core::mem::zeroed() };
__t63.clock = 214750i32;
__t63.hdisplay = 1680u16;
__t63.hsync_start = 1808u16;
__t63.hsync_end = 1984u16;
__t63.htotal = 2288u16;
__t63.hskew = 0u16;
__t63.vdisplay = 1050u16;
__t63.vsync_start = 1053u16;
__t63.vsync_end = 1059u16;
__t63.vtotal = 1105u16;
__t63.vscan = 0u16;
__t63.flags = 6u32;
__t63.r#type = 64u8;
__t63.name = [49u8, 54u8, 56u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t63.status = 0i32;
__t63
}, {
let mut __t64: drm_display_mode = unsafe { core::mem::zeroed() };
__t64.clock = 245500i32;
__t64.hdisplay = 1680u16;
__t64.hsync_start = 1728u16;
__t64.hsync_end = 1760u16;
__t64.htotal = 1840u16;
__t64.hskew = 0u16;
__t64.vdisplay = 1050u16;
__t64.vsync_start = 1053u16;
__t64.vsync_end = 1059u16;
__t64.vtotal = 1112u16;
__t64.vscan = 0u16;
__t64.flags = 9u32;
__t64.r#type = 64u8;
__t64.name = [49u8, 54u8, 56u8, 48u8, 120u8, 49u8, 48u8, 53u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t64.status = 0i32;
__t64
}, {
let mut __t65: drm_display_mode = unsafe { core::mem::zeroed() };
__t65.clock = 204750i32;
__t65.hdisplay = 1792u16;
__t65.hsync_start = 1920u16;
__t65.hsync_end = 2120u16;
__t65.htotal = 2448u16;
__t65.hskew = 0u16;
__t65.vdisplay = 1344u16;
__t65.vsync_start = 1345u16;
__t65.vsync_end = 1348u16;
__t65.vtotal = 1394u16;
__t65.vscan = 0u16;
__t65.flags = 6u32;
__t65.r#type = 64u8;
__t65.name = [49u8, 55u8, 57u8, 50u8, 120u8, 49u8, 51u8, 52u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t65.status = 0i32;
__t65
}, {
let mut __t66: drm_display_mode = unsafe { core::mem::zeroed() };
__t66.clock = 261000i32;
__t66.hdisplay = 1792u16;
__t66.hsync_start = 1888u16;
__t66.hsync_end = 2104u16;
__t66.htotal = 2456u16;
__t66.hskew = 0u16;
__t66.vdisplay = 1344u16;
__t66.vsync_start = 1345u16;
__t66.vsync_end = 1348u16;
__t66.vtotal = 1417u16;
__t66.vscan = 0u16;
__t66.flags = 6u32;
__t66.r#type = 64u8;
__t66.name = [49u8, 55u8, 57u8, 50u8, 120u8, 49u8, 51u8, 52u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t66.status = 0i32;
__t66
}, {
let mut __t67: drm_display_mode = unsafe { core::mem::zeroed() };
__t67.clock = 333250i32;
__t67.hdisplay = 1792u16;
__t67.hsync_start = 1840u16;
__t67.hsync_end = 1872u16;
__t67.htotal = 1952u16;
__t67.hskew = 0u16;
__t67.vdisplay = 1344u16;
__t67.vsync_start = 1347u16;
__t67.vsync_end = 1351u16;
__t67.vtotal = 1423u16;
__t67.vscan = 0u16;
__t67.flags = 9u32;
__t67.r#type = 64u8;
__t67.name = [49u8, 55u8, 57u8, 50u8, 120u8, 49u8, 51u8, 52u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t67.status = 0i32;
__t67
}, {
let mut __t68: drm_display_mode = unsafe { core::mem::zeroed() };
__t68.clock = 218250i32;
__t68.hdisplay = 1856u16;
__t68.hsync_start = 1952u16;
__t68.hsync_end = 2176u16;
__t68.htotal = 2528u16;
__t68.hskew = 0u16;
__t68.vdisplay = 1392u16;
__t68.vsync_start = 1393u16;
__t68.vsync_end = 1396u16;
__t68.vtotal = 1439u16;
__t68.vscan = 0u16;
__t68.flags = 6u32;
__t68.r#type = 64u8;
__t68.name = [49u8, 56u8, 53u8, 54u8, 120u8, 49u8, 51u8, 57u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t68.status = 0i32;
__t68
}, {
let mut __t69: drm_display_mode = unsafe { core::mem::zeroed() };
__t69.clock = 288000i32;
__t69.hdisplay = 1856u16;
__t69.hsync_start = 1984u16;
__t69.hsync_end = 2208u16;
__t69.htotal = 2560u16;
__t69.hskew = 0u16;
__t69.vdisplay = 1392u16;
__t69.vsync_start = 1393u16;
__t69.vsync_end = 1396u16;
__t69.vtotal = 1500u16;
__t69.vscan = 0u16;
__t69.flags = 6u32;
__t69.r#type = 64u8;
__t69.name = [49u8, 56u8, 53u8, 54u8, 120u8, 49u8, 51u8, 57u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t69.status = 0i32;
__t69
}, {
let mut __t70: drm_display_mode = unsafe { core::mem::zeroed() };
__t70.clock = 356500i32;
__t70.hdisplay = 1856u16;
__t70.hsync_start = 1904u16;
__t70.hsync_end = 1936u16;
__t70.htotal = 2016u16;
__t70.hskew = 0u16;
__t70.vdisplay = 1392u16;
__t70.vsync_start = 1395u16;
__t70.vsync_end = 1399u16;
__t70.vtotal = 1474u16;
__t70.vscan = 0u16;
__t70.flags = 9u32;
__t70.r#type = 64u8;
__t70.name = [49u8, 56u8, 53u8, 54u8, 120u8, 49u8, 51u8, 57u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t70.status = 0i32;
__t70
}, {
let mut __t71: drm_display_mode = unsafe { core::mem::zeroed() };
__t71.clock = 148500i32;
__t71.hdisplay = 1920u16;
__t71.hsync_start = 2008u16;
__t71.hsync_end = 2052u16;
__t71.htotal = 2200u16;
__t71.hskew = 0u16;
__t71.vdisplay = 1080u16;
__t71.vsync_start = 1084u16;
__t71.vsync_end = 1089u16;
__t71.vtotal = 1125u16;
__t71.vscan = 0u16;
__t71.flags = 10u32;
__t71.r#type = 64u8;
__t71.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t71.status = 0i32;
__t71
}, {
let mut __t72: drm_display_mode = unsafe { core::mem::zeroed() };
__t72.clock = 154000i32;
__t72.hdisplay = 1920u16;
__t72.hsync_start = 1968u16;
__t72.hsync_end = 2000u16;
__t72.htotal = 2080u16;
__t72.hskew = 0u16;
__t72.vdisplay = 1200u16;
__t72.vsync_start = 1203u16;
__t72.vsync_end = 1209u16;
__t72.vtotal = 1235u16;
__t72.vscan = 0u16;
__t72.flags = 9u32;
__t72.r#type = 64u8;
__t72.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t72.status = 0i32;
__t72
}, {
let mut __t73: drm_display_mode = unsafe { core::mem::zeroed() };
__t73.clock = 193250i32;
__t73.hdisplay = 1920u16;
__t73.hsync_start = 2056u16;
__t73.hsync_end = 2256u16;
__t73.htotal = 2592u16;
__t73.hskew = 0u16;
__t73.vdisplay = 1200u16;
__t73.vsync_start = 1203u16;
__t73.vsync_end = 1209u16;
__t73.vtotal = 1245u16;
__t73.vscan = 0u16;
__t73.flags = 6u32;
__t73.r#type = 64u8;
__t73.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t73.status = 0i32;
__t73
}, {
let mut __t74: drm_display_mode = unsafe { core::mem::zeroed() };
__t74.clock = 245250i32;
__t74.hdisplay = 1920u16;
__t74.hsync_start = 2056u16;
__t74.hsync_end = 2264u16;
__t74.htotal = 2608u16;
__t74.hskew = 0u16;
__t74.vdisplay = 1200u16;
__t74.vsync_start = 1203u16;
__t74.vsync_end = 1209u16;
__t74.vtotal = 1255u16;
__t74.vscan = 0u16;
__t74.flags = 6u32;
__t74.r#type = 64u8;
__t74.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t74.status = 0i32;
__t74
}, {
let mut __t75: drm_display_mode = unsafe { core::mem::zeroed() };
__t75.clock = 281250i32;
__t75.hdisplay = 1920u16;
__t75.hsync_start = 2064u16;
__t75.hsync_end = 2272u16;
__t75.htotal = 2624u16;
__t75.hskew = 0u16;
__t75.vdisplay = 1200u16;
__t75.vsync_start = 1203u16;
__t75.vsync_end = 1209u16;
__t75.vtotal = 1262u16;
__t75.vscan = 0u16;
__t75.flags = 6u32;
__t75.r#type = 64u8;
__t75.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t75.status = 0i32;
__t75
}, {
let mut __t76: drm_display_mode = unsafe { core::mem::zeroed() };
__t76.clock = 317000i32;
__t76.hdisplay = 1920u16;
__t76.hsync_start = 1968u16;
__t76.hsync_end = 2000u16;
__t76.htotal = 2080u16;
__t76.hskew = 0u16;
__t76.vdisplay = 1200u16;
__t76.vsync_start = 1203u16;
__t76.vsync_end = 1209u16;
__t76.vtotal = 1271u16;
__t76.vscan = 0u16;
__t76.flags = 9u32;
__t76.r#type = 64u8;
__t76.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 50u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t76.status = 0i32;
__t76
}, {
let mut __t77: drm_display_mode = unsafe { core::mem::zeroed() };
__t77.clock = 234000i32;
__t77.hdisplay = 1920u16;
__t77.hsync_start = 2048u16;
__t77.hsync_end = 2256u16;
__t77.htotal = 2600u16;
__t77.hskew = 0u16;
__t77.vdisplay = 1440u16;
__t77.vsync_start = 1441u16;
__t77.vsync_end = 1444u16;
__t77.vtotal = 1500u16;
__t77.vscan = 0u16;
__t77.flags = 6u32;
__t77.r#type = 64u8;
__t77.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 52u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t77.status = 0i32;
__t77
}, {
let mut __t78: drm_display_mode = unsafe { core::mem::zeroed() };
__t78.clock = 297000i32;
__t78.hdisplay = 1920u16;
__t78.hsync_start = 2064u16;
__t78.hsync_end = 2288u16;
__t78.htotal = 2640u16;
__t78.hskew = 0u16;
__t78.vdisplay = 1440u16;
__t78.vsync_start = 1441u16;
__t78.vsync_end = 1444u16;
__t78.vtotal = 1500u16;
__t78.vscan = 0u16;
__t78.flags = 6u32;
__t78.r#type = 64u8;
__t78.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 52u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t78.status = 0i32;
__t78
}, {
let mut __t79: drm_display_mode = unsafe { core::mem::zeroed() };
__t79.clock = 380500i32;
__t79.hdisplay = 1920u16;
__t79.hsync_start = 1968u16;
__t79.hsync_end = 2000u16;
__t79.htotal = 2080u16;
__t79.hskew = 0u16;
__t79.vdisplay = 1440u16;
__t79.vsync_start = 1443u16;
__t79.vsync_end = 1447u16;
__t79.vtotal = 1525u16;
__t79.vscan = 0u16;
__t79.flags = 9u32;
__t79.r#type = 64u8;
__t79.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 52u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t79.status = 0i32;
__t79
}, {
let mut __t80: drm_display_mode = unsafe { core::mem::zeroed() };
__t80.clock = 162000i32;
__t80.hdisplay = 2048u16;
__t80.hsync_start = 2074u16;
__t80.hsync_end = 2154u16;
__t80.htotal = 2250u16;
__t80.hskew = 0u16;
__t80.vdisplay = 1152u16;
__t80.vsync_start = 1153u16;
__t80.vsync_end = 1156u16;
__t80.vtotal = 1200u16;
__t80.vscan = 0u16;
__t80.flags = 5u32;
__t80.r#type = 64u8;
__t80.name = [50u8, 48u8, 52u8, 56u8, 120u8, 49u8, 49u8, 53u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t80.status = 0i32;
__t80
}, {
let mut __t81: drm_display_mode = unsafe { core::mem::zeroed() };
__t81.clock = 268500i32;
__t81.hdisplay = 2560u16;
__t81.hsync_start = 2608u16;
__t81.hsync_end = 2640u16;
__t81.htotal = 2720u16;
__t81.hskew = 0u16;
__t81.vdisplay = 1600u16;
__t81.vsync_start = 1603u16;
__t81.vsync_end = 1609u16;
__t81.vtotal = 1646u16;
__t81.vscan = 0u16;
__t81.flags = 9u32;
__t81.r#type = 64u8;
__t81.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t81.status = 0i32;
__t81
}, {
let mut __t82: drm_display_mode = unsafe { core::mem::zeroed() };
__t82.clock = 348500i32;
__t82.hdisplay = 2560u16;
__t82.hsync_start = 2752u16;
__t82.hsync_end = 3032u16;
__t82.htotal = 3504u16;
__t82.hskew = 0u16;
__t82.vdisplay = 1600u16;
__t82.vsync_start = 1603u16;
__t82.vsync_end = 1609u16;
__t82.vtotal = 1658u16;
__t82.vscan = 0u16;
__t82.flags = 6u32;
__t82.r#type = 64u8;
__t82.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t82.status = 0i32;
__t82
}, {
let mut __t83: drm_display_mode = unsafe { core::mem::zeroed() };
__t83.clock = 443250i32;
__t83.hdisplay = 2560u16;
__t83.hsync_start = 2768u16;
__t83.hsync_end = 3048u16;
__t83.htotal = 3536u16;
__t83.hskew = 0u16;
__t83.vdisplay = 1600u16;
__t83.vsync_start = 1603u16;
__t83.vsync_end = 1609u16;
__t83.vtotal = 1672u16;
__t83.vscan = 0u16;
__t83.flags = 6u32;
__t83.r#type = 64u8;
__t83.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t83.status = 0i32;
__t83
}, {
let mut __t84: drm_display_mode = unsafe { core::mem::zeroed() };
__t84.clock = 505250i32;
__t84.hdisplay = 2560u16;
__t84.hsync_start = 2768u16;
__t84.hsync_end = 3048u16;
__t84.htotal = 3536u16;
__t84.hskew = 0u16;
__t84.vdisplay = 1600u16;
__t84.vsync_start = 1603u16;
__t84.vsync_end = 1609u16;
__t84.vtotal = 1682u16;
__t84.vscan = 0u16;
__t84.flags = 6u32;
__t84.r#type = 64u8;
__t84.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t84.status = 0i32;
__t84
}, {
let mut __t85: drm_display_mode = unsafe { core::mem::zeroed() };
__t85.clock = 552750i32;
__t85.hdisplay = 2560u16;
__t85.hsync_start = 2608u16;
__t85.hsync_end = 2640u16;
__t85.htotal = 2720u16;
__t85.hskew = 0u16;
__t85.vdisplay = 1600u16;
__t85.vsync_start = 1603u16;
__t85.vsync_end = 1609u16;
__t85.vtotal = 1694u16;
__t85.vscan = 0u16;
__t85.flags = 9u32;
__t85.r#type = 64u8;
__t85.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t85.status = 0i32;
__t85
}, {
let mut __t86: drm_display_mode = unsafe { core::mem::zeroed() };
__t86.clock = 556744i32;
__t86.hdisplay = 4096u16;
__t86.hsync_start = 4104u16;
__t86.hsync_end = 4136u16;
__t86.htotal = 4176u16;
__t86.hskew = 0u16;
__t86.vdisplay = 2160u16;
__t86.vsync_start = 2208u16;
__t86.vsync_end = 2216u16;
__t86.vtotal = 2222u16;
__t86.vscan = 0u16;
__t86.flags = 9u32;
__t86.r#type = 64u8;
__t86.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t86.status = 0i32;
__t86
}, {
let mut __t87: drm_display_mode = unsafe { core::mem::zeroed() };
__t87.clock = 556188i32;
__t87.hdisplay = 4096u16;
__t87.hsync_start = 4104u16;
__t87.hsync_end = 4136u16;
__t87.htotal = 4176u16;
__t87.hskew = 0u16;
__t87.vdisplay = 2160u16;
__t87.vsync_start = 2208u16;
__t87.vsync_end = 2216u16;
__t87.vtotal = 2222u16;
__t87.vscan = 0u16;
__t87.flags = 9u32;
__t87.r#type = 64u8;
__t87.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t87.status = 0i32;
__t87
}]);

/// These more or less come from the DMT spec.  The 720x400 modes are
/// inferred from historical 80x25 practice.  The 640x480@67 and 832x624@75
/// modes are old-school Mac modes.  The EDID spec says the 1152x864@75 mode
/// should be 1152x870, again for the Mac, but instead we use the x864 DMT
/// mode.
///
/// The DMT modes have been fact-checked; the rest are mild guesses.
/// 800x600@60Hz
/// 800x600@56Hz
/// 640x480@75Hz
/// 640x480@72Hz
/// 640x480@67Hz
/// 640x480@60Hz
/// 720x400@88Hz
/// 720x400@70Hz
/// 1280x1024@75Hz
/// 1024x768@75Hz
/// 1024x768@70Hz
/// 1024x768@60Hz
/// 1024x768@43Hz
/// 832x624@75Hz
/// 800x600@75Hz
/// 800x600@72Hz
/// 1152x864@75Hz
static edid_est_modes: __SlateAlign16<[drm_display_mode; 17]> = __SlateAlign16([{
let mut __t0: drm_display_mode = unsafe { core::mem::zeroed() };
__t0.clock = 40000i32;
__t0.hdisplay = 800u16;
__t0.hsync_start = 840u16;
__t0.hsync_end = 968u16;
__t0.htotal = 1056u16;
__t0.hskew = 0u16;
__t0.vdisplay = 600u16;
__t0.vsync_start = 601u16;
__t0.vsync_end = 605u16;
__t0.vtotal = 628u16;
__t0.vscan = 0u16;
__t0.flags = 5u32;
__t0.r#type = 64u8;
__t0.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.status = 0i32;
__t0
}, {
let mut __t1: drm_display_mode = unsafe { core::mem::zeroed() };
__t1.clock = 36000i32;
__t1.hdisplay = 800u16;
__t1.hsync_start = 824u16;
__t1.hsync_end = 896u16;
__t1.htotal = 1024u16;
__t1.hskew = 0u16;
__t1.vdisplay = 600u16;
__t1.vsync_start = 601u16;
__t1.vsync_end = 603u16;
__t1.vtotal = 625u16;
__t1.vscan = 0u16;
__t1.flags = 5u32;
__t1.r#type = 64u8;
__t1.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.status = 0i32;
__t1
}, {
let mut __t2: drm_display_mode = unsafe { core::mem::zeroed() };
__t2.clock = 31500i32;
__t2.hdisplay = 640u16;
__t2.hsync_start = 656u16;
__t2.hsync_end = 720u16;
__t2.htotal = 840u16;
__t2.hskew = 0u16;
__t2.vdisplay = 480u16;
__t2.vsync_start = 481u16;
__t2.vsync_end = 484u16;
__t2.vtotal = 500u16;
__t2.vscan = 0u16;
__t2.flags = 10u32;
__t2.r#type = 64u8;
__t2.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.status = 0i32;
__t2
}, {
let mut __t3: drm_display_mode = unsafe { core::mem::zeroed() };
__t3.clock = 31500i32;
__t3.hdisplay = 640u16;
__t3.hsync_start = 664u16;
__t3.hsync_end = 704u16;
__t3.htotal = 832u16;
__t3.hskew = 0u16;
__t3.vdisplay = 480u16;
__t3.vsync_start = 489u16;
__t3.vsync_end = 492u16;
__t3.vtotal = 520u16;
__t3.vscan = 0u16;
__t3.flags = 10u32;
__t3.r#type = 64u8;
__t3.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.status = 0i32;
__t3
}, {
let mut __t4: drm_display_mode = unsafe { core::mem::zeroed() };
__t4.clock = 30240i32;
__t4.hdisplay = 640u16;
__t4.hsync_start = 704u16;
__t4.hsync_end = 768u16;
__t4.htotal = 864u16;
__t4.hskew = 0u16;
__t4.vdisplay = 480u16;
__t4.vsync_start = 483u16;
__t4.vsync_end = 486u16;
__t4.vtotal = 525u16;
__t4.vscan = 0u16;
__t4.flags = 10u32;
__t4.r#type = 64u8;
__t4.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.status = 0i32;
__t4
}, {
let mut __t5: drm_display_mode = unsafe { core::mem::zeroed() };
__t5.clock = 25175i32;
__t5.hdisplay = 640u16;
__t5.hsync_start = 656u16;
__t5.hsync_end = 752u16;
__t5.htotal = 800u16;
__t5.hskew = 0u16;
__t5.vdisplay = 480u16;
__t5.vsync_start = 490u16;
__t5.vsync_end = 492u16;
__t5.vtotal = 525u16;
__t5.vscan = 0u16;
__t5.flags = 10u32;
__t5.r#type = 64u8;
__t5.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5.status = 0i32;
__t5
}, {
let mut __t6: drm_display_mode = unsafe { core::mem::zeroed() };
__t6.clock = 35500i32;
__t6.hdisplay = 720u16;
__t6.hsync_start = 738u16;
__t6.hsync_end = 846u16;
__t6.htotal = 900u16;
__t6.hskew = 0u16;
__t6.vdisplay = 400u16;
__t6.vsync_start = 421u16;
__t6.vsync_end = 423u16;
__t6.vtotal = 449u16;
__t6.vscan = 0u16;
__t6.flags = 10u32;
__t6.r#type = 64u8;
__t6.name = [55u8, 50u8, 48u8, 120u8, 52u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.status = 0i32;
__t6
}, {
let mut __t7: drm_display_mode = unsafe { core::mem::zeroed() };
__t7.clock = 28320i32;
__t7.hdisplay = 720u16;
__t7.hsync_start = 738u16;
__t7.hsync_end = 846u16;
__t7.htotal = 900u16;
__t7.hskew = 0u16;
__t7.vdisplay = 400u16;
__t7.vsync_start = 412u16;
__t7.vsync_end = 414u16;
__t7.vtotal = 449u16;
__t7.vscan = 0u16;
__t7.flags = 6u32;
__t7.r#type = 64u8;
__t7.name = [55u8, 50u8, 48u8, 120u8, 52u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.status = 0i32;
__t7
}, {
let mut __t8: drm_display_mode = unsafe { core::mem::zeroed() };
__t8.clock = 135000i32;
__t8.hdisplay = 1280u16;
__t8.hsync_start = 1296u16;
__t8.hsync_end = 1440u16;
__t8.htotal = 1688u16;
__t8.hskew = 0u16;
__t8.vdisplay = 1024u16;
__t8.vsync_start = 1025u16;
__t8.vsync_end = 1028u16;
__t8.vtotal = 1066u16;
__t8.vscan = 0u16;
__t8.flags = 5u32;
__t8.r#type = 64u8;
__t8.name = [49u8, 50u8, 56u8, 48u8, 120u8, 49u8, 48u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8.status = 0i32;
__t8
}, {
let mut __t9: drm_display_mode = unsafe { core::mem::zeroed() };
__t9.clock = 78750i32;
__t9.hdisplay = 1024u16;
__t9.hsync_start = 1040u16;
__t9.hsync_end = 1136u16;
__t9.htotal = 1312u16;
__t9.hskew = 0u16;
__t9.vdisplay = 768u16;
__t9.vsync_start = 769u16;
__t9.vsync_end = 772u16;
__t9.vtotal = 800u16;
__t9.vscan = 0u16;
__t9.flags = 5u32;
__t9.r#type = 64u8;
__t9.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t9.status = 0i32;
__t9
}, {
let mut __t10: drm_display_mode = unsafe { core::mem::zeroed() };
__t10.clock = 75000i32;
__t10.hdisplay = 1024u16;
__t10.hsync_start = 1048u16;
__t10.hsync_end = 1184u16;
__t10.htotal = 1328u16;
__t10.hskew = 0u16;
__t10.vdisplay = 768u16;
__t10.vsync_start = 771u16;
__t10.vsync_end = 777u16;
__t10.vtotal = 806u16;
__t10.vscan = 0u16;
__t10.flags = 10u32;
__t10.r#type = 64u8;
__t10.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10.status = 0i32;
__t10
}, {
let mut __t11: drm_display_mode = unsafe { core::mem::zeroed() };
__t11.clock = 65000i32;
__t11.hdisplay = 1024u16;
__t11.hsync_start = 1048u16;
__t11.hsync_end = 1184u16;
__t11.htotal = 1344u16;
__t11.hskew = 0u16;
__t11.vdisplay = 768u16;
__t11.vsync_start = 771u16;
__t11.vsync_end = 777u16;
__t11.vtotal = 806u16;
__t11.vscan = 0u16;
__t11.flags = 10u32;
__t11.r#type = 64u8;
__t11.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11.status = 0i32;
__t11
}, {
let mut __t12: drm_display_mode = unsafe { core::mem::zeroed() };
__t12.clock = 44900i32;
__t12.hdisplay = 1024u16;
__t12.hsync_start = 1032u16;
__t12.hsync_end = 1208u16;
__t12.htotal = 1264u16;
__t12.hskew = 0u16;
__t12.vdisplay = 768u16;
__t12.vsync_start = 768u16;
__t12.vsync_end = 776u16;
__t12.vtotal = 817u16;
__t12.vscan = 0u16;
__t12.flags = 21u32;
__t12.r#type = 64u8;
__t12.name = [49u8, 48u8, 50u8, 52u8, 120u8, 55u8, 54u8, 56u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12.status = 0i32;
__t12
}, {
let mut __t13: drm_display_mode = unsafe { core::mem::zeroed() };
__t13.clock = 57284i32;
__t13.hdisplay = 832u16;
__t13.hsync_start = 864u16;
__t13.hsync_end = 928u16;
__t13.htotal = 1152u16;
__t13.hskew = 0u16;
__t13.vdisplay = 624u16;
__t13.vsync_start = 625u16;
__t13.vsync_end = 628u16;
__t13.vtotal = 667u16;
__t13.vscan = 0u16;
__t13.flags = 10u32;
__t13.r#type = 64u8;
__t13.name = [56u8, 51u8, 50u8, 120u8, 54u8, 50u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13.status = 0i32;
__t13
}, {
let mut __t14: drm_display_mode = unsafe { core::mem::zeroed() };
__t14.clock = 49500i32;
__t14.hdisplay = 800u16;
__t14.hsync_start = 816u16;
__t14.hsync_end = 896u16;
__t14.htotal = 1056u16;
__t14.hskew = 0u16;
__t14.vdisplay = 600u16;
__t14.vsync_start = 601u16;
__t14.vsync_end = 604u16;
__t14.vtotal = 625u16;
__t14.vscan = 0u16;
__t14.flags = 5u32;
__t14.r#type = 64u8;
__t14.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14.status = 0i32;
__t14
}, {
let mut __t15: drm_display_mode = unsafe { core::mem::zeroed() };
__t15.clock = 50000i32;
__t15.hdisplay = 800u16;
__t15.hsync_start = 856u16;
__t15.hsync_end = 976u16;
__t15.htotal = 1040u16;
__t15.hskew = 0u16;
__t15.vdisplay = 600u16;
__t15.vsync_start = 637u16;
__t15.vsync_end = 643u16;
__t15.vtotal = 666u16;
__t15.vscan = 0u16;
__t15.flags = 5u32;
__t15.r#type = 64u8;
__t15.name = [56u8, 48u8, 48u8, 120u8, 54u8, 48u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15.status = 0i32;
__t15
}, {
let mut __t16: drm_display_mode = unsafe { core::mem::zeroed() };
__t16.clock = 108000i32;
__t16.hdisplay = 1152u16;
__t16.hsync_start = 1216u16;
__t16.hsync_end = 1344u16;
__t16.htotal = 1600u16;
__t16.hskew = 0u16;
__t16.vdisplay = 864u16;
__t16.vsync_start = 865u16;
__t16.vsync_end = 868u16;
__t16.vtotal = 900u16;
__t16.vscan = 0u16;
__t16.flags = 5u32;
__t16.r#type = 64u8;
__t16.name = [49u8, 49u8, 53u8, 50u8, 120u8, 56u8, 54u8, 52u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t16.status = 0i32;
__t16
}]);

#[repr(C)]
#[derive(Clone, Copy)]
struct minimode {
    w: i16,
    h: i16,
    r: i16,
    rb: i16,
}


/// byte 6
/// byte 7
/// byte 8
/// byte 9
/// byte 10
/// byte 11
static est3_modes: __SlateAlign16<[minimode; 44]> = __SlateAlign16([minimode { w: 640i16, h: 350i16, r: 85i16, rb: 0i16 }, minimode { w: 640i16, h: 400i16, r: 85i16, rb: 0i16 }, minimode { w: 720i16, h: 400i16, r: 85i16, rb: 0i16 }, minimode { w: 640i16, h: 480i16, r: 85i16, rb: 0i16 }, minimode { w: 848i16, h: 480i16, r: 60i16, rb: 0i16 }, minimode { w: 800i16, h: 600i16, r: 85i16, rb: 0i16 }, minimode { w: 1024i16, h: 768i16, r: 85i16, rb: 0i16 }, minimode { w: 1152i16, h: 864i16, r: 75i16, rb: 0i16 }, minimode { w: 1280i16, h: 768i16, r: 60i16, rb: 1i16 }, minimode { w: 1280i16, h: 768i16, r: 60i16, rb: 0i16 }, minimode { w: 1280i16, h: 768i16, r: 75i16, rb: 0i16 }, minimode { w: 1280i16, h: 768i16, r: 85i16, rb: 0i16 }, minimode { w: 1280i16, h: 960i16, r: 60i16, rb: 0i16 }, minimode { w: 1280i16, h: 960i16, r: 85i16, rb: 0i16 }, minimode { w: 1280i16, h: 1024i16, r: 60i16, rb: 0i16 }, minimode { w: 1280i16, h: 1024i16, r: 85i16, rb: 0i16 }, minimode { w: 1360i16, h: 768i16, r: 60i16, rb: 0i16 }, minimode { w: 1440i16, h: 900i16, r: 60i16, rb: 1i16 }, minimode { w: 1440i16, h: 900i16, r: 60i16, rb: 0i16 }, minimode { w: 1440i16, h: 900i16, r: 75i16, rb: 0i16 }, minimode { w: 1440i16, h: 900i16, r: 85i16, rb: 0i16 }, minimode { w: 1400i16, h: 1050i16, r: 60i16, rb: 1i16 }, minimode { w: 1400i16, h: 1050i16, r: 60i16, rb: 0i16 }, minimode { w: 1400i16, h: 1050i16, r: 75i16, rb: 0i16 }, minimode { w: 1400i16, h: 1050i16, r: 85i16, rb: 0i16 }, minimode { w: 1680i16, h: 1050i16, r: 60i16, rb: 1i16 }, minimode { w: 1680i16, h: 1050i16, r: 60i16, rb: 0i16 }, minimode { w: 1680i16, h: 1050i16, r: 75i16, rb: 0i16 }, minimode { w: 1680i16, h: 1050i16, r: 85i16, rb: 0i16 }, minimode { w: 1600i16, h: 1200i16, r: 60i16, rb: 0i16 }, minimode { w: 1600i16, h: 1200i16, r: 65i16, rb: 0i16 }, minimode { w: 1600i16, h: 1200i16, r: 70i16, rb: 0i16 }, minimode { w: 1600i16, h: 1200i16, r: 75i16, rb: 0i16 }, minimode { w: 1600i16, h: 1200i16, r: 85i16, rb: 0i16 }, minimode { w: 1792i16, h: 1344i16, r: 60i16, rb: 0i16 }, minimode { w: 1792i16, h: 1344i16, r: 75i16, rb: 0i16 }, minimode { w: 1856i16, h: 1392i16, r: 60i16, rb: 0i16 }, minimode { w: 1856i16, h: 1392i16, r: 75i16, rb: 0i16 }, minimode { w: 1920i16, h: 1200i16, r: 60i16, rb: 1i16 }, minimode { w: 1920i16, h: 1200i16, r: 60i16, rb: 0i16 }, minimode { w: 1920i16, h: 1200i16, r: 75i16, rb: 0i16 }, minimode { w: 1920i16, h: 1200i16, r: 85i16, rb: 0i16 }, minimode { w: 1920i16, h: 1440i16, r: 60i16, rb: 0i16 }, minimode { w: 1920i16, h: 1440i16, r: 75i16, rb: 0i16 }]);

static extra_modes: __SlateAlign16<[minimode; 7]> = __SlateAlign16([minimode { w: 1024i16, h: 576i16, r: 60i16, rb: 0i16 }, minimode { w: 1366i16, h: 768i16, r: 60i16, rb: 0i16 }, minimode { w: 1600i16, h: 900i16, r: 60i16, rb: 0i16 }, minimode { w: 1680i16, h: 945i16, r: 60i16, rb: 0i16 }, minimode { w: 1920i16, h: 1080i16, r: 60i16, rb: 0i16 }, minimode { w: 2048i16, h: 1152i16, r: 60i16, rb: 0i16 }, minimode { w: 2048i16, h: 1536i16, r: 60i16, rb: 0i16 }]);

/// From CEA/CTA-861 spec.
///
/// Do not access directly, instead always use cea_mode_for_vic().
/// 1 - 640x480@60Hz 4:3
/// 2 - 720x480@60Hz 4:3
/// 3 - 720x480@60Hz 16:9
/// 4 - 1280x720@60Hz 16:9
/// 5 - 1920x1080i@60Hz 16:9
/// 6 - 720(1440)x480i@60Hz 4:3
/// 7 - 720(1440)x480i@60Hz 16:9
/// 8 - 720(1440)x240@60Hz 4:3
/// 9 - 720(1440)x240@60Hz 16:9
/// 10 - 2880x480i@60Hz 4:3
/// 11 - 2880x480i@60Hz 16:9
/// 12 - 2880x240@60Hz 4:3
/// 13 - 2880x240@60Hz 16:9
/// 14 - 1440x480@60Hz 4:3
/// 15 - 1440x480@60Hz 16:9
/// 16 - 1920x1080@60Hz 16:9
/// 17 - 720x576@50Hz 4:3
/// 18 - 720x576@50Hz 16:9
/// 19 - 1280x720@50Hz 16:9
/// 20 - 1920x1080i@50Hz 16:9
/// 21 - 720(1440)x576i@50Hz 4:3
/// 22 - 720(1440)x576i@50Hz 16:9
/// 23 - 720(1440)x288@50Hz 4:3
/// 24 - 720(1440)x288@50Hz 16:9
/// 25 - 2880x576i@50Hz 4:3
/// 26 - 2880x576i@50Hz 16:9
/// 27 - 2880x288@50Hz 4:3
/// 28 - 2880x288@50Hz 16:9
/// 29 - 1440x576@50Hz 4:3
/// 30 - 1440x576@50Hz 16:9
/// 31 - 1920x1080@50Hz 16:9
/// 32 - 1920x1080@24Hz 16:9
/// 33 - 1920x1080@25Hz 16:9
/// 34 - 1920x1080@30Hz 16:9
/// 35 - 2880x480@60Hz 4:3
/// 36 - 2880x480@60Hz 16:9
/// 37 - 2880x576@50Hz 4:3
/// 38 - 2880x576@50Hz 16:9
/// 39 - 1920x1080i@50Hz 16:9
/// 40 - 1920x1080i@100Hz 16:9
/// 41 - 1280x720@100Hz 16:9
/// 42 - 720x576@100Hz 4:3
/// 43 - 720x576@100Hz 16:9
/// 44 - 720(1440)x576i@100Hz 4:3
/// 45 - 720(1440)x576i@100Hz 16:9
/// 46 - 1920x1080i@120Hz 16:9
/// 47 - 1280x720@120Hz 16:9
/// 48 - 720x480@120Hz 4:3
/// 49 - 720x480@120Hz 16:9
/// 50 - 720(1440)x480i@120Hz 4:3
/// 51 - 720(1440)x480i@120Hz 16:9
/// 52 - 720x576@200Hz 4:3
/// 53 - 720x576@200Hz 16:9
/// 54 - 720(1440)x576i@200Hz 4:3
/// 55 - 720(1440)x576i@200Hz 16:9
/// 56 - 720x480@240Hz 4:3
/// 57 - 720x480@240Hz 16:9
/// 58 - 720(1440)x480i@240Hz 4:3
/// 59 - 720(1440)x480i@240Hz 16:9
/// 60 - 1280x720@24Hz 16:9
/// 61 - 1280x720@25Hz 16:9
/// 62 - 1280x720@30Hz 16:9
/// 63 - 1920x1080@120Hz 16:9
/// 64 - 1920x1080@100Hz 16:9
/// 65 - 1280x720@24Hz 64:27
/// 66 - 1280x720@25Hz 64:27
/// 67 - 1280x720@30Hz 64:27
/// 68 - 1280x720@50Hz 64:27
/// 69 - 1280x720@60Hz 64:27
/// 70 - 1280x720@100Hz 64:27
/// 71 - 1280x720@120Hz 64:27
/// 72 - 1920x1080@24Hz 64:27
/// 73 - 1920x1080@25Hz 64:27
/// 74 - 1920x1080@30Hz 64:27
/// 75 - 1920x1080@50Hz 64:27
/// 76 - 1920x1080@60Hz 64:27
/// 77 - 1920x1080@100Hz 64:27
/// 78 - 1920x1080@120Hz 64:27
/// 79 - 1680x720@24Hz 64:27
/// 80 - 1680x720@25Hz 64:27
/// 81 - 1680x720@30Hz 64:27
/// 82 - 1680x720@50Hz 64:27
/// 83 - 1680x720@60Hz 64:27
/// 84 - 1680x720@100Hz 64:27
/// 85 - 1680x720@120Hz 64:27
/// 86 - 2560x1080@24Hz 64:27
/// 87 - 2560x1080@25Hz 64:27
/// 88 - 2560x1080@30Hz 64:27
/// 89 - 2560x1080@50Hz 64:27
/// 90 - 2560x1080@60Hz 64:27
/// 91 - 2560x1080@100Hz 64:27
/// 92 - 2560x1080@120Hz 64:27
/// 93 - 3840x2160@24Hz 16:9
/// 94 - 3840x2160@25Hz 16:9
/// 95 - 3840x2160@30Hz 16:9
/// 96 - 3840x2160@50Hz 16:9
/// 97 - 3840x2160@60Hz 16:9
/// 98 - 4096x2160@24Hz 256:135
/// 99 - 4096x2160@25Hz 256:135
/// 100 - 4096x2160@30Hz 256:135
/// 101 - 4096x2160@50Hz 256:135
/// 102 - 4096x2160@60Hz 256:135
/// 103 - 3840x2160@24Hz 64:27
/// 104 - 3840x2160@25Hz 64:27
/// 105 - 3840x2160@30Hz 64:27
/// 106 - 3840x2160@50Hz 64:27
/// 107 - 3840x2160@60Hz 64:27
/// 108 - 1280x720@48Hz 16:9
/// 109 - 1280x720@48Hz 64:27
/// 110 - 1680x720@48Hz 64:27
/// 111 - 1920x1080@48Hz 16:9
/// 112 - 1920x1080@48Hz 64:27
/// 113 - 2560x1080@48Hz 64:27
/// 114 - 3840x2160@48Hz 16:9
/// 115 - 4096x2160@48Hz 256:135
/// 116 - 3840x2160@48Hz 64:27
/// 117 - 3840x2160@100Hz 16:9
/// 118 - 3840x2160@120Hz 16:9
/// 119 - 3840x2160@100Hz 64:27
/// 120 - 3840x2160@120Hz 64:27
/// 121 - 5120x2160@24Hz 64:27
/// 122 - 5120x2160@25Hz 64:27
/// 123 - 5120x2160@30Hz 64:27
/// 124 - 5120x2160@48Hz 64:27
/// 125 - 5120x2160@50Hz 64:27
/// 126 - 5120x2160@60Hz 64:27
/// 127 - 5120x2160@100Hz 64:27
static edid_cea_modes_1: __SlateAlign16<[drm_display_mode; 127]> = __SlateAlign16([{
let mut __t0: drm_display_mode = unsafe { core::mem::zeroed() };
__t0.clock = 25175i32;
__t0.hdisplay = 640u16;
__t0.hsync_start = 656u16;
__t0.hsync_end = 752u16;
__t0.htotal = 800u16;
__t0.hskew = 0u16;
__t0.vdisplay = 480u16;
__t0.vsync_start = 490u16;
__t0.vsync_end = 492u16;
__t0.vtotal = 525u16;
__t0.vscan = 0u16;
__t0.flags = 10u32;
__t0.r#type = 64u8;
__t0.name = [54u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.status = 0i32;
__t0.picture_aspect_ratio = 1u32;
__t0
}, {
let mut __t1: drm_display_mode = unsafe { core::mem::zeroed() };
__t1.clock = 27000i32;
__t1.hdisplay = 720u16;
__t1.hsync_start = 736u16;
__t1.hsync_end = 798u16;
__t1.htotal = 858u16;
__t1.hskew = 0u16;
__t1.vdisplay = 480u16;
__t1.vsync_start = 489u16;
__t1.vsync_end = 495u16;
__t1.vtotal = 525u16;
__t1.vscan = 0u16;
__t1.flags = 10u32;
__t1.r#type = 64u8;
__t1.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.status = 0i32;
__t1.picture_aspect_ratio = 1u32;
__t1
}, {
let mut __t2: drm_display_mode = unsafe { core::mem::zeroed() };
__t2.clock = 27000i32;
__t2.hdisplay = 720u16;
__t2.hsync_start = 736u16;
__t2.hsync_end = 798u16;
__t2.htotal = 858u16;
__t2.hskew = 0u16;
__t2.vdisplay = 480u16;
__t2.vsync_start = 489u16;
__t2.vsync_end = 495u16;
__t2.vtotal = 525u16;
__t2.vscan = 0u16;
__t2.flags = 10u32;
__t2.r#type = 64u8;
__t2.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.status = 0i32;
__t2.picture_aspect_ratio = 2u32;
__t2
}, {
let mut __t3: drm_display_mode = unsafe { core::mem::zeroed() };
__t3.clock = 74250i32;
__t3.hdisplay = 1280u16;
__t3.hsync_start = 1390u16;
__t3.hsync_end = 1430u16;
__t3.htotal = 1650u16;
__t3.hskew = 0u16;
__t3.vdisplay = 720u16;
__t3.vsync_start = 725u16;
__t3.vsync_end = 730u16;
__t3.vtotal = 750u16;
__t3.vscan = 0u16;
__t3.flags = 5u32;
__t3.r#type = 64u8;
__t3.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.status = 0i32;
__t3.picture_aspect_ratio = 2u32;
__t3
}, {
let mut __t4: drm_display_mode = unsafe { core::mem::zeroed() };
__t4.clock = 74250i32;
__t4.hdisplay = 1920u16;
__t4.hsync_start = 2008u16;
__t4.hsync_end = 2052u16;
__t4.htotal = 2200u16;
__t4.hskew = 0u16;
__t4.vdisplay = 1080u16;
__t4.vsync_start = 1084u16;
__t4.vsync_end = 1094u16;
__t4.vtotal = 1125u16;
__t4.vscan = 0u16;
__t4.flags = 21u32;
__t4.r#type = 64u8;
__t4.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.status = 0i32;
__t4.picture_aspect_ratio = 2u32;
__t4
}, {
let mut __t5: drm_display_mode = unsafe { core::mem::zeroed() };
__t5.clock = 13500i32;
__t5.hdisplay = 720u16;
__t5.hsync_start = 739u16;
__t5.hsync_end = 801u16;
__t5.htotal = 858u16;
__t5.hskew = 0u16;
__t5.vdisplay = 480u16;
__t5.vsync_start = 488u16;
__t5.vsync_end = 494u16;
__t5.vtotal = 525u16;
__t5.vscan = 0u16;
__t5.flags = 4122u32;
__t5.r#type = 64u8;
__t5.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5.status = 0i32;
__t5.picture_aspect_ratio = 1u32;
__t5
}, {
let mut __t6: drm_display_mode = unsafe { core::mem::zeroed() };
__t6.clock = 13500i32;
__t6.hdisplay = 720u16;
__t6.hsync_start = 739u16;
__t6.hsync_end = 801u16;
__t6.htotal = 858u16;
__t6.hskew = 0u16;
__t6.vdisplay = 480u16;
__t6.vsync_start = 488u16;
__t6.vsync_end = 494u16;
__t6.vtotal = 525u16;
__t6.vscan = 0u16;
__t6.flags = 4122u32;
__t6.r#type = 64u8;
__t6.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.status = 0i32;
__t6.picture_aspect_ratio = 2u32;
__t6
}, {
let mut __t7: drm_display_mode = unsafe { core::mem::zeroed() };
__t7.clock = 13500i32;
__t7.hdisplay = 720u16;
__t7.hsync_start = 739u16;
__t7.hsync_end = 801u16;
__t7.htotal = 858u16;
__t7.hskew = 0u16;
__t7.vdisplay = 240u16;
__t7.vsync_start = 244u16;
__t7.vsync_end = 247u16;
__t7.vtotal = 262u16;
__t7.vscan = 0u16;
__t7.flags = 4106u32;
__t7.r#type = 64u8;
__t7.name = [55u8, 50u8, 48u8, 120u8, 50u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.status = 0i32;
__t7.picture_aspect_ratio = 1u32;
__t7
}, {
let mut __t8: drm_display_mode = unsafe { core::mem::zeroed() };
__t8.clock = 13500i32;
__t8.hdisplay = 720u16;
__t8.hsync_start = 739u16;
__t8.hsync_end = 801u16;
__t8.htotal = 858u16;
__t8.hskew = 0u16;
__t8.vdisplay = 240u16;
__t8.vsync_start = 244u16;
__t8.vsync_end = 247u16;
__t8.vtotal = 262u16;
__t8.vscan = 0u16;
__t8.flags = 4106u32;
__t8.r#type = 64u8;
__t8.name = [55u8, 50u8, 48u8, 120u8, 50u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8.status = 0i32;
__t8.picture_aspect_ratio = 2u32;
__t8
}, {
let mut __t9: drm_display_mode = unsafe { core::mem::zeroed() };
__t9.clock = 54000i32;
__t9.hdisplay = 2880u16;
__t9.hsync_start = 2956u16;
__t9.hsync_end = 3204u16;
__t9.htotal = 3432u16;
__t9.hskew = 0u16;
__t9.vdisplay = 480u16;
__t9.vsync_start = 488u16;
__t9.vsync_end = 494u16;
__t9.vtotal = 525u16;
__t9.vscan = 0u16;
__t9.flags = 26u32;
__t9.r#type = 64u8;
__t9.name = [50u8, 56u8, 56u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t9.status = 0i32;
__t9.picture_aspect_ratio = 1u32;
__t9
}, {
let mut __t10: drm_display_mode = unsafe { core::mem::zeroed() };
__t10.clock = 54000i32;
__t10.hdisplay = 2880u16;
__t10.hsync_start = 2956u16;
__t10.hsync_end = 3204u16;
__t10.htotal = 3432u16;
__t10.hskew = 0u16;
__t10.vdisplay = 480u16;
__t10.vsync_start = 488u16;
__t10.vsync_end = 494u16;
__t10.vtotal = 525u16;
__t10.vscan = 0u16;
__t10.flags = 26u32;
__t10.r#type = 64u8;
__t10.name = [50u8, 56u8, 56u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10.status = 0i32;
__t10.picture_aspect_ratio = 2u32;
__t10
}, {
let mut __t11: drm_display_mode = unsafe { core::mem::zeroed() };
__t11.clock = 54000i32;
__t11.hdisplay = 2880u16;
__t11.hsync_start = 2956u16;
__t11.hsync_end = 3204u16;
__t11.htotal = 3432u16;
__t11.hskew = 0u16;
__t11.vdisplay = 240u16;
__t11.vsync_start = 244u16;
__t11.vsync_end = 247u16;
__t11.vtotal = 262u16;
__t11.vscan = 0u16;
__t11.flags = 10u32;
__t11.r#type = 64u8;
__t11.name = [50u8, 56u8, 56u8, 48u8, 120u8, 50u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11.status = 0i32;
__t11.picture_aspect_ratio = 1u32;
__t11
}, {
let mut __t12: drm_display_mode = unsafe { core::mem::zeroed() };
__t12.clock = 54000i32;
__t12.hdisplay = 2880u16;
__t12.hsync_start = 2956u16;
__t12.hsync_end = 3204u16;
__t12.htotal = 3432u16;
__t12.hskew = 0u16;
__t12.vdisplay = 240u16;
__t12.vsync_start = 244u16;
__t12.vsync_end = 247u16;
__t12.vtotal = 262u16;
__t12.vscan = 0u16;
__t12.flags = 10u32;
__t12.r#type = 64u8;
__t12.name = [50u8, 56u8, 56u8, 48u8, 120u8, 50u8, 52u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12.status = 0i32;
__t12.picture_aspect_ratio = 2u32;
__t12
}, {
let mut __t13: drm_display_mode = unsafe { core::mem::zeroed() };
__t13.clock = 54000i32;
__t13.hdisplay = 1440u16;
__t13.hsync_start = 1472u16;
__t13.hsync_end = 1596u16;
__t13.htotal = 1716u16;
__t13.hskew = 0u16;
__t13.vdisplay = 480u16;
__t13.vsync_start = 489u16;
__t13.vsync_end = 495u16;
__t13.vtotal = 525u16;
__t13.vscan = 0u16;
__t13.flags = 10u32;
__t13.r#type = 64u8;
__t13.name = [49u8, 52u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13.status = 0i32;
__t13.picture_aspect_ratio = 1u32;
__t13
}, {
let mut __t14: drm_display_mode = unsafe { core::mem::zeroed() };
__t14.clock = 54000i32;
__t14.hdisplay = 1440u16;
__t14.hsync_start = 1472u16;
__t14.hsync_end = 1596u16;
__t14.htotal = 1716u16;
__t14.hskew = 0u16;
__t14.vdisplay = 480u16;
__t14.vsync_start = 489u16;
__t14.vsync_end = 495u16;
__t14.vtotal = 525u16;
__t14.vscan = 0u16;
__t14.flags = 10u32;
__t14.r#type = 64u8;
__t14.name = [49u8, 52u8, 52u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14.status = 0i32;
__t14.picture_aspect_ratio = 2u32;
__t14
}, {
let mut __t15: drm_display_mode = unsafe { core::mem::zeroed() };
__t15.clock = 148500i32;
__t15.hdisplay = 1920u16;
__t15.hsync_start = 2008u16;
__t15.hsync_end = 2052u16;
__t15.htotal = 2200u16;
__t15.hskew = 0u16;
__t15.vdisplay = 1080u16;
__t15.vsync_start = 1084u16;
__t15.vsync_end = 1089u16;
__t15.vtotal = 1125u16;
__t15.vscan = 0u16;
__t15.flags = 5u32;
__t15.r#type = 64u8;
__t15.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15.status = 0i32;
__t15.picture_aspect_ratio = 2u32;
__t15
}, {
let mut __t16: drm_display_mode = unsafe { core::mem::zeroed() };
__t16.clock = 27000i32;
__t16.hdisplay = 720u16;
__t16.hsync_start = 732u16;
__t16.hsync_end = 796u16;
__t16.htotal = 864u16;
__t16.hskew = 0u16;
__t16.vdisplay = 576u16;
__t16.vsync_start = 581u16;
__t16.vsync_end = 586u16;
__t16.vtotal = 625u16;
__t16.vscan = 0u16;
__t16.flags = 10u32;
__t16.r#type = 64u8;
__t16.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t16.status = 0i32;
__t16.picture_aspect_ratio = 1u32;
__t16
}, {
let mut __t17: drm_display_mode = unsafe { core::mem::zeroed() };
__t17.clock = 27000i32;
__t17.hdisplay = 720u16;
__t17.hsync_start = 732u16;
__t17.hsync_end = 796u16;
__t17.htotal = 864u16;
__t17.hskew = 0u16;
__t17.vdisplay = 576u16;
__t17.vsync_start = 581u16;
__t17.vsync_end = 586u16;
__t17.vtotal = 625u16;
__t17.vscan = 0u16;
__t17.flags = 10u32;
__t17.r#type = 64u8;
__t17.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t17.status = 0i32;
__t17.picture_aspect_ratio = 2u32;
__t17
}, {
let mut __t18: drm_display_mode = unsafe { core::mem::zeroed() };
__t18.clock = 74250i32;
__t18.hdisplay = 1280u16;
__t18.hsync_start = 1720u16;
__t18.hsync_end = 1760u16;
__t18.htotal = 1980u16;
__t18.hskew = 0u16;
__t18.vdisplay = 720u16;
__t18.vsync_start = 725u16;
__t18.vsync_end = 730u16;
__t18.vtotal = 750u16;
__t18.vscan = 0u16;
__t18.flags = 5u32;
__t18.r#type = 64u8;
__t18.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t18.status = 0i32;
__t18.picture_aspect_ratio = 2u32;
__t18
}, {
let mut __t19: drm_display_mode = unsafe { core::mem::zeroed() };
__t19.clock = 74250i32;
__t19.hdisplay = 1920u16;
__t19.hsync_start = 2448u16;
__t19.hsync_end = 2492u16;
__t19.htotal = 2640u16;
__t19.hskew = 0u16;
__t19.vdisplay = 1080u16;
__t19.vsync_start = 1084u16;
__t19.vsync_end = 1094u16;
__t19.vtotal = 1125u16;
__t19.vscan = 0u16;
__t19.flags = 21u32;
__t19.r#type = 64u8;
__t19.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t19.status = 0i32;
__t19.picture_aspect_ratio = 2u32;
__t19
}, {
let mut __t20: drm_display_mode = unsafe { core::mem::zeroed() };
__t20.clock = 13500i32;
__t20.hdisplay = 720u16;
__t20.hsync_start = 732u16;
__t20.hsync_end = 795u16;
__t20.htotal = 864u16;
__t20.hskew = 0u16;
__t20.vdisplay = 576u16;
__t20.vsync_start = 580u16;
__t20.vsync_end = 586u16;
__t20.vtotal = 625u16;
__t20.vscan = 0u16;
__t20.flags = 4122u32;
__t20.r#type = 64u8;
__t20.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t20.status = 0i32;
__t20.picture_aspect_ratio = 1u32;
__t20
}, {
let mut __t21: drm_display_mode = unsafe { core::mem::zeroed() };
__t21.clock = 13500i32;
__t21.hdisplay = 720u16;
__t21.hsync_start = 732u16;
__t21.hsync_end = 795u16;
__t21.htotal = 864u16;
__t21.hskew = 0u16;
__t21.vdisplay = 576u16;
__t21.vsync_start = 580u16;
__t21.vsync_end = 586u16;
__t21.vtotal = 625u16;
__t21.vscan = 0u16;
__t21.flags = 4122u32;
__t21.r#type = 64u8;
__t21.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t21.status = 0i32;
__t21.picture_aspect_ratio = 2u32;
__t21
}, {
let mut __t22: drm_display_mode = unsafe { core::mem::zeroed() };
__t22.clock = 13500i32;
__t22.hdisplay = 720u16;
__t22.hsync_start = 732u16;
__t22.hsync_end = 795u16;
__t22.htotal = 864u16;
__t22.hskew = 0u16;
__t22.vdisplay = 288u16;
__t22.vsync_start = 290u16;
__t22.vsync_end = 293u16;
__t22.vtotal = 312u16;
__t22.vscan = 0u16;
__t22.flags = 4106u32;
__t22.r#type = 64u8;
__t22.name = [55u8, 50u8, 48u8, 120u8, 50u8, 56u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t22.status = 0i32;
__t22.picture_aspect_ratio = 1u32;
__t22
}, {
let mut __t23: drm_display_mode = unsafe { core::mem::zeroed() };
__t23.clock = 13500i32;
__t23.hdisplay = 720u16;
__t23.hsync_start = 732u16;
__t23.hsync_end = 795u16;
__t23.htotal = 864u16;
__t23.hskew = 0u16;
__t23.vdisplay = 288u16;
__t23.vsync_start = 290u16;
__t23.vsync_end = 293u16;
__t23.vtotal = 312u16;
__t23.vscan = 0u16;
__t23.flags = 4106u32;
__t23.r#type = 64u8;
__t23.name = [55u8, 50u8, 48u8, 120u8, 50u8, 56u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t23.status = 0i32;
__t23.picture_aspect_ratio = 2u32;
__t23
}, {
let mut __t24: drm_display_mode = unsafe { core::mem::zeroed() };
__t24.clock = 54000i32;
__t24.hdisplay = 2880u16;
__t24.hsync_start = 2928u16;
__t24.hsync_end = 3180u16;
__t24.htotal = 3456u16;
__t24.hskew = 0u16;
__t24.vdisplay = 576u16;
__t24.vsync_start = 580u16;
__t24.vsync_end = 586u16;
__t24.vtotal = 625u16;
__t24.vscan = 0u16;
__t24.flags = 26u32;
__t24.r#type = 64u8;
__t24.name = [50u8, 56u8, 56u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t24.status = 0i32;
__t24.picture_aspect_ratio = 1u32;
__t24
}, {
let mut __t25: drm_display_mode = unsafe { core::mem::zeroed() };
__t25.clock = 54000i32;
__t25.hdisplay = 2880u16;
__t25.hsync_start = 2928u16;
__t25.hsync_end = 3180u16;
__t25.htotal = 3456u16;
__t25.hskew = 0u16;
__t25.vdisplay = 576u16;
__t25.vsync_start = 580u16;
__t25.vsync_end = 586u16;
__t25.vtotal = 625u16;
__t25.vscan = 0u16;
__t25.flags = 26u32;
__t25.r#type = 64u8;
__t25.name = [50u8, 56u8, 56u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t25.status = 0i32;
__t25.picture_aspect_ratio = 2u32;
__t25
}, {
let mut __t26: drm_display_mode = unsafe { core::mem::zeroed() };
__t26.clock = 54000i32;
__t26.hdisplay = 2880u16;
__t26.hsync_start = 2928u16;
__t26.hsync_end = 3180u16;
__t26.htotal = 3456u16;
__t26.hskew = 0u16;
__t26.vdisplay = 288u16;
__t26.vsync_start = 290u16;
__t26.vsync_end = 293u16;
__t26.vtotal = 312u16;
__t26.vscan = 0u16;
__t26.flags = 10u32;
__t26.r#type = 64u8;
__t26.name = [50u8, 56u8, 56u8, 48u8, 120u8, 50u8, 56u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t26.status = 0i32;
__t26.picture_aspect_ratio = 1u32;
__t26
}, {
let mut __t27: drm_display_mode = unsafe { core::mem::zeroed() };
__t27.clock = 54000i32;
__t27.hdisplay = 2880u16;
__t27.hsync_start = 2928u16;
__t27.hsync_end = 3180u16;
__t27.htotal = 3456u16;
__t27.hskew = 0u16;
__t27.vdisplay = 288u16;
__t27.vsync_start = 290u16;
__t27.vsync_end = 293u16;
__t27.vtotal = 312u16;
__t27.vscan = 0u16;
__t27.flags = 10u32;
__t27.r#type = 64u8;
__t27.name = [50u8, 56u8, 56u8, 48u8, 120u8, 50u8, 56u8, 56u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t27.status = 0i32;
__t27.picture_aspect_ratio = 2u32;
__t27
}, {
let mut __t28: drm_display_mode = unsafe { core::mem::zeroed() };
__t28.clock = 54000i32;
__t28.hdisplay = 1440u16;
__t28.hsync_start = 1464u16;
__t28.hsync_end = 1592u16;
__t28.htotal = 1728u16;
__t28.hskew = 0u16;
__t28.vdisplay = 576u16;
__t28.vsync_start = 581u16;
__t28.vsync_end = 586u16;
__t28.vtotal = 625u16;
__t28.vscan = 0u16;
__t28.flags = 10u32;
__t28.r#type = 64u8;
__t28.name = [49u8, 52u8, 52u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t28.status = 0i32;
__t28.picture_aspect_ratio = 1u32;
__t28
}, {
let mut __t29: drm_display_mode = unsafe { core::mem::zeroed() };
__t29.clock = 54000i32;
__t29.hdisplay = 1440u16;
__t29.hsync_start = 1464u16;
__t29.hsync_end = 1592u16;
__t29.htotal = 1728u16;
__t29.hskew = 0u16;
__t29.vdisplay = 576u16;
__t29.vsync_start = 581u16;
__t29.vsync_end = 586u16;
__t29.vtotal = 625u16;
__t29.vscan = 0u16;
__t29.flags = 10u32;
__t29.r#type = 64u8;
__t29.name = [49u8, 52u8, 52u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t29.status = 0i32;
__t29.picture_aspect_ratio = 2u32;
__t29
}, {
let mut __t30: drm_display_mode = unsafe { core::mem::zeroed() };
__t30.clock = 148500i32;
__t30.hdisplay = 1920u16;
__t30.hsync_start = 2448u16;
__t30.hsync_end = 2492u16;
__t30.htotal = 2640u16;
__t30.hskew = 0u16;
__t30.vdisplay = 1080u16;
__t30.vsync_start = 1084u16;
__t30.vsync_end = 1089u16;
__t30.vtotal = 1125u16;
__t30.vscan = 0u16;
__t30.flags = 5u32;
__t30.r#type = 64u8;
__t30.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t30.status = 0i32;
__t30.picture_aspect_ratio = 2u32;
__t30
}, {
let mut __t31: drm_display_mode = unsafe { core::mem::zeroed() };
__t31.clock = 74250i32;
__t31.hdisplay = 1920u16;
__t31.hsync_start = 2558u16;
__t31.hsync_end = 2602u16;
__t31.htotal = 2750u16;
__t31.hskew = 0u16;
__t31.vdisplay = 1080u16;
__t31.vsync_start = 1084u16;
__t31.vsync_end = 1089u16;
__t31.vtotal = 1125u16;
__t31.vscan = 0u16;
__t31.flags = 5u32;
__t31.r#type = 64u8;
__t31.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t31.status = 0i32;
__t31.picture_aspect_ratio = 2u32;
__t31
}, {
let mut __t32: drm_display_mode = unsafe { core::mem::zeroed() };
__t32.clock = 74250i32;
__t32.hdisplay = 1920u16;
__t32.hsync_start = 2448u16;
__t32.hsync_end = 2492u16;
__t32.htotal = 2640u16;
__t32.hskew = 0u16;
__t32.vdisplay = 1080u16;
__t32.vsync_start = 1084u16;
__t32.vsync_end = 1089u16;
__t32.vtotal = 1125u16;
__t32.vscan = 0u16;
__t32.flags = 5u32;
__t32.r#type = 64u8;
__t32.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t32.status = 0i32;
__t32.picture_aspect_ratio = 2u32;
__t32
}, {
let mut __t33: drm_display_mode = unsafe { core::mem::zeroed() };
__t33.clock = 74250i32;
__t33.hdisplay = 1920u16;
__t33.hsync_start = 2008u16;
__t33.hsync_end = 2052u16;
__t33.htotal = 2200u16;
__t33.hskew = 0u16;
__t33.vdisplay = 1080u16;
__t33.vsync_start = 1084u16;
__t33.vsync_end = 1089u16;
__t33.vtotal = 1125u16;
__t33.vscan = 0u16;
__t33.flags = 5u32;
__t33.r#type = 64u8;
__t33.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t33.status = 0i32;
__t33.picture_aspect_ratio = 2u32;
__t33
}, {
let mut __t34: drm_display_mode = unsafe { core::mem::zeroed() };
__t34.clock = 108000i32;
__t34.hdisplay = 2880u16;
__t34.hsync_start = 2944u16;
__t34.hsync_end = 3192u16;
__t34.htotal = 3432u16;
__t34.hskew = 0u16;
__t34.vdisplay = 480u16;
__t34.vsync_start = 489u16;
__t34.vsync_end = 495u16;
__t34.vtotal = 525u16;
__t34.vscan = 0u16;
__t34.flags = 10u32;
__t34.r#type = 64u8;
__t34.name = [50u8, 56u8, 56u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t34.status = 0i32;
__t34.picture_aspect_ratio = 1u32;
__t34
}, {
let mut __t35: drm_display_mode = unsafe { core::mem::zeroed() };
__t35.clock = 108000i32;
__t35.hdisplay = 2880u16;
__t35.hsync_start = 2944u16;
__t35.hsync_end = 3192u16;
__t35.htotal = 3432u16;
__t35.hskew = 0u16;
__t35.vdisplay = 480u16;
__t35.vsync_start = 489u16;
__t35.vsync_end = 495u16;
__t35.vtotal = 525u16;
__t35.vscan = 0u16;
__t35.flags = 10u32;
__t35.r#type = 64u8;
__t35.name = [50u8, 56u8, 56u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t35.status = 0i32;
__t35.picture_aspect_ratio = 2u32;
__t35
}, {
let mut __t36: drm_display_mode = unsafe { core::mem::zeroed() };
__t36.clock = 108000i32;
__t36.hdisplay = 2880u16;
__t36.hsync_start = 2928u16;
__t36.hsync_end = 3184u16;
__t36.htotal = 3456u16;
__t36.hskew = 0u16;
__t36.vdisplay = 576u16;
__t36.vsync_start = 581u16;
__t36.vsync_end = 586u16;
__t36.vtotal = 625u16;
__t36.vscan = 0u16;
__t36.flags = 10u32;
__t36.r#type = 64u8;
__t36.name = [50u8, 56u8, 56u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t36.status = 0i32;
__t36.picture_aspect_ratio = 1u32;
__t36
}, {
let mut __t37: drm_display_mode = unsafe { core::mem::zeroed() };
__t37.clock = 108000i32;
__t37.hdisplay = 2880u16;
__t37.hsync_start = 2928u16;
__t37.hsync_end = 3184u16;
__t37.htotal = 3456u16;
__t37.hskew = 0u16;
__t37.vdisplay = 576u16;
__t37.vsync_start = 581u16;
__t37.vsync_end = 586u16;
__t37.vtotal = 625u16;
__t37.vscan = 0u16;
__t37.flags = 10u32;
__t37.r#type = 64u8;
__t37.name = [50u8, 56u8, 56u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t37.status = 0i32;
__t37.picture_aspect_ratio = 2u32;
__t37
}, {
let mut __t38: drm_display_mode = unsafe { core::mem::zeroed() };
__t38.clock = 72000i32;
__t38.hdisplay = 1920u16;
__t38.hsync_start = 1952u16;
__t38.hsync_end = 2120u16;
__t38.htotal = 2304u16;
__t38.hskew = 0u16;
__t38.vdisplay = 1080u16;
__t38.vsync_start = 1126u16;
__t38.vsync_end = 1136u16;
__t38.vtotal = 1250u16;
__t38.vscan = 0u16;
__t38.flags = 25u32;
__t38.r#type = 64u8;
__t38.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t38.status = 0i32;
__t38.picture_aspect_ratio = 2u32;
__t38
}, {
let mut __t39: drm_display_mode = unsafe { core::mem::zeroed() };
__t39.clock = 148500i32;
__t39.hdisplay = 1920u16;
__t39.hsync_start = 2448u16;
__t39.hsync_end = 2492u16;
__t39.htotal = 2640u16;
__t39.hskew = 0u16;
__t39.vdisplay = 1080u16;
__t39.vsync_start = 1084u16;
__t39.vsync_end = 1094u16;
__t39.vtotal = 1125u16;
__t39.vscan = 0u16;
__t39.flags = 21u32;
__t39.r#type = 64u8;
__t39.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t39.status = 0i32;
__t39.picture_aspect_ratio = 2u32;
__t39
}, {
let mut __t40: drm_display_mode = unsafe { core::mem::zeroed() };
__t40.clock = 148500i32;
__t40.hdisplay = 1280u16;
__t40.hsync_start = 1720u16;
__t40.hsync_end = 1760u16;
__t40.htotal = 1980u16;
__t40.hskew = 0u16;
__t40.vdisplay = 720u16;
__t40.vsync_start = 725u16;
__t40.vsync_end = 730u16;
__t40.vtotal = 750u16;
__t40.vscan = 0u16;
__t40.flags = 5u32;
__t40.r#type = 64u8;
__t40.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t40.status = 0i32;
__t40.picture_aspect_ratio = 2u32;
__t40
}, {
let mut __t41: drm_display_mode = unsafe { core::mem::zeroed() };
__t41.clock = 54000i32;
__t41.hdisplay = 720u16;
__t41.hsync_start = 732u16;
__t41.hsync_end = 796u16;
__t41.htotal = 864u16;
__t41.hskew = 0u16;
__t41.vdisplay = 576u16;
__t41.vsync_start = 581u16;
__t41.vsync_end = 586u16;
__t41.vtotal = 625u16;
__t41.vscan = 0u16;
__t41.flags = 10u32;
__t41.r#type = 64u8;
__t41.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t41.status = 0i32;
__t41.picture_aspect_ratio = 1u32;
__t41
}, {
let mut __t42: drm_display_mode = unsafe { core::mem::zeroed() };
__t42.clock = 54000i32;
__t42.hdisplay = 720u16;
__t42.hsync_start = 732u16;
__t42.hsync_end = 796u16;
__t42.htotal = 864u16;
__t42.hskew = 0u16;
__t42.vdisplay = 576u16;
__t42.vsync_start = 581u16;
__t42.vsync_end = 586u16;
__t42.vtotal = 625u16;
__t42.vscan = 0u16;
__t42.flags = 10u32;
__t42.r#type = 64u8;
__t42.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t42.status = 0i32;
__t42.picture_aspect_ratio = 2u32;
__t42
}, {
let mut __t43: drm_display_mode = unsafe { core::mem::zeroed() };
__t43.clock = 27000i32;
__t43.hdisplay = 720u16;
__t43.hsync_start = 732u16;
__t43.hsync_end = 795u16;
__t43.htotal = 864u16;
__t43.hskew = 0u16;
__t43.vdisplay = 576u16;
__t43.vsync_start = 580u16;
__t43.vsync_end = 586u16;
__t43.vtotal = 625u16;
__t43.vscan = 0u16;
__t43.flags = 4122u32;
__t43.r#type = 64u8;
__t43.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t43.status = 0i32;
__t43.picture_aspect_ratio = 1u32;
__t43
}, {
let mut __t44: drm_display_mode = unsafe { core::mem::zeroed() };
__t44.clock = 27000i32;
__t44.hdisplay = 720u16;
__t44.hsync_start = 732u16;
__t44.hsync_end = 795u16;
__t44.htotal = 864u16;
__t44.hskew = 0u16;
__t44.vdisplay = 576u16;
__t44.vsync_start = 580u16;
__t44.vsync_end = 586u16;
__t44.vtotal = 625u16;
__t44.vscan = 0u16;
__t44.flags = 4122u32;
__t44.r#type = 64u8;
__t44.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t44.status = 0i32;
__t44.picture_aspect_ratio = 2u32;
__t44
}, {
let mut __t45: drm_display_mode = unsafe { core::mem::zeroed() };
__t45.clock = 148500i32;
__t45.hdisplay = 1920u16;
__t45.hsync_start = 2008u16;
__t45.hsync_end = 2052u16;
__t45.htotal = 2200u16;
__t45.hskew = 0u16;
__t45.vdisplay = 1080u16;
__t45.vsync_start = 1084u16;
__t45.vsync_end = 1094u16;
__t45.vtotal = 1125u16;
__t45.vscan = 0u16;
__t45.flags = 21u32;
__t45.r#type = 64u8;
__t45.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t45.status = 0i32;
__t45.picture_aspect_ratio = 2u32;
__t45
}, {
let mut __t46: drm_display_mode = unsafe { core::mem::zeroed() };
__t46.clock = 148500i32;
__t46.hdisplay = 1280u16;
__t46.hsync_start = 1390u16;
__t46.hsync_end = 1430u16;
__t46.htotal = 1650u16;
__t46.hskew = 0u16;
__t46.vdisplay = 720u16;
__t46.vsync_start = 725u16;
__t46.vsync_end = 730u16;
__t46.vtotal = 750u16;
__t46.vscan = 0u16;
__t46.flags = 5u32;
__t46.r#type = 64u8;
__t46.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t46.status = 0i32;
__t46.picture_aspect_ratio = 2u32;
__t46
}, {
let mut __t47: drm_display_mode = unsafe { core::mem::zeroed() };
__t47.clock = 54000i32;
__t47.hdisplay = 720u16;
__t47.hsync_start = 736u16;
__t47.hsync_end = 798u16;
__t47.htotal = 858u16;
__t47.hskew = 0u16;
__t47.vdisplay = 480u16;
__t47.vsync_start = 489u16;
__t47.vsync_end = 495u16;
__t47.vtotal = 525u16;
__t47.vscan = 0u16;
__t47.flags = 10u32;
__t47.r#type = 64u8;
__t47.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t47.status = 0i32;
__t47.picture_aspect_ratio = 1u32;
__t47
}, {
let mut __t48: drm_display_mode = unsafe { core::mem::zeroed() };
__t48.clock = 54000i32;
__t48.hdisplay = 720u16;
__t48.hsync_start = 736u16;
__t48.hsync_end = 798u16;
__t48.htotal = 858u16;
__t48.hskew = 0u16;
__t48.vdisplay = 480u16;
__t48.vsync_start = 489u16;
__t48.vsync_end = 495u16;
__t48.vtotal = 525u16;
__t48.vscan = 0u16;
__t48.flags = 10u32;
__t48.r#type = 64u8;
__t48.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t48.status = 0i32;
__t48.picture_aspect_ratio = 2u32;
__t48
}, {
let mut __t49: drm_display_mode = unsafe { core::mem::zeroed() };
__t49.clock = 27000i32;
__t49.hdisplay = 720u16;
__t49.hsync_start = 739u16;
__t49.hsync_end = 801u16;
__t49.htotal = 858u16;
__t49.hskew = 0u16;
__t49.vdisplay = 480u16;
__t49.vsync_start = 488u16;
__t49.vsync_end = 494u16;
__t49.vtotal = 525u16;
__t49.vscan = 0u16;
__t49.flags = 4122u32;
__t49.r#type = 64u8;
__t49.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t49.status = 0i32;
__t49.picture_aspect_ratio = 1u32;
__t49
}, {
let mut __t50: drm_display_mode = unsafe { core::mem::zeroed() };
__t50.clock = 27000i32;
__t50.hdisplay = 720u16;
__t50.hsync_start = 739u16;
__t50.hsync_end = 801u16;
__t50.htotal = 858u16;
__t50.hskew = 0u16;
__t50.vdisplay = 480u16;
__t50.vsync_start = 488u16;
__t50.vsync_end = 494u16;
__t50.vtotal = 525u16;
__t50.vscan = 0u16;
__t50.flags = 4122u32;
__t50.r#type = 64u8;
__t50.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t50.status = 0i32;
__t50.picture_aspect_ratio = 2u32;
__t50
}, {
let mut __t51: drm_display_mode = unsafe { core::mem::zeroed() };
__t51.clock = 108000i32;
__t51.hdisplay = 720u16;
__t51.hsync_start = 732u16;
__t51.hsync_end = 796u16;
__t51.htotal = 864u16;
__t51.hskew = 0u16;
__t51.vdisplay = 576u16;
__t51.vsync_start = 581u16;
__t51.vsync_end = 586u16;
__t51.vtotal = 625u16;
__t51.vscan = 0u16;
__t51.flags = 10u32;
__t51.r#type = 64u8;
__t51.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t51.status = 0i32;
__t51.picture_aspect_ratio = 1u32;
__t51
}, {
let mut __t52: drm_display_mode = unsafe { core::mem::zeroed() };
__t52.clock = 108000i32;
__t52.hdisplay = 720u16;
__t52.hsync_start = 732u16;
__t52.hsync_end = 796u16;
__t52.htotal = 864u16;
__t52.hskew = 0u16;
__t52.vdisplay = 576u16;
__t52.vsync_start = 581u16;
__t52.vsync_end = 586u16;
__t52.vtotal = 625u16;
__t52.vscan = 0u16;
__t52.flags = 10u32;
__t52.r#type = 64u8;
__t52.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t52.status = 0i32;
__t52.picture_aspect_ratio = 2u32;
__t52
}, {
let mut __t53: drm_display_mode = unsafe { core::mem::zeroed() };
__t53.clock = 54000i32;
__t53.hdisplay = 720u16;
__t53.hsync_start = 732u16;
__t53.hsync_end = 795u16;
__t53.htotal = 864u16;
__t53.hskew = 0u16;
__t53.vdisplay = 576u16;
__t53.vsync_start = 580u16;
__t53.vsync_end = 586u16;
__t53.vtotal = 625u16;
__t53.vscan = 0u16;
__t53.flags = 4122u32;
__t53.r#type = 64u8;
__t53.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t53.status = 0i32;
__t53.picture_aspect_ratio = 1u32;
__t53
}, {
let mut __t54: drm_display_mode = unsafe { core::mem::zeroed() };
__t54.clock = 54000i32;
__t54.hdisplay = 720u16;
__t54.hsync_start = 732u16;
__t54.hsync_end = 795u16;
__t54.htotal = 864u16;
__t54.hskew = 0u16;
__t54.vdisplay = 576u16;
__t54.vsync_start = 580u16;
__t54.vsync_end = 586u16;
__t54.vtotal = 625u16;
__t54.vscan = 0u16;
__t54.flags = 4122u32;
__t54.r#type = 64u8;
__t54.name = [55u8, 50u8, 48u8, 120u8, 53u8, 55u8, 54u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t54.status = 0i32;
__t54.picture_aspect_ratio = 2u32;
__t54
}, {
let mut __t55: drm_display_mode = unsafe { core::mem::zeroed() };
__t55.clock = 108000i32;
__t55.hdisplay = 720u16;
__t55.hsync_start = 736u16;
__t55.hsync_end = 798u16;
__t55.htotal = 858u16;
__t55.hskew = 0u16;
__t55.vdisplay = 480u16;
__t55.vsync_start = 489u16;
__t55.vsync_end = 495u16;
__t55.vtotal = 525u16;
__t55.vscan = 0u16;
__t55.flags = 10u32;
__t55.r#type = 64u8;
__t55.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t55.status = 0i32;
__t55.picture_aspect_ratio = 1u32;
__t55
}, {
let mut __t56: drm_display_mode = unsafe { core::mem::zeroed() };
__t56.clock = 108000i32;
__t56.hdisplay = 720u16;
__t56.hsync_start = 736u16;
__t56.hsync_end = 798u16;
__t56.htotal = 858u16;
__t56.hskew = 0u16;
__t56.vdisplay = 480u16;
__t56.vsync_start = 489u16;
__t56.vsync_end = 495u16;
__t56.vtotal = 525u16;
__t56.vscan = 0u16;
__t56.flags = 10u32;
__t56.r#type = 64u8;
__t56.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t56.status = 0i32;
__t56.picture_aspect_ratio = 2u32;
__t56
}, {
let mut __t57: drm_display_mode = unsafe { core::mem::zeroed() };
__t57.clock = 54000i32;
__t57.hdisplay = 720u16;
__t57.hsync_start = 739u16;
__t57.hsync_end = 801u16;
__t57.htotal = 858u16;
__t57.hskew = 0u16;
__t57.vdisplay = 480u16;
__t57.vsync_start = 488u16;
__t57.vsync_end = 494u16;
__t57.vtotal = 525u16;
__t57.vscan = 0u16;
__t57.flags = 4122u32;
__t57.r#type = 64u8;
__t57.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t57.status = 0i32;
__t57.picture_aspect_ratio = 1u32;
__t57
}, {
let mut __t58: drm_display_mode = unsafe { core::mem::zeroed() };
__t58.clock = 54000i32;
__t58.hdisplay = 720u16;
__t58.hsync_start = 739u16;
__t58.hsync_end = 801u16;
__t58.htotal = 858u16;
__t58.hskew = 0u16;
__t58.vdisplay = 480u16;
__t58.vsync_start = 488u16;
__t58.vsync_end = 494u16;
__t58.vtotal = 525u16;
__t58.vscan = 0u16;
__t58.flags = 4122u32;
__t58.r#type = 64u8;
__t58.name = [55u8, 50u8, 48u8, 120u8, 52u8, 56u8, 48u8, 105u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t58.status = 0i32;
__t58.picture_aspect_ratio = 2u32;
__t58
}, {
let mut __t59: drm_display_mode = unsafe { core::mem::zeroed() };
__t59.clock = 59400i32;
__t59.hdisplay = 1280u16;
__t59.hsync_start = 3040u16;
__t59.hsync_end = 3080u16;
__t59.htotal = 3300u16;
__t59.hskew = 0u16;
__t59.vdisplay = 720u16;
__t59.vsync_start = 725u16;
__t59.vsync_end = 730u16;
__t59.vtotal = 750u16;
__t59.vscan = 0u16;
__t59.flags = 5u32;
__t59.r#type = 64u8;
__t59.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t59.status = 0i32;
__t59.picture_aspect_ratio = 2u32;
__t59
}, {
let mut __t60: drm_display_mode = unsafe { core::mem::zeroed() };
__t60.clock = 74250i32;
__t60.hdisplay = 1280u16;
__t60.hsync_start = 3700u16;
__t60.hsync_end = 3740u16;
__t60.htotal = 3960u16;
__t60.hskew = 0u16;
__t60.vdisplay = 720u16;
__t60.vsync_start = 725u16;
__t60.vsync_end = 730u16;
__t60.vtotal = 750u16;
__t60.vscan = 0u16;
__t60.flags = 5u32;
__t60.r#type = 64u8;
__t60.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t60.status = 0i32;
__t60.picture_aspect_ratio = 2u32;
__t60
}, {
let mut __t61: drm_display_mode = unsafe { core::mem::zeroed() };
__t61.clock = 74250i32;
__t61.hdisplay = 1280u16;
__t61.hsync_start = 3040u16;
__t61.hsync_end = 3080u16;
__t61.htotal = 3300u16;
__t61.hskew = 0u16;
__t61.vdisplay = 720u16;
__t61.vsync_start = 725u16;
__t61.vsync_end = 730u16;
__t61.vtotal = 750u16;
__t61.vscan = 0u16;
__t61.flags = 5u32;
__t61.r#type = 64u8;
__t61.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t61.status = 0i32;
__t61.picture_aspect_ratio = 2u32;
__t61
}, {
let mut __t62: drm_display_mode = unsafe { core::mem::zeroed() };
__t62.clock = 297000i32;
__t62.hdisplay = 1920u16;
__t62.hsync_start = 2008u16;
__t62.hsync_end = 2052u16;
__t62.htotal = 2200u16;
__t62.hskew = 0u16;
__t62.vdisplay = 1080u16;
__t62.vsync_start = 1084u16;
__t62.vsync_end = 1089u16;
__t62.vtotal = 1125u16;
__t62.vscan = 0u16;
__t62.flags = 5u32;
__t62.r#type = 64u8;
__t62.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t62.status = 0i32;
__t62.picture_aspect_ratio = 2u32;
__t62
}, {
let mut __t63: drm_display_mode = unsafe { core::mem::zeroed() };
__t63.clock = 297000i32;
__t63.hdisplay = 1920u16;
__t63.hsync_start = 2448u16;
__t63.hsync_end = 2492u16;
__t63.htotal = 2640u16;
__t63.hskew = 0u16;
__t63.vdisplay = 1080u16;
__t63.vsync_start = 1084u16;
__t63.vsync_end = 1089u16;
__t63.vtotal = 1125u16;
__t63.vscan = 0u16;
__t63.flags = 5u32;
__t63.r#type = 64u8;
__t63.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t63.status = 0i32;
__t63.picture_aspect_ratio = 2u32;
__t63
}, {
let mut __t64: drm_display_mode = unsafe { core::mem::zeroed() };
__t64.clock = 59400i32;
__t64.hdisplay = 1280u16;
__t64.hsync_start = 3040u16;
__t64.hsync_end = 3080u16;
__t64.htotal = 3300u16;
__t64.hskew = 0u16;
__t64.vdisplay = 720u16;
__t64.vsync_start = 725u16;
__t64.vsync_end = 730u16;
__t64.vtotal = 750u16;
__t64.vscan = 0u16;
__t64.flags = 5u32;
__t64.r#type = 64u8;
__t64.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t64.status = 0i32;
__t64.picture_aspect_ratio = 3u32;
__t64
}, {
let mut __t65: drm_display_mode = unsafe { core::mem::zeroed() };
__t65.clock = 74250i32;
__t65.hdisplay = 1280u16;
__t65.hsync_start = 3700u16;
__t65.hsync_end = 3740u16;
__t65.htotal = 3960u16;
__t65.hskew = 0u16;
__t65.vdisplay = 720u16;
__t65.vsync_start = 725u16;
__t65.vsync_end = 730u16;
__t65.vtotal = 750u16;
__t65.vscan = 0u16;
__t65.flags = 5u32;
__t65.r#type = 64u8;
__t65.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t65.status = 0i32;
__t65.picture_aspect_ratio = 3u32;
__t65
}, {
let mut __t66: drm_display_mode = unsafe { core::mem::zeroed() };
__t66.clock = 74250i32;
__t66.hdisplay = 1280u16;
__t66.hsync_start = 3040u16;
__t66.hsync_end = 3080u16;
__t66.htotal = 3300u16;
__t66.hskew = 0u16;
__t66.vdisplay = 720u16;
__t66.vsync_start = 725u16;
__t66.vsync_end = 730u16;
__t66.vtotal = 750u16;
__t66.vscan = 0u16;
__t66.flags = 5u32;
__t66.r#type = 64u8;
__t66.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t66.status = 0i32;
__t66.picture_aspect_ratio = 3u32;
__t66
}, {
let mut __t67: drm_display_mode = unsafe { core::mem::zeroed() };
__t67.clock = 74250i32;
__t67.hdisplay = 1280u16;
__t67.hsync_start = 1720u16;
__t67.hsync_end = 1760u16;
__t67.htotal = 1980u16;
__t67.hskew = 0u16;
__t67.vdisplay = 720u16;
__t67.vsync_start = 725u16;
__t67.vsync_end = 730u16;
__t67.vtotal = 750u16;
__t67.vscan = 0u16;
__t67.flags = 5u32;
__t67.r#type = 64u8;
__t67.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t67.status = 0i32;
__t67.picture_aspect_ratio = 3u32;
__t67
}, {
let mut __t68: drm_display_mode = unsafe { core::mem::zeroed() };
__t68.clock = 74250i32;
__t68.hdisplay = 1280u16;
__t68.hsync_start = 1390u16;
__t68.hsync_end = 1430u16;
__t68.htotal = 1650u16;
__t68.hskew = 0u16;
__t68.vdisplay = 720u16;
__t68.vsync_start = 725u16;
__t68.vsync_end = 730u16;
__t68.vtotal = 750u16;
__t68.vscan = 0u16;
__t68.flags = 5u32;
__t68.r#type = 64u8;
__t68.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t68.status = 0i32;
__t68.picture_aspect_ratio = 3u32;
__t68
}, {
let mut __t69: drm_display_mode = unsafe { core::mem::zeroed() };
__t69.clock = 148500i32;
__t69.hdisplay = 1280u16;
__t69.hsync_start = 1720u16;
__t69.hsync_end = 1760u16;
__t69.htotal = 1980u16;
__t69.hskew = 0u16;
__t69.vdisplay = 720u16;
__t69.vsync_start = 725u16;
__t69.vsync_end = 730u16;
__t69.vtotal = 750u16;
__t69.vscan = 0u16;
__t69.flags = 5u32;
__t69.r#type = 64u8;
__t69.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t69.status = 0i32;
__t69.picture_aspect_ratio = 3u32;
__t69
}, {
let mut __t70: drm_display_mode = unsafe { core::mem::zeroed() };
__t70.clock = 148500i32;
__t70.hdisplay = 1280u16;
__t70.hsync_start = 1390u16;
__t70.hsync_end = 1430u16;
__t70.htotal = 1650u16;
__t70.hskew = 0u16;
__t70.vdisplay = 720u16;
__t70.vsync_start = 725u16;
__t70.vsync_end = 730u16;
__t70.vtotal = 750u16;
__t70.vscan = 0u16;
__t70.flags = 5u32;
__t70.r#type = 64u8;
__t70.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t70.status = 0i32;
__t70.picture_aspect_ratio = 3u32;
__t70
}, {
let mut __t71: drm_display_mode = unsafe { core::mem::zeroed() };
__t71.clock = 74250i32;
__t71.hdisplay = 1920u16;
__t71.hsync_start = 2558u16;
__t71.hsync_end = 2602u16;
__t71.htotal = 2750u16;
__t71.hskew = 0u16;
__t71.vdisplay = 1080u16;
__t71.vsync_start = 1084u16;
__t71.vsync_end = 1089u16;
__t71.vtotal = 1125u16;
__t71.vscan = 0u16;
__t71.flags = 5u32;
__t71.r#type = 64u8;
__t71.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t71.status = 0i32;
__t71.picture_aspect_ratio = 3u32;
__t71
}, {
let mut __t72: drm_display_mode = unsafe { core::mem::zeroed() };
__t72.clock = 74250i32;
__t72.hdisplay = 1920u16;
__t72.hsync_start = 2448u16;
__t72.hsync_end = 2492u16;
__t72.htotal = 2640u16;
__t72.hskew = 0u16;
__t72.vdisplay = 1080u16;
__t72.vsync_start = 1084u16;
__t72.vsync_end = 1089u16;
__t72.vtotal = 1125u16;
__t72.vscan = 0u16;
__t72.flags = 5u32;
__t72.r#type = 64u8;
__t72.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t72.status = 0i32;
__t72.picture_aspect_ratio = 3u32;
__t72
}, {
let mut __t73: drm_display_mode = unsafe { core::mem::zeroed() };
__t73.clock = 74250i32;
__t73.hdisplay = 1920u16;
__t73.hsync_start = 2008u16;
__t73.hsync_end = 2052u16;
__t73.htotal = 2200u16;
__t73.hskew = 0u16;
__t73.vdisplay = 1080u16;
__t73.vsync_start = 1084u16;
__t73.vsync_end = 1089u16;
__t73.vtotal = 1125u16;
__t73.vscan = 0u16;
__t73.flags = 5u32;
__t73.r#type = 64u8;
__t73.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t73.status = 0i32;
__t73.picture_aspect_ratio = 3u32;
__t73
}, {
let mut __t74: drm_display_mode = unsafe { core::mem::zeroed() };
__t74.clock = 148500i32;
__t74.hdisplay = 1920u16;
__t74.hsync_start = 2448u16;
__t74.hsync_end = 2492u16;
__t74.htotal = 2640u16;
__t74.hskew = 0u16;
__t74.vdisplay = 1080u16;
__t74.vsync_start = 1084u16;
__t74.vsync_end = 1089u16;
__t74.vtotal = 1125u16;
__t74.vscan = 0u16;
__t74.flags = 5u32;
__t74.r#type = 64u8;
__t74.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t74.status = 0i32;
__t74.picture_aspect_ratio = 3u32;
__t74
}, {
let mut __t75: drm_display_mode = unsafe { core::mem::zeroed() };
__t75.clock = 148500i32;
__t75.hdisplay = 1920u16;
__t75.hsync_start = 2008u16;
__t75.hsync_end = 2052u16;
__t75.htotal = 2200u16;
__t75.hskew = 0u16;
__t75.vdisplay = 1080u16;
__t75.vsync_start = 1084u16;
__t75.vsync_end = 1089u16;
__t75.vtotal = 1125u16;
__t75.vscan = 0u16;
__t75.flags = 5u32;
__t75.r#type = 64u8;
__t75.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t75.status = 0i32;
__t75.picture_aspect_ratio = 3u32;
__t75
}, {
let mut __t76: drm_display_mode = unsafe { core::mem::zeroed() };
__t76.clock = 297000i32;
__t76.hdisplay = 1920u16;
__t76.hsync_start = 2448u16;
__t76.hsync_end = 2492u16;
__t76.htotal = 2640u16;
__t76.hskew = 0u16;
__t76.vdisplay = 1080u16;
__t76.vsync_start = 1084u16;
__t76.vsync_end = 1089u16;
__t76.vtotal = 1125u16;
__t76.vscan = 0u16;
__t76.flags = 5u32;
__t76.r#type = 64u8;
__t76.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t76.status = 0i32;
__t76.picture_aspect_ratio = 3u32;
__t76
}, {
let mut __t77: drm_display_mode = unsafe { core::mem::zeroed() };
__t77.clock = 297000i32;
__t77.hdisplay = 1920u16;
__t77.hsync_start = 2008u16;
__t77.hsync_end = 2052u16;
__t77.htotal = 2200u16;
__t77.hskew = 0u16;
__t77.vdisplay = 1080u16;
__t77.vsync_start = 1084u16;
__t77.vsync_end = 1089u16;
__t77.vtotal = 1125u16;
__t77.vscan = 0u16;
__t77.flags = 5u32;
__t77.r#type = 64u8;
__t77.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t77.status = 0i32;
__t77.picture_aspect_ratio = 3u32;
__t77
}, {
let mut __t78: drm_display_mode = unsafe { core::mem::zeroed() };
__t78.clock = 59400i32;
__t78.hdisplay = 1680u16;
__t78.hsync_start = 3040u16;
__t78.hsync_end = 3080u16;
__t78.htotal = 3300u16;
__t78.hskew = 0u16;
__t78.vdisplay = 720u16;
__t78.vsync_start = 725u16;
__t78.vsync_end = 730u16;
__t78.vtotal = 750u16;
__t78.vscan = 0u16;
__t78.flags = 5u32;
__t78.r#type = 64u8;
__t78.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t78.status = 0i32;
__t78.picture_aspect_ratio = 3u32;
__t78
}, {
let mut __t79: drm_display_mode = unsafe { core::mem::zeroed() };
__t79.clock = 59400i32;
__t79.hdisplay = 1680u16;
__t79.hsync_start = 2908u16;
__t79.hsync_end = 2948u16;
__t79.htotal = 3168u16;
__t79.hskew = 0u16;
__t79.vdisplay = 720u16;
__t79.vsync_start = 725u16;
__t79.vsync_end = 730u16;
__t79.vtotal = 750u16;
__t79.vscan = 0u16;
__t79.flags = 5u32;
__t79.r#type = 64u8;
__t79.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t79.status = 0i32;
__t79.picture_aspect_ratio = 3u32;
__t79
}, {
let mut __t80: drm_display_mode = unsafe { core::mem::zeroed() };
__t80.clock = 59400i32;
__t80.hdisplay = 1680u16;
__t80.hsync_start = 2380u16;
__t80.hsync_end = 2420u16;
__t80.htotal = 2640u16;
__t80.hskew = 0u16;
__t80.vdisplay = 720u16;
__t80.vsync_start = 725u16;
__t80.vsync_end = 730u16;
__t80.vtotal = 750u16;
__t80.vscan = 0u16;
__t80.flags = 5u32;
__t80.r#type = 64u8;
__t80.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t80.status = 0i32;
__t80.picture_aspect_ratio = 3u32;
__t80
}, {
let mut __t81: drm_display_mode = unsafe { core::mem::zeroed() };
__t81.clock = 82500i32;
__t81.hdisplay = 1680u16;
__t81.hsync_start = 1940u16;
__t81.hsync_end = 1980u16;
__t81.htotal = 2200u16;
__t81.hskew = 0u16;
__t81.vdisplay = 720u16;
__t81.vsync_start = 725u16;
__t81.vsync_end = 730u16;
__t81.vtotal = 750u16;
__t81.vscan = 0u16;
__t81.flags = 5u32;
__t81.r#type = 64u8;
__t81.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t81.status = 0i32;
__t81.picture_aspect_ratio = 3u32;
__t81
}, {
let mut __t82: drm_display_mode = unsafe { core::mem::zeroed() };
__t82.clock = 99000i32;
__t82.hdisplay = 1680u16;
__t82.hsync_start = 1940u16;
__t82.hsync_end = 1980u16;
__t82.htotal = 2200u16;
__t82.hskew = 0u16;
__t82.vdisplay = 720u16;
__t82.vsync_start = 725u16;
__t82.vsync_end = 730u16;
__t82.vtotal = 750u16;
__t82.vscan = 0u16;
__t82.flags = 5u32;
__t82.r#type = 64u8;
__t82.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t82.status = 0i32;
__t82.picture_aspect_ratio = 3u32;
__t82
}, {
let mut __t83: drm_display_mode = unsafe { core::mem::zeroed() };
__t83.clock = 165000i32;
__t83.hdisplay = 1680u16;
__t83.hsync_start = 1740u16;
__t83.hsync_end = 1780u16;
__t83.htotal = 2000u16;
__t83.hskew = 0u16;
__t83.vdisplay = 720u16;
__t83.vsync_start = 725u16;
__t83.vsync_end = 730u16;
__t83.vtotal = 825u16;
__t83.vscan = 0u16;
__t83.flags = 5u32;
__t83.r#type = 64u8;
__t83.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t83.status = 0i32;
__t83.picture_aspect_ratio = 3u32;
__t83
}, {
let mut __t84: drm_display_mode = unsafe { core::mem::zeroed() };
__t84.clock = 198000i32;
__t84.hdisplay = 1680u16;
__t84.hsync_start = 1740u16;
__t84.hsync_end = 1780u16;
__t84.htotal = 2000u16;
__t84.hskew = 0u16;
__t84.vdisplay = 720u16;
__t84.vsync_start = 725u16;
__t84.vsync_end = 730u16;
__t84.vtotal = 825u16;
__t84.vscan = 0u16;
__t84.flags = 5u32;
__t84.r#type = 64u8;
__t84.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t84.status = 0i32;
__t84.picture_aspect_ratio = 3u32;
__t84
}, {
let mut __t85: drm_display_mode = unsafe { core::mem::zeroed() };
__t85.clock = 99000i32;
__t85.hdisplay = 2560u16;
__t85.hsync_start = 3558u16;
__t85.hsync_end = 3602u16;
__t85.htotal = 3750u16;
__t85.hskew = 0u16;
__t85.vdisplay = 1080u16;
__t85.vsync_start = 1084u16;
__t85.vsync_end = 1089u16;
__t85.vtotal = 1100u16;
__t85.vscan = 0u16;
__t85.flags = 5u32;
__t85.r#type = 64u8;
__t85.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t85.status = 0i32;
__t85.picture_aspect_ratio = 3u32;
__t85
}, {
let mut __t86: drm_display_mode = unsafe { core::mem::zeroed() };
__t86.clock = 90000i32;
__t86.hdisplay = 2560u16;
__t86.hsync_start = 3008u16;
__t86.hsync_end = 3052u16;
__t86.htotal = 3200u16;
__t86.hskew = 0u16;
__t86.vdisplay = 1080u16;
__t86.vsync_start = 1084u16;
__t86.vsync_end = 1089u16;
__t86.vtotal = 1125u16;
__t86.vscan = 0u16;
__t86.flags = 5u32;
__t86.r#type = 64u8;
__t86.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t86.status = 0i32;
__t86.picture_aspect_ratio = 3u32;
__t86
}, {
let mut __t87: drm_display_mode = unsafe { core::mem::zeroed() };
__t87.clock = 118800i32;
__t87.hdisplay = 2560u16;
__t87.hsync_start = 3328u16;
__t87.hsync_end = 3372u16;
__t87.htotal = 3520u16;
__t87.hskew = 0u16;
__t87.vdisplay = 1080u16;
__t87.vsync_start = 1084u16;
__t87.vsync_end = 1089u16;
__t87.vtotal = 1125u16;
__t87.vscan = 0u16;
__t87.flags = 5u32;
__t87.r#type = 64u8;
__t87.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t87.status = 0i32;
__t87.picture_aspect_ratio = 3u32;
__t87
}, {
let mut __t88: drm_display_mode = unsafe { core::mem::zeroed() };
__t88.clock = 185625i32;
__t88.hdisplay = 2560u16;
__t88.hsync_start = 3108u16;
__t88.hsync_end = 3152u16;
__t88.htotal = 3300u16;
__t88.hskew = 0u16;
__t88.vdisplay = 1080u16;
__t88.vsync_start = 1084u16;
__t88.vsync_end = 1089u16;
__t88.vtotal = 1125u16;
__t88.vscan = 0u16;
__t88.flags = 5u32;
__t88.r#type = 64u8;
__t88.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t88.status = 0i32;
__t88.picture_aspect_ratio = 3u32;
__t88
}, {
let mut __t89: drm_display_mode = unsafe { core::mem::zeroed() };
__t89.clock = 198000i32;
__t89.hdisplay = 2560u16;
__t89.hsync_start = 2808u16;
__t89.hsync_end = 2852u16;
__t89.htotal = 3000u16;
__t89.hskew = 0u16;
__t89.vdisplay = 1080u16;
__t89.vsync_start = 1084u16;
__t89.vsync_end = 1089u16;
__t89.vtotal = 1100u16;
__t89.vscan = 0u16;
__t89.flags = 5u32;
__t89.r#type = 64u8;
__t89.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t89.status = 0i32;
__t89.picture_aspect_ratio = 3u32;
__t89
}, {
let mut __t90: drm_display_mode = unsafe { core::mem::zeroed() };
__t90.clock = 371250i32;
__t90.hdisplay = 2560u16;
__t90.hsync_start = 2778u16;
__t90.hsync_end = 2822u16;
__t90.htotal = 2970u16;
__t90.hskew = 0u16;
__t90.vdisplay = 1080u16;
__t90.vsync_start = 1084u16;
__t90.vsync_end = 1089u16;
__t90.vtotal = 1250u16;
__t90.vscan = 0u16;
__t90.flags = 5u32;
__t90.r#type = 64u8;
__t90.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t90.status = 0i32;
__t90.picture_aspect_ratio = 3u32;
__t90
}, {
let mut __t91: drm_display_mode = unsafe { core::mem::zeroed() };
__t91.clock = 495000i32;
__t91.hdisplay = 2560u16;
__t91.hsync_start = 3108u16;
__t91.hsync_end = 3152u16;
__t91.htotal = 3300u16;
__t91.hskew = 0u16;
__t91.vdisplay = 1080u16;
__t91.vsync_start = 1084u16;
__t91.vsync_end = 1089u16;
__t91.vtotal = 1250u16;
__t91.vscan = 0u16;
__t91.flags = 5u32;
__t91.r#type = 64u8;
__t91.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t91.status = 0i32;
__t91.picture_aspect_ratio = 3u32;
__t91
}, {
let mut __t92: drm_display_mode = unsafe { core::mem::zeroed() };
__t92.clock = 297000i32;
__t92.hdisplay = 3840u16;
__t92.hsync_start = 5116u16;
__t92.hsync_end = 5204u16;
__t92.htotal = 5500u16;
__t92.hskew = 0u16;
__t92.vdisplay = 2160u16;
__t92.vsync_start = 2168u16;
__t92.vsync_end = 2178u16;
__t92.vtotal = 2250u16;
__t92.vscan = 0u16;
__t92.flags = 5u32;
__t92.r#type = 64u8;
__t92.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t92.status = 0i32;
__t92.picture_aspect_ratio = 2u32;
__t92
}, {
let mut __t93: drm_display_mode = unsafe { core::mem::zeroed() };
__t93.clock = 297000i32;
__t93.hdisplay = 3840u16;
__t93.hsync_start = 4896u16;
__t93.hsync_end = 4984u16;
__t93.htotal = 5280u16;
__t93.hskew = 0u16;
__t93.vdisplay = 2160u16;
__t93.vsync_start = 2168u16;
__t93.vsync_end = 2178u16;
__t93.vtotal = 2250u16;
__t93.vscan = 0u16;
__t93.flags = 5u32;
__t93.r#type = 64u8;
__t93.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t93.status = 0i32;
__t93.picture_aspect_ratio = 2u32;
__t93
}, {
let mut __t94: drm_display_mode = unsafe { core::mem::zeroed() };
__t94.clock = 297000i32;
__t94.hdisplay = 3840u16;
__t94.hsync_start = 4016u16;
__t94.hsync_end = 4104u16;
__t94.htotal = 4400u16;
__t94.hskew = 0u16;
__t94.vdisplay = 2160u16;
__t94.vsync_start = 2168u16;
__t94.vsync_end = 2178u16;
__t94.vtotal = 2250u16;
__t94.vscan = 0u16;
__t94.flags = 5u32;
__t94.r#type = 64u8;
__t94.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t94.status = 0i32;
__t94.picture_aspect_ratio = 2u32;
__t94
}, {
let mut __t95: drm_display_mode = unsafe { core::mem::zeroed() };
__t95.clock = 594000i32;
__t95.hdisplay = 3840u16;
__t95.hsync_start = 4896u16;
__t95.hsync_end = 4984u16;
__t95.htotal = 5280u16;
__t95.hskew = 0u16;
__t95.vdisplay = 2160u16;
__t95.vsync_start = 2168u16;
__t95.vsync_end = 2178u16;
__t95.vtotal = 2250u16;
__t95.vscan = 0u16;
__t95.flags = 5u32;
__t95.r#type = 64u8;
__t95.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t95.status = 0i32;
__t95.picture_aspect_ratio = 2u32;
__t95
}, {
let mut __t96: drm_display_mode = unsafe { core::mem::zeroed() };
__t96.clock = 594000i32;
__t96.hdisplay = 3840u16;
__t96.hsync_start = 4016u16;
__t96.hsync_end = 4104u16;
__t96.htotal = 4400u16;
__t96.hskew = 0u16;
__t96.vdisplay = 2160u16;
__t96.vsync_start = 2168u16;
__t96.vsync_end = 2178u16;
__t96.vtotal = 2250u16;
__t96.vscan = 0u16;
__t96.flags = 5u32;
__t96.r#type = 64u8;
__t96.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t96.status = 0i32;
__t96.picture_aspect_ratio = 2u32;
__t96
}, {
let mut __t97: drm_display_mode = unsafe { core::mem::zeroed() };
__t97.clock = 297000i32;
__t97.hdisplay = 4096u16;
__t97.hsync_start = 5116u16;
__t97.hsync_end = 5204u16;
__t97.htotal = 5500u16;
__t97.hskew = 0u16;
__t97.vdisplay = 2160u16;
__t97.vsync_start = 2168u16;
__t97.vsync_end = 2178u16;
__t97.vtotal = 2250u16;
__t97.vscan = 0u16;
__t97.flags = 5u32;
__t97.r#type = 64u8;
__t97.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t97.status = 0i32;
__t97.picture_aspect_ratio = 4u32;
__t97
}, {
let mut __t98: drm_display_mode = unsafe { core::mem::zeroed() };
__t98.clock = 297000i32;
__t98.hdisplay = 4096u16;
__t98.hsync_start = 5064u16;
__t98.hsync_end = 5152u16;
__t98.htotal = 5280u16;
__t98.hskew = 0u16;
__t98.vdisplay = 2160u16;
__t98.vsync_start = 2168u16;
__t98.vsync_end = 2178u16;
__t98.vtotal = 2250u16;
__t98.vscan = 0u16;
__t98.flags = 5u32;
__t98.r#type = 64u8;
__t98.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t98.status = 0i32;
__t98.picture_aspect_ratio = 4u32;
__t98
}, {
let mut __t99: drm_display_mode = unsafe { core::mem::zeroed() };
__t99.clock = 297000i32;
__t99.hdisplay = 4096u16;
__t99.hsync_start = 4184u16;
__t99.hsync_end = 4272u16;
__t99.htotal = 4400u16;
__t99.hskew = 0u16;
__t99.vdisplay = 2160u16;
__t99.vsync_start = 2168u16;
__t99.vsync_end = 2178u16;
__t99.vtotal = 2250u16;
__t99.vscan = 0u16;
__t99.flags = 5u32;
__t99.r#type = 64u8;
__t99.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t99.status = 0i32;
__t99.picture_aspect_ratio = 4u32;
__t99
}, {
let mut __t100: drm_display_mode = unsafe { core::mem::zeroed() };
__t100.clock = 594000i32;
__t100.hdisplay = 4096u16;
__t100.hsync_start = 5064u16;
__t100.hsync_end = 5152u16;
__t100.htotal = 5280u16;
__t100.hskew = 0u16;
__t100.vdisplay = 2160u16;
__t100.vsync_start = 2168u16;
__t100.vsync_end = 2178u16;
__t100.vtotal = 2250u16;
__t100.vscan = 0u16;
__t100.flags = 5u32;
__t100.r#type = 64u8;
__t100.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t100.status = 0i32;
__t100.picture_aspect_ratio = 4u32;
__t100
}, {
let mut __t101: drm_display_mode = unsafe { core::mem::zeroed() };
__t101.clock = 594000i32;
__t101.hdisplay = 4096u16;
__t101.hsync_start = 4184u16;
__t101.hsync_end = 4272u16;
__t101.htotal = 4400u16;
__t101.hskew = 0u16;
__t101.vdisplay = 2160u16;
__t101.vsync_start = 2168u16;
__t101.vsync_end = 2178u16;
__t101.vtotal = 2250u16;
__t101.vscan = 0u16;
__t101.flags = 5u32;
__t101.r#type = 64u8;
__t101.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t101.status = 0i32;
__t101.picture_aspect_ratio = 4u32;
__t101
}, {
let mut __t102: drm_display_mode = unsafe { core::mem::zeroed() };
__t102.clock = 297000i32;
__t102.hdisplay = 3840u16;
__t102.hsync_start = 5116u16;
__t102.hsync_end = 5204u16;
__t102.htotal = 5500u16;
__t102.hskew = 0u16;
__t102.vdisplay = 2160u16;
__t102.vsync_start = 2168u16;
__t102.vsync_end = 2178u16;
__t102.vtotal = 2250u16;
__t102.vscan = 0u16;
__t102.flags = 5u32;
__t102.r#type = 64u8;
__t102.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t102.status = 0i32;
__t102.picture_aspect_ratio = 3u32;
__t102
}, {
let mut __t103: drm_display_mode = unsafe { core::mem::zeroed() };
__t103.clock = 297000i32;
__t103.hdisplay = 3840u16;
__t103.hsync_start = 4896u16;
__t103.hsync_end = 4984u16;
__t103.htotal = 5280u16;
__t103.hskew = 0u16;
__t103.vdisplay = 2160u16;
__t103.vsync_start = 2168u16;
__t103.vsync_end = 2178u16;
__t103.vtotal = 2250u16;
__t103.vscan = 0u16;
__t103.flags = 5u32;
__t103.r#type = 64u8;
__t103.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t103.status = 0i32;
__t103.picture_aspect_ratio = 3u32;
__t103
}, {
let mut __t104: drm_display_mode = unsafe { core::mem::zeroed() };
__t104.clock = 297000i32;
__t104.hdisplay = 3840u16;
__t104.hsync_start = 4016u16;
__t104.hsync_end = 4104u16;
__t104.htotal = 4400u16;
__t104.hskew = 0u16;
__t104.vdisplay = 2160u16;
__t104.vsync_start = 2168u16;
__t104.vsync_end = 2178u16;
__t104.vtotal = 2250u16;
__t104.vscan = 0u16;
__t104.flags = 5u32;
__t104.r#type = 64u8;
__t104.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t104.status = 0i32;
__t104.picture_aspect_ratio = 3u32;
__t104
}, {
let mut __t105: drm_display_mode = unsafe { core::mem::zeroed() };
__t105.clock = 594000i32;
__t105.hdisplay = 3840u16;
__t105.hsync_start = 4896u16;
__t105.hsync_end = 4984u16;
__t105.htotal = 5280u16;
__t105.hskew = 0u16;
__t105.vdisplay = 2160u16;
__t105.vsync_start = 2168u16;
__t105.vsync_end = 2178u16;
__t105.vtotal = 2250u16;
__t105.vscan = 0u16;
__t105.flags = 5u32;
__t105.r#type = 64u8;
__t105.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t105.status = 0i32;
__t105.picture_aspect_ratio = 3u32;
__t105
}, {
let mut __t106: drm_display_mode = unsafe { core::mem::zeroed() };
__t106.clock = 594000i32;
__t106.hdisplay = 3840u16;
__t106.hsync_start = 4016u16;
__t106.hsync_end = 4104u16;
__t106.htotal = 4400u16;
__t106.hskew = 0u16;
__t106.vdisplay = 2160u16;
__t106.vsync_start = 2168u16;
__t106.vsync_end = 2178u16;
__t106.vtotal = 2250u16;
__t106.vscan = 0u16;
__t106.flags = 5u32;
__t106.r#type = 64u8;
__t106.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t106.status = 0i32;
__t106.picture_aspect_ratio = 3u32;
__t106
}, {
let mut __t107: drm_display_mode = unsafe { core::mem::zeroed() };
__t107.clock = 90000i32;
__t107.hdisplay = 1280u16;
__t107.hsync_start = 2240u16;
__t107.hsync_end = 2280u16;
__t107.htotal = 2500u16;
__t107.hskew = 0u16;
__t107.vdisplay = 720u16;
__t107.vsync_start = 725u16;
__t107.vsync_end = 730u16;
__t107.vtotal = 750u16;
__t107.vscan = 0u16;
__t107.flags = 5u32;
__t107.r#type = 64u8;
__t107.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t107.status = 0i32;
__t107.picture_aspect_ratio = 2u32;
__t107
}, {
let mut __t108: drm_display_mode = unsafe { core::mem::zeroed() };
__t108.clock = 90000i32;
__t108.hdisplay = 1280u16;
__t108.hsync_start = 2240u16;
__t108.hsync_end = 2280u16;
__t108.htotal = 2500u16;
__t108.hskew = 0u16;
__t108.vdisplay = 720u16;
__t108.vsync_start = 725u16;
__t108.vsync_end = 730u16;
__t108.vtotal = 750u16;
__t108.vscan = 0u16;
__t108.flags = 5u32;
__t108.r#type = 64u8;
__t108.name = [49u8, 50u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t108.status = 0i32;
__t108.picture_aspect_ratio = 3u32;
__t108
}, {
let mut __t109: drm_display_mode = unsafe { core::mem::zeroed() };
__t109.clock = 99000i32;
__t109.hdisplay = 1680u16;
__t109.hsync_start = 2490u16;
__t109.hsync_end = 2530u16;
__t109.htotal = 2750u16;
__t109.hskew = 0u16;
__t109.vdisplay = 720u16;
__t109.vsync_start = 725u16;
__t109.vsync_end = 730u16;
__t109.vtotal = 750u16;
__t109.vscan = 0u16;
__t109.flags = 5u32;
__t109.r#type = 64u8;
__t109.name = [49u8, 54u8, 56u8, 48u8, 120u8, 55u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t109.status = 0i32;
__t109.picture_aspect_ratio = 3u32;
__t109
}, {
let mut __t110: drm_display_mode = unsafe { core::mem::zeroed() };
__t110.clock = 148500i32;
__t110.hdisplay = 1920u16;
__t110.hsync_start = 2558u16;
__t110.hsync_end = 2602u16;
__t110.htotal = 2750u16;
__t110.hskew = 0u16;
__t110.vdisplay = 1080u16;
__t110.vsync_start = 1084u16;
__t110.vsync_end = 1089u16;
__t110.vtotal = 1125u16;
__t110.vscan = 0u16;
__t110.flags = 5u32;
__t110.r#type = 64u8;
__t110.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t110.status = 0i32;
__t110.picture_aspect_ratio = 2u32;
__t110
}, {
let mut __t111: drm_display_mode = unsafe { core::mem::zeroed() };
__t111.clock = 148500i32;
__t111.hdisplay = 1920u16;
__t111.hsync_start = 2558u16;
__t111.hsync_end = 2602u16;
__t111.htotal = 2750u16;
__t111.hskew = 0u16;
__t111.vdisplay = 1080u16;
__t111.vsync_start = 1084u16;
__t111.vsync_end = 1089u16;
__t111.vtotal = 1125u16;
__t111.vscan = 0u16;
__t111.flags = 5u32;
__t111.r#type = 64u8;
__t111.name = [49u8, 57u8, 50u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t111.status = 0i32;
__t111.picture_aspect_ratio = 3u32;
__t111
}, {
let mut __t112: drm_display_mode = unsafe { core::mem::zeroed() };
__t112.clock = 198000i32;
__t112.hdisplay = 2560u16;
__t112.hsync_start = 3558u16;
__t112.hsync_end = 3602u16;
__t112.htotal = 3750u16;
__t112.hskew = 0u16;
__t112.vdisplay = 1080u16;
__t112.vsync_start = 1084u16;
__t112.vsync_end = 1089u16;
__t112.vtotal = 1100u16;
__t112.vscan = 0u16;
__t112.flags = 5u32;
__t112.r#type = 64u8;
__t112.name = [50u8, 53u8, 54u8, 48u8, 120u8, 49u8, 48u8, 56u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t112.status = 0i32;
__t112.picture_aspect_ratio = 3u32;
__t112
}, {
let mut __t113: drm_display_mode = unsafe { core::mem::zeroed() };
__t113.clock = 594000i32;
__t113.hdisplay = 3840u16;
__t113.hsync_start = 5116u16;
__t113.hsync_end = 5204u16;
__t113.htotal = 5500u16;
__t113.hskew = 0u16;
__t113.vdisplay = 2160u16;
__t113.vsync_start = 2168u16;
__t113.vsync_end = 2178u16;
__t113.vtotal = 2250u16;
__t113.vscan = 0u16;
__t113.flags = 5u32;
__t113.r#type = 64u8;
__t113.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t113.status = 0i32;
__t113.picture_aspect_ratio = 2u32;
__t113
}, {
let mut __t114: drm_display_mode = unsafe { core::mem::zeroed() };
__t114.clock = 594000i32;
__t114.hdisplay = 4096u16;
__t114.hsync_start = 5116u16;
__t114.hsync_end = 5204u16;
__t114.htotal = 5500u16;
__t114.hskew = 0u16;
__t114.vdisplay = 2160u16;
__t114.vsync_start = 2168u16;
__t114.vsync_end = 2178u16;
__t114.vtotal = 2250u16;
__t114.vscan = 0u16;
__t114.flags = 5u32;
__t114.r#type = 64u8;
__t114.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t114.status = 0i32;
__t114.picture_aspect_ratio = 4u32;
__t114
}, {
let mut __t115: drm_display_mode = unsafe { core::mem::zeroed() };
__t115.clock = 594000i32;
__t115.hdisplay = 3840u16;
__t115.hsync_start = 5116u16;
__t115.hsync_end = 5204u16;
__t115.htotal = 5500u16;
__t115.hskew = 0u16;
__t115.vdisplay = 2160u16;
__t115.vsync_start = 2168u16;
__t115.vsync_end = 2178u16;
__t115.vtotal = 2250u16;
__t115.vscan = 0u16;
__t115.flags = 5u32;
__t115.r#type = 64u8;
__t115.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t115.status = 0i32;
__t115.picture_aspect_ratio = 3u32;
__t115
}, {
let mut __t116: drm_display_mode = unsafe { core::mem::zeroed() };
__t116.clock = 1188000i32;
__t116.hdisplay = 3840u16;
__t116.hsync_start = 4896u16;
__t116.hsync_end = 4984u16;
__t116.htotal = 5280u16;
__t116.hskew = 0u16;
__t116.vdisplay = 2160u16;
__t116.vsync_start = 2168u16;
__t116.vsync_end = 2178u16;
__t116.vtotal = 2250u16;
__t116.vscan = 0u16;
__t116.flags = 5u32;
__t116.r#type = 64u8;
__t116.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t116.status = 0i32;
__t116.picture_aspect_ratio = 2u32;
__t116
}, {
let mut __t117: drm_display_mode = unsafe { core::mem::zeroed() };
__t117.clock = 1188000i32;
__t117.hdisplay = 3840u16;
__t117.hsync_start = 4016u16;
__t117.hsync_end = 4104u16;
__t117.htotal = 4400u16;
__t117.hskew = 0u16;
__t117.vdisplay = 2160u16;
__t117.vsync_start = 2168u16;
__t117.vsync_end = 2178u16;
__t117.vtotal = 2250u16;
__t117.vscan = 0u16;
__t117.flags = 5u32;
__t117.r#type = 64u8;
__t117.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t117.status = 0i32;
__t117.picture_aspect_ratio = 2u32;
__t117
}, {
let mut __t118: drm_display_mode = unsafe { core::mem::zeroed() };
__t118.clock = 1188000i32;
__t118.hdisplay = 3840u16;
__t118.hsync_start = 4896u16;
__t118.hsync_end = 4984u16;
__t118.htotal = 5280u16;
__t118.hskew = 0u16;
__t118.vdisplay = 2160u16;
__t118.vsync_start = 2168u16;
__t118.vsync_end = 2178u16;
__t118.vtotal = 2250u16;
__t118.vscan = 0u16;
__t118.flags = 5u32;
__t118.r#type = 64u8;
__t118.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t118.status = 0i32;
__t118.picture_aspect_ratio = 3u32;
__t118
}, {
let mut __t119: drm_display_mode = unsafe { core::mem::zeroed() };
__t119.clock = 1188000i32;
__t119.hdisplay = 3840u16;
__t119.hsync_start = 4016u16;
__t119.hsync_end = 4104u16;
__t119.htotal = 4400u16;
__t119.hskew = 0u16;
__t119.vdisplay = 2160u16;
__t119.vsync_start = 2168u16;
__t119.vsync_end = 2178u16;
__t119.vtotal = 2250u16;
__t119.vscan = 0u16;
__t119.flags = 5u32;
__t119.r#type = 64u8;
__t119.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t119.status = 0i32;
__t119.picture_aspect_ratio = 3u32;
__t119
}, {
let mut __t120: drm_display_mode = unsafe { core::mem::zeroed() };
__t120.clock = 396000i32;
__t120.hdisplay = 5120u16;
__t120.hsync_start = 7116u16;
__t120.hsync_end = 7204u16;
__t120.htotal = 7500u16;
__t120.hskew = 0u16;
__t120.vdisplay = 2160u16;
__t120.vsync_start = 2168u16;
__t120.vsync_end = 2178u16;
__t120.vtotal = 2200u16;
__t120.vscan = 0u16;
__t120.flags = 5u32;
__t120.r#type = 64u8;
__t120.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t120.status = 0i32;
__t120.picture_aspect_ratio = 3u32;
__t120
}, {
let mut __t121: drm_display_mode = unsafe { core::mem::zeroed() };
__t121.clock = 396000i32;
__t121.hdisplay = 5120u16;
__t121.hsync_start = 6816u16;
__t121.hsync_end = 6904u16;
__t121.htotal = 7200u16;
__t121.hskew = 0u16;
__t121.vdisplay = 2160u16;
__t121.vsync_start = 2168u16;
__t121.vsync_end = 2178u16;
__t121.vtotal = 2200u16;
__t121.vscan = 0u16;
__t121.flags = 5u32;
__t121.r#type = 64u8;
__t121.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t121.status = 0i32;
__t121.picture_aspect_ratio = 3u32;
__t121
}, {
let mut __t122: drm_display_mode = unsafe { core::mem::zeroed() };
__t122.clock = 396000i32;
__t122.hdisplay = 5120u16;
__t122.hsync_start = 5784u16;
__t122.hsync_end = 5872u16;
__t122.htotal = 6000u16;
__t122.hskew = 0u16;
__t122.vdisplay = 2160u16;
__t122.vsync_start = 2168u16;
__t122.vsync_end = 2178u16;
__t122.vtotal = 2200u16;
__t122.vscan = 0u16;
__t122.flags = 5u32;
__t122.r#type = 64u8;
__t122.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t122.status = 0i32;
__t122.picture_aspect_ratio = 3u32;
__t122
}, {
let mut __t123: drm_display_mode = unsafe { core::mem::zeroed() };
__t123.clock = 742500i32;
__t123.hdisplay = 5120u16;
__t123.hsync_start = 5866u16;
__t123.hsync_end = 5954u16;
__t123.htotal = 6250u16;
__t123.hskew = 0u16;
__t123.vdisplay = 2160u16;
__t123.vsync_start = 2168u16;
__t123.vsync_end = 2178u16;
__t123.vtotal = 2475u16;
__t123.vscan = 0u16;
__t123.flags = 5u32;
__t123.r#type = 64u8;
__t123.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t123.status = 0i32;
__t123.picture_aspect_ratio = 3u32;
__t123
}, {
let mut __t124: drm_display_mode = unsafe { core::mem::zeroed() };
__t124.clock = 742500i32;
__t124.hdisplay = 5120u16;
__t124.hsync_start = 6216u16;
__t124.hsync_end = 6304u16;
__t124.htotal = 6600u16;
__t124.hskew = 0u16;
__t124.vdisplay = 2160u16;
__t124.vsync_start = 2168u16;
__t124.vsync_end = 2178u16;
__t124.vtotal = 2250u16;
__t124.vscan = 0u16;
__t124.flags = 5u32;
__t124.r#type = 64u8;
__t124.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t124.status = 0i32;
__t124.picture_aspect_ratio = 3u32;
__t124
}, {
let mut __t125: drm_display_mode = unsafe { core::mem::zeroed() };
__t125.clock = 742500i32;
__t125.hdisplay = 5120u16;
__t125.hsync_start = 5284u16;
__t125.hsync_end = 5372u16;
__t125.htotal = 5500u16;
__t125.hskew = 0u16;
__t125.vdisplay = 2160u16;
__t125.vsync_start = 2168u16;
__t125.vsync_end = 2178u16;
__t125.vtotal = 2250u16;
__t125.vscan = 0u16;
__t125.flags = 5u32;
__t125.r#type = 64u8;
__t125.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t125.status = 0i32;
__t125.picture_aspect_ratio = 3u32;
__t125
}, {
let mut __t126: drm_display_mode = unsafe { core::mem::zeroed() };
__t126.clock = 1485000i32;
__t126.hdisplay = 5120u16;
__t126.hsync_start = 6216u16;
__t126.hsync_end = 6304u16;
__t126.htotal = 6600u16;
__t126.hskew = 0u16;
__t126.vdisplay = 2160u16;
__t126.vsync_start = 2168u16;
__t126.vsync_end = 2178u16;
__t126.vtotal = 2250u16;
__t126.vscan = 0u16;
__t126.flags = 5u32;
__t126.r#type = 64u8;
__t126.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t126.status = 0i32;
__t126.picture_aspect_ratio = 3u32;
__t126
}]);

/// From CEA/CTA-861 spec.
///
/// Do not access directly, instead always use cea_mode_for_vic().
/// 193 - 5120x2160@120Hz 64:27
/// 194 - 7680x4320@24Hz 16:9
/// 195 - 7680x4320@25Hz 16:9
/// 196 - 7680x4320@30Hz 16:9
/// 197 - 7680x4320@48Hz 16:9
/// 198 - 7680x4320@50Hz 16:9
/// 199 - 7680x4320@60Hz 16:9
/// 200 - 7680x4320@100Hz 16:9
/// 201 - 7680x4320@120Hz 16:9
/// 202 - 7680x4320@24Hz 64:27
/// 203 - 7680x4320@25Hz 64:27
/// 204 - 7680x4320@30Hz 64:27
/// 205 - 7680x4320@48Hz 64:27
/// 206 - 7680x4320@50Hz 64:27
/// 207 - 7680x4320@60Hz 64:27
/// 208 - 7680x4320@100Hz 64:27
/// 209 - 7680x4320@120Hz 64:27
/// 210 - 10240x4320@24Hz 64:27
/// 211 - 10240x4320@25Hz 64:27
/// 212 - 10240x4320@30Hz 64:27
/// 213 - 10240x4320@48Hz 64:27
/// 214 - 10240x4320@50Hz 64:27
/// 215 - 10240x4320@60Hz 64:27
/// 216 - 10240x4320@100Hz 64:27
/// 217 - 10240x4320@120Hz 64:27
/// 218 - 4096x2160@100Hz 256:135
/// 219 - 4096x2160@120Hz 256:135
static edid_cea_modes_193: __SlateAlign16<[drm_display_mode; 27]> = __SlateAlign16([{
let mut __t0: drm_display_mode = unsafe { core::mem::zeroed() };
__t0.clock = 1485000i32;
__t0.hdisplay = 5120u16;
__t0.hsync_start = 5284u16;
__t0.hsync_end = 5372u16;
__t0.htotal = 5500u16;
__t0.hskew = 0u16;
__t0.vdisplay = 2160u16;
__t0.vsync_start = 2168u16;
__t0.vsync_end = 2178u16;
__t0.vtotal = 2250u16;
__t0.vscan = 0u16;
__t0.flags = 5u32;
__t0.r#type = 64u8;
__t0.name = [53u8, 49u8, 50u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.status = 0i32;
__t0.picture_aspect_ratio = 3u32;
__t0
}, {
let mut __t1: drm_display_mode = unsafe { core::mem::zeroed() };
__t1.clock = 1188000i32;
__t1.hdisplay = 7680u16;
__t1.hsync_start = 10232u16;
__t1.hsync_end = 10408u16;
__t1.htotal = 11000u16;
__t1.hskew = 0u16;
__t1.vdisplay = 4320u16;
__t1.vsync_start = 4336u16;
__t1.vsync_end = 4356u16;
__t1.vtotal = 4500u16;
__t1.vscan = 0u16;
__t1.flags = 5u32;
__t1.r#type = 64u8;
__t1.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.status = 0i32;
__t1.picture_aspect_ratio = 2u32;
__t1
}, {
let mut __t2: drm_display_mode = unsafe { core::mem::zeroed() };
__t2.clock = 1188000i32;
__t2.hdisplay = 7680u16;
__t2.hsync_start = 10032u16;
__t2.hsync_end = 10208u16;
__t2.htotal = 10800u16;
__t2.hskew = 0u16;
__t2.vdisplay = 4320u16;
__t2.vsync_start = 4336u16;
__t2.vsync_end = 4356u16;
__t2.vtotal = 4400u16;
__t2.vscan = 0u16;
__t2.flags = 5u32;
__t2.r#type = 64u8;
__t2.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.status = 0i32;
__t2.picture_aspect_ratio = 2u32;
__t2
}, {
let mut __t3: drm_display_mode = unsafe { core::mem::zeroed() };
__t3.clock = 1188000i32;
__t3.hdisplay = 7680u16;
__t3.hsync_start = 8232u16;
__t3.hsync_end = 8408u16;
__t3.htotal = 9000u16;
__t3.hskew = 0u16;
__t3.vdisplay = 4320u16;
__t3.vsync_start = 4336u16;
__t3.vsync_end = 4356u16;
__t3.vtotal = 4400u16;
__t3.vscan = 0u16;
__t3.flags = 5u32;
__t3.r#type = 64u8;
__t3.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.status = 0i32;
__t3.picture_aspect_ratio = 2u32;
__t3
}, {
let mut __t4: drm_display_mode = unsafe { core::mem::zeroed() };
__t4.clock = 2376000i32;
__t4.hdisplay = 7680u16;
__t4.hsync_start = 10232u16;
__t4.hsync_end = 10408u16;
__t4.htotal = 11000u16;
__t4.hskew = 0u16;
__t4.vdisplay = 4320u16;
__t4.vsync_start = 4336u16;
__t4.vsync_end = 4356u16;
__t4.vtotal = 4500u16;
__t4.vscan = 0u16;
__t4.flags = 5u32;
__t4.r#type = 64u8;
__t4.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.status = 0i32;
__t4.picture_aspect_ratio = 2u32;
__t4
}, {
let mut __t5: drm_display_mode = unsafe { core::mem::zeroed() };
__t5.clock = 2376000i32;
__t5.hdisplay = 7680u16;
__t5.hsync_start = 10032u16;
__t5.hsync_end = 10208u16;
__t5.htotal = 10800u16;
__t5.hskew = 0u16;
__t5.vdisplay = 4320u16;
__t5.vsync_start = 4336u16;
__t5.vsync_end = 4356u16;
__t5.vtotal = 4400u16;
__t5.vscan = 0u16;
__t5.flags = 5u32;
__t5.r#type = 64u8;
__t5.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5.status = 0i32;
__t5.picture_aspect_ratio = 2u32;
__t5
}, {
let mut __t6: drm_display_mode = unsafe { core::mem::zeroed() };
__t6.clock = 2376000i32;
__t6.hdisplay = 7680u16;
__t6.hsync_start = 8232u16;
__t6.hsync_end = 8408u16;
__t6.htotal = 9000u16;
__t6.hskew = 0u16;
__t6.vdisplay = 4320u16;
__t6.vsync_start = 4336u16;
__t6.vsync_end = 4356u16;
__t6.vtotal = 4400u16;
__t6.vscan = 0u16;
__t6.flags = 5u32;
__t6.r#type = 64u8;
__t6.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.status = 0i32;
__t6.picture_aspect_ratio = 2u32;
__t6
}, {
let mut __t7: drm_display_mode = unsafe { core::mem::zeroed() };
__t7.clock = 4752000i32;
__t7.hdisplay = 7680u16;
__t7.hsync_start = 9792u16;
__t7.hsync_end = 9968u16;
__t7.htotal = 10560u16;
__t7.hskew = 0u16;
__t7.vdisplay = 4320u16;
__t7.vsync_start = 4336u16;
__t7.vsync_end = 4356u16;
__t7.vtotal = 4500u16;
__t7.vscan = 0u16;
__t7.flags = 5u32;
__t7.r#type = 64u8;
__t7.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.status = 0i32;
__t7.picture_aspect_ratio = 2u32;
__t7
}, {
let mut __t8: drm_display_mode = unsafe { core::mem::zeroed() };
__t8.clock = 4752000i32;
__t8.hdisplay = 7680u16;
__t8.hsync_start = 8032u16;
__t8.hsync_end = 8208u16;
__t8.htotal = 8800u16;
__t8.hskew = 0u16;
__t8.vdisplay = 4320u16;
__t8.vsync_start = 4336u16;
__t8.vsync_end = 4356u16;
__t8.vtotal = 4500u16;
__t8.vscan = 0u16;
__t8.flags = 5u32;
__t8.r#type = 64u8;
__t8.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8.status = 0i32;
__t8.picture_aspect_ratio = 2u32;
__t8
}, {
let mut __t9: drm_display_mode = unsafe { core::mem::zeroed() };
__t9.clock = 1188000i32;
__t9.hdisplay = 7680u16;
__t9.hsync_start = 10232u16;
__t9.hsync_end = 10408u16;
__t9.htotal = 11000u16;
__t9.hskew = 0u16;
__t9.vdisplay = 4320u16;
__t9.vsync_start = 4336u16;
__t9.vsync_end = 4356u16;
__t9.vtotal = 4500u16;
__t9.vscan = 0u16;
__t9.flags = 5u32;
__t9.r#type = 64u8;
__t9.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t9.status = 0i32;
__t9.picture_aspect_ratio = 3u32;
__t9
}, {
let mut __t10: drm_display_mode = unsafe { core::mem::zeroed() };
__t10.clock = 1188000i32;
__t10.hdisplay = 7680u16;
__t10.hsync_start = 10032u16;
__t10.hsync_end = 10208u16;
__t10.htotal = 10800u16;
__t10.hskew = 0u16;
__t10.vdisplay = 4320u16;
__t10.vsync_start = 4336u16;
__t10.vsync_end = 4356u16;
__t10.vtotal = 4400u16;
__t10.vscan = 0u16;
__t10.flags = 5u32;
__t10.r#type = 64u8;
__t10.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10.status = 0i32;
__t10.picture_aspect_ratio = 3u32;
__t10
}, {
let mut __t11: drm_display_mode = unsafe { core::mem::zeroed() };
__t11.clock = 1188000i32;
__t11.hdisplay = 7680u16;
__t11.hsync_start = 8232u16;
__t11.hsync_end = 8408u16;
__t11.htotal = 9000u16;
__t11.hskew = 0u16;
__t11.vdisplay = 4320u16;
__t11.vsync_start = 4336u16;
__t11.vsync_end = 4356u16;
__t11.vtotal = 4400u16;
__t11.vscan = 0u16;
__t11.flags = 5u32;
__t11.r#type = 64u8;
__t11.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11.status = 0i32;
__t11.picture_aspect_ratio = 3u32;
__t11
}, {
let mut __t12: drm_display_mode = unsafe { core::mem::zeroed() };
__t12.clock = 2376000i32;
__t12.hdisplay = 7680u16;
__t12.hsync_start = 10232u16;
__t12.hsync_end = 10408u16;
__t12.htotal = 11000u16;
__t12.hskew = 0u16;
__t12.vdisplay = 4320u16;
__t12.vsync_start = 4336u16;
__t12.vsync_end = 4356u16;
__t12.vtotal = 4500u16;
__t12.vscan = 0u16;
__t12.flags = 5u32;
__t12.r#type = 64u8;
__t12.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12.status = 0i32;
__t12.picture_aspect_ratio = 3u32;
__t12
}, {
let mut __t13: drm_display_mode = unsafe { core::mem::zeroed() };
__t13.clock = 2376000i32;
__t13.hdisplay = 7680u16;
__t13.hsync_start = 10032u16;
__t13.hsync_end = 10208u16;
__t13.htotal = 10800u16;
__t13.hskew = 0u16;
__t13.vdisplay = 4320u16;
__t13.vsync_start = 4336u16;
__t13.vsync_end = 4356u16;
__t13.vtotal = 4400u16;
__t13.vscan = 0u16;
__t13.flags = 5u32;
__t13.r#type = 64u8;
__t13.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13.status = 0i32;
__t13.picture_aspect_ratio = 3u32;
__t13
}, {
let mut __t14: drm_display_mode = unsafe { core::mem::zeroed() };
__t14.clock = 2376000i32;
__t14.hdisplay = 7680u16;
__t14.hsync_start = 8232u16;
__t14.hsync_end = 8408u16;
__t14.htotal = 9000u16;
__t14.hskew = 0u16;
__t14.vdisplay = 4320u16;
__t14.vsync_start = 4336u16;
__t14.vsync_end = 4356u16;
__t14.vtotal = 4400u16;
__t14.vscan = 0u16;
__t14.flags = 5u32;
__t14.r#type = 64u8;
__t14.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14.status = 0i32;
__t14.picture_aspect_ratio = 3u32;
__t14
}, {
let mut __t15: drm_display_mode = unsafe { core::mem::zeroed() };
__t15.clock = 4752000i32;
__t15.hdisplay = 7680u16;
__t15.hsync_start = 9792u16;
__t15.hsync_end = 9968u16;
__t15.htotal = 10560u16;
__t15.hskew = 0u16;
__t15.vdisplay = 4320u16;
__t15.vsync_start = 4336u16;
__t15.vsync_end = 4356u16;
__t15.vtotal = 4500u16;
__t15.vscan = 0u16;
__t15.flags = 5u32;
__t15.r#type = 64u8;
__t15.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15.status = 0i32;
__t15.picture_aspect_ratio = 3u32;
__t15
}, {
let mut __t16: drm_display_mode = unsafe { core::mem::zeroed() };
__t16.clock = 4752000i32;
__t16.hdisplay = 7680u16;
__t16.hsync_start = 8032u16;
__t16.hsync_end = 8208u16;
__t16.htotal = 8800u16;
__t16.hskew = 0u16;
__t16.vdisplay = 4320u16;
__t16.vsync_start = 4336u16;
__t16.vsync_end = 4356u16;
__t16.vtotal = 4500u16;
__t16.vscan = 0u16;
__t16.flags = 5u32;
__t16.r#type = 64u8;
__t16.name = [55u8, 54u8, 56u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t16.status = 0i32;
__t16.picture_aspect_ratio = 3u32;
__t16
}, {
let mut __t17: drm_display_mode = unsafe { core::mem::zeroed() };
__t17.clock = 1485000i32;
__t17.hdisplay = 10240u16;
__t17.hsync_start = 11732u16;
__t17.hsync_end = 11908u16;
__t17.htotal = 12500u16;
__t17.hskew = 0u16;
__t17.vdisplay = 4320u16;
__t17.vsync_start = 4336u16;
__t17.vsync_end = 4356u16;
__t17.vtotal = 4950u16;
__t17.vscan = 0u16;
__t17.flags = 5u32;
__t17.r#type = 64u8;
__t17.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t17.status = 0i32;
__t17.picture_aspect_ratio = 3u32;
__t17
}, {
let mut __t18: drm_display_mode = unsafe { core::mem::zeroed() };
__t18.clock = 1485000i32;
__t18.hdisplay = 10240u16;
__t18.hsync_start = 12732u16;
__t18.hsync_end = 12908u16;
__t18.htotal = 13500u16;
__t18.hskew = 0u16;
__t18.vdisplay = 4320u16;
__t18.vsync_start = 4336u16;
__t18.vsync_end = 4356u16;
__t18.vtotal = 4400u16;
__t18.vscan = 0u16;
__t18.flags = 5u32;
__t18.r#type = 64u8;
__t18.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t18.status = 0i32;
__t18.picture_aspect_ratio = 3u32;
__t18
}, {
let mut __t19: drm_display_mode = unsafe { core::mem::zeroed() };
__t19.clock = 1485000i32;
__t19.hdisplay = 10240u16;
__t19.hsync_start = 10528u16;
__t19.hsync_end = 10704u16;
__t19.htotal = 11000u16;
__t19.hskew = 0u16;
__t19.vdisplay = 4320u16;
__t19.vsync_start = 4336u16;
__t19.vsync_end = 4356u16;
__t19.vtotal = 4500u16;
__t19.vscan = 0u16;
__t19.flags = 5u32;
__t19.r#type = 64u8;
__t19.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t19.status = 0i32;
__t19.picture_aspect_ratio = 3u32;
__t19
}, {
let mut __t20: drm_display_mode = unsafe { core::mem::zeroed() };
__t20.clock = 2970000i32;
__t20.hdisplay = 10240u16;
__t20.hsync_start = 11732u16;
__t20.hsync_end = 11908u16;
__t20.htotal = 12500u16;
__t20.hskew = 0u16;
__t20.vdisplay = 4320u16;
__t20.vsync_start = 4336u16;
__t20.vsync_end = 4356u16;
__t20.vtotal = 4950u16;
__t20.vscan = 0u16;
__t20.flags = 5u32;
__t20.r#type = 64u8;
__t20.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t20.status = 0i32;
__t20.picture_aspect_ratio = 3u32;
__t20
}, {
let mut __t21: drm_display_mode = unsafe { core::mem::zeroed() };
__t21.clock = 2970000i32;
__t21.hdisplay = 10240u16;
__t21.hsync_start = 12732u16;
__t21.hsync_end = 12908u16;
__t21.htotal = 13500u16;
__t21.hskew = 0u16;
__t21.vdisplay = 4320u16;
__t21.vsync_start = 4336u16;
__t21.vsync_end = 4356u16;
__t21.vtotal = 4400u16;
__t21.vscan = 0u16;
__t21.flags = 5u32;
__t21.r#type = 64u8;
__t21.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t21.status = 0i32;
__t21.picture_aspect_ratio = 3u32;
__t21
}, {
let mut __t22: drm_display_mode = unsafe { core::mem::zeroed() };
__t22.clock = 2970000i32;
__t22.hdisplay = 10240u16;
__t22.hsync_start = 10528u16;
__t22.hsync_end = 10704u16;
__t22.htotal = 11000u16;
__t22.hskew = 0u16;
__t22.vdisplay = 4320u16;
__t22.vsync_start = 4336u16;
__t22.vsync_end = 4356u16;
__t22.vtotal = 4500u16;
__t22.vscan = 0u16;
__t22.flags = 5u32;
__t22.r#type = 64u8;
__t22.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t22.status = 0i32;
__t22.picture_aspect_ratio = 3u32;
__t22
}, {
let mut __t23: drm_display_mode = unsafe { core::mem::zeroed() };
__t23.clock = 5940000i32;
__t23.hdisplay = 10240u16;
__t23.hsync_start = 12432u16;
__t23.hsync_end = 12608u16;
__t23.htotal = 13200u16;
__t23.hskew = 0u16;
__t23.vdisplay = 4320u16;
__t23.vsync_start = 4336u16;
__t23.vsync_end = 4356u16;
__t23.vtotal = 4500u16;
__t23.vscan = 0u16;
__t23.flags = 5u32;
__t23.r#type = 64u8;
__t23.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t23.status = 0i32;
__t23.picture_aspect_ratio = 3u32;
__t23
}, {
let mut __t24: drm_display_mode = unsafe { core::mem::zeroed() };
__t24.clock = 5940000i32;
__t24.hdisplay = 10240u16;
__t24.hsync_start = 10528u16;
__t24.hsync_end = 10704u16;
__t24.htotal = 11000u16;
__t24.hskew = 0u16;
__t24.vdisplay = 4320u16;
__t24.vsync_start = 4336u16;
__t24.vsync_end = 4356u16;
__t24.vtotal = 4500u16;
__t24.vscan = 0u16;
__t24.flags = 5u32;
__t24.r#type = 64u8;
__t24.name = [49u8, 48u8, 50u8, 52u8, 48u8, 120u8, 52u8, 51u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t24.status = 0i32;
__t24.picture_aspect_ratio = 3u32;
__t24
}, {
let mut __t25: drm_display_mode = unsafe { core::mem::zeroed() };
__t25.clock = 1188000i32;
__t25.hdisplay = 4096u16;
__t25.hsync_start = 4896u16;
__t25.hsync_end = 4984u16;
__t25.htotal = 5280u16;
__t25.hskew = 0u16;
__t25.vdisplay = 2160u16;
__t25.vsync_start = 2168u16;
__t25.vsync_end = 2178u16;
__t25.vtotal = 2250u16;
__t25.vscan = 0u16;
__t25.flags = 5u32;
__t25.r#type = 64u8;
__t25.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t25.status = 0i32;
__t25.picture_aspect_ratio = 4u32;
__t25
}, {
let mut __t26: drm_display_mode = unsafe { core::mem::zeroed() };
__t26.clock = 1188000i32;
__t26.hdisplay = 4096u16;
__t26.hsync_start = 4184u16;
__t26.hsync_end = 4272u16;
__t26.htotal = 4400u16;
__t26.hskew = 0u16;
__t26.vdisplay = 2160u16;
__t26.vsync_start = 2168u16;
__t26.vsync_end = 2178u16;
__t26.vtotal = 2250u16;
__t26.vscan = 0u16;
__t26.flags = 5u32;
__t26.r#type = 64u8;
__t26.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t26.status = 0i32;
__t26.picture_aspect_ratio = 4u32;
__t26
}]);

/// HDMI 1.4 4k modes. Index using the VIC.
/// 0 - dummy, VICs start at 1
/// 1 - 3840x2160@30Hz
/// 2 - 3840x2160@25Hz
/// 3 - 3840x2160@24Hz
/// 4 - 4096x2160@24Hz (SMPTE)
static edid_4k_modes: __SlateAlign16<[drm_display_mode; 5]> = __SlateAlign16([{
let mut __t0: drm_display_mode = unsafe { core::mem::zeroed() };
__t0
}, {
let mut __t1: drm_display_mode = unsafe { core::mem::zeroed() };
__t1.clock = 297000i32;
__t1.hdisplay = 3840u16;
__t1.hsync_start = 4016u16;
__t1.hsync_end = 4104u16;
__t1.htotal = 4400u16;
__t1.hskew = 0u16;
__t1.vdisplay = 2160u16;
__t1.vsync_start = 2168u16;
__t1.vsync_end = 2178u16;
__t1.vtotal = 2250u16;
__t1.vscan = 0u16;
__t1.flags = 5u32;
__t1.r#type = 64u8;
__t1.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.status = 0i32;
__t1.picture_aspect_ratio = 2u32;
__t1
}, {
let mut __t2: drm_display_mode = unsafe { core::mem::zeroed() };
__t2.clock = 297000i32;
__t2.hdisplay = 3840u16;
__t2.hsync_start = 4896u16;
__t2.hsync_end = 4984u16;
__t2.htotal = 5280u16;
__t2.hskew = 0u16;
__t2.vdisplay = 2160u16;
__t2.vsync_start = 2168u16;
__t2.vsync_end = 2178u16;
__t2.vtotal = 2250u16;
__t2.vscan = 0u16;
__t2.flags = 5u32;
__t2.r#type = 64u8;
__t2.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.status = 0i32;
__t2.picture_aspect_ratio = 2u32;
__t2
}, {
let mut __t3: drm_display_mode = unsafe { core::mem::zeroed() };
__t3.clock = 297000i32;
__t3.hdisplay = 3840u16;
__t3.hsync_start = 5116u16;
__t3.hsync_end = 5204u16;
__t3.htotal = 5500u16;
__t3.hskew = 0u16;
__t3.vdisplay = 2160u16;
__t3.vsync_start = 2168u16;
__t3.vsync_end = 2178u16;
__t3.vtotal = 2250u16;
__t3.vscan = 0u16;
__t3.flags = 5u32;
__t3.r#type = 64u8;
__t3.name = [51u8, 56u8, 52u8, 48u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.status = 0i32;
__t3.picture_aspect_ratio = 2u32;
__t3
}, {
let mut __t4: drm_display_mode = unsafe { core::mem::zeroed() };
__t4.clock = 297000i32;
__t4.hdisplay = 4096u16;
__t4.hsync_start = 5116u16;
__t4.hsync_end = 5204u16;
__t4.htotal = 5500u16;
__t4.hskew = 0u16;
__t4.vdisplay = 2160u16;
__t4.vsync_start = 2168u16;
__t4.vsync_end = 2178u16;
__t4.vtotal = 2250u16;
__t4.vscan = 0u16;
__t4.flags = 5u32;
__t4.r#type = 64u8;
__t4.name = [52u8, 48u8, 57u8, 54u8, 120u8, 50u8, 49u8, 54u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.status = 0i32;
__t4.picture_aspect_ratio = 4u32;
__t4
}]);

// DDC fetch and block validation
/// The opaque EDID type, internal to drm_edid.c.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
    /// Size allocated for edid
    size: u64,
    edid: *const edid,
}


fn edid_hfeeodb_block_count(edid: *const edid) -> i32 {
let eeodb: i32 = edid_hfeeodb_extension_block_count(edid);
if eeodb != 0 { eeodb.wrapping_add(1) } else { 0 }
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_header_is_valid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_is_valid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_valid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_override_connector_update: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_raw: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_alloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_dup: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_free: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_probe_ddc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_get_edid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_read_custom: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_read_ddc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_read: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_get_product_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_print_product_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_get_panel_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_read_base_block: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_get_edid_switcheroo: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_read_switcheroo: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_duplicate: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_has_quirk: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_find_dmt: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_match_cea_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_display_mode_from_cea_vic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_match: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_get_monitor_name: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_to_sad: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_to_speaker_allocation: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_av_sync_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_detect_hdmi_monitor: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_detect_monitor_audio: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_default_rgb_quant_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_connector_update: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_connector_add_modes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_connector_update_edid_property: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_add_edid_modes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_add_modes_noedid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_hdmi_avi_infoframe_from_display_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_hdmi_avi_infoframe_quant_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_hdmi_vendor_infoframe_from_display_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edid_is_digital: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous", sym drm_edid_header_is_valid, sym drm_edid_is_valid, sym drm_edid_valid, sym drm_edid_override_connector_update, sym drm_edid_raw, sym drm_edid_alloc, sym drm_edid_dup, sym drm_edid_free, sym drm_probe_ddc, sym drm_get_edid, sym drm_edid_read_custom, sym drm_edid_read_ddc, sym drm_edid_read, sym drm_edid_get_product_id, sym drm_edid_print_product_id, sym drm_edid_get_panel_id, sym drm_edid_read_base_block, sym drm_get_edid_switcheroo, sym drm_edid_read_switcheroo, sym drm_edid_duplicate, sym drm_edid_has_quirk, sym drm_mode_find_dmt, sym drm_match_cea_mode, sym drm_display_mode_from_cea_vic, sym drm_edid_match, sym drm_edid_get_monitor_name, sym drm_edid_to_sad, sym drm_edid_to_speaker_allocation, sym drm_av_sync_delay, sym drm_detect_hdmi_monitor, sym drm_detect_monitor_audio, sym drm_default_rgb_quant_range, sym drm_edid_connector_update, sym drm_edid_connector_add_modes, sym drm_connector_update_edid_property, sym drm_add_edid_modes, sym drm_add_modes_noedid, sym drm_hdmi_avi_infoframe_from_display_mode, sym drm_hdmi_avi_infoframe_quant_range, sym drm_hdmi_vendor_infoframe_from_display_mode, sym drm_edid_is_digital, options(att_syntax));

fn edid_extension_block_count(edid: *const edid) -> i32 {
(unsafe { (*edid).extensions }) as i32
}

fn edid_block_count(edid: *const edid) -> i32 {
edid_extension_block_count(edid).wrapping_add(1)
}

fn edid_size_by_blocks(num_blocks: i32) -> i32 {
num_blocks.wrapping_mul(128)
}

fn edid_size(edid: *const edid) -> i32 {
edid_size_by_blocks(edid_block_count(edid))
}

fn edid_block_data(edid: *const edid, index: i32) -> *const () {
loop {
if ((const { core::mem::size_of::<edid>() }) as u64) != 128 {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
edid.wrapping_offset(index as isize) as *const ()
}

fn edid_extension_block_data(edid: *const edid, index: i32) -> *const () {
edid_block_data(edid, index.wrapping_add(1))
}

/// EDID block count indicated in EDID, may exceed allocated size
fn __drm_edid_block_count(drm_edid: *const drm_edid) -> i32 {
let mut num_blocks: i32 = 0;
// Starting point
num_blocks = edid_block_count(unsafe { (*drm_edid).edid });
// HF-EEODB override
if (unsafe { (*drm_edid).size }) >= (edid_size_by_blocks(2i32) as u64) {
let mut eeodb: i32 = 0;
// Note: HF-EEODB may specify a smaller extension count than the
// regular one. Unlike in buffer allocation, here we can use it.
eeodb = edid_hfeeodb_block_count(unsafe { (*drm_edid).edid });
if eeodb != 0 {
num_blocks = eeodb;
}
}
num_blocks
}

// EDID block count, limited by allocated size
fn drm_edid_block_count(drm_edid: *const drm_edid) -> i32 {
// Limit by allocated size
let mut __v0: i32;
let __UNIQUE_ID_x__678: i32 = __drm_edid_block_count(drm_edid);
let __UNIQUE_ID_y__679: i32 = ((unsafe { (*drm_edid).size }) as i32) / 128;
loop {
if __types_ok!(__UNIQUE_ID_x__678, __UNIQUE_ID_y__679) == 0 {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__678 < __UNIQUE_ID_y__679 { __UNIQUE_ID_x__678 } else { __UNIQUE_ID_y__679 };
__v0
}

/// EDID extension block count, limited by allocated size
fn drm_edid_extension_block_count(drm_edid: *const drm_edid) -> i32 {
drm_edid_block_count(drm_edid).wrapping_sub(1)
}

fn drm_edid_block_data(drm_edid: *const drm_edid, index: i32) -> *const () {
edid_block_data(unsafe { (*drm_edid).edid }, index)
}

fn drm_edid_extension_block_data(drm_edid: *const drm_edid, index: i32) -> *const () {
edid_extension_block_data(unsafe { (*drm_edid).edid }, index)
}

/// Initializer helper for legacy interfaces, where we have no choice but to
/// trust edid size. Not for general purpose use.
fn drm_edid_legacy_init(drm_edid: *mut drm_edid, edid: *const edid) -> *const drm_edid {
if edid.is_null() {
return core::ptr::null::<drm_edid>();
}
unsafe { memset(drm_edid as *mut (), 0i32, core::mem::size_of::<drm_edid>() as u64) };
unsafe {
(*drm_edid).edid = edid;
}
unsafe {
(*drm_edid).size = edid_size(edid) as u64;
}
drm_edid as *const drm_edid
}

/// EDID base and extension block iterator.
///
/// struct drm_edid_iter iter;
/// const u8 *block;
///
/// drm_edid_iter_begin(drm_edid, &iter);
/// drm_edid_iter_for_each(block, &iter) {
///         // do stuff with block
/// }
/// drm_edid_iter_end(&iter);
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid_iter {
    drm_edid: *const drm_edid,
    /// Current block index.
    index: i32,
}


fn drm_edid_iter_begin(drm_edid: *const drm_edid, iter: *mut drm_edid_iter) {
unsafe { memset(iter as *mut (), 0i32, core::mem::size_of::<drm_edid_iter>() as u64) };
unsafe {
(*iter).drm_edid = drm_edid;
}
}

fn __drm_edid_iter_next(iter: *mut drm_edid_iter) -> *const () {
let mut block: *const () = core::ptr::null::<()>();
if unsafe { (*iter).drm_edid }.is_null() {
return core::ptr::null::<()>();
}
if (unsafe { (*iter).index }) < drm_edid_block_count(unsafe { (*iter).drm_edid }) {
let __v1: i32 = unsafe { (*iter).index };
unsafe {
(*iter).index = __v1.wrapping_add(1);
}
block = drm_edid_block_data(unsafe { (*iter).drm_edid }, __v1);
}
block
}

fn drm_edid_iter_end(iter: *mut drm_edid_iter) {
unsafe { memset(iter as *mut (), 0i32, core::mem::size_of::<drm_edid_iter>() as u64) };
}

static edid_header: [u8; 8] = [0, 255, 255, 255, 255, 255, 255, 0];

fn edid_header_fix(edid: *mut ()) {
unsafe { memcpy(edid, (core::ptr::addr_of!(edid_header) as *const u8) as *const (), core::mem::size_of::<[u8; 8]>() as u64) };
}

/// drm_edid_header_is_valid - sanity check the header of the base EDID block
/// @_edid: pointer to raw base EDID block
///
/// Sanity check the header of the base EDID block.
///
/// Return: 8 if the header is perfect, down to 0 if it's totally wrong.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_header_is_valid")]
extern "C" fn drm_edid_header_is_valid(_edid: *const ()) -> i32 {
let mut i: i32 = 0;
let mut score: i32 = 0;
while (i as u64) < (core::mem::size_of::<[u8; 8]>() as u64) {
if ((unsafe { *((*(_edid as *const edid)).header.as_ptr() as *const u8).wrapping_offset(i as isize) }) as i32) == ((unsafe { edid_header[i as usize] }) as i32) {
score = score.wrapping_add(1);
}
i = i.wrapping_add(1);
}
score
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_header_is_valid_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const ()) -> i32>, *mut ()>(Some(drm_edid_header_is_valid)) };

#[unsafe(link_section = ".data..read_mostly")]
static mut edid_fixup: i32 = 6;

fn __check_edid_fixup() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(edid_fixup) }
}

static __param_str_edid_fixup: [u8; 15] = [100, 114, 109, 46, 101, 100, 105, 100, 95, 102, 105, 120, 117, 112, 0];

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_edid_fixup: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_edid_fixup) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_drm_edid_c_1823>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_edid_c_1823_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_edid_c_1823_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_edid_c_1823_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_edid_c_1823_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_edid_c_1823_6>() as u64) as i32).wrapping_add(256i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(edid_fixup) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_682: [u8; 28] = [100, 114, 109, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 101, 100, 105, 100, 95, 102, 105, 120, 117, 112, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_683: [u8; 79] = [100, 114, 109, 46, 112, 97, 114, 109, 61, 101, 100, 105, 100, 95, 102, 105, 120, 117, 112, 58, 77, 105, 110, 105, 109, 117, 109, 32, 110, 117, 109, 98, 101, 114, 32, 111, 102, 32, 118, 97, 108, 105, 100, 32, 69, 68, 73, 68, 32, 104, 101, 97, 100, 101, 114, 32, 98, 121, 116, 101, 115, 32, 40, 48, 45, 56, 44, 32, 100, 101, 102, 97, 117, 108, 116, 32, 54, 41, 0];

fn edid_block_compute_checksum(_block: *const ()) -> i32 {
let block: *const u8 = _block as *const u8;
let mut i: i32 = 0;
let mut csum: u8 = 0;
let mut crc: u8 = 0;
while i < 127i32 {
csum = (csum as i32).wrapping_add((unsafe { *block.wrapping_offset(i as isize) }) as i32) as u8;
i = i.wrapping_add(1);
}
crc = 256i32.wrapping_sub(csum as i32) as u8;
crc as i32
}

fn edid_block_get_checksum(_block: *const ()) -> i32 {
(unsafe { (*(_block as *const edid)).checksum }) as i32
}

fn edid_block_tag(_block: *const ()) -> i32 {
let block: *const u8 = _block as *const u8;
(unsafe { *block.wrapping_offset(0 as isize) }) as i32
}

fn edid_block_is_zero(edid: *const ()) -> bool {
mem_is_zero(edid, 128u64)
}

fn drm_edid_eq(drm_edid: *const drm_edid, raw_edid: *const (), raw_edid_size: u64) -> bool {
let edid1_present: bool = !drm_edid.is_null() && !unsafe { (*drm_edid).edid }.is_null() && (unsafe { (*drm_edid).size }) != 0;
let edid2_present: bool = !raw_edid.is_null() && raw_edid_size != 0;
if (edid1_present as i32) != (edid2_present as i32) {
return false;
}
if edid1_present {
if (unsafe { (*drm_edid).size }) != raw_edid_size {
return false;
}
if (unsafe { memcmp((*drm_edid).edid as *const (), raw_edid, (*drm_edid).size) }) != 0 {
return false;
}
}
true
}

fn edid_block_check(_block: *const (), is_base_block: bool) -> u32 {
let block: *const edid = _block as *const edid;
if block.is_null() {
return 2;
}
if is_base_block {
let score: i32 = drm_edid_header_is_valid(block as *const ());
let mut __v0: i32;
let __UNIQUE_ID_v__684: i32 = unsafe { edid_fixup };
let __UNIQUE_ID_l__685: i32 = 0;
let __UNIQUE_ID_h__686: i32 = 8;
loop {
if !(!statically_true!(__UNIQUE_ID_l__685 > __UNIQUE_ID_h__686)) {
unsafe { __compiletime_assert_687() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__684, __UNIQUE_ID_l__685, __UNIQUE_ID_h__686) == 0 {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__684 >= __UNIQUE_ID_h__686 { __UNIQUE_ID_h__686 } else { if __UNIQUE_ID_v__684 <= __UNIQUE_ID_l__685 { __UNIQUE_ID_l__685 } else { __UNIQUE_ID_v__684 } };
if score < __v0 {
if edid_block_is_zero(block as *const ()) {
return 3;
} else {
return 4;
}
}
if score < 8 {
return 5;
}
}
if edid_block_compute_checksum(block as *const ()) != edid_block_get_checksum(block as *const ()) {
if edid_block_is_zero(block as *const ()) {
return 3;
} else {
return 7;
}
}
if is_base_block {
if ((unsafe { (*block).version }) as i32) != 1 {
return 8;
}
}
0
}

fn edid_block_status_valid(status: u32, tag: i32) -> bool {
(status as u32) == 0 || (status as u32) == 6 || (status as u32) == 7 && tag == 2
}

fn edid_block_valid(block: *const (), base: bool) -> bool {
edid_block_status_valid(edid_block_check(block, base), edid_block_tag(block))
}

fn edid_block_status_print(status: u32, block: *const edid, block_num: i32) {
match status as u32 {
0 => {
}
1 => {
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017EDID block %d read failed\n".as_ptr() as *const u8, block_num) };
}
__v1 = 0;
}
2 => {
let mut __v2: i32;
if false {
unsafe { _printk(c"\x017EDID block %d pointer is NULL\n".as_ptr() as *const u8, block_num) };
}
__v2 = 0;
}
3 => {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x015EDID block %d is all zeroes\n".as_ptr() as *const u8, block_num) };
}
4 => {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x015EDID has corrupt header\n".as_ptr() as *const u8) };
}
5 => {
let mut __v7: i32;
if false {
unsafe { _printk(c"\x017EDID corrupt header needs repair\n".as_ptr() as *const u8) };
}
__v7 = 0;
}
6 => {
let mut __v8: i32;
if false {
unsafe { _printk(c"\x017EDID corrupt header fixed\n".as_ptr() as *const u8) };
}
__v8 = 0;
}
7 => {
if edid_block_status_valid(status, edid_block_tag(block as *const ())) {
let mut __v9: i32;
if false {
unsafe { _printk(c"\x017EDID block %d (tag 0x%02x) checksum is invalid, remainder is %d, ignoring\n".as_ptr() as *const u8, block_num, edid_block_tag(block as *const ()), edid_block_compute_checksum(block as *const ())) };
}
__v9 = 0;
} else {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x015EDID block %d (tag 0x%02x) checksum is invalid, remainder is %d\n".as_ptr() as *const u8, block_num, edid_block_tag(block as *const ()), edid_block_compute_checksum(block as *const ())) };
}
}
8 => {
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x015EDID has major version %d, instead of 1\n".as_ptr() as *const u8, (*block).version as i32) };
}
_ => {
let mut __v14: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"EDID block %d unknown edid block status code %d\n".as_ptr() as *const u8, block_num, status as u32) };
let mut __v15: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v15 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v16: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_edid_block_status_print_17, sym __slate_asm_string_edid_block_status_print_18, const 1984 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v16 = bug;
unsafe { __v15.unwrap()(__v16, block_num, status as u32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v14 = (__ret_warn_on != 0) as i64;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_689: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn edid_block_dump(level: *const u8, block: *const (), block_num: i32) {
let mut status: u32 = 0;
let mut prefix: __SlateAlign16<[u8; 20]> = __SlateAlign16([0u8; 20]);
status = edid_block_check(block, block_num == 0);
if (status as u32) == 3 {
unsafe { sprintf(prefix.0.as_mut_ptr() as *mut u8, c"\t[%02x] ZERO ".as_ptr() as *const u8, block_num) };
} else {
if !edid_block_status_valid(status, edid_block_tag(block)) {
unsafe { sprintf(prefix.0.as_mut_ptr() as *mut u8, c"\t[%02x] BAD  ".as_ptr() as *const u8, block_num) };
} else {
unsafe { sprintf(prefix.0.as_mut_ptr() as *mut u8, c"\t[%02x] GOOD ".as_ptr() as *const u8, block_num) };
}
}
unsafe { print_hex_dump(level, prefix.0.as_mut_ptr() as *const u8, 0i32, 16i32, 1i32, block, 128u64, false) };
}

/// Validate a base or extension EDID block and optionally dump bad blocks to
/// the console.
fn drm_edid_block_valid(_block: *mut (), block_num: i32, print_bad_edid: bool, edid_corrupt: *mut bool) -> bool {
let block: *mut edid = _block as *mut edid;
let mut status: u32 = 0;
let is_base_block: bool = block_num == 0;
let mut valid: bool = false;
let mut __v0: i64;
let __ret_warn_on: i32 = block.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_edid_block_valid_1, sym __slate_asm_string_drm_edid_block_valid_2, const 2018 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("691: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 691b - ., 4; .popsection\n# {0}", const 691 as i32, options(nostack, att_syntax));
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
status = edid_block_check(block as *const (), is_base_block);
if (status as u32) == 5 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Fixing EDID header, your hardware may be failing\n".as_ptr() as *const u8) };
edid_header_fix(block as *mut ());
// Retry with fixed header, update status if that worked.
status = edid_block_check(block as *const (), is_base_block);
if (status as u32) == 0 {
status = 6;
}
}
if !edid_corrupt.is_null() {
// Unknown major version isn't corrupt but we can't use it. Only
// the base block can reset edid_corrupt to false.
if is_base_block && ((status as u32) == 0 || (status as u32) == 8) {
unsafe {
*edid_corrupt = false;
}
} else {
if (status as u32) != 0 {
unsafe {
*edid_corrupt = true;
}
}
}
}
edid_block_status_print(status, block as *const edid, block_num);
// Determine whether we can use this block with this status.
valid = edid_block_status_valid(status, edid_block_tag(block as *const ()));
if !valid && print_bad_edid && (status as u32) != 3 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x015Raw EDID:\n".as_ptr() as *const u8) };
edid_block_dump(c"\x015".as_ptr() as *const u8, block as *const (), block_num);
}
valid
}

/// drm_edid_is_valid - sanity check EDID data
/// @edid: EDID data
///
/// Sanity-check an entire EDID record (including extensions)
///
/// Return: True if the EDID data is valid, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_is_valid")]
extern "C" fn drm_edid_is_valid(edid: *mut edid) -> bool {
let mut i: i32 = 0;
if edid.is_null() {
return false;
}
while i < edid_block_count(edid as *const edid) {
let block: *mut () = edid_block_data(edid as *const edid, i) as *mut ();
if !drm_edid_block_valid(block, i, true, core::ptr::null_mut::<bool>()) {
return false;
}
i = i.wrapping_add(1);
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_is_valid_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut edid) -> bool>, *mut ()>(Some(drm_edid_is_valid)) };

/// drm_edid_valid - sanity check EDID data
/// @drm_edid: EDID data
///
/// Sanity check an EDID. Cross check block count against allocated size and
/// checksum the blocks.
///
/// Return: True if the EDID data is valid, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_valid")]
extern "C" fn drm_edid_valid(drm_edid: *const drm_edid) -> bool {
let mut i: i32 = 0;
if drm_edid.is_null() {
return false;
}
if (edid_size_by_blocks(__drm_edid_block_count(drm_edid)) as u64) != unsafe { (*drm_edid).size } {
return false;
}
while i < drm_edid_block_count(drm_edid) {
let block: *const () = drm_edid_block_data(drm_edid, i);
if !edid_block_valid(block, i == 0) {
return false;
}
i = i.wrapping_add(1);
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_valid_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid) -> bool>, *mut ()>(Some(drm_edid_valid)) };

fn edid_filter_invalid_blocks(edid: *mut edid, alloc_size: *mut u64) -> *mut edid {
// Note: If the EDID uses HF-EEODB, but has invalid blocks, we'll revert
// back to regular extension count here. We don't want to start
// modifying the HF-EEODB extension too.
let mut i: i32 = 0;
let mut valid_blocks: i32 = 0;
while i < edid_block_count(edid as *const edid) {
let src_block: *const () = edid_block_data(edid as *const edid, i);
if edid_block_valid(src_block, i == 0) {
let dst_block: *mut () = edid_block_data(edid as *const edid, valid_blocks) as *mut ();
unsafe { memmove(dst_block, src_block, 128u64) };
valid_blocks = valid_blocks.wrapping_add(1);
}
i = i.wrapping_add(1);
}
// We already trusted the base block to be valid here...
let mut __v5: i64;
let __ret_warn_on: i32 = (valid_blocks == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("694: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 694b - ., 3; .popsection\n# {0}", const 694 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_edid_filter_invalid_blocks_6, sym __slate_asm_string_edid_filter_invalid_blocks_7, const 2136 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 4; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
unsafe { kfree(edid as *const ()) };
return core::ptr::null_mut::<edid>();
}
unsafe {
(*edid).extensions = valid_blocks.wrapping_sub(1) as u8;
}
unsafe {
(*edid).checksum = edid_block_compute_checksum(edid as *const ()) as u8;
}
unsafe {
*alloc_size = edid_size_by_blocks(valid_blocks) as u64;
}
let new: *mut edid = krealloc_node!(edid, unsafe { *alloc_size }, 3264u32, 1i32.wrapping_neg()) as *mut edid;
if new.is_null() {
unsafe { kfree(edid as *const ()) };
}
new
}

/// drm_do_probe_ddc_edid() - get EDID information via I2C
/// @data: I2C device adapter
/// @buf: EDID data buffer to be filled
/// @block: 128 byte EDID block to start fetching from
/// @len: EDID data buffer length to fetch
///
/// Try to fetch EDID information by calling I2C driver functions.
///
/// Return: 0 on success or -1 on failure.
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_do_probe_ddc_edid")]
extern "C" fn drm_do_probe_ddc_edid(data: *mut (), buf: *mut u8, block: u32, len: u64) -> i32 {
let adapter: *mut i2c_adapter = data as *mut i2c_adapter;
let mut start: u8 = block.wrapping_mul(128) as u8;
let mut segment: u8 = (block >> 1i32) as u8;
let xfers: u8 = (if segment != 0 { 3i32 } else { 2i32 }) as u8;
let mut ret: i32 = 0;
let mut retries: i32 = 5;
// The core I2C driver will automatically retry the transfer if the
// adapter reports EAGAIN. However, we find that bit-banging transfers
// are susceptible to errors under a heavily loaded machine and
// generate spurious NAKs and timeouts. Retrying the transfer
// of the individual block a few times seems to overcome this.
'__slate_break_0: loop {
let mut msgs: __SlateAlign16<[i2c_msg; 3]> = __SlateAlign16([{
let mut __t0: i2c_msg = unsafe { core::mem::zeroed() };
__t0.addr = 48u16;
__t0.flags = 0u16;
__t0.len = 1u16;
__t0.buf = core::ptr::addr_of_mut!(segment);
__t0
}, {
let mut __t1: i2c_msg = unsafe { core::mem::zeroed() };
__t1.addr = 80u16;
__t1.flags = 0u16;
__t1.len = 1u16;
__t1.buf = core::ptr::addr_of_mut!(start);
__t1
}, {
let mut __t2: i2c_msg = unsafe { core::mem::zeroed() };
__t2.addr = 80u16;
__t2.flags = 1u16;
__t2.len = len as u16;
__t2.buf = buf;
__t2
}]);
// Avoid sending the segment addr to not upset non-compliant
// DDC monitors.
ret = unsafe { i2c_transfer(adapter, (msgs.0.as_mut_ptr() as *mut i2c_msg).wrapping_offset(3i32.wrapping_sub(xfers as i32) as isize), xfers as i32) };
if ret == 6i32.wrapping_neg() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"drm: skipping non-existent adapter %s\n".as_ptr() as *const u8, (*adapter).name.as_mut_ptr() as *mut u8) };
break '__slate_break_0;
}
let __v1: bool = if ret != (xfers as i32) {
let __v3: i32 = retries.wrapping_sub(1);
retries = __v3;
__v3 != 0
} else {
false
};
if !__v1 {
break;
}
}
if ret == (xfers as i32) { 0 } else { 1i32.wrapping_neg() }
}

fn connector_bad_edid(connector: *mut drm_connector, edid: *const edid, num_blocks: i32) {
let mut last_block: u8 = 0;
// 0x7e in the EDID is the number of extension blocks. The EDID
// is 1 (base block) + num_ext_blocks big. That means we can think
// of 0x7e in the EDID of the _index_ of the last block in the
// combined chunk of memory.
last_block = unsafe { (*edid).extensions };
// Calculate real checksum for the last edid extension block data
if (last_block as i32) < num_blocks {
unsafe {
(*connector).real_edid_checksum = edid_block_compute_checksum(edid.wrapping_offset((last_block as i32) as isize) as *const ()) as u8;
}
}
let __v1: u32 = unsafe { (*connector).bad_edid_counter };
unsafe {
(*connector).bad_edid_counter = __v1.wrapping_add(1);
}
let __v3: bool = __v1 != 0 && !drm_debug_enabled!(2i32);
if __v3 {
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID is invalid:\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
for i in 0..num_blocks {
edid_block_dump(c"\x017".as_ptr() as *const u8, edid.wrapping_offset(i as isize) as *const (), i);
}
}

/// Get override or firmware EDID
fn drm_edid_override_get(connector: *mut drm_connector) -> *const drm_edid {
let mut r#override: *const drm_edid = core::ptr::null::<drm_edid>();
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
if !unsafe { (*connector).edid_override }.is_null() {
r#override = drm_edid_dup(unsafe { (*connector).edid_override });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
if r#override.is_null() {
r#override = drm_edid_load_firmware(connector);
}
if IS_ERR(r#override as *const ()) { core::ptr::null::<drm_edid>() } else { r#override }
}

/// For debugfs edid_override implementation
#[unsafe(no_mangle)]
extern "C" fn drm_edid_override_show(connector: *mut drm_connector, m: *mut seq_file) -> i32 {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
let drm_edid: *const drm_edid = unsafe { (*connector).edid_override };
if !drm_edid.is_null() {
unsafe { seq_write(m, (*drm_edid).edid as *const (), (*drm_edid).size) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
0
}

/// For debugfs edid_override implementation
#[unsafe(no_mangle)]
extern "C" fn drm_edid_override_set(connector: *mut drm_connector, edid: *const (), size: u64) -> i32 {
let drm_edid: *const drm_edid = drm_edid_alloc(edid, size);
if !drm_edid_valid(drm_edid) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID override invalid\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
drm_edid_free(drm_edid);
return 22i32.wrapping_neg();
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID override set\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
drm_edid_free(unsafe { (*connector).edid_override });
unsafe {
(*connector).edid_override = drm_edid;
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
0
}

/// For debugfs edid_override implementation
#[unsafe(no_mangle)]
extern "C" fn drm_edid_override_reset(connector: *mut drm_connector) -> i32 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID override reset\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
drm_edid_free(unsafe { (*connector).edid_override });
unsafe {
(*connector).edid_override = core::ptr::null::<drm_edid>();
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).edid_override_mutex)) };
0
}

/// drm_edid_override_connector_update - add modes from override/firmware EDID
/// @connector: connector we're probing
///
/// Add modes from the override/firmware EDID, if available. Only to be used from
/// drm_helper_probe_single_connector_modes() as a fallback for when DDC probe
/// failed during drm_get_edid() and caused the override/firmware EDID to be
/// skipped.
///
/// Return: The number of modes added or 0 if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_override_connector_update")]
extern "C" fn drm_edid_override_connector_update(connector: *mut drm_connector) -> i32 {
let mut num_modes: i32 = 0;
let r#override: *const drm_edid = drm_edid_override_get(connector);
if !r#override.is_null() {
if drm_edid_connector_update(connector, r#override) == 0 {
num_modes = drm_edid_connector_add_modes(connector);
}
drm_edid_free(r#override);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] adding %d modes via fallback override/firmware EDID\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, num_modes) };
}
num_modes
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_override_connector_update_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector) -> i32>, *mut ()>(Some(drm_edid_override_connector_update)) };

fn edid_block_read(block: *mut (), block_num: u32, read_block: Option<unsafe extern "C" fn(*mut (), *mut u8, u32, u64) -> i32>, context: *mut ()) -> u32 {
let mut status: u32 = 0;
let is_base_block: bool = block_num == 0;
for r#try in 0i32..4i32 {
if (unsafe { read_block.unwrap()(context, block as *mut u8, block_num, 128u64) }) != 0 {
return 1;
}
status = edid_block_check(block as *const (), is_base_block);
if (status as u32) == 5 {
edid_header_fix(block);
// Retry with fixed header, update status if that worked.
status = edid_block_check(block as *const (), is_base_block);
if (status as u32) == 0 {
status = 6;
}
}
if edid_block_status_valid(status, edid_block_tag(block as *const ())) {
break;
}
// Fail early for unrepairable base block all zeros.
if r#try == 0 && is_base_block && (status as u32) == 3 {
break;
}
}
status
}

fn _drm_do_get_edid(connector: *mut drm_connector, read_block: Option<unsafe extern "C" fn(*mut (), *mut u8, u32, u64) -> i32>, context: *mut (), size: *mut u64) -> *mut edid {
let mut __slate_compound_31: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_33: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
// If the first EDID extension is a CTA extension, and
// the first Data Block is HF-EEODB, override the
// extension block count.
//
// Note: HF-EEODB could specify a smaller extension
// count too, but we can't risk allocating a smaller
// amount.
let mut eeodb: i32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut alloc_size: u64 = unsafe { core::mem::zeroed() };
let mut new: core::mem::MaybeUninit<*mut edid> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut edid = core::ptr::addr_of_mut!(new) as *mut *mut edid;
let mut edid: core::mem::MaybeUninit<*mut edid> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut edid = core::ptr::addr_of_mut!(edid) as *mut *mut edid;
let mut r#override: core::mem::MaybeUninit<*const drm_edid> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *const drm_edid = core::ptr::addr_of_mut!(r#override) as *mut *const drm_edid;
let mut invalid_blocks: i32 = unsafe { core::mem::zeroed() };
let mut num_blocks: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(num_blocks) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
unsafe {
'__join_3: {
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
invalid_blocks = 0;
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
alloc_size = 128;
*__slate_slot_16 = drm_edid_override_get(connector);
if !(*__slate_slot_16).is_null() {
alloc_size = (*(*__slate_slot_16)).size;
*__slate_slot_15 = alloc_hooks!(kmemdup_noprof((*(*__slate_slot_16)).edid as *const (), alloc_size, 3264u32)) as *mut edid;
drm_edid_free(*__slate_slot_16);
if (*__slate_slot_15).is_null() {
return core::ptr::null_mut::<edid>();
}
} else {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_10 = _kmalloc_noprof(alloc_size, 3264u32, *{
__slate_compound_31 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_31)
});
alloc_tag_restore!();
} else {
*__slate_slot_10 = _kmalloc_noprof(alloc_size, 3264u32, *{
__slate_compound_33 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_33)
});
}
*__slate_slot_15 = *__slate_slot_10 as *mut edid;
if (*__slate_slot_15).is_null() {
return core::ptr::null_mut::<edid>();
} else {
'__join_0: {
*__slate_slot_20 = edid_block_read(*__slate_slot_15 as *mut (), 0u32, read_block, context);
edid_block_status_print(*__slate_slot_20, *__slate_slot_15 as *const edid, 0i32);
if (*__slate_slot_20 as u32) != 1 {
// FIXME: Clarify what a corrupt EDID actually means.
if (*__slate_slot_20 as u32) == 0 || (*__slate_slot_20 as u32) == 8 {
(*connector).edid_corrupt = false;
} else {
(*connector).edid_corrupt = true;
}
if !edid_block_status_valid(*__slate_slot_20, edid_block_tag(*__slate_slot_15 as *const ())) {
if (*__slate_slot_20 as u32) == 3 {
let __v7: i32 = (*connector).null_edid_counter;
(*connector).null_edid_counter = __v7.wrapping_add(1);
}
connector_bad_edid(connector, *__slate_slot_15 as *const edid, 1i32);
} else {
if edid_extension_block_count(*__slate_slot_15 as *const edid) == 0 {
break '__join_3;
} else {
alloc_size = edid_size(*__slate_slot_15 as *const edid) as u64;
*__slate_slot_14 = krealloc_node!(*__slate_slot_15, alloc_size, 3264u32, 1i32.wrapping_neg()) as *mut edid;
if !(*__slate_slot_14).is_null() {
*__slate_slot_15 = *__slate_slot_14;
*__slate_slot_18 = edid_block_count(*__slate_slot_15 as *const edid);
*__slate_slot_19 = 1;
while *__slate_slot_19 < *__slate_slot_18 {
let block: *mut () = edid_block_data(*__slate_slot_15 as *const edid, *__slate_slot_19) as *mut ();
*__slate_slot_20 = edid_block_read(block, *__slate_slot_19 as u32, read_block, context);
edid_block_status_print(*__slate_slot_20, block as *const edid, *__slate_slot_19);
if !edid_block_status_valid(*__slate_slot_20, edid_block_tag(block as *const ())) {
if (*__slate_slot_20 as u32) == 1 {
break '__join_0;
} else {
invalid_blocks = invalid_blocks.wrapping_add(1);
}
} else {
if *__slate_slot_19 == 1 {
eeodb = edid_hfeeodb_block_count(*__slate_slot_15 as *const edid);
if eeodb > *__slate_slot_18 {
*__slate_slot_18 = eeodb;
alloc_size = edid_size_by_blocks(*__slate_slot_18) as u64;
*__slate_slot_14 = krealloc_node!(*__slate_slot_15, alloc_size, 3264u32, 1i32.wrapping_neg()) as *mut edid;
if (*__slate_slot_14).is_null() {
break '__join_0;
} else {
*__slate_slot_15 = *__slate_slot_14;
}
}
}
}
let __v1: i32 = *__slate_slot_19;
*__slate_slot_19 = __v1.wrapping_add(1);
}
if invalid_blocks != 0 {
connector_bad_edid(connector, *__slate_slot_15 as *const edid, *__slate_slot_18);
*__slate_slot_15 = edid_filter_invalid_blocks(*__slate_slot_15, core::ptr::addr_of_mut!(alloc_size));
break '__join_3;
} else {
break '__join_3;
}
}
}
}
}
}
kfree(*__slate_slot_15 as *const ());
return core::ptr::null_mut::<edid>();
}
}
}
if !size.is_null() {
*size = alloc_size;
}
return *__slate_slot_15;
}
}

/// drm_edid_raw - Get a pointer to the raw EDID data.
/// @drm_edid: drm_edid container
///
/// Get a pointer to the raw EDID data.
///
/// This is for transition only. Avoid using this like the plague.
///
/// Return: Pointer to raw EDID data.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_raw")]
extern "C" fn drm_edid_raw(drm_edid: *const drm_edid) -> *const edid {
if drm_edid.is_null() || (unsafe { (*drm_edid).size }) == 0 {
return core::ptr::null::<edid>();
}
// Do not return pointers where relying on EDID extension count would
// lead to buffer overflow.
let mut __v0: i64;
let __ret_warn_on: i32 = ((edid_size(unsafe { (*drm_edid).edid }) as u64) > unsafe { (*drm_edid).size }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 3; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_edid_raw_1, sym __slate_asm_string_drm_edid_raw_2, const 2514 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("698: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 698b - ., 4; .popsection\n# {0}", const 698 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return core::ptr::null::<edid>();
}
unsafe { (*drm_edid).edid }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_raw_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid) -> *const edid>, *mut ()>(Some(drm_edid_raw)) };

/// Allocate struct drm_edid container *without* duplicating the edid data
fn _drm_edid_alloc(edid: *const (), size: u64) -> *const drm_edid {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
if edid.is_null() || size == 0 || size < 128 {
return core::ptr::null::<drm_edid>();
}
let mut __v0: *mut drm_edid;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_edid>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_edid;
let drm_edid: *mut drm_edid = __v0;
if !drm_edid.is_null() {
unsafe {
(*drm_edid).edid = edid as *const edid;
}
unsafe {
(*drm_edid).size = size;
}
}
drm_edid as *const drm_edid
}

/// drm_edid_alloc - Allocate a new drm_edid container
/// @edid: Pointer to raw EDID data
/// @size: Size of memory allocated for EDID
///
/// Allocate a new drm_edid container. Do not calculate edid size from edid, pass
/// the actual size that has been allocated for the data. There is no validation
/// of the raw EDID data against the size, but at least the EDID base block must
/// fit in the buffer.
///
/// The returned pointer must be freed using drm_edid_free().
///
/// Return: drm_edid container, or NULL on errors
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_alloc")]
extern "C" fn drm_edid_alloc(mut edid: *const (), size: u64) -> *const drm_edid {
if edid.is_null() || size == 0 || size < 128 {
return core::ptr::null::<drm_edid>();
}
edid = alloc_hooks!(unsafe { kmemdup_noprof(edid, size, 3264u32) }) as *const ();
if edid.is_null() {
return core::ptr::null::<drm_edid>();
}
let drm_edid: *const drm_edid = _drm_edid_alloc(edid, size);
if drm_edid.is_null() {
unsafe { kfree(edid) };
}
drm_edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_alloc_700: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), u64) -> *const drm_edid>, *mut ()>(Some(drm_edid_alloc)) };

/// drm_edid_dup - Duplicate a drm_edid container
/// @drm_edid: EDID to duplicate
///
/// The returned pointer must be freed using drm_edid_free().
///
/// Returns: drm_edid container copy, or NULL on errors
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_dup")]
extern "C" fn drm_edid_dup(drm_edid: *const drm_edid) -> *const drm_edid {
if drm_edid.is_null() {
return core::ptr::null::<drm_edid>();
}
drm_edid_alloc((unsafe { (*drm_edid).edid }) as *const (), unsafe { (*drm_edid).size })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_dup_701: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid) -> *const drm_edid>, *mut ()>(Some(drm_edid_dup)) };

/// drm_edid_free - Free the drm_edid container
/// @drm_edid: EDID to free
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_free")]
extern "C" fn drm_edid_free(drm_edid: *const drm_edid) {
if drm_edid.is_null() {
return;
}
unsafe { kfree((*drm_edid).edid as *const ()) };
unsafe { kfree(drm_edid as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_free_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid)>, *mut ()>(Some(drm_edid_free)) };

/// drm_probe_ddc() - probe DDC presence
/// @adapter: I2C adapter to probe
///
/// Return: True on success, false on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_probe_ddc")]
extern "C" fn drm_probe_ddc(adapter: *mut i2c_adapter) -> bool {
let mut out: u8 = 0;
drm_do_probe_ddc_edid(adapter as *mut (), core::ptr::addr_of_mut!(out), 0u32, 1u64) == 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_probe_ddc_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut i2c_adapter) -> bool>, *mut ()>(Some(drm_probe_ddc)) };

/// drm_get_edid - get EDID data, if available
/// @connector: connector we're probing
/// @adapter: I2C adapter to use for DDC
///
/// Poke the given I2C channel to grab EDID data if possible.  If found,
/// attach it to the connector.
///
/// Return: Pointer to valid EDID or NULL if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_get_edid")]
extern "C" fn drm_get_edid(connector: *mut drm_connector, adapter: *mut i2c_adapter) -> *mut edid {
if ((unsafe { (*connector).force }) as u32) == 1 {
return core::ptr::null_mut::<edid>();
}
let __v0: bool = ((unsafe { (*connector).force }) as u32) == 0 && !drm_probe_ddc(adapter);
if __v0 {
return core::ptr::null_mut::<edid>();
}
let edid: *mut edid = _drm_do_get_edid(connector, Some(drm_do_probe_ddc_edid), adapter as *mut (), core::ptr::null_mut::<u64>());
drm_connector_update_edid_property(connector, edid as *const edid);
edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_get_edid_704: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut i2c_adapter) -> *mut edid>, *mut ()>(Some(drm_get_edid)) };

/// drm_edid_read_custom - Read EDID data using given EDID block read function
/// @connector: Connector to use
/// @read_block: EDID block read function
/// @context: Private data passed to the block read function
///
/// When the I2C adapter connected to the DDC bus is hidden behind a device that
/// exposes a different interface to read EDID blocks this function can be used
/// to get EDID data using a custom block read function.
///
/// As in the general case the DDC bus is accessible by the kernel at the I2C
/// level, drivers must make all reasonable efforts to expose it as an I2C
/// adapter and use drm_edid_read() or drm_edid_read_ddc() instead of abusing
/// this function.
///
/// The EDID may be overridden using debugfs override_edid or firmware EDID
/// (drm_edid_load_firmware() and drm.edid_firmware parameter), in this priority
/// order. Having either of them bypasses actual EDID reads.
///
/// The returned pointer must be freed using drm_edid_free().
///
/// Return: Pointer to EDID, or NULL if probe/read failed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_read_custom")]
extern "C" fn drm_edid_read_custom(connector: *mut drm_connector, read_block: Option<unsafe extern "C" fn(*mut (), *mut u8, u32, u64) -> i32>, context: *mut ()) -> *const drm_edid {
let mut size: u64 = 0;
let edid: *mut edid = _drm_do_get_edid(connector, read_block, context, core::ptr::addr_of_mut!(size));
if edid.is_null() {
return core::ptr::null::<drm_edid>();
}
// Sanity check for now
let mut __v0: i64;
let __ret_warn_on: i32 = (size == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*connector).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*connector).dev as *const drm_device) as *const device), c"drm_WARN_ON(!size)".as_ptr() as *mut u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_edid_read_custom_3, sym __slate_asm_string_drm_edid_read_custom_4, const 2680 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(__drm_to_dev((*connector).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*connector).dev as *const drm_device) as *const device), c"drm_WARN_ON(!size)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let drm_edid: *const drm_edid = _drm_edid_alloc(edid as *const (), size);
if drm_edid.is_null() {
unsafe { kfree(edid as *const ()) };
}
drm_edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_705: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_read_custom_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, Option<unsafe extern "C" fn(*mut (), *mut u8, u32, u64) -> i32>, *mut ()) -> *const drm_edid>, *mut ()>(Some(drm_edid_read_custom)) };

/// drm_edid_read_ddc - Read EDID data using given I2C adapter
/// @connector: Connector to use
/// @adapter: I2C adapter to use for DDC
///
/// Read EDID using the given I2C adapter.
///
/// The EDID may be overridden using debugfs override_edid or firmware EDID
/// (drm_edid_load_firmware() and drm.edid_firmware parameter), in this priority
/// order. Having either of them bypasses actual EDID reads.
///
/// Prefer initializing connector->ddc with drm_connector_init_with_ddc() and
/// using drm_edid_read() instead of this function.
///
/// The returned pointer must be freed using drm_edid_free().
///
/// Return: Pointer to EDID, or NULL if probe/read failed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_read_ddc")]
extern "C" fn drm_edid_read_ddc(connector: *mut drm_connector, adapter: *mut i2c_adapter) -> *const drm_edid {
if ((unsafe { (*connector).force }) as u32) == 1 {
return core::ptr::null::<drm_edid>();
}
let __v0: bool = ((unsafe { (*connector).force }) as u32) == 0 && !drm_probe_ddc(adapter);
if __v0 {
return core::ptr::null::<drm_edid>();
}
let drm_edid: *const drm_edid = drm_edid_read_custom(connector, Some(drm_do_probe_ddc_edid), adapter as *mut ());
// Note: Do *not* call connector updates here.
drm_edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_read_ddc_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut i2c_adapter) -> *const drm_edid>, *mut ()>(Some(drm_edid_read_ddc)) };

/// drm_edid_read - Read EDID data using connector's I2C adapter
/// @connector: Connector to use
///
/// Read EDID using the connector's I2C adapter.
///
/// The EDID may be overridden using debugfs override_edid or firmware EDID
/// (drm_edid_load_firmware() and drm.edid_firmware parameter), in this priority
/// order. Having either of them bypasses actual EDID reads.
///
/// The returned pointer must be freed using drm_edid_free().
///
/// Return: Pointer to EDID, or NULL if probe/read failed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_read")]
extern "C" fn drm_edid_read(connector: *mut drm_connector) -> *const drm_edid {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*connector).ddc }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*connector).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*connector).dev as *const drm_device) as *const device), c"drm_WARN_ON(!connector->ddc)".as_ptr() as *mut u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_edid_read_3, sym __slate_asm_string_drm_edid_read_4, const 2743 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(__drm_to_dev((*connector).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*connector).dev as *const drm_device) as *const device), c"drm_WARN_ON(!connector->ddc)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return core::ptr::null::<drm_edid>();
}
drm_edid_read_ddc(connector, unsafe { (*connector).ddc })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_708: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_read_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector) -> *const drm_edid>, *mut ()>(Some(drm_edid_read)) };

/// drm_edid_get_product_id - Get the vendor and product identification
/// @drm_edid: EDID
/// @id: Where to place the product id
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_get_product_id")]
extern "C" fn drm_edid_get_product_id(drm_edid: *const drm_edid, id: *mut drm_edid_product_id) {
if !drm_edid.is_null() && !unsafe { (*drm_edid).edid }.is_null() && (unsafe { (*drm_edid).size }) >= 128 {
unsafe { memcpy(id as *mut (), core::ptr::addr_of!((*(*drm_edid).edid).__slate_anon_1.product_id) as *const (), core::mem::size_of::<drm_edid_product_id>() as u64) };
} else {
unsafe { memset(id as *mut (), 0i32, core::mem::size_of::<drm_edid_product_id>() as u64) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_get_product_id_710: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid, *mut drm_edid_product_id)>, *mut ()>(Some(drm_edid_get_product_id)) };

fn decode_date(s: *mut seq_buf, id: *const drm_edid_product_id) {
let week: i32 = (unsafe { (*id).week_of_manufacture }) as i32;
let year: i32 = ((unsafe { (*id).year_of_manufacture }) as i32).wrapping_add(1990);
if week == 255 {
unsafe { seq_buf_printf(s, c"model year: %d".as_ptr() as *const u8, year) };
} else {
if week == 0 {
unsafe { seq_buf_printf(s, c"year of manufacture: %d".as_ptr() as *const u8, year) };
} else {
unsafe { seq_buf_printf(s, c"week/year of manufacture: %d/%d".as_ptr() as *const u8, week, year) };
}
}
}

// drm_edid_print_product_id - Print decoded product id to printer
// @p: drm printer
// @id: EDID product id
// @raw: If true, also print the raw hex
//
// See VESA E-EDID 1.4 section 3.4.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_print_product_id_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_printer, *const drm_edid_product_id, bool)>, *mut ()>(Some(drm_edid_print_product_id)) };

/// drm_edid_get_panel_id - Get a panel's ID from EDID
/// @drm_edid: EDID that contains panel ID.
///
/// This function uses the first block of the EDID of a panel and (assuming
/// that the EDID is valid) extracts the ID out of it. The ID is a 32-bit value
/// (16 bits of manufacturer ID and 16 bits of per-manufacturer ID) that's
/// supposed to be different for each different modem of panel.
///
/// Return: A 32-bit ID that should be different for each make/model of panel.
///         See the functions drm_edid_encode_panel_id() and
///         drm_edid_decode_panel_id() for some details on the structure of this
///         ID. Return 0 if the EDID size is less than a base block.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_get_panel_id")]
extern "C" fn drm_edid_get_panel_id(drm_edid: *const drm_edid) -> u32 {
let edid: *const edid = unsafe { (*drm_edid).edid };
if (unsafe { (*drm_edid).size }) < 128 {
return 0;
}
// We represent the ID as a 32-bit number so it can easily be compared
// with "==".
//
// NOTE that we deal with endianness differently for the top half
// of this ID than for the bottom half. The bottom half (the product
// id) gets decoded as little endian by the EDID_PRODUCT_ID because
// that's how everyone seems to interpret it. The top half (the mfg_id)
// gets stored as big endian because that makes
// drm_edid_encode_panel_id() and drm_edid_decode_panel_id() easier
// to write (it's easier to extract the ASCII). It doesn't really
// matter, though, as long as the number here is unique.
((unsafe { (core::ptr::addr_of!((*edid).__slate_anon_1.__slate_anon_1.mfg_id) as *const u8).wrapping_offset(0 as isize).read_unaligned() }) as u32) << 24i32 | ((unsafe { (core::ptr::addr_of!((*edid).__slate_anon_1.__slate_anon_1.mfg_id) as *const u8).wrapping_offset(1 as isize).read_unaligned() }) as u32) << 16i32 | (EDID_PRODUCT_ID!(edid) as u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_get_panel_id_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid) -> u32>, *mut ()>(Some(drm_edid_get_panel_id)) };

/// drm_edid_read_base_block - Get a panel's EDID base block
/// @adapter: I2C adapter to use for DDC
///
/// This function returns the drm_edid containing the first block of the EDID of
/// a panel.
///
/// This function is intended to be used during early probing on devices where
/// more than one panel might be present. Because of its intended use it must
/// assume that the EDID of the panel is correct, at least as far as the base
/// block is concerned (in other words, we don't process any overrides here).
///
/// Caller should call drm_edid_free() after use.
///
/// NOTE: it's expected that this function and drm_do_get_edid() will both
/// be read the EDID, but there is no caching between them. Since we're only
/// reading the first block, hopefully this extra overhead won't be too big.
///
/// WARNING: Only use this function when the connector is unknown. For example,
/// during the early probe of panel. The EDID read from the function is temporary
/// and should be replaced by the full EDID returned from other drm_edid_read.
///
/// Return: Pointer to allocated EDID base block, or NULL on any failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_read_base_block")]
extern "C" fn drm_edid_read_base_block(adapter: *mut i2c_adapter) -> *const drm_edid {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut status: u32 = 0;
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(128u64, 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(128u64, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let base_block: *mut () = __v0;
if base_block.is_null() {
return core::ptr::null::<drm_edid>();
}
status = edid_block_read(base_block, 0u32, Some(drm_do_probe_ddc_edid), adapter as *mut ());
edid_block_status_print(status, base_block as *const edid, 0i32);
if !edid_block_status_valid(status, edid_block_tag(base_block as *const ())) {
edid_block_dump(c"\x015".as_ptr() as *const u8, base_block as *const (), 0i32);
unsafe { kfree(base_block as *const ()) };
return core::ptr::null::<drm_edid>();
}
_drm_edid_alloc(base_block as *const (), 128u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_read_base_block_715: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut i2c_adapter) -> *const drm_edid>, *mut ()>(Some(drm_edid_read_base_block)) };

/// drm_get_edid_switcheroo - get EDID data for a vga_switcheroo output
/// @connector: connector we're probing
/// @adapter: I2C adapter to use for DDC
///
/// Wrapper around drm_get_edid() for laptops with dual GPUs using one set of
/// outputs. The wrapper adds the requisite vga_switcheroo calls to temporarily
/// switch DDC to the GPU which is retrieving EDID.
///
/// Return: Pointer to valid EDID or %NULL if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_get_edid_switcheroo")]
extern "C" fn drm_get_edid_switcheroo(connector: *mut drm_connector, adapter: *mut i2c_adapter) -> *mut edid {
let dev: *mut drm_device = unsafe { (*connector).dev };
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*dev).dev });
let mut __v1: i32;
let __ret_warn_on: i32 = !(!(!dev_is_pci!(unsafe { (*dev).dev }))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!((dev->dev)->bus == &pci_bus_type))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_get_edid_switcheroo_4, sym __slate_asm_string_drm_get_edid_switcheroo_5, const 2912 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!((dev->dev)->bus == &pci_bus_type))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = __ret_warn_on;
if __v1 != 0 {
return core::ptr::null_mut::<edid>();
}
vga_switcheroo_lock_ddc(pdev);
let edid: *mut edid = drm_get_edid(connector, adapter);
vga_switcheroo_unlock_ddc(pdev);
edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_716: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_get_edid_switcheroo_717: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut i2c_adapter) -> *mut edid>, *mut ()>(Some(drm_get_edid_switcheroo)) };

/// drm_edid_read_switcheroo - get EDID data for a vga_switcheroo output
/// @connector: connector we're probing
/// @adapter: I2C adapter to use for DDC
///
/// Wrapper around drm_edid_read_ddc() for laptops with dual GPUs using one set
/// of outputs. The wrapper adds the requisite vga_switcheroo calls to
/// temporarily switch DDC to the GPU which is retrieving EDID.
///
/// Return: Pointer to valid EDID or %NULL if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_read_switcheroo")]
extern "C" fn drm_edid_read_switcheroo(connector: *mut drm_connector, adapter: *mut i2c_adapter) -> *const drm_edid {
let dev: *mut drm_device = unsafe { (*connector).dev };
let pdev: *mut pci_dev = to_pci_dev!(unsafe { (*dev).dev });
let mut __v1: i32;
let __ret_warn_on: i32 = !(!(!dev_is_pci!(unsafe { (*dev).dev }))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!((dev->dev)->bus == &pci_bus_type))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_edid_read_switcheroo_4, sym __slate_asm_string_drm_edid_read_switcheroo_5, const 2941 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!((dev->dev)->bus == &pci_bus_type))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = __ret_warn_on;
if __v1 != 0 {
return core::ptr::null::<drm_edid>();
}
vga_switcheroo_lock_ddc(pdev);
let drm_edid: *const drm_edid = drm_edid_read_ddc(connector, adapter);
vga_switcheroo_unlock_ddc(pdev);
drm_edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_718: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_read_switcheroo_719: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut i2c_adapter) -> *const drm_edid>, *mut ()>(Some(drm_edid_read_switcheroo)) };

/// drm_edid_duplicate - duplicate an EDID and the extensions
/// @edid: EDID to duplicate
///
/// Return: Pointer to duplicated EDID or NULL on allocation failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_duplicate")]
extern "C" fn drm_edid_duplicate(edid: *const edid) -> *mut edid {
if edid.is_null() {
return core::ptr::null_mut::<edid>();
}
alloc_hooks!(unsafe { kmemdup_noprof(edid as *const (), edid_size(edid) as u64, 3264u32) }) as *mut edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_duplicate_720: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid) -> *mut edid>, *mut ()>(Some(drm_edid_duplicate)) };

// EDID parsing
/// edid_get_quirks - return quirk flags for a given EDID
/// @drm_edid: EDID to process
///
/// This tells subsequent routines what fixes they need to apply.
///
/// Return: A u32 represents the quirks to apply.
fn edid_get_quirks(drm_edid: *const drm_edid) -> u32 {
let mut quirk: *const edid_quirk = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__4!() {
quirk = unsafe { core::ptr::addr_of!(edid_quirk_list.0) as *const edid_quirk }.wrapping_offset(i as isize);
if drm_edid_match(drm_edid, unsafe { core::ptr::addr_of!((*quirk).ident) }) {
return unsafe { (*quirk).quirks };
}
i = i.wrapping_add(1);
}
0
}

fn drm_edid_has_internal_quirk(connector: *mut drm_connector, quirk: u32) -> bool {
((unsafe { (*connector).display_info.quirks }) as u64) & BIT__2!(quirk) != 0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_has_quirk")]
extern "C" fn drm_edid_has_quirk(connector: *mut drm_connector, quirk: u32) -> bool {
((unsafe { (*connector).display_info.quirks }) as u64) & BIT__2!(quirk) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_has_quirk_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, u32) -> bool>, *mut ()>(Some(drm_edid_has_quirk)) };

/// Walk the mode list for connector, clearing the preferred status on existing
/// modes and setting it anew for the right mode ala quirks.
fn edid_fixup_preferred(connector: *mut drm_connector) {
let mut t: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut cur_mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut preferred_mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut target_refresh: i32 = 0;
let mut cur_vrefresh: i32 = 0;
let mut preferred_vrefresh: i32 = 0;
if list_empty((unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }) as *const list_head) != 0 {
return;
}
if drm_edid_has_internal_quirk(connector, 1u32) {
target_refresh = 60;
}
if drm_edid_has_internal_quirk(connector, 3u32) {
target_refresh = 75;
}
preferred_mode = list_first_entry__9!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, drm_display_mode);
cur_mode = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
t = list_next_entry!(cur_mode, head);
while list_entry_is_head!(cur_mode, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
let __v6: u8 = (((unsafe { (*cur_mode).r#type }) as i32) & -9i32) as u8;
unsafe {
(*cur_mode).r#type = __v6;
}
if cur_mode != preferred_mode {
// Largest mode is preferred
if MODE_SIZE!(cur_mode) > MODE_SIZE!(preferred_mode) {
preferred_mode = cur_mode;
}
cur_vrefresh = unsafe { drm_mode_vrefresh(cur_mode as *const drm_display_mode) };
preferred_vrefresh = unsafe { drm_mode_vrefresh(preferred_mode as *const drm_display_mode) };
// At a given size, try to get closest to target refresh
let __v8: bool = MODE_SIZE!(cur_mode) == MODE_SIZE!(preferred_mode) && MODE_REFRESH_DIFF!(cur_vrefresh, target_refresh) < MODE_REFRESH_DIFF!(preferred_vrefresh, target_refresh);
if __v8 {
preferred_mode = cur_mode;
}
}
cur_mode = t;
t = list_next_entry!(t, head);
}
unsafe {
(*preferred_mode).r#type = (((*preferred_mode).r#type as i32) | 8i32) as u8;
}
}

fn mode_is_rb(mode: *const drm_display_mode) -> bool {
((unsafe { (*mode).htotal }) as i32).wrapping_sub((unsafe { (*mode).hdisplay }) as i32) == 160 && ((unsafe { (*mode).hsync_end }) as i32).wrapping_sub((unsafe { (*mode).hdisplay }) as i32) == 80 && ((unsafe { (*mode).hsync_end }) as i32).wrapping_sub((unsafe { (*mode).hsync_start }) as i32) == 32 && ((unsafe { (*mode).vsync_start }) as i32).wrapping_sub((unsafe { (*mode).vdisplay }) as i32) == 3
}

/// drm_mode_find_dmt - Create a copy of a mode if present in DMT
/// @dev: Device to duplicate against
/// @hsize: Mode width
/// @vsize: Mode height
/// @fresh: Mode refresh rate
/// @rb: Mode reduced-blanking-ness
///
/// Walk the DMT mode list looking for a match for the given parameters.
///
/// Return: A newly allocated copy of the mode, or NULL if not found.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_mode_find_dmt")]
extern "C" fn drm_mode_find_dmt(dev: *mut drm_device, hsize: i32, vsize: i32, fresh: i32, rb: bool) -> *mut drm_display_mode {
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__14!() {
let ptr: *const drm_display_mode = unsafe { core::ptr::addr_of!(drm_dmt_modes.0) as *const drm_display_mode }.wrapping_offset(i as isize);
if hsize == ((unsafe { (*ptr).hdisplay }) as i32) {
if vsize == ((unsafe { (*ptr).vdisplay }) as i32) {
if fresh == unsafe { drm_mode_vrefresh(ptr) } {
if (rb as i32) == (mode_is_rb(ptr) as i32) {
return unsafe { drm_mode_duplicate(dev, ptr) };
}
}
}
}
i = i.wrapping_add(1);
}
core::ptr::null_mut::<drm_display_mode>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_find_dmt_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, i32, i32, i32, bool) -> *mut drm_display_mode>, *mut ()>(Some(drm_mode_find_dmt)) };

fn is_display_descriptor(descriptor: *const detailed_timing, r#type: u8) -> bool {
loop {
if offsetof__22!() != 0 {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if offsetof__63!() != 2 {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if offsetof__53!() != 3 {
unsafe { __compiletime_assert_725() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
((unsafe { (*descriptor).pixel_clock }) as i32) == 0 && ((unsafe { (*descriptor).data.other_data.pad1 }) as i32) == 0 && ((unsafe { (*descriptor).data.other_data.r#type }) as i32) == (r#type as i32)
}

fn is_detailed_timing_descriptor(descriptor: *const detailed_timing) -> bool {
loop {
if offsetof__22!() != 0 {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
((unsafe { (*descriptor).pixel_clock }) as i32) != 0
}

fn cea_for_each_detailed_block(ext: *const u8, cb: Option<unsafe extern "C" fn(*const detailed_timing, *mut ())>, closure: *mut ()) {
let mut n: i32 = 0;
let d: u8 = unsafe { *ext.wrapping_offset(2 as isize) };
let det_base: *const u8 = ext.wrapping_offset((d as i32) as isize);
if (d as i32) < 4 || (d as i32) > 127 {
return;
}
n = 127i32.wrapping_sub(d as i32) / 18;
for i in 0..n {
unsafe { cb.unwrap()(det_base.wrapping_offset(18i32.wrapping_mul(i) as isize) as *const detailed_timing, closure) };
}
}

fn vtb_for_each_detailed_block(ext: *const u8, cb: Option<unsafe extern "C" fn(*const detailed_timing, *mut ())>, closure: *mut ()) {
let mut n: u32 = 0;
let mut __v0: i32;
let __UNIQUE_ID_x__727: i32 = (unsafe { *ext.wrapping_offset(2 as isize) }) as i32;
let __UNIQUE_ID_y__728: i32 = 6;
loop {
if __types_ok!(__UNIQUE_ID_x__727, __UNIQUE_ID_y__728) == 0 {
unsafe { __compiletime_assert_729() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__727 < __UNIQUE_ID_y__728 { __UNIQUE_ID_x__727 } else { __UNIQUE_ID_y__728 };
n = __v0 as u32;
let det_base: *const u8 = ext.wrapping_offset(5 as isize);
if ((unsafe { *ext.wrapping_offset(1 as isize) }) as i32) != 1 {
return;
}
// unknown version
for i in 0..n {
unsafe { cb.unwrap()(det_base.wrapping_offset(18u32.wrapping_mul(i) as isize) as *const detailed_timing, closure) };
}
}

fn drm_for_each_detailed_block(drm_edid: *const drm_edid, cb: Option<unsafe extern "C" fn(*const detailed_timing, *mut ())>, closure: *mut ()) {
let mut edid_iter: drm_edid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(edid_iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if drm_edid.is_null() {
return;
}
for i in 0i32..4i32 {
unsafe { cb.unwrap()(((*(*drm_edid).edid).detailed_timings.as_ptr() as *const detailed_timing).wrapping_offset(i as isize), closure) };
}
drm_edid_iter_begin(drm_edid, core::ptr::addr_of_mut!(edid_iter));
loop {
let __v4: *const u8 = __drm_edid_iter_next(core::ptr::addr_of_mut!(edid_iter)) as *const u8;
let ext: *const u8 = __v4;
if __v4.is_null() {
break;
}
match (unsafe { *ext }) as i32 {
2 => {
cea_for_each_detailed_block(ext, cb, closure);
}
16 => {
vtb_for_each_detailed_block(ext, cb, closure);
}
_ => {
}
}
}
drm_edid_iter_end(core::ptr::addr_of_mut!(edid_iter));
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.is_rb")]
extern "C" fn is_rb(descriptor: *const detailed_timing, data: *mut ()) {
if !is_display_descriptor(descriptor, 253u8) {
return;
}
loop {
if offsetof__12!() != 10 {
unsafe { __compiletime_assert_730() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if offsetof__50!() != 15 {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if ((unsafe { (*descriptor).data.other_data.data.range.flags }) as i32) == 4 && ((unsafe { (*descriptor).data.other_data.data.range.formula.cvt.flags }) as i32) & 16i32 != 0 {
unsafe {
*(data as *mut bool) = true;
}
}
}

/// EDID 1.4 defines this explicitly.  For EDID 1.3, we guess, badly.
fn drm_monitor_supports_rb(drm_edid: *const drm_edid) -> bool {
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) >= 4 {
let mut ret: bool = false;
drm_for_each_detailed_block(drm_edid, Some(is_rb), core::ptr::addr_of_mut!(ret) as *mut ());
return ret;
}
drm_edid_is_digital(drm_edid)
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.find_gtf2")]
extern "C" fn find_gtf2(descriptor: *const detailed_timing, data: *mut ()) {
if !is_display_descriptor(descriptor, 253u8) {
return;
}
loop {
if offsetof__12!() != 10 {
unsafe { __compiletime_assert_732() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if ((unsafe { (*descriptor).data.other_data.data.range.flags }) as i32) == 2 {
unsafe {
*(data as *mut *const detailed_timing) = descriptor;
}
}
}

// Secondary GTF curve kicks in above some break frequency
fn drm_gtf2_hbreak(drm_edid: *const drm_edid) -> i32 {
let mut descriptor: *const detailed_timing = core::ptr::null::<detailed_timing>();
drm_for_each_detailed_block(drm_edid, Some(find_gtf2), core::ptr::addr_of_mut!(descriptor) as *mut ());
loop {
if offsetof__65!() != 12 {
unsafe { __compiletime_assert_733() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if !descriptor.is_null() { ((unsafe { (*descriptor).data.other_data.data.range.formula.gtf2.hfreq_start_khz }) as i32).wrapping_mul(2) } else { 0 }
}

fn drm_gtf2_2c(drm_edid: *const drm_edid) -> i32 {
let mut descriptor: *const detailed_timing = core::ptr::null::<detailed_timing>();
drm_for_each_detailed_block(drm_edid, Some(find_gtf2), core::ptr::addr_of_mut!(descriptor) as *mut ());
loop {
if offsetof__40!() != 13 {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if !descriptor.is_null() { (unsafe { (*descriptor).data.other_data.data.range.formula.gtf2.c }) as i32 } else { 0 }
}

fn drm_gtf2_m(drm_edid: *const drm_edid) -> i32 {
let mut descriptor: *const detailed_timing = core::ptr::null::<detailed_timing>();
drm_for_each_detailed_block(drm_edid, Some(find_gtf2), core::ptr::addr_of_mut!(descriptor) as *mut ());
loop {
if offsetof__83!() != 14 {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if !descriptor.is_null() { (unsafe { (*descriptor).data.other_data.data.range.formula.gtf2.m }) as i32 } else { 0 }
}

fn drm_gtf2_k(drm_edid: *const drm_edid) -> i32 {
let mut descriptor: *const detailed_timing = core::ptr::null::<detailed_timing>();
drm_for_each_detailed_block(drm_edid, Some(find_gtf2), core::ptr::addr_of_mut!(descriptor) as *mut ());
loop {
if offsetof__41!() != 16 {
unsafe { __compiletime_assert_736() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if !descriptor.is_null() { (unsafe { (*descriptor).data.other_data.data.range.formula.gtf2.k }) as i32 } else { 0 }
}

fn drm_gtf2_2j(drm_edid: *const drm_edid) -> i32 {
let mut descriptor: *const detailed_timing = core::ptr::null::<detailed_timing>();
drm_for_each_detailed_block(drm_edid, Some(find_gtf2), core::ptr::addr_of_mut!(descriptor) as *mut ());
loop {
if offsetof__44!() != 17 {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if !descriptor.is_null() { (unsafe { (*descriptor).data.other_data.data.range.formula.gtf2.j }) as i32 } else { 0 }
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.get_timing_level")]
extern "C" fn get_timing_level(descriptor: *const detailed_timing, data: *mut ()) {
if !is_display_descriptor(descriptor, 253u8) {
return;
}
loop {
if offsetof__12!() != 10 {
unsafe { __compiletime_assert_738() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
match (unsafe { (*descriptor).data.other_data.data.range.flags }) as i32 {
0 => {
unsafe {
*(data as *mut i32) = 1;
}
}
2 => {
unsafe {
*(data as *mut i32) = 2;
}
}
4 => {
unsafe {
*(data as *mut i32) = 3;
}
}
_ => {
}
}
}

/// Get standard timing level (CVT/GTF/DMT).
fn standard_timing_level(drm_edid: *const drm_edid) -> i32 {
let edid: *const edid = unsafe { (*drm_edid).edid };
if ((unsafe { (*edid).revision }) as i32) >= 4 {
// If the range descriptor doesn't
// indicate otherwise default to CVT
let mut ret: i32 = 3;
drm_for_each_detailed_block(drm_edid, Some(get_timing_level), core::ptr::addr_of_mut!(ret) as *mut ());
return ret;
} else {
let __v0: bool = ((unsafe { (*edid).revision }) as i32) >= 3 && drm_gtf2_hbreak(drm_edid) != 0;
if __v0 {
return 2;
} else {
if ((unsafe { (*edid).revision }) as i32) >= 2 {
return 1;
} else {
return 0;
}
}
}
}

/// 0 is reserved.  The spec says 0x01 fill for unused timings.  Some old
/// monitors fill with ascii space (0x20) instead.
fn bad_std_timing(a: u8, b: u8) -> i32 {
((a as i32) == 0 && (b as i32) == 0 || (a as i32) == 1 && (b as i32) == 1 || (a as i32) == 32 && (b as i32) == 32) as i32
}

fn drm_mode_hsync(mode: *const drm_display_mode) -> i32 {
if ((unsafe { (*mode).htotal }) as i32) <= 0 {
return 0;
}
let mut __v0: i32;
let __x: i32 = unsafe { (*mode).clock };
let __d: u16 = unsafe { (*mode).htotal };
__v0 = if 1i32.wrapping_neg() > 0 || ((1i32.wrapping_neg() as u16) as i32) > 0 || ((__x > 0) as i32) == (((__d as i32) > 0) as i32) { __x.wrapping_add((__d as i32) / 2) / (__d as i32) } else { __x.wrapping_sub((__d as i32) / 2) / (__d as i32) };
__v0
}

fn drm_gtf2_mode(dev: *mut drm_device, drm_edid: *const drm_edid, hsize: i32, vsize: i32, vrefresh_rate: i32) -> *mut drm_display_mode {
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
// This is potentially wrong if there's ever a monitor with
// more than one ranges section, each claiming a different
// secondary GTF curve.  Please don't do that.
mode = unsafe { drm_gtf_mode(dev, hsize, vsize, vrefresh_rate, false, 0i32) };
if mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
if drm_mode_hsync(mode as *const drm_display_mode) > drm_gtf2_hbreak(drm_edid) {
unsafe { drm_mode_destroy(dev, mode) };
mode = unsafe { drm_gtf_mode_complex(dev, hsize, vsize, vrefresh_rate, false, 0i32, drm_gtf2_m(drm_edid), drm_gtf2_2c(drm_edid), drm_gtf2_k(drm_edid), drm_gtf2_2j(drm_edid)) };
}
mode
}

/// Take the standard timing params (in this case width, aspect, and refresh)
/// and convert them into a real mode using CVT/GTF/DMT.
fn drm_mode_std(connector: *mut drm_connector, drm_edid: *const drm_edid, t: *const std_timing) -> *mut drm_display_mode {
let dev: *mut drm_device = unsafe { (*connector).dev };
let mut m: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut mode: *mut drm_display_mode = core::ptr::null_mut::<drm_display_mode>();
let mut hsize: i32 = 0;
let mut vsize: i32 = 0;
let mut vrefresh_rate: i32 = 0;
let aspect_ratio: u32 = ((((unsafe { (*t).vfreq_aspect }) as i32) & 192i32) >> 6i32) as u32;
let vfreq: u32 = ((((unsafe { (*t).vfreq_aspect }) as i32) & 63i32) >> 0i32) as u32;
let timing_level: i32 = standard_timing_level(drm_edid);
if bad_std_timing(unsafe { (*t).hsize }, unsafe { (*t).vfreq_aspect }) != 0 {
return core::ptr::null_mut::<drm_display_mode>();
}
// According to the EDID spec, the hdisplay = hsize * 8 + 248
hsize = ((unsafe { (*t).hsize }) as i32).wrapping_mul(8).wrapping_add(248);
// vrefresh_rate = vfreq + 60
vrefresh_rate = vfreq.wrapping_add(60) as i32;
// the vdisplay is calculated based on the aspect ratio
if aspect_ratio == 0 {
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) < 3 {
vsize = hsize;
} else {
vsize = hsize.wrapping_mul(10) / 16;
}
} else {
if aspect_ratio == 1 {
vsize = hsize.wrapping_mul(3) / 4;
} else {
if aspect_ratio == 2 {
vsize = hsize.wrapping_mul(4) / 5;
} else {
vsize = hsize.wrapping_mul(9) / 16;
}
}
}
// HDTV hack, part 1
if vrefresh_rate == 60 && (hsize == 1360 && vsize == 765 || hsize == 1368 && vsize == 769) {
hsize = 1366;
vsize = 768;
}
// If this connector already has a mode for this size and refresh
// rate (because it came from detailed or CVT info), use that
// instead.  This way we don't have to guess at interlace or
// reduced blanking.
m = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
while list_entry_is_head!(m, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
let __v1: bool = ((unsafe { (*m).hdisplay }) as i32) == hsize && ((unsafe { (*m).vdisplay }) as i32) == vsize && (unsafe { drm_mode_vrefresh(m as *const drm_display_mode) }) == vrefresh_rate;
if __v1 {
return core::ptr::null_mut::<drm_display_mode>();
}
m = list_next_entry!(m, head);
}
// HDTV hack, part 2
if hsize == 1366 && vsize == 768 && vrefresh_rate == 60 {
mode = unsafe { drm_cvt_mode(dev, 1366i32, 768i32, vrefresh_rate, false, false, false) };
if mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
unsafe {
(*mode).hdisplay = 1366;
}
unsafe {
(*mode).hsync_start = ((*mode).hsync_start as i32).wrapping_sub(1) as u16;
}
unsafe {
(*mode).hsync_end = ((*mode).hsync_end as i32).wrapping_sub(1) as u16;
}
return mode;
}
// check whether it can be found in default mode table
if drm_monitor_supports_rb(drm_edid) {
mode = drm_mode_find_dmt(dev, hsize, vsize, vrefresh_rate, true);
if !mode.is_null() {
return mode;
}
}
mode = drm_mode_find_dmt(dev, hsize, vsize, vrefresh_rate, false);
if !mode.is_null() {
return mode;
}
// okay, generate it
match timing_level {
0 => {
}
1 => {
mode = unsafe { drm_gtf_mode(dev, hsize, vsize, vrefresh_rate, false, 0i32) };
}
2 => {
mode = drm_gtf2_mode(dev, drm_edid, hsize, vsize, vrefresh_rate);
}
3 => {
mode = unsafe { drm_cvt_mode(dev, hsize, vsize, vrefresh_rate, false, false, false) };
}
_ => {
}
}
mode
}

/// EDID is delightfully ambiguous about how interlaced modes are to be
/// encoded.  Our internal representation is of frame height, but some
/// HDTV detailed timings are encoded as field height.
///
/// The format list here is from CEA, in frame size.  Technically we
/// should be checking refresh rate too.  Whatever.
fn drm_mode_do_interlace_quirk(mode: *mut drm_display_mode, pt: *const detailed_pixel_timing) {
let mut i: i32 = 0;
if ((unsafe { (*pt).misc }) as i32) & 128i32 == 0 {
return;
}
while (i as u64) < ARRAY_SIZE__88!() {
if ((unsafe { (*mode).hdisplay }) as i32) == unsafe { cea_interlaced.0[i as usize].w } && ((unsafe { (*mode).vdisplay }) as i32) == (unsafe { cea_interlaced.0[i as usize].h }) / 2 {
unsafe {
(*mode).vdisplay = ((*mode).vdisplay as i32).wrapping_mul(2) as u16;
}
unsafe {
(*mode).vsync_start = ((*mode).vsync_start as i32).wrapping_mul(2) as u16;
}
unsafe {
(*mode).vsync_end = ((*mode).vsync_end as i32).wrapping_mul(2) as u16;
}
unsafe {
(*mode).vtotal = ((*mode).vtotal as i32).wrapping_mul(2) as u16;
}
unsafe {
(*mode).vtotal = (((*mode).vtotal as i32) | 1) as u16;
}
}
i = i.wrapping_add(1);
}
unsafe {
(*mode).flags = (*mode).flags | 16u32;
}
}

static cea_interlaced: __SlateAlign16<[__SlateRecord_drm_edid_c_3489; 7]> = __SlateAlign16([__SlateRecord_drm_edid_c_3489 { w: 1920i32, h: 1080i32 }, __SlateRecord_drm_edid_c_3489 { w: 720i32, h: 480i32 }, __SlateRecord_drm_edid_c_3489 { w: 1440i32, h: 480i32 }, __SlateRecord_drm_edid_c_3489 { w: 2880i32, h: 480i32 }, __SlateRecord_drm_edid_c_3489 { w: 720i32, h: 576i32 }, __SlateRecord_drm_edid_c_3489 { w: 1440i32, h: 576i32 }, __SlateRecord_drm_edid_c_3489 { w: 2880i32, h: 576i32 }]);

/// Create a new mode from an EDID detailed timing section. An EDID detailed
/// timing block contains enough info for us to create and return a new struct
/// drm_display_mode.
fn drm_mode_detailed(connector: *mut drm_connector, drm_edid: *const drm_edid, timing: *const detailed_timing) -> *mut drm_display_mode {
let mut vsync_pulse_width: u32 = unsafe { core::mem::zeroed() };
let mut vsync_offset: u32 = unsafe { core::mem::zeroed() };
let mut hsync_pulse_width: u32 = unsafe { core::mem::zeroed() };
let mut hsync_offset: u32 = unsafe { core::mem::zeroed() };
let mut vblank: u32 = unsafe { core::mem::zeroed() };
let mut hblank: u32 = unsafe { core::mem::zeroed() };
let mut vactive: u32 = unsafe { core::mem::zeroed() };
let mut hactive: u32 = unsafe { core::mem::zeroed() };
let mut pt: *const detailed_pixel_timing = unsafe { core::mem::zeroed() };
let mut mode: core::mem::MaybeUninit<*mut drm_display_mode> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut drm_display_mode = core::ptr::addr_of_mut!(mode) as *mut *mut drm_display_mode;
let mut dev: *mut drm_device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*connector).dev;
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
pt = core::ptr::addr_of!((*timing).data.pixel_data);
hactive = ((((*pt).hactive_hblank_hi as i32) & 240) << 4i32 | ((*pt).hactive_lo as i32)) as u32;
vactive = ((((*pt).vactive_vblank_hi as i32) & 240) << 4i32 | ((*pt).vactive_lo as i32)) as u32;
hblank = ((((*pt).hactive_hblank_hi as i32) & 15) << 8i32 | ((*pt).hblank_lo as i32)) as u32;
vblank = ((((*pt).vactive_vblank_hi as i32) & 15) << 8i32 | ((*pt).vblank_lo as i32)) as u32;
hsync_offset = ((((*pt).hsync_vsync_offset_pulse_width_hi as i32) & 192) << 2i32 | ((*pt).hsync_offset_lo as i32)) as u32;
hsync_pulse_width = ((((*pt).hsync_vsync_offset_pulse_width_hi as i32) & 48) << 4i32 | ((*pt).hsync_pulse_width_lo as i32)) as u32;
vsync_offset = ((((*pt).hsync_vsync_offset_pulse_width_hi as i32) & 12) << 2i32 | ((*pt).vsync_offset_pulse_width_lo as i32) >> 4i32) as u32;
vsync_pulse_width = ((((*pt).hsync_vsync_offset_pulse_width_hi as i32) & 3) << 4i32 | ((*pt).vsync_offset_pulse_width_lo as i32) & 15) as u32;
// ignore tiny modes
if hactive < 64 || vactive < 64 {
return core::ptr::null_mut::<drm_display_mode>();
} else {
if ((*pt).misc as i32) & 32i32 != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Stereo mode not supported\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name);
return core::ptr::null_mut::<drm_display_mode>();
} else {
if ((*pt).misc as i32) & 24i32 == 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Composite sync not supported\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name);
}
// it is incorrect if hsync/vsync width is zero
if hsync_pulse_width == 0 || vsync_pulse_width == 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Incorrect Detailed timing. Wrong Hsync/Vsync pulse width\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name);
return core::ptr::null_mut::<drm_display_mode>();
} else {
if drm_edid_has_internal_quirk(connector, 7u32) {
*__slate_slot_24 = drm_cvt_mode(dev, hactive as i32, vactive as i32, 60i32, true, false, false);
if (*__slate_slot_24).is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
} else {
*__slate_slot_24 = drm_mode_create(dev);
if (*__slate_slot_24).is_null() {
return core::ptr::null_mut::<drm_display_mode>();
} else {
if drm_edid_has_internal_quirk(connector, 2u32) {
(*(*__slate_slot_24)).clock = 10880i32;
} else {
(*(*__slate_slot_24)).clock = ((*timing).pixel_clock as i32).wrapping_mul(10);
}
(*(*__slate_slot_24)).hdisplay = hactive as u16;
(*(*__slate_slot_24)).hsync_start = (((*(*__slate_slot_24)).hdisplay as i32) as u32).wrapping_add(hsync_offset) as u16;
(*(*__slate_slot_24)).hsync_end = (((*(*__slate_slot_24)).hsync_start as i32) as u32).wrapping_add(hsync_pulse_width) as u16;
(*(*__slate_slot_24)).htotal = (((*(*__slate_slot_24)).hdisplay as i32) as u32).wrapping_add(hblank) as u16;
(*(*__slate_slot_24)).vdisplay = vactive as u16;
(*(*__slate_slot_24)).vsync_start = (((*(*__slate_slot_24)).vdisplay as i32) as u32).wrapping_add(vsync_offset) as u16;
(*(*__slate_slot_24)).vsync_end = (((*(*__slate_slot_24)).vsync_start as i32) as u32).wrapping_add(vsync_pulse_width) as u16;
(*(*__slate_slot_24)).vtotal = (((*(*__slate_slot_24)).vdisplay as i32) as u32).wrapping_add(vblank) as u16;
// Some EDIDs have bogus h/vsync_end values
if ((*(*__slate_slot_24)).hsync_end as i32) > ((*(*__slate_slot_24)).htotal as i32) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] reducing hsync_end %d->%d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*(*__slate_slot_24)).hsync_end as i32, (*(*__slate_slot_24)).htotal as i32);
(*(*__slate_slot_24)).hsync_end = (*(*__slate_slot_24)).htotal;
}
if ((*(*__slate_slot_24)).vsync_end as i32) > ((*(*__slate_slot_24)).vtotal as i32) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] reducing vsync_end %d->%d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*(*__slate_slot_24)).vsync_end as i32, (*(*__slate_slot_24)).vtotal as i32);
(*(*__slate_slot_24)).vsync_end = (*(*__slate_slot_24)).vtotal;
}
drm_mode_do_interlace_quirk(*__slate_slot_24, pt);
if drm_edid_has_internal_quirk(connector, 6u32) {
let __v8: *mut drm_display_mode = *__slate_slot_24;
(*__v8).flags = (*__v8).flags | 5u32;
} else {
let __v14: *mut drm_display_mode = *__slate_slot_24;
(*__v14).flags = (*__v14).flags | ((if ((*pt).misc as i32) & 2i32 != 0 { 1i32 } else { 2i32 }) as u32);
let __v11: *mut drm_display_mode = *__slate_slot_24;
(*__v11).flags = (*__v11).flags | ((if ((*pt).misc as i32) & 4i32 != 0 { 4i32 } else { 8i32 }) as u32);
}
}
}
(*(*__slate_slot_24)).width_mm = (((*pt).width_mm_lo as i32) | (((*pt).width_height_mm_hi as i32) & 240) << 4i32) as u16;
(*(*__slate_slot_24)).height_mm = (((*pt).height_mm_lo as i32) | (((*pt).width_height_mm_hi as i32) & 15) << 8i32) as u16;
if drm_edid_has_internal_quirk(connector, 4u32) {
let __v5: *mut drm_display_mode = *__slate_slot_24;
(*__v5).width_mm = ((*__v5).width_mm as i32).wrapping_mul(10) as u16;
let __v2: *mut drm_display_mode = *__slate_slot_24;
(*__v2).height_mm = ((*__v2).height_mm as i32).wrapping_mul(10) as u16;
}
if drm_edid_has_internal_quirk(connector, 5u32) {
(*(*__slate_slot_24)).width_mm = ((*(*drm_edid).edid).width_cm as i32).wrapping_mul(10) as u16;
(*(*__slate_slot_24)).height_mm = ((*(*drm_edid).edid).height_cm as i32).wrapping_mul(10) as u16;
}
(*(*__slate_slot_24)).r#type = 64u8;
drm_mode_set_name(*__slate_slot_24);
return *__slate_slot_24;
}
}
}
}
}

fn mode_in_hsync_range(mode: *const drm_display_mode, edid: *const edid, t: *const u8) -> bool {
let mut hsync: i32 = 0;
let mut hmin: i32 = (unsafe { *t.wrapping_offset(7 as isize) }) as i32;
let mut hmax: i32 = 0;
if ((unsafe { (*edid).revision }) as i32) >= 4 {
hmin = hmin.wrapping_add(if ((unsafe { *t.wrapping_offset(4 as isize) }) as i32) & 4 != 0 { 255 } else { 0 });
}
hmax = (unsafe { *t.wrapping_offset(8 as isize) }) as i32;
if ((unsafe { (*edid).revision }) as i32) >= 4 {
hmax = hmax.wrapping_add(if ((unsafe { *t.wrapping_offset(4 as isize) }) as i32) & 8 != 0 { 255 } else { 0 });
}
hsync = drm_mode_hsync(mode);
hsync <= hmax && hsync >= hmin
}

fn mode_in_vsync_range(mode: *const drm_display_mode, edid: *const edid, t: *const u8) -> bool {
let mut vsync: i32 = 0;
let mut vmin: i32 = (unsafe { *t.wrapping_offset(5 as isize) }) as i32;
let mut vmax: i32 = 0;
if ((unsafe { (*edid).revision }) as i32) >= 4 {
vmin = vmin.wrapping_add(if ((unsafe { *t.wrapping_offset(4 as isize) }) as i32) & 1 != 0 { 255 } else { 0 });
}
vmax = (unsafe { *t.wrapping_offset(6 as isize) }) as i32;
if ((unsafe { (*edid).revision }) as i32) >= 4 {
vmax = vmax.wrapping_add(if ((unsafe { *t.wrapping_offset(4 as isize) }) as i32) & 2 != 0 { 255 } else { 0 });
}
vsync = unsafe { drm_mode_vrefresh(mode) };
vsync <= vmax && vsync >= vmin
}

fn range_pixel_clock(edid: *const edid, t: *const u8) -> u32 {
// unspecified
if ((unsafe { *t.wrapping_offset(9 as isize) }) as i32) == 0 || ((unsafe { *t.wrapping_offset(9 as isize) }) as i32) == 255 {
return 0;
}
// 1.4 with CVT support gives us real precision, yay
if ((unsafe { (*edid).revision }) as i32) >= 4 && ((unsafe { *t.wrapping_offset(10 as isize) }) as i32) == 4 {
return ((unsafe { *t.wrapping_offset(9 as isize) }) as i32).wrapping_mul(10000).wrapping_sub((((unsafe { *t.wrapping_offset(12 as isize) }) as i32) >> 2i32).wrapping_mul(250)) as u32;
}
// 1.3 is pathetic, so fuzz up a bit
((unsafe { *t.wrapping_offset(9 as isize) }) as i32).wrapping_mul(10000).wrapping_add(5001) as u32
}

fn mode_in_range(mode: *const drm_display_mode, drm_edid: *const drm_edid, timing: *const detailed_timing) -> bool {
let edid: *const edid = unsafe { (*drm_edid).edid };
let mut max_clock: u32 = 0;
let t: *const u8 = timing as *const u8;
if !mode_in_hsync_range(mode, edid, t) {
return false;
}
if !mode_in_vsync_range(mode, edid, t) {
return false;
}
max_clock = range_pixel_clock(edid, t);
if max_clock != 0 {
if ((unsafe { (*mode).clock }) as u32) > max_clock {
return false;
}
}
// 1.4 max horizontal check
if ((unsafe { (*edid).revision }) as i32) >= 4 && ((unsafe { *t.wrapping_offset(10 as isize) }) as i32) == 4 {
if (unsafe { *t.wrapping_offset(13 as isize) }) != 0 && ((unsafe { (*mode).hdisplay }) as i32) > 8i32.wrapping_mul(((unsafe { *t.wrapping_offset(13 as isize) }) as i32).wrapping_add(256i32.wrapping_mul(((unsafe { *t.wrapping_offset(12 as isize) }) as i32) & 3))) {
return false;
}
}
let __v0: bool = mode_is_rb(mode) && !drm_monitor_supports_rb(drm_edid);
if __v0 {
return false;
}
true
}

fn valid_inferred_mode(connector: *const drm_connector, mode: *const drm_display_mode) -> bool {
let mut m: *const drm_display_mode = unsafe { core::mem::zeroed() };
let mut ok: bool = false;
m = list_first_entry__12!(unsafe { core::ptr::addr_of!((*connector).probed_modes) });
while list_entry_is_head__2!(m, unsafe { core::ptr::addr_of!((*connector).probed_modes) }, head) == 0 {
let __v1: bool = ((unsafe { (*mode).hdisplay }) as i32) == ((unsafe { (*m).hdisplay }) as i32) && ((unsafe { (*mode).vdisplay }) as i32) == ((unsafe { (*m).vdisplay }) as i32) && (unsafe { drm_mode_vrefresh(mode) }) == unsafe { drm_mode_vrefresh(m) };
if __v1 {
return false;
}
// duplicated
if ((unsafe { (*mode).hdisplay }) as i32) <= ((unsafe { (*m).hdisplay }) as i32) && ((unsafe { (*mode).vdisplay }) as i32) <= ((unsafe { (*m).vdisplay }) as i32) {
ok = true;
}
m = list_next_entry__17!(m, head);
}
ok
}

fn drm_dmt_modes_for_range(connector: *mut drm_connector, drm_edid: *const drm_edid, timing: *const detailed_timing) -> i32 {
let mut i: i32 = 0;
let mut modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
while (i as u64) < ARRAY_SIZE__85!() {
let __v0: bool = mode_in_range(unsafe { core::ptr::addr_of!(drm_dmt_modes.0) as *const drm_display_mode }.wrapping_offset(i as isize), drm_edid, timing) && valid_inferred_mode(connector as *const drm_connector, unsafe { core::ptr::addr_of!(drm_dmt_modes.0) as *const drm_display_mode }.wrapping_offset(i as isize));
if __v0 {
let newmode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, (core::ptr::addr_of!(drm_dmt_modes.0) as *const drm_display_mode).wrapping_offset(i as isize)) };
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
modes
}

/// fix up 1366x768 mode from 1368x768;
/// GFT/CVT can't express 1366 width which isn't dividable by 8
#[unsafe(no_mangle)]
extern "C" fn drm_mode_fixup_1366x768(mode: *mut drm_display_mode) {
if ((unsafe { (*mode).hdisplay }) as i32) == 1368 && ((unsafe { (*mode).vdisplay }) as i32) == 768 {
unsafe {
(*mode).hdisplay = 1366;
}
unsafe {
(*mode).hsync_start = ((*mode).hsync_start as i32).wrapping_sub(1) as u16;
}
unsafe {
(*mode).hsync_end = ((*mode).hsync_end as i32).wrapping_sub(1) as u16;
}
unsafe { drm_mode_set_name(mode) };
}
}

fn drm_gtf_modes_for_range(connector: *mut drm_connector, drm_edid: *const drm_edid, timing: *const detailed_timing) -> i32 {
let mut i: i32 = 0;
let mut modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
while (i as u64) < ARRAY_SIZE__69!() {
let m: *const minimode = unsafe { core::ptr::addr_of!(extra_modes.0) as *const minimode }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = unsafe { drm_gtf_mode(dev, (*m).w as i32, (*m).h as i32, (*m).r as i32, false, 0i32) };
if newmode.is_null() {
return modes;
}
drm_mode_fixup_1366x768(newmode);
let __v0: bool = !mode_in_range(newmode as *const drm_display_mode, drm_edid, timing) || !valid_inferred_mode(connector as *const drm_connector, newmode as *const drm_display_mode);
if __v0 {
unsafe { drm_mode_destroy(dev, newmode) };
} else {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
i = i.wrapping_add(1);
}
modes
}

fn drm_gtf2_modes_for_range(connector: *mut drm_connector, drm_edid: *const drm_edid, timing: *const detailed_timing) -> i32 {
let mut i: i32 = 0;
let mut modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
while (i as u64) < ARRAY_SIZE__32!() {
let m: *const minimode = unsafe { core::ptr::addr_of!(extra_modes.0) as *const minimode }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = drm_gtf2_mode(dev, drm_edid, (unsafe { (*m).w }) as i32, (unsafe { (*m).h }) as i32, (unsafe { (*m).r }) as i32);
if newmode.is_null() {
return modes;
}
drm_mode_fixup_1366x768(newmode);
let __v0: bool = !mode_in_range(newmode as *const drm_display_mode, drm_edid, timing) || !valid_inferred_mode(connector as *const drm_connector, newmode as *const drm_display_mode);
if __v0 {
unsafe { drm_mode_destroy(dev, newmode) };
} else {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
i = i.wrapping_add(1);
}
modes
}

fn drm_cvt_modes_for_range(connector: *mut drm_connector, drm_edid: *const drm_edid, timing: *const detailed_timing) -> i32 {
let mut i: i32 = 0;
let mut modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
let rb: bool = drm_monitor_supports_rb(drm_edid);
while (i as u64) < ARRAY_SIZE__38!() {
let m: *const minimode = unsafe { core::ptr::addr_of!(extra_modes.0) as *const minimode }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = unsafe { drm_cvt_mode(dev, (*m).w as i32, (*m).h as i32, (*m).r as i32, rb, false, false) };
if newmode.is_null() {
return modes;
}
drm_mode_fixup_1366x768(newmode);
let __v0: bool = !mode_in_range(newmode as *const drm_display_mode, drm_edid, timing) || !valid_inferred_mode(connector as *const drm_connector, newmode as *const drm_display_mode);
if __v0 {
unsafe { drm_mode_destroy(dev, newmode) };
} else {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
i = i.wrapping_add(1);
}
modes
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.do_inferred_modes")]
extern "C" fn do_inferred_modes(timing: *const detailed_timing, c: *mut ()) {
let data: *const detailed_non_pixel = unsafe { core::ptr::addr_of!((*timing).data.other_data) };
let range: *const detailed_data_monitor_range = unsafe { core::ptr::addr_of!((*data).data.range) };
if !is_display_descriptor(timing, 253u8) {
return;
}
let __v0: i32 = drm_dmt_modes_for_range(unsafe { (*(c as *mut detailed_mode_closure)).connector }, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, timing);
let __v2: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v2.wrapping_add(__v0);
}
if ((unsafe { (*(*(*(c as *mut detailed_mode_closure)).drm_edid).edid).revision }) as i32) < 2 {
return;
}
// GTF not defined yet
match (unsafe { (*range).flags }) as i32 {
2 => {
let __v4: i32 = drm_gtf2_modes_for_range(unsafe { (*(c as *mut detailed_mode_closure)).connector }, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, timing);
let __v6: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v6.wrapping_add(__v4);
}
}
0 => {
let __v9: i32 = drm_gtf_modes_for_range(unsafe { (*(c as *mut detailed_mode_closure)).connector }, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, timing);
let __v11: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v11.wrapping_add(__v9);
}
}
4 => {
if ((unsafe { (*(*(*(c as *mut detailed_mode_closure)).drm_edid).edid).revision }) as i32) >= 4 {
let __v13: i32 = drm_cvt_modes_for_range(unsafe { (*(c as *mut detailed_mode_closure)).connector }, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, timing);
let __v15: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v15.wrapping_add(__v13);
}
}
}
_ => {
}
}
}

fn add_inferred_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) >= 1 {
drm_for_each_detailed_block(drm_edid, Some(do_inferred_modes), core::ptr::addr_of_mut!(closure) as *mut ());
}
closure.modes
}

fn drm_est3_modes(connector: *mut drm_connector, timing: *const detailed_timing) -> i32 {
let mut j: i32 = 0;
let mut m: i32 = 0;
let mut modes: i32 = 0;
let est: *const u8 = (timing as *const u8).wrapping_offset(6 as isize);
for i in 0i32..6i32 {
j = 7;
while j >= 0 {
m = i.wrapping_mul(8).wrapping_add(7i32.wrapping_sub(j));
if (m as u64) >= ARRAY_SIZE__73!() {
break;
}
if ((unsafe { *est.wrapping_offset(i as isize) }) as i32) & 1i32 << j != 0 {
let mode: *mut drm_display_mode = drm_mode_find_dmt(unsafe { (*connector).dev }, (unsafe { est3_modes.0[m as usize].w }) as i32, (unsafe { est3_modes.0[m as usize].h }) as i32, (unsafe { est3_modes.0[m as usize].r }) as i32, (unsafe { est3_modes.0[m as usize].rb }) != 0);
if !mode.is_null() {
unsafe { drm_mode_probed_add(connector, mode) };
modes = modes.wrapping_add(1);
}
}
j = j.wrapping_sub(1);
}
}
modes
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.do_established_modes")]
extern "C" fn do_established_modes(timing: *const detailed_timing, c: *mut ()) {
if !is_display_descriptor(timing, 247u8) {
return;
}
let __v0: i32 = drm_est3_modes(unsafe { (*(c as *mut detailed_mode_closure)).connector }, timing);
let __v2: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v2.wrapping_add(__v0);
}
}

/// Get established modes from EDID and add them. Each EDID block contains a
/// bitmap of the supported "established modes" list (defined above). Tease them
/// out and add them to the global modes list.
fn add_established_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let dev: *mut drm_device = unsafe { (*connector).dev };
let edid: *const edid = unsafe { (*drm_edid).edid };
let est_bits: u64 = (((unsafe { (*edid).established_timings.t1 }) as i32) | ((unsafe { (*edid).established_timings.t2 }) as i32) << 8i32 | (((unsafe { (*edid).established_timings.mfg_rsvd }) as i32) & 128) << 9i32) as u64;
let mut i: i32 = 0;
let mut modes: i32 = 0;
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
while i <= 16 {
if est_bits & ((1i32 << i) as u64) != 0 {
let newmode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, (core::ptr::addr_of!(edid_est_modes.0) as *const drm_display_mode).wrapping_offset(i as isize)) };
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
if ((unsafe { (*edid).revision }) as i32) >= 1 {
drm_for_each_detailed_block(drm_edid, Some(do_established_modes), core::ptr::addr_of_mut!(closure) as *mut ());
}
modes.wrapping_add(closure.modes)
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.do_standard_modes")]
extern "C" fn do_standard_modes(timing: *const detailed_timing, c: *mut ()) {
let data: *const detailed_non_pixel = unsafe { core::ptr::addr_of!((*timing).data.other_data) };
let connector: *mut drm_connector = unsafe { (*(c as *mut detailed_mode_closure)).connector };
if !is_display_descriptor(timing, 250u8) {
return;
}
for i in 0i32..6i32 {
let std: *const std_timing = unsafe { (*data).data.timings.as_ptr() as *const std_timing }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = drm_mode_std(connector, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, std);
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
let __v1: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v1.wrapping_add(1);
}
}
}
}

/// Get standard modes from EDID and add them. Standard modes can be calculated
/// using the appropriate standard (DMT, GTF, or CVT). Grab them from EDID and
/// add them to the list.
fn add_standard_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut modes: i32 = 0;
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
for i in 0i32..8i32 {
let newmode: *mut drm_display_mode = drm_mode_std(connector, drm_edid, unsafe { (*(*drm_edid).edid).standard_timings.as_ptr() as *const std_timing }.wrapping_offset(i as isize));
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) >= 1 {
drm_for_each_detailed_block(drm_edid, Some(do_standard_modes), core::ptr::addr_of_mut!(closure) as *mut ());
}
// XXX should also look for standard codes in VTB blocks
modes.wrapping_add(closure.modes)
}

fn drm_cvt_modes(connector: *mut drm_connector, timing: *const detailed_timing) -> i32 {
let mut modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
let mut empty: [u8; 3] = [0, 0, 0];
for i in 0i32..4i32 {
let mut width: i32 = 0;
let mut height: i32 = 0;
let cvt: *const cvt_timing = unsafe { core::ptr::addr_of!((*timing).data.other_data.data.cvt) as *const cvt_timing }.wrapping_offset(i as isize);
if (unsafe { memcmp(((*cvt).code.as_ptr() as *const u8) as *const (), empty.as_ptr() as *const (), 3u64) }) != 0 {
height = ((unsafe { *((*cvt).code.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32).wrapping_add((((unsafe { *((*cvt).code.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) & 240) << 4i32).wrapping_add(1).wrapping_mul(2);
match ((unsafe { *((*cvt).code.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) & 12 {
4 => {
width = height.wrapping_mul(16) / 9;
}
8 => {
width = height.wrapping_mul(16) / 10;
}
12 => {
width = height.wrapping_mul(15) / 9;
}
_ => {
// default - because compiler doesn't see that we've enumerated all cases
width = height.wrapping_mul(4) / 3;
}
}
for j in 1i32..5i32 {
if ((unsafe { *((*cvt).code.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) & 1i32 << j != 0 {
let newmode: *mut drm_display_mode = unsafe { drm_cvt_mode(dev, width, height, rates.0[j as usize], j == 0, false, false) };
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
}
}
}
modes
}

static rates: __SlateAlign16<[i32; 5]> = __SlateAlign16([60i32, 85i32, 75i32, 60i32, 50i32]);

#[unsafe(link_section = ".text.slate_distinct.drm_edid.do_cvt_mode")]
extern "C" fn do_cvt_mode(timing: *const detailed_timing, c: *mut ()) {
if !is_display_descriptor(timing, 248u8) {
return;
}
let __v0: i32 = drm_cvt_modes(unsafe { (*(c as *mut detailed_mode_closure)).connector }, timing);
let __v2: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v2.wrapping_add(__v0);
}
}

fn add_cvt_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) >= 3 {
drm_for_each_detailed_block(drm_edid, Some(do_cvt_mode), core::ptr::addr_of_mut!(closure) as *mut ());
}
// XXX should also look for CVT codes in VTB blocks
closure.modes
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.do_detailed_mode")]
extern "C" fn do_detailed_mode(timing: *const detailed_timing, c: *mut ()) {
if !is_detailed_timing_descriptor(timing) {
return;
}
let newmode: *mut drm_display_mode = drm_mode_detailed(unsafe { (*(c as *mut detailed_mode_closure)).connector }, unsafe { (*(c as *mut detailed_mode_closure)).drm_edid }, timing);
if newmode.is_null() {
return;
}
if unsafe { (*(c as *mut detailed_mode_closure)).preferred } {
unsafe {
(*newmode).r#type = (((*newmode).r#type as i32) | 8i32) as u8;
}
}
// Detailed modes are limited to 10kHz pixel clock resolution,
// so fix up anything that looks like CEA/HDMI mode, but the clock
// is just slightly off.
fixup_detailed_cea_mode_clock(unsafe { (*(c as *mut detailed_mode_closure)).connector }, newmode);
unsafe { drm_mode_probed_add((*(c as *mut detailed_mode_closure)).connector, newmode) };
let __v4: i32 = unsafe { (*(c as *mut detailed_mode_closure)).modes };
unsafe {
(*(c as *mut detailed_mode_closure)).modes = __v4.wrapping_add(1);
}
unsafe {
(*(c as *mut detailed_mode_closure)).preferred = false;
}
}

/// add_detailed_modes - Add modes from detailed timings
/// @connector: attached connector
/// @drm_edid: EDID block to scan
fn add_detailed_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) >= 4 {
closure.preferred = true;
} else {
closure.preferred = ((unsafe { (*(*drm_edid).edid).features }) as i32) & 2i32 != 0;
}
// first detailed timing is always preferred
drm_for_each_detailed_block(drm_edid, Some(do_detailed_mode), core::ptr::addr_of_mut!(closure) as *mut ());
closure.modes
}

// CTA-861-H Table 60 - CTA Tag Codes
// CTA-861-H Table 62 - CTA Extended Tag Codes
/// Search EDID for CEA extension block.
///
/// FIXME: Prefer not returning pointers to raw EDID data.
#[unsafe(no_mangle)]
extern "C" fn drm_edid_find_extension(drm_edid: *const drm_edid, ext_id: i32, ext_index: *mut i32) -> *const u8 {
let mut edid_ext: *const u8 = core::ptr::null::<u8>();
let mut i: i32 = 0;
// No EDID or EDID extensions
let __v0: bool = drm_edid.is_null() || drm_edid_extension_block_count(drm_edid) == 0;
if __v0 {
return core::ptr::null::<u8>();
}
// Find CEA extension
i = unsafe { *ext_index };
while i < drm_edid_extension_block_count(drm_edid) {
edid_ext = drm_edid_extension_block_data(drm_edid, i) as *const u8;
if edid_block_tag(edid_ext as *const ()) == ext_id {
break;
}
i = i.wrapping_add(1);
}
if i >= drm_edid_extension_block_count(drm_edid) {
return core::ptr::null::<u8>();
}
unsafe {
*ext_index = i.wrapping_add(1);
}
edid_ext
}

/// Return true if the EDID has a CTA extension or a DisplayID CTA data block
fn drm_edid_has_cta_extension(drm_edid: *const drm_edid) -> bool {
let mut iter: displayid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut edid_iter: drm_edid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(edid_iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut found: bool = false;
// Look for a top level CEA extension block
drm_edid_iter_begin(drm_edid, core::ptr::addr_of_mut!(edid_iter));
loop {
let __v1: *const u8 = __drm_edid_iter_next(core::ptr::addr_of_mut!(edid_iter)) as *const u8;
let ext: *const u8 = __v1;
if __v1.is_null() {
break;
}
if ((unsafe { *ext.wrapping_offset(0 as isize) }) as i32) == 2 {
found = true;
break;
}
}
drm_edid_iter_end(core::ptr::addr_of_mut!(edid_iter));
if found {
return true;
}
// CEA blocks can also be found embedded in a DisplayID block
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter)) };
loop {
let __v3: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!(iter)) };
if __v3.is_null() {
break;
}
if ((unsafe { (*__v3).tag }) as i32) == 129 {
found = true;
break;
}
}
unsafe { displayid_iter_end(core::ptr::addr_of_mut!(iter)) };
found
}

#[inline(always)]
fn cea_mode_for_vic(vic: u8) -> *const drm_display_mode {
loop {
if 1u64.wrapping_add(ARRAY_SIZE__89!()).wrapping_sub(1) != 127 {
unsafe { __compiletime_assert_739() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 193u64.wrapping_add(ARRAY_SIZE__70!()).wrapping_sub(1) != 219 {
unsafe { __compiletime_assert_740() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if (vic as i32) >= 1 && ((vic as i32) as u64) < 1u64.wrapping_add(ARRAY_SIZE__56!()) {
return unsafe { core::ptr::addr_of!(edid_cea_modes_1.0) as *const drm_display_mode }.wrapping_offset((vic as i32).wrapping_sub(1) as isize);
}
if (vic as i32) >= 193 && ((vic as i32) as u64) < 193u64.wrapping_add(ARRAY_SIZE__54!()) {
return unsafe { core::ptr::addr_of!(edid_cea_modes_193.0) as *const drm_display_mode }.wrapping_offset((vic as i32).wrapping_sub(193) as isize);
}
core::ptr::null::<drm_display_mode>()
}

fn cea_num_vics() -> u8 {
193u64.wrapping_add(ARRAY_SIZE__9!()) as u8
}

fn cea_next_vic(mut vic: u8) -> u8 {
let __v1: u8 = (vic as i32).wrapping_add(1) as u8;
vic = __v1;
if ((__v1 as i32) as u64) == 1u64.wrapping_add(ARRAY_SIZE__84!()) {
vic = 193;
}
vic
}

/// Calculate the alternate clock for the CEA mode
/// (60Hz vs. 59.94Hz etc.)
fn cea_mode_alternate_clock(cea_mode: *const drm_display_mode) -> u32 {
let mut clock: u32 = (unsafe { (*cea_mode).clock }) as u32;
if (unsafe { drm_mode_vrefresh(cea_mode) }) % 6 != 0 {
return clock;
}
// edid_cea_modes contains the 59.94Hz
// variant for 240 and 480 line modes,
// and the 60Hz variant otherwise.
if ((unsafe { (*cea_mode).vdisplay }) as i32) == 240 || ((unsafe { (*cea_mode).vdisplay }) as i32) == 480 {
let mut __v0: u32;
let __x: u32 = clock.wrapping_mul(1001);
let __d: i32 = 1000;
__v0 = if (1i32.wrapping_neg() as u32) > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add((__d / 2) as u32) / (__d as u32) } else { __x.wrapping_sub((__d / 2) as u32) / (__d as u32) };
clock = __v0;
} else {
let mut __v1: u32;
let __x: u32 = clock.wrapping_mul(1000);
let __d: i32 = 1001;
__v1 = if (1i32.wrapping_neg() as u32) > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add((__d / 2) as u32) / (__d as u32) } else { __x.wrapping_sub((__d / 2) as u32) / (__d as u32) };
clock = __v1;
}
clock
}

fn cea_mode_alternate_timings(vic: u8, mode: &mut drm_display_mode) -> bool {
// For certain VICs the spec allows the vertical
// front porch to vary by one or two lines.
//
// cea_modes[] stores the variant with the shortest
// vertical front porch. We can adjust the mode to
// get the other variants by simply increasing the
// vertical front porch length.
loop {
let __v0: bool = ((unsafe { (*cea_mode_for_vic(8u8)).vtotal }) as i32) != 262 || ((unsafe { (*cea_mode_for_vic(9u8)).vtotal }) as i32) != 262;
let __v1: bool = __v0 || ((unsafe { (*cea_mode_for_vic(12u8)).vtotal }) as i32) != 262;
let __v2: bool = __v1 || ((unsafe { (*cea_mode_for_vic(13u8)).vtotal }) as i32) != 262;
let __v3: bool = __v2 || ((unsafe { (*cea_mode_for_vic(23u8)).vtotal }) as i32) != 312;
let __v4: bool = __v3 || ((unsafe { (*cea_mode_for_vic(24u8)).vtotal }) as i32) != 312;
let __v5: bool = __v4 || ((unsafe { (*cea_mode_for_vic(27u8)).vtotal }) as i32) != 312;
let __v6: bool = __v5 || ((unsafe { (*cea_mode_for_vic(28u8)).vtotal }) as i32) != 312;
if __v6 {
unsafe { __compiletime_assert_741() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if ((vic as i32) == 8 || (vic as i32) == 9 || (vic as i32) == 12 || (vic as i32) == 13) && ((unsafe { (*(mode as *mut drm_display_mode)).vtotal }) as i32) < 263 || ((vic as i32) == 23 || (vic as i32) == 24 || (vic as i32) == 27 || (vic as i32) == 28) && ((unsafe { (*(mode as *mut drm_display_mode)).vtotal }) as i32) < 314 {
unsafe {
(*mode).vsync_start = ((*mode).vsync_start as i32).wrapping_add(1) as u16;
}
unsafe {
(*mode).vsync_end = ((*mode).vsync_end as i32).wrapping_add(1) as u16;
}
unsafe {
(*mode).vtotal = ((*mode).vtotal as i32).wrapping_add(1) as u16;
}
return true;
}
false
}

fn drm_match_cea_mode_clock_tolerance(to_match: *const drm_display_mode, clock_tolerance: u32) -> u8 {
let mut match_flags: u32 = 5u32;
let mut vic: u8 = 1;
if (unsafe { (*to_match).clock }) == 0 {
return 0;
}
if ((unsafe { (*to_match).picture_aspect_ratio }) as u32) != 0 {
match_flags |= 16u32;
}
while (vic as i32) < (cea_num_vics() as i32) {
let mut cea_mode: drm_display_mode = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cea_mode) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
let mut clock1: u32 = 0;
let mut clock2: u32 = 0;
unsafe { drm_mode_init(core::ptr::addr_of_mut!(cea_mode), cea_mode_for_vic(vic)) };
// Check both 60Hz and 59.94Hz
clock1 = cea_mode.clock as u32;
clock2 = cea_mode_alternate_clock(core::ptr::addr_of_mut!(cea_mode) as *const drm_display_mode);
let __v2: bool = (abs!(((unsafe { (*to_match).clock }) as u32).wrapping_sub(clock1)) as u32) > clock_tolerance && (abs!(((unsafe { (*to_match).clock }) as u32).wrapping_sub(clock2)) as u32) > clock_tolerance;
if !__v2 {
loop {
if unsafe { drm_mode_match(to_match, core::ptr::addr_of_mut!(cea_mode) as *const drm_display_mode, match_flags) } {
return vic;
}
if !cea_mode_alternate_timings(vic, unsafe { &mut (*core::ptr::addr_of_mut!(cea_mode)) }) {
break;
}
}
}
vic = cea_next_vic(vic);
}
0
}

/// drm_match_cea_mode - look for a CEA mode matching given mode
/// @to_match: display mode
///
/// Return: The CEA Video ID (VIC) of the mode or 0 if it isn't a CEA-861
/// mode.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_match_cea_mode")]
extern "C" fn drm_match_cea_mode(to_match: *const drm_display_mode) -> u8 {
let mut match_flags: u32 = 5u32;
let mut vic: u8 = 1;
if (unsafe { (*to_match).clock }) == 0 {
return 0;
}
if ((unsafe { (*to_match).picture_aspect_ratio }) as u32) != 0 {
match_flags |= 16u32;
}
while (vic as i32) < (cea_num_vics() as i32) {
let mut cea_mode: drm_display_mode = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cea_mode) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
let mut clock1: u32 = 0;
let mut clock2: u32 = 0;
unsafe { drm_mode_init(core::ptr::addr_of_mut!(cea_mode), cea_mode_for_vic(vic)) };
// Check both 60Hz and 59.94Hz
clock1 = cea_mode.clock as u32;
clock2 = cea_mode_alternate_clock(core::ptr::addr_of_mut!(cea_mode) as *const drm_display_mode);
if !(KHZ2PICOS!(unsafe { (*to_match).clock }) != KHZ2PICOS!(clock1) && KHZ2PICOS!(unsafe { (*to_match).clock }) != KHZ2PICOS!(clock2)) {
loop {
if unsafe { drm_mode_match(to_match, core::ptr::addr_of_mut!(cea_mode) as *const drm_display_mode, match_flags) } {
return vic;
}
if !cea_mode_alternate_timings(vic, unsafe { &mut (*core::ptr::addr_of_mut!(cea_mode)) }) {
break;
}
}
}
vic = cea_next_vic(vic);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_match_cea_mode_742: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode) -> u8>, *mut ()>(Some(drm_match_cea_mode)) };

fn drm_valid_cea_vic(vic: u8) -> bool {
!cea_mode_for_vic(vic).is_null()
}

fn drm_get_cea_aspect_ratio(video_code: u8) -> u32 {
let mode: *const drm_display_mode = cea_mode_for_vic(video_code);
if !mode.is_null() {
return unsafe { (*mode).picture_aspect_ratio };
}
0
}

fn drm_get_hdmi_aspect_ratio(video_code: u8) -> u32 {
unsafe { edid_4k_modes.0[(video_code as i32) as usize].picture_aspect_ratio }
}

/// Calculate the alternate clock for HDMI modes (those from the HDMI vendor
/// specific block).
fn hdmi_mode_alternate_clock(hdmi_mode: *const drm_display_mode) -> u32 {
cea_mode_alternate_clock(hdmi_mode)
}

fn drm_match_hdmi_mode_clock_tolerance(to_match: *const drm_display_mode, clock_tolerance: u32) -> u8 {
let mut match_flags: u32 = 5u32;
let mut vic: u8 = 1;
if (unsafe { (*to_match).clock }) == 0 {
return 0;
}
if ((unsafe { (*to_match).picture_aspect_ratio }) as u32) != 0 {
match_flags |= 16u32;
}
while ((vic as i32) as u64) < ARRAY_SIZE__79!() {
let hdmi_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!(edid_4k_modes.0) as *const drm_display_mode }.wrapping_offset((vic as i32) as isize);
let mut clock1: u32 = 0;
let mut clock2: u32 = 0;
// Make sure to also match alternate clocks
clock1 = (unsafe { (*hdmi_mode).clock }) as u32;
clock2 = hdmi_mode_alternate_clock(hdmi_mode);
let __v2: bool = (abs!(((unsafe { (*to_match).clock }) as u32).wrapping_sub(clock1)) as u32) > clock_tolerance && (abs!(((unsafe { (*to_match).clock }) as u32).wrapping_sub(clock2)) as u32) > clock_tolerance;
if !__v2 {
if unsafe { drm_mode_match(to_match, hdmi_mode, match_flags) } {
return vic;
}
}
vic = (vic as i32).wrapping_add(1) as u8;
}
0
}

/// drm_match_hdmi_mode - look for a HDMI mode matching given mode
/// @to_match: display mode
///
/// An HDMI mode is one defined in the HDMI vendor specific block.
///
/// Returns the HDMI Video ID (VIC) of the mode or 0 if it isn't one.
fn drm_match_hdmi_mode(to_match: *const drm_display_mode) -> u8 {
let mut match_flags: u32 = 5u32;
let mut vic: u8 = 1;
if (unsafe { (*to_match).clock }) == 0 {
return 0;
}
if ((unsafe { (*to_match).picture_aspect_ratio }) as u32) != 0 {
match_flags |= 16u32;
}
while ((vic as i32) as u64) < ARRAY_SIZE__41!() {
let hdmi_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!(edid_4k_modes.0) as *const drm_display_mode }.wrapping_offset((vic as i32) as isize);
let mut clock1: u32 = 0;
let mut clock2: u32 = 0;
// Make sure to also match alternate clocks
clock1 = (unsafe { (*hdmi_mode).clock }) as u32;
clock2 = hdmi_mode_alternate_clock(hdmi_mode);
let __v2: bool = (KHZ2PICOS!(unsafe { (*to_match).clock }) == KHZ2PICOS!(clock1) || KHZ2PICOS!(unsafe { (*to_match).clock }) == KHZ2PICOS!(clock2)) && unsafe { drm_mode_match(to_match, hdmi_mode, match_flags) };
if __v2 {
return vic;
}
vic = (vic as i32).wrapping_add(1) as u8;
}
0
}

fn drm_valid_hdmi_vic(vic: u8) -> bool {
(vic as i32) > 0 && ((vic as i32) as u64) < ARRAY_SIZE__45!()
}

fn add_alternate_cea_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let dev: *mut drm_device = unsafe { (*connector).dev };
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut tmp: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut list: list_head = unsafe { core::mem::zeroed() };
list = list_head { next: core::ptr::addr_of_mut!(list), prev: core::ptr::addr_of_mut!(list) };
let mut modes: i32 = 0;
// Don't add CTA modes if the CTA extension block is missing
if !drm_edid_has_cta_extension(drm_edid) {
return 0;
}
// Go through all probed modes and create a new mode
// with the alternate clock for certain CEA modes.
mode = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
while list_entry_is_head!(mode, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
let mut cea_mode: *const drm_display_mode = core::ptr::null::<drm_display_mode>();
let mut newmode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut vic: u8 = drm_match_cea_mode(mode as *const drm_display_mode);
let mut clock1: u32 = 0;
let mut clock2: u32 = 0;
if drm_valid_cea_vic(vic) {
cea_mode = cea_mode_for_vic(vic);
clock2 = cea_mode_alternate_clock(cea_mode);
} else {
vic = drm_match_hdmi_mode(mode as *const drm_display_mode);
if drm_valid_hdmi_vic(vic) {
cea_mode = unsafe { core::ptr::addr_of!(edid_4k_modes.0) as *const drm_display_mode }.wrapping_offset((vic as i32) as isize);
clock2 = hdmi_mode_alternate_clock(cea_mode);
}
}
if !cea_mode.is_null() {
clock1 = (unsafe { (*cea_mode).clock }) as u32;
if clock1 != clock2 {
if !(((unsafe { (*mode).clock }) as u32) != clock1 && ((unsafe { (*mode).clock }) as u32) != clock2) {
newmode = unsafe { drm_mode_duplicate(dev, cea_mode) };
if !newmode.is_null() {
// Carry over the stereo flags
unsafe {
(*newmode).flags = (*newmode).flags | (*mode).flags & 507904u32;
}
// The current mode could be either variant. Make
// sure to pick the "other" clock for the new mode.
if ((unsafe { (*mode).clock }) as u32) != clock1 {
unsafe {
(*newmode).clock = clock1 as i32;
}
} else {
unsafe {
(*newmode).clock = clock2 as i32;
}
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*newmode).head) }, core::ptr::addr_of_mut!(list));
}
}
}
}
mode = list_next_entry!(mode, head);
}
mode = list_first_entry__4!(core::ptr::addr_of_mut!(list));
tmp = list_next_entry!(mode, head);
while list_entry_is_head!(mode, core::ptr::addr_of_mut!(list), head) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*mode).head) });
unsafe { drm_mode_probed_add(connector, mode) };
modes = modes.wrapping_add(1);
mode = tmp;
tmp = list_next_entry!(tmp, head);
}
modes
}

fn svd_to_vic(svd: u8) -> u8 {
// 0-6 bit vic, 7th bit native mode indicator
if (svd as i32) >= 1 && (svd as i32) <= 64 || (svd as i32) >= 129 && (svd as i32) <= 192 {
return ((svd as i32) & 127) as u8;
}
svd
}

/// Return a display mode for the 0-based vic_index'th VIC across all CTA VDBs in
/// the EDID, or NULL on errors.
fn drm_display_mode_from_vic_index(connector: *mut drm_connector, vic_index: i32) -> *mut drm_display_mode {
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).display_info) }) as *const drm_display_info;
let dev: *mut drm_device = unsafe { (*connector).dev };
if unsafe { (*info).vics }.is_null() || vic_index >= unsafe { (*info).vics_len } || (unsafe { *unsafe { (*info).vics }.wrapping_offset(vic_index as isize) }) == 0 {
return core::ptr::null_mut::<drm_display_mode>();
}
drm_display_mode_from_cea_vic(dev, unsafe { *unsafe { (*info).vics }.wrapping_offset(vic_index as isize) })
}

/// do_y420vdb_modes - Parse YCBCR 420 only modes
/// @connector: connector corresponding to the HDMI sink
/// @svds: start of the data block of CEA YCBCR 420 VDB
/// @len: length of the CEA YCBCR 420 VDB
///
/// Parse the CEA-861-F YCBCR 420 Video Data Block (Y420VDB)
/// which contains modes which can be supported in YCBCR 420
/// output format only.
fn do_y420vdb_modes(connector: *mut drm_connector, svds: *const u8, svds_len: u8) -> i32 {
let dev: *mut drm_device = unsafe { (*connector).dev };
let mut modes: i32 = 0;
for i in 0..svds_len as i32 {
let vic: u8 = svd_to_vic(unsafe { *svds.wrapping_offset(i as isize) });
if drm_valid_cea_vic(vic) {
let newmode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, cea_mode_for_vic(vic)) };
if newmode.is_null() {
break;
}
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
modes
}

/// drm_display_mode_from_cea_vic() - return a mode for CEA VIC
/// @dev: DRM device
/// @video_code: CEA VIC of the mode
///
/// Creates a new mode matching the specified CEA VIC.
///
/// Returns: A new drm_display_mode on success or NULL on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_display_mode_from_cea_vic")]
extern "C" fn drm_display_mode_from_cea_vic(dev: *mut drm_device, video_code: u8) -> *mut drm_display_mode {
let cea_mode: *const drm_display_mode = cea_mode_for_vic(video_code);
if cea_mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
let newmode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, cea_mode) };
if newmode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
newmode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_display_mode_from_cea_vic_743: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, u8) -> *mut drm_display_mode>, *mut ()>(Some(drm_display_mode_from_cea_vic)) };

/// Add modes based on VICs parsed in parse_cta_vdb()
fn add_cta_vdb_modes(connector: *mut drm_connector) -> i32 {
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).display_info) }) as *const drm_display_info;
let mut i: i32 = 0;
let mut modes: i32 = 0;
if unsafe { (*info).vics }.is_null() {
return 0;
}
while i < unsafe { (*info).vics_len } {
let mode: *mut drm_display_mode = drm_display_mode_from_vic_index(connector, i);
if !mode.is_null() {
unsafe { drm_mode_probed_add(connector, mode) };
modes = modes.wrapping_add(1);
}
i = i.wrapping_add(1);
}
modes
}

#[repr(C)]
#[derive(Clone, Copy)]
struct stereo_mandatory_mode {
    width: i32,
    height: i32,
    vrefresh: i32,
    flags: u32,
}


static stereo_mandatory_modes: __SlateAlign16<[stereo_mandatory_mode; 8]> = __SlateAlign16([stereo_mandatory_mode { width: 1920i32, height: 1080i32, vrefresh: 24i32, flags: 114688u32 }, stereo_mandatory_mode { width: 1920i32, height: 1080i32, vrefresh: 24i32, flags: 16384u32 }, stereo_mandatory_mode { width: 1920i32, height: 1080i32, vrefresh: 50i32, flags: 131088u32 }, stereo_mandatory_mode { width: 1920i32, height: 1080i32, vrefresh: 60i32, flags: 131088u32 }, stereo_mandatory_mode { width: 1280i32, height: 720i32, vrefresh: 50i32, flags: 114688u32 }, stereo_mandatory_mode { width: 1280i32, height: 720i32, vrefresh: 50i32, flags: 16384u32 }, stereo_mandatory_mode { width: 1280i32, height: 720i32, vrefresh: 60i32, flags: 114688u32 }, stereo_mandatory_mode { width: 1280i32, height: 720i32, vrefresh: 60i32, flags: 16384u32 }]);

fn stereo_match_mandatory(mode: *const drm_display_mode, stereo_mode: *const stereo_mandatory_mode) -> bool {
let interlaced: u32 = (unsafe { (*mode).flags }) & 16u32;
((unsafe { (*mode).hdisplay }) as i32) == unsafe { (*stereo_mode).width } && ((unsafe { (*mode).vdisplay }) as i32) == unsafe { (*stereo_mode).height } && interlaced == (unsafe { (*stereo_mode).flags }) & 16u32 && (unsafe { drm_mode_vrefresh(mode) }) == unsafe { (*stereo_mode).vrefresh }
}

fn add_hdmi_mandatory_stereo_modes(connector: *mut drm_connector) -> i32 {
let dev: *mut drm_device = unsafe { (*connector).dev };
let mut mode: *const drm_display_mode = unsafe { core::mem::zeroed() };
let mut stereo_modes: list_head = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(stereo_modes) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut modes: i32 = 0;
let mut i: i32 = 0;
INIT_LIST_HEAD(core::ptr::addr_of_mut!(stereo_modes));
mode = list_first_entry__12!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
while list_entry_is_head__3!(mode, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
i = 0;
while (i as u64) < ARRAY_SIZE__80!() {
let mut new_mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
if stereo_match_mandatory(mode, unsafe { core::ptr::addr_of!(stereo_mandatory_modes.0) as *const stereo_mandatory_mode }.wrapping_offset(i as isize)) {
let mandatory: *const stereo_mandatory_mode = unsafe { core::ptr::addr_of!(stereo_mandatory_modes.0) as *const stereo_mandatory_mode }.wrapping_offset(i as isize);
new_mode = unsafe { drm_mode_duplicate(dev, mode) };
if !new_mode.is_null() {
unsafe {
(*new_mode).flags = (*new_mode).flags | (*mandatory).flags;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*new_mode).head) }, core::ptr::addr_of_mut!(stereo_modes));
modes = modes.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
mode = list_next_entry__17!(mode, head);
}
list_splice_tail(unsafe { &(*core::ptr::addr_of_mut!(stereo_modes)) }, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
modes
}

fn add_hdmi_mode(connector: *mut drm_connector, vic: u8) -> i32 {
let dev: *mut drm_device = unsafe { (*connector).dev };
if !drm_valid_hdmi_vic(vic) {
while false {
}
unsafe { _dev_err(__drm_to_dev((*connector).dev as *const drm_device) as *const device, c"[drm] *ERROR* [CONNECTOR:%d:%s] Unknown HDMI VIC: %d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, vic as i32) };
return 0;
}
let newmode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, (core::ptr::addr_of!(edid_4k_modes.0) as *const drm_display_mode).wrapping_offset((vic as i32) as isize)) };
if newmode.is_null() {
return 0;
}
unsafe { drm_mode_probed_add(connector, newmode) };
1
}

fn add_3d_struct_modes(connector: *mut drm_connector, structure: u16, vic_index: i32) -> i32 {
let mut newmode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut modes: i32 = 0;
if (structure as i32) & 1i32 != 0 {
newmode = drm_display_mode_from_vic_index(connector, vic_index);
if !newmode.is_null() {
unsafe {
(*newmode).flags = (*newmode).flags | 16384u32;
}
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
if (structure as i32) & 64i32 != 0 {
newmode = drm_display_mode_from_vic_index(connector, vic_index);
if !newmode.is_null() {
unsafe {
(*newmode).flags = (*newmode).flags | 114688u32;
}
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
if (structure as i32) & 256i32 != 0 {
newmode = drm_display_mode_from_vic_index(connector, vic_index);
if !newmode.is_null() {
unsafe {
(*newmode).flags = (*newmode).flags | 131072u32;
}
unsafe { drm_mode_probed_add(connector, newmode) };
modes = modes.wrapping_add(1);
}
}
modes
}

fn hdmi_vsdb_latency_present(db: *const u8) -> bool {
(((unsafe { *db.wrapping_offset(8 as isize) }) as i32) as u64) & BIT!(7i32) != 0
}

fn hdmi_vsdb_i_latency_present(db: *const u8) -> bool {
hdmi_vsdb_latency_present(db) && (((unsafe { *db.wrapping_offset(8 as isize) }) as i32) as u64) & BIT!(6i32) != 0
}

fn hdmi_vsdb_latency_length(db: *const u8) -> i32 {
if hdmi_vsdb_i_latency_present(db) {
return 4;
} else {
if hdmi_vsdb_latency_present(db) {
return 2;
} else {
return 0;
}
}
}

/// do_hdmi_vsdb_modes - Parse the HDMI Vendor Specific data block
/// @connector: connector corresponding to the HDMI sink
/// @db: start of the CEA vendor specific block
/// @len: length of the CEA block payload, ie. one can access up to db[len]
///
/// Parses the HDMI VSDB looking for modes to add to @connector. This function
/// also adds the stereo 3d modes when applicable.
fn do_hdmi_vsdb_modes(connector: *mut drm_connector, db: *const u8, len: u8) -> i32 {
let mut detail_present: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut bool = core::ptr::addr_of_mut!(detail_present) as *mut bool;
let mut newflag: u32 = unsafe { core::mem::zeroed() };
let mut newmode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut vic_index: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(vic_index) as *mut i32;
let mut vic: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u8 = core::ptr::addr_of_mut!(vic) as *mut u8;
let mut structure_all: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut u16 = core::ptr::addr_of_mut!(structure_all) as *mut u16;
let mut mask: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u16 = core::ptr::addr_of_mut!(mask) as *mut u16;
let mut hdmi_3d_len: u8 = unsafe { core::mem::zeroed() };
let mut vic_len: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u8 = core::ptr::addr_of_mut!(vic_len) as *mut u8;
let mut multi_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut i32 = core::ptr::addr_of_mut!(multi_len) as *mut i32;
let mut multi_present: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut offset: i32 = unsafe { core::mem::zeroed() };
let mut modes: i32 = unsafe { core::mem::zeroed() };
unsafe {
modes = 0;
offset = 0;
let __slate_auto_bytes_1 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
multi_present = 0;
let __slate_auto_bytes_2 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
hdmi_3d_len = 0;
let __slate_auto_bytes_4 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_5 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
if (len as i32) >= 8 {
// no HDMI_Video_Present
if (*db.wrapping_offset(8 as isize) as i32) & 32i32 != 0 {
let __v37: i32 = hdmi_vsdb_latency_length(db);
offset = offset.wrapping_add(__v37);
// the declared length is not long enough for the 2 first bytes
// of additional video format capabilities
if (len as i32) >= 8i32.wrapping_add(offset).wrapping_add(2) {
// 3D_Present
offset = offset.wrapping_add(1);
if (*db.wrapping_offset(8i32.wrapping_add(offset) as isize) as i32) & 128i32 != 0 {
let __v32: i32 = add_hdmi_mandatory_stereo_modes(connector);
modes = modes.wrapping_add(__v32);
// 3D_Multi_present
multi_present = ((*db.wrapping_offset(8i32.wrapping_add(offset) as isize) as i32) & 96) >> 5i32;
}
offset = offset.wrapping_add(1);
*__slate_slot_41 = ((*db.wrapping_offset(8i32.wrapping_add(offset) as isize) as i32) >> 5i32) as u8;
hdmi_3d_len = ((*db.wrapping_offset(8i32.wrapping_add(offset) as isize) as i32) & 31) as u8;
*__slate_slot_44 = 0;
while *__slate_slot_44 < (*__slate_slot_41 as i32) && (len as i32) >= 9i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) {
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_27 = *db.wrapping_offset(9i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) as isize);
let __v26: i32 = add_hdmi_mode(connector, *__slate_slot_27);
modes = modes.wrapping_add(__v26);
let __v23: i32 = *__slate_slot_44;
*__slate_slot_44 = __v23.wrapping_add(1);
}
offset = offset.wrapping_add(1i32.wrapping_add(*__slate_slot_41 as i32));
if multi_present == 1 {
*__slate_slot_42 = 2;
} else {
if multi_present == 2 {
*__slate_slot_42 = 4;
} else {
*__slate_slot_42 = 0;
}
}
if (len as i32) >= 8i32.wrapping_add(offset).wrapping_add(hdmi_3d_len as i32).wrapping_sub(1) {
if (hdmi_3d_len as i32) >= *__slate_slot_42 {
if multi_present == 1 || multi_present == 2 {
// 3D_Structure_ALL
*__slate_slot_38 = ((*db.wrapping_offset(8i32.wrapping_add(offset) as isize) as i32) << 8i32 | (*db.wrapping_offset(9i32.wrapping_add(offset) as isize) as i32)) as u16;
// check if 3D_MASK is present
if multi_present == 2 {
*__slate_slot_39 = ((*db.wrapping_offset(10i32.wrapping_add(offset) as isize) as i32) << 8i32 | (*db.wrapping_offset(11i32.wrapping_add(offset) as isize) as i32)) as u16;
} else {
*__slate_slot_39 = 65535;
}
*__slate_slot_44 = 0;
while *__slate_slot_44 < 16 {
if (*__slate_slot_39 as i32) & 1i32 << *__slate_slot_44 != 0 {
let __v19: i32 = add_3d_struct_modes(connector, *__slate_slot_38, *__slate_slot_44);
modes = modes.wrapping_add(__v19);
}
let __v16: i32 = *__slate_slot_44;
*__slate_slot_44 = __v16.wrapping_add(1);
}
}
offset = offset.wrapping_add(*__slate_slot_42);
*__slate_slot_44 = 0;
while *__slate_slot_44 < (hdmi_3d_len as i32).wrapping_sub(*__slate_slot_42) {
let __slate_auto_bytes_6 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
newmode = core::ptr::null_mut::<drm_display_mode>();
newflag = 0;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_9 = (*db.wrapping_offset(8i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) as isize) as i32) & 15 > 7;
if *__slate_slot_9 && (*__slate_slot_44).wrapping_add(1) == (hdmi_3d_len as i32).wrapping_sub(*__slate_slot_42) {
break;
} else {
// 2D_VIC_order_X
*__slate_slot_12 = (*db.wrapping_offset(8i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) as isize) as i32) >> 4i32;
// 3D_Structure_X
let __t0: i32 = (*db.wrapping_offset(8i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) as isize) as i32) & 15;
if __t0 == 0 {
newflag = 16384u32;
} else {
if __t0 == 6 {
newflag = 114688u32;
} else {
if __t0 == 8 {
if (*db.wrapping_offset(9i32.wrapping_add(offset).wrapping_add(*__slate_slot_44) as isize) as i32) >> 4i32 == 1 {
newflag = 131072u32;
}
// 3D_Detail_X
}
}
}
if newflag != 0 {
newmode = drm_display_mode_from_vic_index(connector, *__slate_slot_12);
if !newmode.is_null() {
(*newmode).flags = (*newmode).flags | newflag;
drm_mode_probed_add(connector, newmode);
modes = modes.wrapping_add(1);
}
}
if *__slate_slot_9 {
let __v3: i32 = *__slate_slot_44;
*__slate_slot_44 = __v3.wrapping_add(1);
}
let __v1: i32 = *__slate_slot_44;
*__slate_slot_44 = __v1.wrapping_add(1);
}
}
}
}
}
}
}
return modes;
}
}

fn cea_revision(cea: *const u8) -> i32 {
// FIXME is this correct for the DispID variant?
// The DispID spec doesn't really specify whether
// this is the revision of the CEA extension or
// the DispID CEA data block. And the only value
// given as an example is 0.
(unsafe { *cea.wrapping_offset(1 as isize) }) as i32
}

/// CTA Data Block iterator.
///
/// Iterate through all CTA Data Blocks in both EDID CTA Extensions and DisplayID
/// CTA Data Blocks.
///
/// struct cea_db *db:
/// struct cea_db_iter iter;
///
/// cea_db_iter_edid_begin(edid, &iter);
/// cea_db_iter_for_each(db, &iter) {
///         // do stuff with db
/// }
/// cea_db_iter_end(&iter);
#[repr(C)]
#[derive(Clone, Copy)]
struct cea_db_iter {
    edid_iter: drm_edid_iter,
    displayid_iter: displayid_iter,
    /// Current Data Block Collection.
    collection: *const u8,
    /// Current Data Block index in current collection.
    index: i32,
    /// End index in current collection.
    end: i32,
}


/// CTA-861-H section 7.4 CTA Data BLock Collection
#[repr(C)]
#[derive(Clone, Copy)]
struct cea_db {
    tag_length: u8,
    data: [u8; 0],
}


fn cea_db_tag(db: *const cea_db) -> i32 {
((unsafe { (*db).tag_length }) as i32) >> 5i32
}

fn cea_db_payload_len(_db: *const ()) -> i32 {
// FIXME: Transition to passing struct cea_db * everywhere.
((unsafe { (*(_db as *const cea_db)).tag_length }) as i32) & 31
}

fn cea_db_data(db: *const cea_db) -> *const () {
(unsafe { core::ptr::addr_of!((*db).data) as *const u8 }) as *const ()
}

fn cea_db_is_extended_tag(db: *const cea_db, tag: i32) -> bool {
let __v0: bool = cea_db_tag(db) == 7 && cea_db_payload_len(db as *const ()) >= 1;
__v0 && ((unsafe { *(core::ptr::addr_of!((*db).data) as *const u8).wrapping_offset(0 as isize) }) as i32) == tag
}

fn cea_db_is_vendor(db: *const cea_db, vendor_oui: i32) -> bool {
let data: *const u8 = cea_db_data(db) as *const u8;
let __v0: bool = cea_db_tag(db) == 3 && cea_db_payload_len(db as *const ()) >= 3;
__v0 && oui(unsafe { *data.wrapping_offset(2 as isize) }, unsafe { *data.wrapping_offset(1 as isize) }, unsafe { *data.wrapping_offset(0 as isize) }) == vendor_oui
}

fn cea_db_iter_edid_begin(drm_edid: *const drm_edid, iter: *mut cea_db_iter) {
unsafe { memset(iter as *mut (), 0i32, core::mem::size_of::<cea_db_iter>() as u64) };
drm_edid_iter_begin(drm_edid, unsafe { core::ptr::addr_of_mut!((*iter).edid_iter) });
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!((*iter).displayid_iter)) };
}

fn __cea_db_iter_current_block(iter: *const cea_db_iter) -> *const cea_db {
if unsafe { (*iter).collection }.is_null() {
return core::ptr::null::<cea_db>();
}
let db: *const cea_db = unsafe { (*iter).collection }.wrapping_offset((unsafe { (*iter).index }) as isize) as *const cea_db;
let __v0: bool = ((unsafe { (*iter).index }) as u64).wrapping_add(1) <= ((unsafe { (*iter).end }) as u64) && ((unsafe { (*iter).index }) as u64).wrapping_add(1).wrapping_add(cea_db_payload_len(db as *const ()) as u64) <= ((unsafe { (*iter).end }) as u64);
if __v0 {
return db;
}
core::ptr::null::<cea_db>()
}

/// References:
/// - CTA-861-H section 7.3.3 CTA Extension Version 3
fn cea_db_collection_size(cta: *const u8) -> i32 {
let d: u8 = unsafe { *cta.wrapping_offset(2 as isize) };
if (d as i32) < 4 || (d as i32) > 127 {
return 0;
}
(d as i32).wrapping_sub(4)
}

/// References:
/// - VESA E-EDID v1.4
/// - CTA-861-H section 7.3.3 CTA Extension Version 3
fn __cea_db_iter_edid_next(iter: &mut cea_db_iter) -> *const () {
loop {
let __v2: *const u8 = __drm_edid_iter_next(unsafe { core::ptr::addr_of_mut!((*(iter as *mut cea_db_iter)).edid_iter) }) as *const u8;
let ext: *const u8 = __v2;
if __v2.is_null() {
break;
}
let mut size: i32 = 0;
// Only support CTA Extension revision 3+
let __v0: bool = ((unsafe { *ext.wrapping_offset(0 as isize) }) as i32) != 2 || cea_revision(ext) < 3;
if !__v0 {
size = cea_db_collection_size(ext);
if size != 0 {
unsafe {
(*(iter as *mut cea_db_iter)).index = 4;
}
unsafe {
(*(iter as *mut cea_db_iter)).end = unsafe { (*(iter as *mut cea_db_iter)).index }.wrapping_add(size);
}
return ext as *const ();
}
}
}
core::ptr::null::<()>()
}

/// References:
/// - DisplayID v1.3 Appendix C: CEA Data Block within a DisplayID Data Block
/// - DisplayID v2.0 section 4.10 CTA DisplayID Data Block
///
/// Note that the above do not specify any connection between DisplayID Data
/// Block revision and CTA Extension versions.
fn __cea_db_iter_displayid_next(iter: &mut cea_db_iter) -> *const () {
loop {
let __v1: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!((*(iter as *mut cea_db_iter)).displayid_iter)) };
let block: *const displayid_block = __v1;
if __v1.is_null() {
break;
}
if ((unsafe { (*block).tag }) as i32) == 129 {
// The displayid iterator has already verified the block bounds
// in displayid_iter_block().
unsafe {
(*(iter as *mut cea_db_iter)).index = core::mem::size_of::<displayid_block>() as i32;
}
unsafe {
(*(iter as *mut cea_db_iter)).end = unsafe { (*(iter as *mut cea_db_iter)).index }.wrapping_add((*block).num_bytes as i32);
}
return block as *const ();
}
}
core::ptr::null::<()>()
}

fn __cea_db_iter_next(iter: &mut cea_db_iter) -> *const cea_db {
let mut db: *const cea_db = unsafe { core::mem::zeroed() };
if !unsafe { (*(iter as *mut cea_db_iter)).collection }.is_null() {
// Current collection should always be valid.
db = __cea_db_iter_current_block((iter as *mut cea_db_iter) as *const cea_db_iter);
let mut __v0: i64;
let __ret_warn_on: i32 = db.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("744: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 744b - ., 3; .popsection\n# {0}", const 744 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___cea_db_iter_next_1, sym __slate_asm_string___cea_db_iter_next_2, const 5161 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("745: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 745b - ., 4; .popsection\n# {0}", const 745 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
unsafe {
(*(iter as *mut cea_db_iter)).collection = core::ptr::null::<u8>();
}
return core::ptr::null::<cea_db>();
}
// Next block in CTA Data Block Collection
let __v5: u64 = 1u64.wrapping_add(cea_db_payload_len(db as *const ()) as u64);
unsafe {
(*iter).index = ((*iter).index as u64).wrapping_add(__v5) as i32;
}
db = __cea_db_iter_current_block((iter as *mut cea_db_iter) as *const cea_db_iter);
if !db.is_null() {
return db;
}
}
loop {
// Find the next CTA Data Block Collection. First iterate all
// the EDID CTA Extensions, then all the DisplayID CTA blocks.
//
// Per DisplayID v1.3 Appendix B: DisplayID as an EDID
// Extension, it's recommended that DisplayID extensions are
// exposed after all of the CTA Extensions.
unsafe {
(*(iter as *mut cea_db_iter)).collection = __cea_db_iter_edid_next(iter) as *const u8;
}
if unsafe { (*(iter as *mut cea_db_iter)).collection }.is_null() {
unsafe {
(*(iter as *mut cea_db_iter)).collection = __cea_db_iter_displayid_next(iter) as *const u8;
}
}
if unsafe { (*(iter as *mut cea_db_iter)).collection }.is_null() {
return core::ptr::null::<cea_db>();
}
db = __cea_db_iter_current_block((iter as *mut cea_db_iter) as *const cea_db_iter);
if !db.is_null() {
return db;
}
}
unsafe { core::mem::zeroed() }
}

fn cea_db_iter_end(iter: *mut cea_db_iter) {
unsafe { displayid_iter_end(core::ptr::addr_of_mut!((*iter).displayid_iter)) };
drm_edid_iter_end(unsafe { core::ptr::addr_of_mut!((*iter).edid_iter) });
unsafe { memset(iter as *mut (), 0i32, core::mem::size_of::<cea_db_iter>() as u64) };
}

fn cea_db_is_hdmi_vsdb(db: *const cea_db) -> bool {
cea_db_is_vendor(db, 3075i32) && cea_db_payload_len(db as *const ()) >= 5
}

fn cea_db_is_hdmi_forum_vsdb(db: *const cea_db) -> bool {
cea_db_is_vendor(db, 12869080i32) && cea_db_payload_len(db as *const ()) >= 7
}

fn cea_db_is_hdmi_forum_eeodb(db: *const ()) -> bool {
cea_db_is_extended_tag(db as *const cea_db, 120i32) && cea_db_payload_len(db) >= 2
}

fn cea_db_is_microsoft_vsdb(db: *const cea_db) -> bool {
cea_db_is_vendor(db, 13242972i32) && cea_db_payload_len(db as *const ()) == 21
}

fn cea_db_is_amd_vsdb(db: *const cea_db) -> bool {
let __v0: bool = cea_db_is_vendor(db, 26i32) && cea_db_payload_len(db as *const ()) >= 15;
__v0 && cea_db_payload_len(db as *const ()) <= 20
}

fn cea_db_is_vcdb(db: *const cea_db) -> bool {
cea_db_is_extended_tag(db, 0i32) && cea_db_payload_len(db as *const ()) == 2
}

fn cea_db_is_hdmi_forum_scdb(db: *const cea_db) -> bool {
cea_db_is_extended_tag(db, 121i32) && cea_db_payload_len(db as *const ()) >= 7
}

fn cea_db_is_y420cmdb(db: *const cea_db) -> bool {
cea_db_is_extended_tag(db, 15i32)
}

fn cea_db_is_y420vdb(db: *const cea_db) -> bool {
cea_db_is_extended_tag(db, 14i32)
}

fn cea_db_is_hdmi_hdr_metadata_block(db: *const cea_db) -> bool {
cea_db_is_extended_tag(db, 6i32) && cea_db_payload_len(db as *const ()) >= 3
}

/// Get the HF-EEODB override extension block count from EDID.
///
/// The passed in EDID may be partially read, as long as it has at least two
/// blocks (base block and one extension block) if EDID extension count is > 0.
///
/// Note that this is *not* how you should parse CTA Data Blocks in general; this
/// is only to handle partially read EDIDs. Normally, use the CTA Data Block
/// iterators instead.
///
/// References:
/// - HDMI 2.1 section 10.3.6 HDMI Forum EDID Extension Override Data Block
fn edid_hfeeodb_extension_block_count(edid: *const edid) -> i32 {
// No extensions according to base block, no HF-EEODB.
if edid_extension_block_count(edid) == 0 {
return 0;
}
// HF-EEODB is always in the first EDID extension block only
let cta: *const u8 = edid_extension_block_data(edid, 0i32) as *const u8;
let __v0: bool = edid_block_tag(cta as *const ()) != 2 || cea_revision(cta) < 3;
if __v0 {
return 0;
}
// Need to have the data block collection, and at least 3 bytes.
if cea_db_collection_size(cta) < 3 {
return 0;
}
// Sinks that include the HF-EEODB in their E-EDID shall include one and
// only one instance of the HF-EEODB in the E-EDID, occupying bytes 4
// through 6 of Block 1 of the E-EDID.
if !cea_db_is_hdmi_forum_eeodb(cta.wrapping_offset(4 as isize) as *const ()) {
return 0;
}
(unsafe { *cta.wrapping_offset(6i32 as isize) }) as i32
}

/// CTA-861 YCbCr 4:2:0 Capability Map Data Block (CTA Y420CMDB)
///
/// Y420CMDB contains a bitmap which gives the index of CTA modes from CTA VDB,
/// which can support YCBCR 420 sampling output also (apart from RGB/YCBCR444
/// etc). For example, if the bit 0 in bitmap is set, first mode in VDB can
/// support YCBCR420 output too.
fn parse_cta_y420cmdb(connector: *mut drm_connector, db: *const cea_db, y420cmdb_map: &mut u64) {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut map: u64 = unsafe { core::mem::zeroed() };
let mut data: *const u8 = unsafe { core::mem::zeroed() };
let mut map_len: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut info: *mut drm_display_info = unsafe { core::mem::zeroed() };
unsafe {
info = core::ptr::addr_of_mut!((*connector).display_info);
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
map_len = cea_db_payload_len(db as *const ()).wrapping_sub(1);
data = ((cea_db_data(db) as *const u8).wrapping_offset(1 as isize) as *const ()) as *const u8;
map = 0;
if map_len == 0 {
// All CEA modes support ycbcr420 sampling also.
map = !0;
} else {
__ret_warn_on = (map_len > 8) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("746: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 746b - ., 3; .popsection\n# {0}", const 746 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_parse_cta_y420cmdb_15, sym __slate_asm_string_parse_cta_y420cmdb_16, const 5341 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("747: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 747b - ., 4; .popsection\n# {0}", const 747 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// This map indicates which of the existing CEA block modes
// from VDB can support YCBCR420 output too. So if bit=0 is
// set, first mode from VDB can support YCBCR420 output too.
// We will parse and keep this map, before parsing VDB itself
// to avoid going through the same block again and again.
//
// Spec is not clear about max possible size of this block.
// Clamping max bitmap block size at 8 bytes. Every byte can
// address 8 CEA modes, in this way this map can address
// 8*8 = first 64 SVDs.
if __ret_warn_on != 0 {
map_len = 8;
}
*__slate_slot_13 = 0;
while *__slate_slot_13 < map_len {
map |= (*data.wrapping_offset(*__slate_slot_13 as isize) as u64) << 8i32.wrapping_mul(*__slate_slot_13);
let __v4: i32 = *__slate_slot_13;
*__slate_slot_13 = __v4.wrapping_add(1);
}
}
if map != 0 {
let __v1: u32 = (*info).color_formats;
let __v0: u32 = ((__v1 as u64) | BIT!(3i32)) as u32;
(*info).color_formats = __v0;
}
*(y420cmdb_map as *mut u64) = map;
}
}

fn add_cea_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut modes: i32 = 0;
// CTA VDB block VICs parsed earlier
modes = add_cta_vdb_modes(connector);
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
loop {
let __v6: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
let db: *const cea_db = __v6;
if __v6.is_null() {
break;
}
if cea_db_is_hdmi_vsdb(db) {
let __v0: i32 = do_hdmi_vsdb_modes(connector, db as *const u8, cea_db_payload_len(db as *const ()) as u8);
modes = modes.wrapping_add(__v0);
} else {
if cea_db_is_y420vdb(db) {
let vdb420: *const u8 = ((unsafe { (cea_db_data(db) as *const u8).wrapping_offset(1 as isize) }) as *const ()) as *const u8;
// Add 4:2:0(only) modes present in EDID
let __v3: i32 = do_y420vdb_modes(connector, vdb420, cea_db_payload_len(db as *const ()).wrapping_sub(1) as u8);
modes = modes.wrapping_add(__v3);
}
}
}
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
modes
}

fn fixup_detailed_cea_mode_clock(connector: *mut drm_connector, mode: *mut drm_display_mode) {
let mut cea_mode: *const drm_display_mode = unsafe { core::mem::zeroed() };
let mut clock1: i32 = 0;
let mut clock2: i32 = 0;
let mut clock: i32 = 0;
let mut vic: u8 = 0;
let mut r#type: *const u8 = unsafe { core::mem::zeroed() };
// allow 5kHz clock difference either way to account for
// the 10kHz clock resolution limit of detailed timings.
vic = drm_match_cea_mode_clock_tolerance(mode as *const drm_display_mode, 5u32);
if drm_valid_cea_vic(vic) {
r#type = c"CEA".as_ptr() as *const u8;
cea_mode = cea_mode_for_vic(vic);
clock1 = unsafe { (*cea_mode).clock };
clock2 = cea_mode_alternate_clock(cea_mode) as i32;
} else {
vic = drm_match_hdmi_mode_clock_tolerance(mode as *const drm_display_mode, 5u32);
if drm_valid_hdmi_vic(vic) {
r#type = c"HDMI".as_ptr() as *const u8;
cea_mode = unsafe { core::ptr::addr_of!(edid_4k_modes.0) as *const drm_display_mode }.wrapping_offset((vic as i32) as isize);
clock1 = unsafe { (*cea_mode).clock };
clock2 = hdmi_mode_alternate_clock(cea_mode) as i32;
} else {
return;
}
}
// pick whichever is closest
if abs__2!(unsafe { (*mode).clock }.wrapping_sub(clock1)) < abs__2!(unsafe { (*mode).clock }.wrapping_sub(clock2)) {
clock = clock1;
} else {
clock = clock2;
}
if (unsafe { (*mode).clock }) == clock {
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] detailed mode matches %s VIC %d, adjusting clock %d -> %d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, r#type, vic as i32, (*mode).clock, clock) };
unsafe {
(*mode).clock = clock;
}
}

fn drm_calculate_luminance_range(connector: *mut drm_connector) {
let hdr_metadata: *const hdr_static_metadata = (unsafe { core::ptr::addr_of_mut!((*connector).display_info.hdr_sink_metadata.__slate_anon_1.hdmi_type1) }) as *const hdr_static_metadata;
let luminance_range: *mut drm_luminance_range_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info.luminance_range) };
let mut max_avg: u32 = 0;
let mut min_cll: u32 = 0;
let mut max: u32 = 0;
let mut min: u32 = 0;
let mut q: u32 = 0;
let mut r: u32 = 0;
if (((unsafe { (*hdr_metadata).metadata_type }) as i32) as u64) & BIT!(0i32) == 0 {
return;
}
max_avg = (unsafe { (*hdr_metadata).max_fall }) as u32;
min_cll = (unsafe { (*hdr_metadata).min_cll }) as u32;
// From the specification (CTA-861-G), for calculating the maximum
// luminance we need to use:
// 	Luminance = 50*2**(CV/32)
// Where CV is a one-byte value.
// For calculating this expression we may need float point precision;
// to avoid this complexity level, we take advantage that CV is divided
// by a constant. From the Euclids division algorithm, we know that CV
// can be written as: CV = 32*q + r. Next, we replace CV in the
// Luminance expression and get 50*(2**q)*(2**(r/32)), hence we just
// need to pre-compute the value of r/32. For pre-computing the values
// We just used the following Ruby line:
// 	(0...32).each {|cv| puts (50*2**(cv/32.0)).round}
// The results of the above expressions can be verified at
// pre_computed_values.
q = max_avg >> 5i32;
r = max_avg % 32;
max = (1i32 << q).wrapping_mul((unsafe { pre_computed_values.0[r as usize] }) as i32) as u32;
// min luminance: maxLum * (CV/255)^2 / 100
let mut __v0: u32;
{
let __x: u32 = min_cll;
let __d: i32 = 255;
__v0 = if (1i32.wrapping_neg() as u32) > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add((__d / 2) as u32) / (__d as u32) } else { __x.wrapping_sub((__d / 2) as u32) / (__d as u32) };
}
q = __v0;
let mut __v1: u32;
let __x: u32 = q.wrapping_mul(q);
let __d: i32 = 100;
__v1 = if (1i32.wrapping_neg() as u32) > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add((__d / 2) as u32) / (__d as u32) } else { __x.wrapping_sub((__d / 2) as u32) / (__d as u32) };
min = max.wrapping_mul(__v1);
unsafe {
(*luminance_range).min_luminance = min;
}
unsafe {
(*luminance_range).max_luminance = max;
}
}

static pre_computed_values: __SlateAlign16<[u8; 32]> = __SlateAlign16([50u8, 51u8, 52u8, 53u8, 55u8, 56u8, 57u8, 58u8, 59u8, 61u8, 62u8, 63u8, 65u8, 66u8, 68u8, 69u8, 71u8, 72u8, 74u8, 75u8, 77u8, 79u8, 81u8, 82u8, 84u8, 86u8, 88u8, 90u8, 92u8, 94u8, 96u8, 98u8]);

fn eotf_supported(edid_ext: *const u8) -> u8 {
((((unsafe { *edid_ext.wrapping_offset(2 as isize) }) as i32) as u64) & (BIT!(0i32) | BIT!(1i32) | BIT!(2i32) | BIT!(3i32))) as u8
}

fn hdr_metadata_type(edid_ext: *const u8) -> u8 {
((((unsafe { *edid_ext.wrapping_offset(3 as isize) }) as i32) as u64) & BIT!(0i32)) as u8
}

fn drm_parse_hdr_metadata_block(connector: *mut drm_connector, db: *const u8) {
let hdr_metadata: *mut hdr_static_metadata = unsafe { core::ptr::addr_of_mut!((*connector).display_info.hdr_sink_metadata.__slate_anon_1.hdmi_type1) };
let len: u16 = cea_db_payload_len(db as *const ()) as u16;
unsafe {
(*hdr_metadata).eotf = eotf_supported(db);
}
unsafe {
(*hdr_metadata).metadata_type = hdr_metadata_type(db);
}
if (len as i32) >= 4 {
unsafe {
(*hdr_metadata).max_cll = *db.wrapping_offset(4 as isize) as u16;
}
}
if (len as i32) >= 5 {
unsafe {
(*hdr_metadata).max_fall = *db.wrapping_offset(5 as isize) as u16;
}
}
if (len as i32) >= 6 {
unsafe {
(*hdr_metadata).min_cll = *db.wrapping_offset(6 as isize) as u16;
}
// Calculate only when all values are available
drm_calculate_luminance_range(connector);
}
}

/// HDMI Vendor-Specific Data Block (HDMI VSDB, H14b-VSDB)
fn drm_parse_hdmi_vsdb_audio(connector: *mut drm_connector, db: *const u8) {
let len: u8 = cea_db_payload_len(db as *const ()) as u8;
if (len as i32) >= 6 && ((unsafe { *db.wrapping_offset(6 as isize) }) as i32) & 128i32 != 0 {
let __v0: *mut u8 = unsafe { (*connector).eld.as_mut_ptr() as *mut u8 }.wrapping_offset(5 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 2i32) as u8;
}
}
let __v3: bool = (len as i32) >= 10 && hdmi_vsdb_latency_present(db);
if __v3 {
unsafe {
*((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(0 as isize) = true;
}
unsafe {
*((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = *db.wrapping_offset(9 as isize) as i32;
}
unsafe {
*((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = *db.wrapping_offset(10 as isize) as i32;
}
}
let __v4: bool = (len as i32) >= 12 && hdmi_vsdb_i_latency_present(db);
if __v4 {
unsafe {
*((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(1 as isize) = true;
}
unsafe {
*((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = *db.wrapping_offset(11 as isize) as i32;
}
unsafe {
*((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = *db.wrapping_offset(12 as isize) as i32;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI: latency present %d %d, video latency %d %d, audio latency %d %d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, *((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(0 as isize) as i32, *((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(1 as isize) as i32, *((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize), *((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize), *((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize), *((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize)) };
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.match_identity")]
extern "C" fn match_identity(timing: *const detailed_timing, data: *mut ()) {
let name: *const u8 = unsafe { (*(*(data as *mut drm_edid_match_closure)).ident).name };
let name_len: u32 = (unsafe { strlen(name) }) as u32;
let desc: *const u8 = (unsafe { core::ptr::addr_of!((*timing).data.other_data.data.str.str) as *const u8 }) as *const u8;
let desc_len: u32 = ARRAY_SIZE__90!() as u32;
let __v0: bool = if name_len > desc_len {
true
} else {
let __v1: bool = is_display_descriptor(timing, 252u8) || is_display_descriptor(timing, 254u8);
!__v1
};
if __v0 {
return;
}
if (unsafe { strncmp(name, desc, name_len as u64) }) != 0 {
return;
}
for i in name_len..desc_len {
if ((unsafe { *desc.wrapping_offset(i as isize) }) as i32) == 10 {
break;
}
// Allow white space before EDID string terminator.
if !isspace!(unsafe { *desc.wrapping_offset(i as isize) }) {
return;
}
}
unsafe {
(*(data as *mut drm_edid_match_closure)).matched = true;
}
}

/// drm_edid_match - match drm_edid with given identity
/// @drm_edid: EDID
/// @ident: the EDID identity to match with
///
/// Check if the EDID matches with the given identity.
///
/// Return: True if the given identity matched with EDID, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_match")]
extern "C" fn drm_edid_match(drm_edid: *const drm_edid, ident: *const drm_edid_ident) -> bool {
let __v0: bool = drm_edid.is_null() || drm_edid_get_panel_id(drm_edid) != unsafe { (*ident).panel_id };
if __v0 {
return false;
}
// Match with name only if it's not NULL.
if !unsafe { (*ident).name }.is_null() {
let mut closure: drm_edid_match_closure = {
let mut __t0: drm_edid_match_closure = unsafe { core::mem::zeroed() };
__t0.ident = ident;
__t0.matched = false;
__t0
};
drm_for_each_detailed_block(drm_edid, Some(match_identity), core::ptr::addr_of_mut!(closure) as *mut ());
return closure.matched;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_match_748: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid, *const drm_edid_ident) -> bool>, *mut ()>(Some(drm_edid_match)) };

#[unsafe(link_section = ".text.slate_distinct.drm_edid.monitor_name")]
extern "C" fn monitor_name(timing: *const detailed_timing, data: *mut ()) {
if !is_display_descriptor(timing, 252u8) {
return;
}
unsafe {
*(data as *mut *const u8) = (core::ptr::addr_of!((*timing).data.other_data.data.str.str) as *const u8) as *const u8;
}
}

fn get_monitor_name(drm_edid: *const drm_edid, name: *mut u8) -> i32 {
let mut edid_name: *const u8 = core::ptr::null::<u8>();
let mut mnl: i32 = 0;
if drm_edid.is_null() || name.is_null() {
return 0;
}
drm_for_each_detailed_block(drm_edid, Some(monitor_name), core::ptr::addr_of_mut!(edid_name) as *mut ());
while !edid_name.is_null() && mnl < 13 {
if ((unsafe { *edid_name.wrapping_offset(mnl as isize) }) as i32) == 10 {
break;
}
unsafe {
*name.wrapping_offset(mnl as isize) = *edid_name.wrapping_offset(mnl as isize);
}
mnl = mnl.wrapping_add(1);
}
mnl
}

// drm_edid_get_monitor_name - fetch the monitor name from the edid
// @edid: monitor EDID information
// @name: pointer to a character array to hold the name of the monitor
// @bufsize: The size of the name buffer (should be at least 14 chars.)
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_get_monitor_name_752: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid, *mut u8, i32)>, *mut ()>(Some(drm_edid_get_monitor_name)) };

fn clear_eld(connector: *mut drm_connector) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).eld_mutex)) };
unsafe { memset(((*connector).eld.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 128]>() as u64) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).eld_mutex)) };
unsafe {
*((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(0 as isize) = false;
}
unsafe {
*((*connector).latency_present.as_mut_ptr() as *mut bool).wrapping_offset(1 as isize) = false;
}
unsafe {
*((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = 0;
}
unsafe {
*((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = 0;
}
unsafe {
*((*connector).video_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = 0;
}
unsafe {
*((*connector).audio_latency.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = 0;
}
}

/// Get 3-byte SAD buffer from struct cea_sad.
#[unsafe(no_mangle)]
extern "C" fn drm_edid_cta_sad_get(cta_sad: *const cea_sad, sad: *mut u8) {
unsafe {
*sad.wrapping_offset(0 as isize) = (((*cta_sad).format as i32) << 3i32 | ((*cta_sad).channels as i32)) as u8;
}
unsafe {
*sad.wrapping_offset(1 as isize) = (*cta_sad).freq;
}
unsafe {
*sad.wrapping_offset(2 as isize) = (*cta_sad).byte2;
}
}

/// Set struct cea_sad from 3-byte SAD buffer.
#[unsafe(no_mangle)]
extern "C" fn drm_edid_cta_sad_set(cta_sad: *mut cea_sad, sad: *const u8) {
unsafe {
(*cta_sad).format = (((*sad.wrapping_offset(0 as isize) as i32) & 120) >> 3i32) as u8;
}
unsafe {
(*cta_sad).channels = ((*sad.wrapping_offset(0 as isize) as i32) & 7) as u8;
}
unsafe {
(*cta_sad).freq = ((*sad.wrapping_offset(1 as isize) as i32) & 127) as u8;
}
unsafe {
(*cta_sad).byte2 = *sad.wrapping_offset(2 as isize);
}
}

// drm_edid_to_eld - build ELD from EDID
// @connector: connector corresponding to the HDMI/DP sink
// @drm_edid: EDID to parse
//
// Fill the ELD (EDID-Like Data) buffer for passing to the audio driver. The
// HDCP and Port_ID ELD fields are left for the graphics driver to fill in.
fn drm_edid_to_eld(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).display_info) }) as *const drm_display_info;
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let eld: *mut u8 = unsafe { (*connector).eld.as_mut_ptr() as *mut u8 };
let mut total_sad_count: i32 = 0;
let mut mnl: i32 = 0;
if drm_edid.is_null() {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*connector).eld_mutex)) };
mnl = get_monitor_name(drm_edid, eld.wrapping_offset(20 as isize) as *mut u8);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] ELD monitor %s\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, eld.wrapping_offset(20 as isize)) };
unsafe {
*eld.wrapping_offset(4 as isize) = (((*info).cea_rev as i32) << 5i32) as u8;
}
let __v0: *mut u8 = eld.wrapping_offset(4 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | mnl) as u8;
}
unsafe {
*eld.wrapping_offset(0 as isize) = 16u8;
}
unsafe {
*eld.wrapping_offset(16 as isize) = (core::ptr::addr_of!((*(*drm_edid).edid).__slate_anon_1.__slate_anon_1.mfg_id) as *const u8).wrapping_offset(0 as isize).read_unaligned();
}
unsafe {
*eld.wrapping_offset(17 as isize) = (core::ptr::addr_of!((*(*drm_edid).edid).__slate_anon_1.__slate_anon_1.mfg_id) as *const u8).wrapping_offset(1 as isize).read_unaligned();
}
unsafe {
*eld.wrapping_offset(18 as isize) = (core::ptr::addr_of!((*(*drm_edid).edid).__slate_anon_1.__slate_anon_1.prod_code) as *const u8).wrapping_offset(0 as isize).read_unaligned();
}
unsafe {
*eld.wrapping_offset(19 as isize) = (core::ptr::addr_of!((*(*drm_edid).edid).__slate_anon_1.__slate_anon_1.prod_code) as *const u8).wrapping_offset(1 as isize).read_unaligned();
}
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
loop {
let __v8: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
let db: *const cea_db = __v8;
if __v8.is_null() {
break;
}
let data: *const u8 = cea_db_data(db) as *const u8;
let len: i32 = cea_db_payload_len(db as *const ());
let mut sad_count: i32 = 0;
match cea_db_tag(db) {
1 => {
let mut __v3: i32;
let __UNIQUE_ID_x__753: i32 = len / 3;
let __UNIQUE_ID_y__754: i32 = 15i32.wrapping_sub(total_sad_count);
loop {
if __types_ok!(__UNIQUE_ID_x__753, __UNIQUE_ID_y__754) == 0 {
unsafe { __compiletime_assert_755() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__753 < __UNIQUE_ID_y__754 { __UNIQUE_ID_x__753 } else { __UNIQUE_ID_y__754 };
sad_count = __v3; // Audio Data Block, contains SADs
if sad_count >= 1 {
unsafe { memcpy(eld.wrapping_offset(DRM_ELD_CEA_SAD!(mnl, total_sad_count) as isize) as *mut (), data as *const (), sad_count.wrapping_mul(3) as u64) };
}
total_sad_count = total_sad_count.wrapping_add(sad_count);
}
4 => {
if len >= 1 {
unsafe {
*eld.wrapping_offset(7 as isize) = *data.wrapping_offset(0 as isize);
}
}
// Speaker Allocation Data Block
}
3 => {
if cea_db_is_hdmi_vsdb(db) {
drm_parse_hdmi_vsdb_audio(connector, db as *const u8);
}
// HDMI Vendor-Specific Data Block
}
_ => {
}
}
}
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
let __v10: *mut u8 = eld.wrapping_offset(5 as isize);
unsafe {
*__v10 = ((*__v10 as i32) | total_sad_count << 4i32) as u8;
}
if (unsafe { (*connector).connector_type }) == 10 || (unsafe { (*connector).connector_type }) == 14 {
let __v13: *mut u8 = eld.wrapping_offset(5 as isize);
unsafe {
*__v13 = ((*__v13 as i32) | 4i32) as u8;
}
} else {
let __v16: *mut u8 = eld.wrapping_offset(5 as isize);
unsafe {
*__v16 = ((*__v16 as i32) | 0i32) as u8;
}
}
unsafe {
*eld.wrapping_offset(2 as isize) = (drm_eld_calc_baseline_block_size(eld as *const u8).wrapping_add(4).wrapping_sub(1) / 4) as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] ELD size %d, SAD count %d\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, drm_eld_size(eld as *const u8), total_sad_count) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*connector).eld_mutex)) };
}

fn _drm_edid_to_sad(drm_edid: *const drm_edid, psads: *mut *mut cea_sad) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut db: *const cea_db = unsafe { core::mem::zeroed() };
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut count: i32 = 0;
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
'__slate_break_9: loop {
let __v10: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
db = __v10;
if __v10.is_null() {
break;
}
if cea_db_tag(db) == 1 {
count = cea_db_payload_len(db as *const ()) / 3; // SAD is 3B
let mut __v0: *mut cea_sad;
let __obj_size: u64 = size_mul(core::mem::size_of::<cea_sad>() as u64, count as u64);
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
__v0 = __v1 as *mut cea_sad;
let sads: *mut cea_sad = __v0;
unsafe {
*psads = sads;
}
if sads.is_null() {
return 12i32.wrapping_neg();
}
for i in 0..count {
drm_edid_cta_sad_set(sads.wrapping_offset(i as isize), unsafe { core::ptr::addr_of!((*db).data) as *const u8 }.wrapping_offset(i.wrapping_mul(3) as isize));
}
break '__slate_break_9;
}
}
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Found %d Short Audio Descriptors\n".as_ptr() as *const u8, count) };
count
}

/// drm_edid_to_sad - extracts SADs from EDID
/// @edid: EDID to parse
/// @sads: pointer that will be set to the extracted SADs
///
/// Looks for CEA EDID block and extracts SADs (Short Audio Descriptors) from it.
///
/// Note: The returned pointer needs to be freed using kfree().
///
/// Return: The number of found SADs or negative number on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_to_sad")]
extern "C" fn drm_edid_to_sad(edid: *const edid, sads: *mut *mut cea_sad) -> i32 {
let mut drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
_drm_edid_to_sad(drm_edid_legacy_init(core::ptr::addr_of_mut!(drm_edid), edid), sads)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_to_sad_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid, *mut *mut cea_sad) -> i32>, *mut ()>(Some(drm_edid_to_sad)) };

fn _drm_edid_to_speaker_allocation(drm_edid: *const drm_edid, sadb: *mut *mut u8) -> i32 {
let mut db: *const cea_db = unsafe { core::mem::zeroed() };
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut count: i32 = 0;
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
loop {
let __v7: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
db = __v7;
if __v7.is_null() {
break;
}
let __v0: bool = cea_db_tag(db) == 4 && cea_db_payload_len(db as *const ()) == 3;
if __v0 {
unsafe {
*sadb = alloc_hooks!(kmemdup_noprof((core::ptr::addr_of!((*db).data) as *const u8) as *const (), cea_db_payload_len(db as *const ()) as u64, 3264u32)) as *mut u8;
}
if unsafe { *sadb }.is_null() {
return 12i32.wrapping_neg();
}
count = cea_db_payload_len(db as *const ());
break;
}
}
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Found %d Speaker Allocation Data Blocks\n".as_ptr() as *const u8, count) };
count
}

/// drm_edid_to_speaker_allocation - extracts Speaker Allocation Data Blocks from EDID
/// @edid: EDID to parse
/// @sadb: pointer to the speaker block
///
/// Looks for CEA EDID block and extracts the Speaker Allocation Data Block from it.
///
/// Note: The returned pointer needs to be freed using kfree().
///
/// Return: The number of found Speaker Allocation Blocks or negative number on
/// error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_to_speaker_allocation")]
extern "C" fn drm_edid_to_speaker_allocation(edid: *const edid, sadb: *mut *mut u8) -> i32 {
let mut drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
_drm_edid_to_speaker_allocation(drm_edid_legacy_init(core::ptr::addr_of_mut!(drm_edid), edid), sadb)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_to_speaker_allocation_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid, *mut *mut u8) -> i32>, *mut ()>(Some(drm_edid_to_speaker_allocation)) };

// drm_av_sync_delay - compute the HDMI/DP sink audio-video sync delay
// @connector: connector associated with the HDMI/DP sink
// @mode: the display mode
//
// Return: The HDMI/DP sink's audio-video sync delay in milliseconds or 0 if
// the sink doesn't support audio or video.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_av_sync_delay_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *const drm_display_mode) -> i32>, *mut ()>(Some(drm_av_sync_delay)) };

fn _drm_detect_hdmi_monitor(drm_edid: *const drm_edid) -> bool {
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut hdmi: bool = false;
// Because HDMI identifier is in Vendor Specific Block,
// search it from all data blocks of CEA extension.
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
loop {
let __v1: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
let db: *const cea_db = __v1;
if __v1.is_null() {
break;
}
if cea_db_is_hdmi_vsdb(db) {
hdmi = true;
break;
}
}
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
hdmi
}

/// drm_detect_hdmi_monitor - detect whether monitor is HDMI
/// @edid: monitor EDID information
///
/// Parse the CEA extension according to CEA-861-B.
///
/// Drivers that have added the modes parsed from EDID to drm_display_info
/// should use &drm_display_info.is_hdmi instead of calling this function.
///
/// Return: True if the monitor is HDMI, false if not or unknown.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_detect_hdmi_monitor")]
extern "C" fn drm_detect_hdmi_monitor(edid: *const edid) -> bool {
let mut drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
_drm_detect_hdmi_monitor(drm_edid_legacy_init(core::ptr::addr_of_mut!(drm_edid), edid))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_detect_hdmi_monitor_768: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid) -> bool>, *mut ()>(Some(drm_detect_hdmi_monitor)) };

fn _drm_detect_monitor_audio(drm_edid: *const drm_edid) -> bool {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut data: *const u8 = unsafe { core::mem::zeroed() };
let mut has_audio: bool = unsafe { core::mem::zeroed() };
let mut edid_ext: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *const u8 = core::ptr::addr_of_mut!(edid_ext) as *mut *const u8;
let mut iter: core::mem::MaybeUninit<cea_db_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut cea_db_iter = core::ptr::addr_of_mut!(iter) as *mut cea_db_iter;
let mut db: core::mem::MaybeUninit<*const cea_db> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *const cea_db = core::ptr::addr_of_mut!(db) as *mut *const cea_db;
let mut edid_iter: core::mem::MaybeUninit<drm_edid_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut drm_edid_iter = core::ptr::addr_of_mut!(edid_iter) as *mut drm_edid_iter;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 64usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
has_audio = false;
drm_edid_iter_begin(drm_edid, core::ptr::addr_of_mut!(*__slate_slot_10));
loop {
let __v5: *const u8 = __drm_edid_iter_next(core::ptr::addr_of_mut!(*__slate_slot_10)) as *const u8;
*__slate_slot_7 = __v5;
if !__v5.is_null() {
if (*(*__slate_slot_7).wrapping_offset(0 as isize) as i32) == 2 {
has_audio = (*(*__slate_slot_7).wrapping_offset(3 as isize) as i32) & 64i32 != 0;
if has_audio {
break;
}
}
} else {
break;
}
}
drm_edid_iter_end(core::ptr::addr_of_mut!(*__slate_slot_10));
if has_audio {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Monitor has basic audio support\n".as_ptr() as *const u8);
} else {
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(*__slate_slot_8));
'__join_1: {
loop {
let __v0: *const cea_db = __cea_db_iter_next(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8)));
*__slate_slot_9 = __v0;
if !__v0.is_null() {
if cea_db_tag(*__slate_slot_9) == 1 {
break;
}
} else {
break '__join_1;
}
}
data = cea_db_data(*__slate_slot_9) as *const u8;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_3 = 0;
while *__slate_slot_3 < cea_db_payload_len(*__slate_slot_9 as *const ()) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"CEA audio format %d\n".as_ptr() as *const u8, (*data.wrapping_offset(*__slate_slot_3 as isize) as i32) >> 3i32 & 15);
let __v2: i32 = *__slate_slot_3;
*__slate_slot_3 = __v2.wrapping_add(3);
}
has_audio = true;
}
cea_db_iter_end(core::ptr::addr_of_mut!(*__slate_slot_8));
}
return has_audio;
}
}

/// drm_detect_monitor_audio - check monitor audio capability
/// @edid: EDID block to scan
///
/// Monitor should have CEA extension block.
/// If monitor has 'basic audio', but no CEA audio blocks, it's 'basic
/// audio' only. If there is any audio extension block and supported
/// audio format, assume at least 'basic audio' support, even if 'basic
/// audio' is not defined in EDID.
///
/// Return: True if the monitor supports audio, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_detect_monitor_audio")]
extern "C" fn drm_detect_monitor_audio(edid: *const edid) -> bool {
let mut drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
_drm_detect_monitor_audio(drm_edid_legacy_init(core::ptr::addr_of_mut!(drm_edid), edid))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_detect_monitor_audio_769: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const edid) -> bool>, *mut ()>(Some(drm_detect_monitor_audio)) };

/// drm_default_rgb_quant_range - default RGB quantization range
/// @mode: display mode
///
/// Determine the default RGB quantization range for the mode,
/// as specified in CEA-861.
///
/// Return: The default RGB quantization range for the mode
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_default_rgb_quant_range")]
extern "C" fn drm_default_rgb_quant_range(mode: *const drm_display_mode) -> u32 {
// All CEA modes other than VIC 1 use limited quantization range.
(if (drm_match_cea_mode(mode) as i32) > 1 { 1i32 } else { 2i32 }) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_default_rgb_quant_range_770: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode) -> u32>, *mut ()>(Some(drm_default_rgb_quant_range)) };

/// CTA-861 Video Data Block (CTA VDB)
fn parse_cta_vdb(connector: *mut drm_connector, db: *const cea_db) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let mut vic_index: i32 = 0;
let len: i32 = cea_db_payload_len(db as *const ());
let svds: *const u8 = cea_db_data(db) as *const u8;
if len == 0 {
return;
}
// Gracefully handle multiple VDBs, however unlikely that is
let vics: *mut u8 = krealloc_node__2!(unsafe { (*info).vics }, unsafe { (*info).vics_len }.wrapping_add(len), 3264u32, 1i32.wrapping_neg()) as *mut u8;
if vics.is_null() {
return;
}
vic_index = unsafe { (*info).vics_len };
let __v6: i32 = unsafe { (*info).vics_len };
unsafe {
(*info).vics_len = __v6.wrapping_add(len);
}
unsafe {
(*info).vics = vics;
}
for i in 0..len {
let mut vic: u8 = svd_to_vic(unsafe { *svds.wrapping_offset(i as isize) });
if !drm_valid_cea_vic(vic) {
vic = 0;
}
let __v8: i32 = vic_index;
vic_index = __v8.wrapping_add(1);
unsafe {
*unsafe { (*info).vics }.wrapping_offset(__v8 as isize) = vic;
}
}
}

/// Update y420_cmdb_modes based on previously parsed CTA VDB and Y420CMDB.
///
/// Translate the y420cmdb_map based on VIC indexes to y420_cmdb_modes indexed
/// using the VICs themselves.
fn update_cta_y420cmdb(connector: *mut drm_connector, y420cmdb_map: u64) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let hdmi: *mut drm_hdmi_info = unsafe { core::ptr::addr_of_mut!((*info).hdmi) };
let len: i32 = min_t__2!(i32, unsafe { (*info).vics_len }, BITS_PER_TYPE__2!());
for i in 0..len {
let vic: u8 = unsafe { *unsafe { (*info).vics }.wrapping_offset(i as isize) };
if vic != 0 && y420cmdb_map & BIT_ULL!(i) != 0 {
bitmap_set(unsafe { (*hdmi).y420_cmdb_modes.as_mut_ptr() as *mut u64 }, vic as u32, 1u32);
}
}
}

fn cta_vdb_has_vic(connector: *const drm_connector, vic: u8) -> bool {
let info: *const drm_display_info = unsafe { core::ptr::addr_of!((*connector).display_info) };
let mut i: i32 = 0;
if vic == 0 || unsafe { (*info).vics }.is_null() {
return false;
}
while i < unsafe { (*info).vics_len } {
if ((unsafe { *unsafe { (*info).vics }.wrapping_offset(i as isize) }) as i32) == (vic as i32) {
return true;
}
i = i.wrapping_add(1);
}
false
}

/// CTA-861-H YCbCr 4:2:0 Video Data Block (CTA Y420VDB)
fn parse_cta_y420vdb(connector: *mut drm_connector, db: *const cea_db) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let hdmi: *mut drm_hdmi_info = unsafe { core::ptr::addr_of_mut!((*info).hdmi) };
let svds: *const u8 = ((unsafe { (cea_db_data(db) as *const u8).wrapping_offset(1 as isize) }) as *const ()) as *const u8;
let mut i: i32 = 0;
while i < cea_db_payload_len(db as *const ()).wrapping_sub(1) {
let vic: u8 = svd_to_vic(unsafe { *svds.wrapping_offset(i as isize) });
if drm_valid_cea_vic(vic) {
bitmap_set(unsafe { (*hdmi).y420_vdb_modes.as_mut_ptr() as *mut u64 }, vic as u32, 1u32);
let __v2: u32 = unsafe { (*info).color_formats };
let __v3: u32 = ((__v2 as u64) | BIT!(3i32)) as u32;
unsafe {
(*info).color_formats = __v3;
}
}
i = i.wrapping_add(1);
}
}

fn drm_parse_vcdb(connector: *mut drm_connector, db: *const u8) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] CEA VCDB 0x%02x\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, *db.wrapping_offset(2 as isize) as i32) };
if ((unsafe { *db.wrapping_offset(2 as isize) }) as i32) & 64i32 != 0 {
unsafe {
(*info).rgb_quant_range_selectable = true;
}
}
}

fn drm_get_max_frl_rate(max_frl_rate: i32, max_lanes: *mut u8, max_rate_per_lane: *mut u8) {
match max_frl_rate {
1 => {
unsafe {
*max_lanes = 3;
}
unsafe {
*max_rate_per_lane = 3;
}
}
2 => {
unsafe {
*max_lanes = 3;
}
unsafe {
*max_rate_per_lane = 6;
}
}
3 => {
unsafe {
*max_lanes = 4;
}
unsafe {
*max_rate_per_lane = 6;
}
}
4 => {
unsafe {
*max_lanes = 4;
}
unsafe {
*max_rate_per_lane = 8;
}
}
5 => {
unsafe {
*max_lanes = 4;
}
unsafe {
*max_rate_per_lane = 10;
}
}
6 => {
unsafe {
*max_lanes = 4;
}
unsafe {
*max_rate_per_lane = 12;
}
}
_ => {
unsafe {
*max_lanes = 0;
}
unsafe {
*max_rate_per_lane = 0;
}
}
}
}

fn drm_parse_ycbcr420_deep_color_info(connector: *mut drm_connector, db: *const u8) {
let mut dc_mask: u8 = 0;
let hdmi: *mut drm_hdmi_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info.hdmi) };
dc_mask = (((unsafe { *db.wrapping_offset(7 as isize) }) as i32) & 7i32) as u8;
unsafe {
(*hdmi).y420_dc_modes = dc_mask;
}
}

fn drm_parse_dsc_info(hdmi_dsc: *mut drm_hdmi_dsc_cap, hf_scds: *const u8) {
unsafe {
(*hdmi_dsc).v_1p2 = (*hf_scds.wrapping_offset(11 as isize) as i32) & 128i32 != 0;
}
if !unsafe { (*hdmi_dsc).v_1p2 } {
return;
}
unsafe {
(*hdmi_dsc).native_420 = (*hf_scds.wrapping_offset(11 as isize) as i32) & 64i32 != 0;
}
unsafe {
(*hdmi_dsc).all_bpp = (*hf_scds.wrapping_offset(11 as isize) as i32) & 8i32 != 0;
}
if ((unsafe { *hf_scds.wrapping_offset(11 as isize) }) as i32) & 4i32 != 0 {
unsafe {
(*hdmi_dsc).bpc_supported = 16;
}
} else {
if ((unsafe { *hf_scds.wrapping_offset(11 as isize) }) as i32) & 2i32 != 0 {
unsafe {
(*hdmi_dsc).bpc_supported = 12;
}
} else {
if ((unsafe { *hf_scds.wrapping_offset(11 as isize) }) as i32) & 1i32 != 0 {
unsafe {
(*hdmi_dsc).bpc_supported = 10;
}
} else {
unsafe {
(*hdmi_dsc).bpc_supported = 8;
}
}
}
}
// Supports min 8 BPC if DSC 1.2 is supported
if cea_db_payload_len(hf_scds as *const ()) >= 12 && (unsafe { *hf_scds.wrapping_offset(12 as isize) }) != 0 {
let mut dsc_max_slices: u8 = 0;
let dsc_max_frl_rate: u8 = ((((unsafe { *hf_scds.wrapping_offset(12 as isize) }) as i32) & 240) >> 4i32) as u8;
drm_get_max_frl_rate(dsc_max_frl_rate as i32, unsafe { core::ptr::addr_of_mut!((*hdmi_dsc).max_lanes) }, unsafe { core::ptr::addr_of_mut!((*hdmi_dsc).max_frl_rate_per_lane) });
dsc_max_slices = (((unsafe { *hf_scds.wrapping_offset(12 as isize) }) as i32) & 15) as u8;
match dsc_max_slices as i32 {
1 => {
unsafe {
(*hdmi_dsc).max_slices = 1;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 340;
}
}
2 => {
unsafe {
(*hdmi_dsc).max_slices = 2;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 340;
}
}
3 => {
unsafe {
(*hdmi_dsc).max_slices = 4;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 340;
}
}
4 => {
unsafe {
(*hdmi_dsc).max_slices = 8;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 340;
}
}
5 => {
unsafe {
(*hdmi_dsc).max_slices = 8;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 400;
}
}
6 => {
unsafe {
(*hdmi_dsc).max_slices = 12;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 400;
}
}
7 => {
unsafe {
(*hdmi_dsc).max_slices = 16;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 400;
}
}
_ => {
unsafe {
(*hdmi_dsc).max_slices = 0;
}
unsafe {
(*hdmi_dsc).clk_per_slice = 0;
}
}
}
}
if cea_db_payload_len(hf_scds as *const ()) >= 13 && (unsafe { *hf_scds.wrapping_offset(13 as isize) }) != 0 {
unsafe {
(*hdmi_dsc).total_chunk_kbytes = ((*hf_scds.wrapping_offset(13 as isize) as i32) & 63) as u8;
}
}
}

/// Sink Capability Data Structure
fn drm_parse_hdmi_forum_scds(connector: *mut drm_connector, hf_scds: *const u8) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let hdmi: *mut drm_hdmi_info = unsafe { core::ptr::addr_of_mut!((*info).hdmi) };
let hdmi_dsc: *mut drm_hdmi_dsc_cap = unsafe { core::ptr::addr_of_mut!((*hdmi).dsc_cap) };
let mut max_tmds_clock: i32 = 0;
let mut max_frl_rate: u8 = 0;
let mut dsc_support: bool = false;
unsafe {
(*info).has_hdmi_infoframe = true;
}
if ((unsafe { *hf_scds.wrapping_offset(6 as isize) }) as i32) & 128 != 0 {
unsafe {
(*hdmi).scdc.supported = true;
}
if ((unsafe { *hf_scds.wrapping_offset(6 as isize) }) as i32) & 64 != 0 {
unsafe {
(*hdmi).scdc.read_request = true;
}
}
}
// All HDMI 2.0 monitors must support scrambling at rates > 340 MHz.
// And as per the spec, three factors confirm this:
// * Availability of a HF-VSDB block in EDID (check)
// * Non zero Max_TMDS_Char_Rate filed in HF-VSDB (let's check)
// * SCDC support available (let's check)
// Lets check it out.
if (unsafe { *hf_scds.wrapping_offset(5 as isize) }) != 0 {
let scdc: *mut drm_scdc = unsafe { core::ptr::addr_of_mut!((*hdmi).scdc) };
// max clock is 5000 KHz times block value
max_tmds_clock = ((unsafe { *hf_scds.wrapping_offset(5 as isize) }) as i32).wrapping_mul(5000);
if max_tmds_clock > 340000 {
unsafe {
(*info).max_tmds_clock = max_tmds_clock;
}
}
if unsafe { (*scdc).supported } {
unsafe {
(*scdc).scrambling.supported = true;
}
// Few sinks support scrambling for clocks < 340M
if ((unsafe { *hf_scds.wrapping_offset(6 as isize) }) as i32) & 8 != 0 {
unsafe {
(*scdc).scrambling.low_rates = true;
}
}
}
}
if (unsafe { *hf_scds.wrapping_offset(7 as isize) }) != 0 {
max_frl_rate = ((((unsafe { *hf_scds.wrapping_offset(7 as isize) }) as i32) & 240) >> 4i32) as u8;
drm_get_max_frl_rate(max_frl_rate as i32, unsafe { core::ptr::addr_of_mut!((*hdmi).max_lanes) }, unsafe { core::ptr::addr_of_mut!((*hdmi).max_frl_rate_per_lane) });
}
drm_parse_ycbcr420_deep_color_info(connector, hf_scds);
if cea_db_payload_len(hf_scds as *const ()) >= 11 && (unsafe { *hf_scds.wrapping_offset(11 as isize) }) != 0 {
drm_parse_dsc_info(hdmi_dsc, hf_scds);
dsc_support = true;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HF-VSDB: max TMDS clock: %d KHz, HDMI 2.1 support: %s, DSC 1.2 support: %s\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, max_tmds_clock, str_yes_no(max_frl_rate != 0), str_yes_no(dsc_support)) };
}

fn drm_parse_hdmi_deep_color_info(connector: *mut drm_connector, hdmi: *const u8) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let mut dc_bpc: u32 = 0;
// HDMI supports at least 8 bpc
unsafe {
(*info).bpc = 8;
}
if cea_db_payload_len(hdmi as *const ()) < 6 {
return;
}
if ((unsafe { *hdmi.wrapping_offset(6 as isize) }) as i32) & 16i32 != 0 {
dc_bpc = 10;
unsafe {
(*info).edid_hdmi_rgb444_dc_modes = (((*info).edid_hdmi_rgb444_dc_modes as i32) | 16i32) as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI sink does deep color 30.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
}
if ((unsafe { *hdmi.wrapping_offset(6 as isize) }) as i32) & 32i32 != 0 {
dc_bpc = 12;
unsafe {
(*info).edid_hdmi_rgb444_dc_modes = (((*info).edid_hdmi_rgb444_dc_modes as i32) | 32i32) as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI sink does deep color 36.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
}
if ((unsafe { *hdmi.wrapping_offset(6 as isize) }) as i32) & 64i32 != 0 {
dc_bpc = 16;
unsafe {
(*info).edid_hdmi_rgb444_dc_modes = (((*info).edid_hdmi_rgb444_dc_modes as i32) | 64i32) as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI sink does deep color 48.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
}
if dc_bpc == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] No deep color support on this HDMI sink.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Assigning HDMI sink color depth as %d bpc.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, dc_bpc) };
unsafe {
(*info).bpc = dc_bpc;
}
// YCRCB444 is optional according to spec.
if ((unsafe { *hdmi.wrapping_offset(6 as isize) }) as i32) & 8i32 != 0 {
unsafe {
(*info).edid_hdmi_ycbcr444_dc_modes = (*info).edid_hdmi_rgb444_dc_modes;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI sink does YCRCB444 in deep color.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
}
// Spec says that if any deep color mode is supported at all,
// then deep color 36 bit must be supported.
if ((unsafe { *hdmi.wrapping_offset(6 as isize) }) as i32) & 32i32 == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI sink should do DC_36, but does not!\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
}
}

/// HDMI Vendor-Specific Data Block (HDMI VSDB, H14b-VSDB)
fn drm_parse_hdmi_vsdb_video(connector: *mut drm_connector, db: *const u8) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let len: u8 = cea_db_payload_len(db as *const ()) as u8;
unsafe {
(*info).is_hdmi = true;
}
unsafe {
(*info).source_physical_address = ((*db.wrapping_offset(4 as isize) as i32) << 8i32 | (*db.wrapping_offset(5 as isize) as i32)) as u16;
}
if (len as i32) >= 6 {
unsafe {
(*info).dvi_dual = (*db.wrapping_offset(6 as isize) as i32) & 1 != 0;
}
}
if (len as i32) >= 7 {
unsafe {
(*info).max_tmds_clock = (*db.wrapping_offset(7 as isize) as i32).wrapping_mul(5000);
}
}
// Try to infer whether the sink supports HDMI infoframes.
//
// HDMI infoframe support was first added in HDMI 1.4. Assume the sink
// supports infoframes if HDMI_Video_present is set.
if (len as i32) >= 8 && (((unsafe { *db.wrapping_offset(8 as isize) }) as i32) as u64) & BIT!(5i32) != 0 {
unsafe {
(*info).has_hdmi_infoframe = true;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HDMI: DVI dual %d, max TMDS clock %d kHz\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).dvi_dual as i32, (*info).max_tmds_clock) };
drm_parse_hdmi_deep_color_info(connector, db);
}

/// See EDID extension for head-mounted and specialized monitors, specified at:
/// https://docs.microsoft.com/en-us/windows-hardware/drivers/display/specialized-monitors-edid-extension
fn drm_parse_microsoft_vsdb(connector: *mut drm_connector, db: *const u8) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let version: u8 = unsafe { *db.wrapping_offset(4 as isize) };
let desktop_usage: bool = (((unsafe { *db.wrapping_offset(5 as isize) }) as i32) as u64) & BIT!(6i32) != 0;
// Version 1 and 2 for HMDs, version 3 flags desktop usage explicitly
if (version as i32) == 1 || (version as i32) == 2 || (version as i32) == 3 && !desktop_usage {
unsafe {
(*info).non_desktop = true;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] HMD or specialized display VSDB version %u: 0x%02x\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, version as i32, *db.wrapping_offset(5 as isize) as i32) };
}

fn drm_parse_amd_vsdb(connector: *mut drm_connector, db: *const cea_db) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let data: *const u8 = cea_db_data(db) as *const u8;
if ((unsafe { (*(data as *const amd_vsdb_v3_payload)).version }) as i32) != 3 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Unsupported AMD VSDB version %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*(data as *const amd_vsdb_v3_payload)).version as i32) };
return;
}
unsafe {
(*info).amd_vsdb.version = (*(data as *const amd_vsdb_v3_payload)).version;
}
unsafe {
(*info).amd_vsdb.replay_mode = ((*(data as *const amd_vsdb_v3_payload)).feature_caps as i32) & 64 != 0;
}
unsafe {
(*info).amd_vsdb.panel_type = ((((*(data as *const amd_vsdb_v3_payload)).cs_eotf_support as i32) & 192) >> 6i32) as u8;
}
unsafe {
(*info).amd_vsdb.luminance_range1.max_luminance = (*(data as *const amd_vsdb_v3_payload)).lum1_max as u32;
}
unsafe {
(*info).amd_vsdb.luminance_range1.min_luminance = (*(data as *const amd_vsdb_v3_payload)).lum1_min as u32;
}
unsafe {
(*info).amd_vsdb.luminance_range2.max_luminance = (*(data as *const amd_vsdb_v3_payload)).lum2_max as u32;
}
unsafe {
(*info).amd_vsdb.luminance_range2.min_luminance = (*(data as *const amd_vsdb_v3_payload)).lum2_min as u32;
}
// The AMD VSDB v3 payload length is variable (15..20 bytes).
// All fields through p->rsvd1 (byte 14) are always present,
// but p->extra[] (bytes 15+) may not be.  Any future access to
// extra[] must be guarded with a runtime length check to avoid
// out-of-bounds reads on shorter (but spec-valid) payloads.
// For example:
//
//   int len = cea_db_payload_len(db);
//
//   if (len > AMD_VSDB_V3_PAYLOAD_MIN_LEN)
//       info->amd_vsdb.foo = p->extra[0];
}

fn drm_parse_cea_ext(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let mut edid_iter: drm_edid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(edid_iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut iter: cea_db_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut y420cmdb_map: u64 = 0;
drm_edid_iter_begin(drm_edid, core::ptr::addr_of_mut!(edid_iter));
loop {
let __v7: *const u8 = __drm_edid_iter_next(core::ptr::addr_of_mut!(edid_iter)) as *const u8;
let edid_ext: *const u8 = __v7;
if __v7.is_null() {
break;
}
if ((unsafe { *edid_ext.wrapping_offset(0 as isize) }) as i32) == 2 {
if (unsafe { (*info).cea_rev }) == 0 {
unsafe {
(*info).cea_rev = *edid_ext.wrapping_offset(1 as isize);
}
}
if ((unsafe { (*info).cea_rev }) as i32) != ((unsafe { *edid_ext.wrapping_offset(1 as isize) }) as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] CEA extension version mismatch %u != %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).cea_rev as i32, *edid_ext.wrapping_offset(1 as isize) as i32) };
}
// The existence of a CTA extension should imply RGB support
unsafe {
(*info).color_formats = BIT!(0i32) as u32;
}
if ((unsafe { *edid_ext.wrapping_offset(3 as isize) }) as i32) & 32i32 != 0 {
let __v2: u32 = unsafe { (*info).color_formats };
let __v3: u32 = ((__v2 as u64) | BIT!(1i32)) as u32;
unsafe {
(*info).color_formats = __v3;
}
}
if ((unsafe { *edid_ext.wrapping_offset(3 as isize) }) as i32) & 16i32 != 0 {
let __v5: u32 = unsafe { (*info).color_formats };
let __v6: u32 = ((__v5 as u64) | BIT!(2i32)) as u32;
unsafe {
(*info).color_formats = __v6;
}
}
if ((unsafe { *edid_ext.wrapping_offset(3 as isize) }) as i32) & 64i32 != 0 {
unsafe {
(*info).has_audio = true;
}
}
}
}
drm_edid_iter_end(core::ptr::addr_of_mut!(edid_iter));
cea_db_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter));
loop {
let __v9: *const cea_db = __cea_db_iter_next(unsafe { &mut (*core::ptr::addr_of_mut!(iter)) });
let db: *const cea_db = __v9;
if __v9.is_null() {
break;
}
// FIXME: convert parsers to use struct cea_db
let data: *const u8 = db as *const u8;
if cea_db_is_hdmi_vsdb(db) {
drm_parse_hdmi_vsdb_video(connector, data);
} else {
let __v8: bool = cea_db_is_hdmi_forum_vsdb(db) || cea_db_is_hdmi_forum_scdb(db);
if __v8 {
drm_parse_hdmi_forum_scds(connector, data);
} else {
if cea_db_is_microsoft_vsdb(db) {
drm_parse_microsoft_vsdb(connector, data);
} else {
if cea_db_is_amd_vsdb(db) {
drm_parse_amd_vsdb(connector, db);
} else {
if cea_db_is_y420cmdb(db) {
parse_cta_y420cmdb(connector, db, unsafe { &mut (*core::ptr::addr_of_mut!(y420cmdb_map)) });
} else {
if cea_db_is_y420vdb(db) {
parse_cta_y420vdb(connector, db);
} else {
if cea_db_is_vcdb(db) {
drm_parse_vcdb(connector, data);
} else {
if cea_db_is_hdmi_hdr_metadata_block(db) {
drm_parse_hdr_metadata_block(connector, data);
} else {
if cea_db_tag(db) == 2 {
parse_cta_vdb(connector, db);
} else {
if cea_db_tag(db) == 1 {
unsafe {
(*info).has_audio = true;
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
cea_db_iter_end(core::ptr::addr_of_mut!(iter));
if y420cmdb_map != 0 {
update_cta_y420cmdb(connector, y420cmdb_map);
}
}

#[unsafe(link_section = ".text.slate_distinct.drm_edid.get_monitor_range")]
extern "C" fn get_monitor_range(timing: *const detailed_timing, c: *mut ()) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*(*(c as *mut detailed_mode_closure)).connector).display_info) };
let monitor_range: *mut drm_monitor_range_info = unsafe { core::ptr::addr_of_mut!((*info).monitor_range) };
let data: *const detailed_non_pixel = unsafe { core::ptr::addr_of!((*timing).data.other_data) };
let range: *const detailed_data_monitor_range = unsafe { core::ptr::addr_of!((*data).data.range) };
let edid: *const edid = unsafe { (*(*(c as *mut detailed_mode_closure)).drm_edid).edid };
if !is_display_descriptor(timing, 253u8) {
return;
}
// These limits are used to determine the VRR refresh
// rate range. Only the "range limits only" variant
// of the range descriptor seems to guarantee that
// any and all timings are accepted by the sink, as
// opposed to just timings conforming to the indicated
// formula (GTF/GTF2/CVT). Thus other variants of the
// range descriptor are not accepted here.
if ((unsafe { (*range).flags }) as i32) != 1 {
return;
}
unsafe {
(*monitor_range).min_vfreq = (*range).min_vfreq as u16;
}
unsafe {
(*monitor_range).max_vfreq = (*range).max_vfreq as u16;
}
if ((unsafe { (*edid).revision }) as i32) >= 4 {
if ((unsafe { (*data).pad2 }) as i32) & 1i32 != 0 {
unsafe {
(*monitor_range).min_vfreq = ((*monitor_range).min_vfreq as i32).wrapping_add(255) as u16;
}
}
if ((unsafe { (*data).pad2 }) as i32) & 2i32 != 0 {
unsafe {
(*monitor_range).max_vfreq = ((*monitor_range).max_vfreq as i32).wrapping_add(255) as u16;
}
}
}
}

fn drm_get_monitor_range(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).display_info) }) as *const drm_display_info;
let mut closure: detailed_mode_closure = {
let mut __t0: detailed_mode_closure = unsafe { core::mem::zeroed() };
__t0.connector = connector;
__t0.drm_edid = drm_edid;
__t0
};
if ((unsafe { (*(*drm_edid).edid).revision }) as i32) < 4 {
return;
}
if ((unsafe { (*(*drm_edid).edid).features }) as i32) & 1i32 == 0 {
return;
}
drm_for_each_detailed_block(drm_edid, Some(get_monitor_range), core::ptr::addr_of_mut!(closure) as *mut ());
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Supported Monitor Refresh rate range is %d Hz - %d Hz\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).monitor_range.min_vfreq as i32, (*info).monitor_range.max_vfreq as i32) };
}

fn drm_parse_vesa_mso_data(connector: *mut drm_connector, block: *const displayid_block) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
if ((unsafe { (*block).num_bytes }) as i32) < 3 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Unexpected vendor block size %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*block).num_bytes as i32) };
return;
}
if oui(unsafe { *((*(block as *mut displayid_vesa_vendor_specific_block)).oui.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }, unsafe { *((*(block as *mut displayid_vesa_vendor_specific_block)).oui.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }, unsafe { *((*(block as *mut displayid_vesa_vendor_specific_block)).oui.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) != 3801746 {
return;
}
if (core::mem::size_of::<displayid_vesa_vendor_specific_block>() as u64) != (core::mem::size_of::<displayid_block>() as u64).wrapping_add(((unsafe { (*block).num_bytes }) as i32) as u64) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Unexpected VESA vendor block size\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return;
}
let mut __v0: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6625_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6))) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_773() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_774() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6625_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) == 0 {
unsafe { __compiletime_assert_775() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_drm_edid_c_6625_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6625_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_776() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_edid_c_6625_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6625_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6625_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6625_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_777() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((((unsafe { (*(block as *mut displayid_vesa_vendor_specific_block)).mso }) as i32) as u64) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6625_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6))) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6625_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 5i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32);
__v0 = __v2;
// 2 or 4 links
// 4 links
match __v0 {
0 => {
unsafe {
(*info).mso_stream_count = 0;
}
}
1 => {
unsafe {
(*info).mso_stream_count = 2;
}
// 2 or 4 links
}
2 => {
unsafe {
(*info).mso_stream_count = 4;
}
// 4 links
}
_ => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Reserved MSO mode value\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
unsafe {
(*info).mso_stream_count = 0;
}
}
}
if (unsafe { (*info).mso_stream_count }) == 0 {
unsafe {
(*info).mso_pixel_overlap = 0;
}
return;
}
let mut __v8: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6646_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3))) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_778() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v10: u64;
loop {
if false {
unsafe { __compiletime_assert_779() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6646_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)) == 0 {
unsafe { __compiletime_assert_780() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_drm_edid_c_6646_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6646_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_781() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_edid_c_6646_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6646_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6646_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6646_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_782() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = ((((unsafe { (*(block as *mut displayid_vesa_vendor_specific_block)).mso }) as i32) as u64) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6646_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3))) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6646_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)).trailing_zeros() as i32);
__v8 = __v10;
unsafe {
(*info).mso_pixel_overlap = __v8 as u8;
}
if ((unsafe { (*info).mso_pixel_overlap }) as i32) > 8 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Reserved MSO pixel overlap value %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).mso_pixel_overlap as i32) };
unsafe {
(*info).mso_pixel_overlap = 8;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] MSO stream count %u, pixel overlap %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).mso_stream_count as i32, (*info).mso_pixel_overlap as i32) };
}

fn drm_update_mso(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let mut iter: displayid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter)) };
loop {
let __v0: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!(iter)) };
let block: *const displayid_block = __v0;
if __v0.is_null() {
break;
}
if ((unsafe { (*block).tag }) as i32) == 126 {
drm_parse_vesa_mso_data(connector, block);
}
}
unsafe { displayid_iter_end(core::ptr::addr_of_mut!(iter)) };
}

/// A connector has no EDID information, so we've got no EDID to compute quirks from. Reset
/// all of the values which would have been set from EDID
fn drm_reset_display_info(connector: *mut drm_connector) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
unsafe {
(*info).width_mm = 0;
}
unsafe {
(*info).height_mm = 0;
}
unsafe {
(*info).bpc = 0;
}
unsafe {
(*info).color_formats = 0;
}
unsafe {
(*info).cea_rev = 0;
}
unsafe {
(*info).max_tmds_clock = 0;
}
unsafe {
(*info).dvi_dual = false;
}
unsafe {
(*info).is_hdmi = false;
}
unsafe {
(*info).has_audio = false;
}
unsafe {
(*info).has_hdmi_infoframe = false;
}
unsafe {
(*info).rgb_quant_range_selectable = false;
}
unsafe { memset(core::ptr::addr_of_mut!((*info).hdmi) as *mut (), 0i32, core::mem::size_of::<drm_hdmi_info>() as u64) };
unsafe { memset(core::ptr::addr_of_mut!((*info).hdr_sink_metadata) as *mut (), 0i32, core::mem::size_of::<hdr_sink_metadata>() as u64) };
unsafe {
(*info).edid_hdmi_rgb444_dc_modes = 0;
}
unsafe {
(*info).edid_hdmi_ycbcr444_dc_modes = 0;
}
unsafe {
(*info).non_desktop = false;
}
unsafe { memset(core::ptr::addr_of_mut!((*info).monitor_range) as *mut (), 0i32, core::mem::size_of::<drm_monitor_range_info>() as u64) };
unsafe { memset(core::ptr::addr_of_mut!((*info).luminance_range) as *mut (), 0i32, core::mem::size_of::<drm_luminance_range_info>() as u64) };
unsafe {
(*info).mso_stream_count = 0;
}
unsafe {
(*info).mso_pixel_overlap = 0;
}
unsafe {
(*info).max_dsc_bpp = 0;
}
unsafe { kfree((*info).vics as *const ()) };
unsafe {
(*info).vics = core::ptr::null_mut::<u8>();
}
unsafe {
(*info).vics_len = 0;
}
unsafe {
(*info).quirks = 0;
}
unsafe {
(*info).source_physical_address = 65535;
}
unsafe { memset(core::ptr::addr_of_mut!((*info).amd_vsdb) as *mut (), 0i32, core::mem::size_of::<drm_amd_vsdb_info>() as u64) };
unsafe {
(*info).panel_type = 0;
}
}

fn drm_displayid_process_base_section_header(connector: *mut drm_connector, iter: *const displayid_iter) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] DisplayID extension version 0x%02x, primary use 0x%02x\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, displayid_version(iter) as i32, displayid_primary_use(iter) as i32) };
let __v0: bool = ((unsafe { displayid_version(iter) }) as i32) == 32 && (((unsafe { displayid_primary_use(iter) }) as i32) == 7 || ((unsafe { displayid_primary_use(iter) }) as i32) == 8);
if __v0 {
unsafe {
(*info).non_desktop = true;
}
}
}

fn drm_displayid_parse_display_params(connector: *mut drm_connector, block: *const displayid_block) {
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info) };
let mut tech: u8 = 0;
if (((unsafe { (*block).num_bytes }) as i32) as u64) < (core::mem::size_of::<displayid_display_params_block>() as u64).wrapping_sub(core::mem::size_of::<displayid_block>() as u64) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] DisplayID Display Parameters block too short (%u < %zu)\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*block).num_bytes as i32, (core::mem::size_of::<displayid_display_params_block>() as u64).wrapping_sub(core::mem::size_of::<displayid_block>() as u64)) };
return;
}
let mut __v0: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6754_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6))) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_783() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u64;
loop {
if false {
unsafe { __compiletime_assert_784() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_edid_c_6754_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) == 0 {
unsafe { __compiletime_assert_785() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_drm_edid_c_6754_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6754_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_786() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_edid_c_6754_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6754_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6754_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_edid_c_6754_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_787() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((((unsafe { (*(block as *const displayid_display_params_block)).color_depth_and_tech }) as i32) as u64) & (core::mem::size_of::<__SlateRecord_drm_edid_c_6754_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6))) >> ((core::mem::size_of::<__SlateRecord_drm_edid_c_6754_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 4i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)).trailing_zeros() as i32);
__v0 = __v2;
tech = __v0 as u8;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] DisplayID Display Parameters: device technology %s\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, if (tech as i32) == 1 { c"LCD".as_ptr() as *mut u8 } else { if (tech as i32) == 2 { c"OLED".as_ptr() as *mut u8 } else { c"unspecified".as_ptr() as *mut u8 } }) };
match tech as i32 {
1 => {
unsafe {
(*info).panel_type = 2;
}
}
2 => {
unsafe {
(*info).panel_type = 1;
}
}
_ => {
}
}
}

fn update_displayid_info(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let mut iter: displayid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut base_section_header_processed: bool = false;
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter)) };
loop {
let __v0: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!(iter)) };
let block: *const displayid_block = __v0;
if __v0.is_null() {
break;
}
if !base_section_header_processed {
drm_displayid_process_base_section_header(connector, core::ptr::addr_of_mut!(iter) as *const displayid_iter);
base_section_header_processed = true;
}
if ((unsafe { displayid_version(core::ptr::addr_of_mut!(iter) as *const displayid_iter) }) as i32) == 32 && ((unsafe { (*block).tag }) as i32) == 33 {
drm_displayid_parse_display_params(connector, block);
}
}
unsafe { displayid_iter_end(core::ptr::addr_of_mut!(iter)) };
}

fn update_display_info(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let mut edid: core::mem::MaybeUninit<*const edid> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *const edid = core::ptr::addr_of_mut!(edid) as *mut *const edid;
let mut info: *mut drm_display_info = unsafe { core::mem::zeroed() };
unsafe {
info = core::ptr::addr_of_mut!((*connector).display_info);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
drm_reset_display_info(connector);
clear_eld(connector);
if drm_edid.is_null() {
return;
} else {
*__slate_slot_9 = (*drm_edid).edid;
(*info).quirks = edid_get_quirks(drm_edid);
(*info).width_mm = ((*(*__slate_slot_9)).width_cm as i32).wrapping_mul(10) as u32;
(*info).height_mm = ((*(*__slate_slot_9)).height_cm as i32).wrapping_mul(10) as u32;
drm_get_monitor_range(connector, drm_edid);
if ((*(*__slate_slot_9)).revision as i32) >= 3 {
if drm_edid_is_digital(drm_edid) {
let __v7: u32 = (*info).color_formats;
let __v6: u32 = ((__v7 as u64) | BIT!(0i32)) as u32;
(*info).color_formats = __v6;
drm_parse_cea_ext(connector, drm_edid);
update_displayid_info(connector, drm_edid);
// Digital sink with "DFP 1.x compliant TMDS" according to EDID 1.3?
//
// For such displays, the DFP spec 1.0, section 3.10 "EDID support"
// tells us to assume 8 bpc color depth if the EDID doesn't have
// extensions which tell otherwise.
if (*info).bpc == 0 && ((*(*__slate_slot_9)).revision as i32) == 3 && ((*(*__slate_slot_9)).input as i32) & 1i32 != 0 {
(*info).bpc = 8;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Assigning DFP sink color depth as %d bpc.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).bpc);
}
// Only defined for 1.4 with digital displays
if ((*(*__slate_slot_9)).revision as i32) >= 4 {
'__join_17: {
let __t0: i32 = ((*(*__slate_slot_9)).input as i32) & 112i32;
if __t0 == 16 {
(*info).bpc = 6;
break '__join_17;
} else {
if __t0 == 32 {
(*info).bpc = 8;
break '__join_17;
} else {
if __t0 == 48 {
(*info).bpc = 10;
break '__join_17;
} else {
if __t0 == 64 {
(*info).bpc = 12;
break '__join_17;
} else {
if __t0 == 80 {
(*info).bpc = 14;
break '__join_17;
} else {
if __t0 == 96 {
(*info).bpc = 16;
break '__join_17;
}
}
}
}
}
}
(*info).bpc = 0;
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Assigning EDID-1.4 digital sink color depth as %d bpc.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*info).bpc);
if ((*(*__slate_slot_9)).features as i32) & 8i32 != 0 {
let __v4: u32 = (*info).color_formats;
let __v3: u32 = ((__v4 as u64) | BIT!(1i32)) as u32;
(*info).color_formats = __v3;
}
if ((*(*__slate_slot_9)).features as i32) & 16i32 != 0 {
let __v1: u32 = (*info).color_formats;
let __v0: u32 = ((__v1 as u64) | BIT!(2i32)) as u32;
(*info).color_formats = __v0;
}
drm_update_mso(connector, drm_edid);
}
}
}
if drm_edid_has_internal_quirk(connector, 12u32) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Non-desktop display%s\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, if (*info).non_desktop { c" (redundant quirk)".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
(*info).non_desktop = true;
}
if drm_edid_has_internal_quirk(connector, 13u32) {
(*info).max_dsc_bpp = 15;
}
if drm_edid_has_internal_quirk(connector, 10u32) {
(*info).bpc = 6;
}
if drm_edid_has_internal_quirk(connector, 8u32) {
(*info).bpc = 8;
}
if drm_edid_has_internal_quirk(connector, 11u32) {
(*info).bpc = 10;
}
if drm_edid_has_internal_quirk(connector, 9u32) {
(*info).bpc = 12;
}
// Depends on info->cea_rev set by drm_parse_cea_ext() above
drm_edid_to_eld(connector, drm_edid);
}
}
}

fn drm_mode_displayid_detailed(dev: *mut drm_device, timings: *const displayid_detailed_timings_1, type_7: bool) -> *mut drm_display_mode {
let pixel_clock: u32 = (((unsafe { (core::ptr::addr_of!((*timings).pixel_clock) as *const u8).wrapping_offset(0 as isize).read_unaligned() }) as i32) | ((unsafe { (core::ptr::addr_of!((*timings).pixel_clock) as *const u8).wrapping_offset(1 as isize).read_unaligned() }) as i32) << 8i32 | ((unsafe { (core::ptr::addr_of!((*timings).pixel_clock) as *const u8).wrapping_offset(2 as isize).read_unaligned() }) as i32) << 16i32).wrapping_add(1) as u32;
let hactive: u32 = ((unsafe { (*timings).hactive }) as i32).wrapping_add(1) as u32;
let hblank: u32 = ((unsafe { (*timings).hblank }) as i32).wrapping_add(1) as u32;
let hsync: u32 = (((unsafe { (*timings).hsync }) as i32) & 32767).wrapping_add(1) as u32;
let hsync_width: u32 = ((unsafe { (*timings).hsw }) as i32).wrapping_add(1) as u32;
let vactive: u32 = ((unsafe { (*timings).vactive }) as i32).wrapping_add(1) as u32;
let vblank: u32 = ((unsafe { (*timings).vblank }) as i32).wrapping_add(1) as u32;
let vsync: u32 = (((unsafe { (*timings).vsync }) as i32) & 32767).wrapping_add(1) as u32;
let vsync_width: u32 = ((unsafe { (*timings).vsw }) as i32).wrapping_add(1) as u32;
let hsync_positive: bool = ((unsafe { (*timings).hsync }) as i32) & 32768i32 != 0;
let vsync_positive: bool = ((unsafe { (*timings).vsync }) as i32) & 32768i32 != 0;
let mode: *mut drm_display_mode = unsafe { drm_mode_create(dev) };
if mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
// resolution is kHz for type VII, and 10 kHz for type I
unsafe {
(*mode).clock = (if type_7 { pixel_clock } else { pixel_clock.wrapping_mul(10) }) as i32;
}
unsafe {
(*mode).hdisplay = hactive as u16;
}
unsafe {
(*mode).hsync_start = (((*mode).hdisplay as i32) as u32).wrapping_add(hsync) as u16;
}
unsafe {
(*mode).hsync_end = (((*mode).hsync_start as i32) as u32).wrapping_add(hsync_width) as u16;
}
unsafe {
(*mode).htotal = (((*mode).hdisplay as i32) as u32).wrapping_add(hblank) as u16;
}
unsafe {
(*mode).vdisplay = vactive as u16;
}
unsafe {
(*mode).vsync_start = (((*mode).vdisplay as i32) as u32).wrapping_add(vsync) as u16;
}
unsafe {
(*mode).vsync_end = (((*mode).vsync_start as i32) as u32).wrapping_add(vsync_width) as u16;
}
unsafe {
(*mode).vtotal = (((*mode).vdisplay as i32) as u32).wrapping_add(vblank) as u16;
}
unsafe {
(*mode).flags = 0;
}
unsafe {
(*mode).flags = (*mode).flags | ((if hsync_positive { 1i32 } else { 2i32 }) as u32);
}
unsafe {
(*mode).flags = (*mode).flags | ((if vsync_positive { 4i32 } else { 8i32 }) as u32);
}
unsafe {
(*mode).r#type = 64u8;
}
if ((unsafe { (*timings).flags }) as i32) & 128 != 0 {
unsafe {
(*mode).r#type = (((*mode).r#type as i32) | 8i32) as u8;
}
}
unsafe { drm_mode_set_name(mode) };
mode
}

fn add_displayid_detailed_1_modes(connector: *mut drm_connector, block: *const displayid_block) -> i32 {
let mut num_timings: i32 = 0;
let mut num_modes: i32 = 0;
let type_7: bool = ((unsafe { (*block).tag }) as i32) == 34;
// blocks must be multiple of 20 bytes length
if ((unsafe { (*block).num_bytes }) as i32) % 20 != 0 {
return 0;
}
num_timings = ((unsafe { (*block).num_bytes }) as i32) / 20;
for i in 0..num_timings {
let timings: *mut displayid_detailed_timings_1 = unsafe { core::ptr::addr_of_mut!((*(block as *mut displayid_detailed_timing_block)).timings) as *mut displayid_detailed_timings_1 }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = drm_mode_displayid_detailed(unsafe { (*connector).dev }, timings as *const displayid_detailed_timings_1, type_7);
if !newmode.is_null() {
unsafe { drm_mode_probed_add(connector, newmode) };
num_modes = num_modes.wrapping_add(1);
}
}
num_modes
}

fn drm_mode_displayid_formula(dev: *mut drm_device, timings: *const displayid_formula_timings_9, type_10: bool) -> *mut drm_display_mode {
let hactive: u16 = ((unsafe { (*timings).hactive }) as i32).wrapping_add(1) as u16;
let vactive: u16 = ((unsafe { (*timings).vactive }) as i32).wrapping_add(1) as u16;
let timing_formula: u8 = (((unsafe { (*timings).flags }) as i32) & 7) as u8;
// TODO: support RB-v2 & RB-v3
if (timing_formula as i32) > 1 {
return core::ptr::null_mut::<drm_display_mode>();
}
// TODO: support video-optimized refresh rate
if ((unsafe { (*timings).flags }) as i32) & 16i32 != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Fractional vrefresh is not implemented, proceeding with non-video-optimized refresh rate".as_ptr() as *const u8) };
}
let mode: *mut drm_display_mode = unsafe { drm_cvt_mode(dev, hactive as i32, vactive as i32, ((*timings).vrefresh as i32).wrapping_add(1), (timing_formula as i32) == 1, false, false) };
if mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
// TODO: interpret S3D flags
unsafe {
(*mode).r#type = 64u8;
}
unsafe { drm_mode_set_name(mode) };
mode
}

fn add_displayid_formula_modes(connector: *mut drm_connector, block: *const displayid_block) -> i32 {
let mut num_timings: i32 = 0;
let mut num_modes: i32 = 0;
let type_10: bool = ((unsafe { (*block).tag }) as i32) == 42;
let timing_size: i32 = 6i32.wrapping_add((((unsafe { (*(block as *const displayid_formula_timing_block)).base.rev }) as i32) & 112) >> 4i32);
// extended blocks are not supported yet
if timing_size != 6 {
return 0;
}
if ((unsafe { (*block).num_bytes }) as i32) % timing_size != 0 {
return 0;
}
num_timings = ((unsafe { (*block).num_bytes }) as i32) / timing_size;
for i in 0..num_timings {
let timings: *const displayid_formula_timings_9 = unsafe { core::ptr::addr_of!((*(block as *const displayid_formula_timing_block)).timings) as *const displayid_formula_timings_9 }.wrapping_offset(i as isize);
let newmode: *mut drm_display_mode = drm_mode_displayid_formula(unsafe { (*connector).dev }, timings, type_10);
if newmode.is_null() {
continue;
}
unsafe { drm_mode_probed_add(connector, newmode) };
num_modes = num_modes.wrapping_add(1);
}
num_modes
}

fn add_displayid_detailed_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut iter: displayid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut num_modes: i32 = 0;
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter)) };
loop {
let __v6: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!(iter)) };
let block: *const displayid_block = __v6;
if __v6.is_null() {
break;
}
if ((unsafe { (*block).tag }) as i32) == 3 || ((unsafe { (*block).tag }) as i32) == 34 {
let __v0: i32 = add_displayid_detailed_1_modes(connector, block);
num_modes = num_modes.wrapping_add(__v0);
} else {
if ((unsafe { (*block).tag }) as i32) == 36 || ((unsafe { (*block).tag }) as i32) == 42 {
let __v3: i32 = add_displayid_formula_modes(connector, block);
num_modes = num_modes.wrapping_add(__v3);
}
}
}
unsafe { displayid_iter_end(core::ptr::addr_of_mut!(iter)) };
num_modes
}

fn _drm_edid_connector_add_modes(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut num_modes: i32 = 0;
if drm_edid.is_null() {
return 0;
}
// EDID spec says modes should be preferred in this order:
// - preferred detailed mode
// - other detailed modes from base block
// - detailed modes from extension blocks
// - CVT 3-byte code modes
// - standard timing codes
// - established timing codes
// - modes inferred from GTF or CVT range information
//
// We get this pretty much right.
//
// XXX order for additional mode types in extension blocks?
let __v0: i32 = add_detailed_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v0);
let __v3: i32 = add_cvt_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v3);
let __v6: i32 = add_standard_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v6);
let __v9: i32 = add_established_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v9);
let __v12: i32 = add_cea_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v12);
let __v15: i32 = add_alternate_cea_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v15);
let __v18: i32 = add_displayid_detailed_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v18);
if ((unsafe { (*(*drm_edid).edid).features }) as i32) & 1i32 != 0 {
let __v21: i32 = add_inferred_modes(connector, drm_edid);
num_modes = num_modes.wrapping_add(__v21);
}
let __v24: bool = drm_edid_has_internal_quirk(connector, 1u32) || drm_edid_has_internal_quirk(connector, 3u32);
if __v24 {
edid_fixup_preferred(connector);
}
num_modes
}

fn _drm_edid_connector_property_update(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut old_edid_size: u64 = unsafe { core::mem::zeroed() };
let mut old_edid: *const () = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut dev: *mut drm_device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*connector).dev;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !unsafe { (*connector).edid_blob_ptr }.is_null() {
old_edid = (*(*connector).edid_blob_ptr).data as *const ();
old_edid_size = (*(*connector).edid_blob_ptr).length;
if !old_edid.is_null() {
__v3 = !drm_edid_eq(drm_edid, old_edid, old_edid_size);
} else {
__v3 = false;
}
if __v3 {
let __v1: u64 = (*connector).epoch_counter;
(*connector).epoch_counter = __v1.wrapping_add(1);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID changed, epoch counter %llu\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*connector).epoch_counter);
}
}
*__slate_slot_6 = drm_property_replace_global_blob(dev, core::ptr::addr_of_mut!((*connector).edid_blob_ptr), if !drm_edid.is_null() { (*drm_edid).size } else { 0u64 }, (if !drm_edid.is_null() { (*drm_edid).edid } else { core::ptr::null::<edid>() }) as *const (), core::ptr::addr_of_mut!((*connector).base), (*dev).mode_config.edid_property);
if *__slate_slot_6 != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] EDID property update failed (%d)\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, *__slate_slot_6);
} else {
*__slate_slot_6 = drm_object_property_set_value(core::ptr::addr_of_mut!((*connector).base), (*dev).mode_config.non_desktop_property, (*connector).display_info.non_desktop as u64);
if *__slate_slot_6 != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Non-desktop property update failed (%d)\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, *__slate_slot_6);
} else {
*__slate_slot_6 = drm_connector_set_tile_property(connector);
if *__slate_slot_6 != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Tile property update failed (%d)\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, *__slate_slot_6);
}
}
}
return *__slate_slot_6;
}
}

/// For sysfs edid show implementation
#[unsafe(no_mangle)]
extern "C" fn drm_edid_connector_property_show(connector: *mut drm_connector, buf: *mut u8, off: i64, mut count: u64) -> i64 {
let mut ret: i64 = unsafe { core::mem::zeroed() };
let mut size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(size) as *mut u64;
let mut edid: core::mem::MaybeUninit<*const ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *const () = core::ptr::addr_of_mut!(edid) as *mut *const ();
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
mutex_lock(core::ptr::addr_of_mut!((*(*connector).dev).mode_config.mutex));
if !unsafe { (*connector).edid_blob_ptr }.is_null() {
*__slate_slot_2 = (*(*connector).edid_blob_ptr).data as *const ();
*__slate_slot_1 = (*(*connector).edid_blob_ptr).length;
if !(*__slate_slot_2).is_null() {
if (off as u64) < *__slate_slot_1 {
if (off as u64).wrapping_add(count) > *__slate_slot_1 {
count = (*__slate_slot_1).wrapping_sub(off as u64);
}
memcpy(buf as *mut (), (*__slate_slot_2 as *const u8).wrapping_offset(off as isize) as *const (), count);
ret = count as i64;
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*(*connector).dev).mode_config.mutex));
return ret;
}
}

/// drm_edid_connector_update - Update connector information from EDID
/// @connector: Connector
/// @drm_edid: EDID
///
/// Update the connector display info, ELD, HDR metadata, relevant properties,
/// etc. from the passed in EDID.
///
/// If EDID is NULL, reset the information.
///
/// Must be called before calling drm_edid_connector_add_modes().
///
/// Return: 0 on success, negative error on errors.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_connector_update")]
extern "C" fn drm_edid_connector_update(connector: *mut drm_connector, drm_edid: *const drm_edid) -> i32 {
update_display_info(connector, drm_edid);
_drm_update_tile_info(connector, drm_edid);
_drm_edid_connector_property_update(connector, drm_edid)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_connector_update_788: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *const drm_edid) -> i32>, *mut ()>(Some(drm_edid_connector_update)) };

/// drm_edid_connector_add_modes - Update probed modes from the EDID property
/// @connector: Connector
///
/// Add the modes from the previously updated EDID property to the connector
/// probed modes list.
///
/// drm_edid_connector_update() must have been called before this to update the
/// EDID property.
///
/// Return: The number of modes added, or 0 if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_connector_add_modes")]
extern "C" fn drm_edid_connector_add_modes(connector: *mut drm_connector) -> i32 {
let mut drm_edid: *const drm_edid = core::ptr::null::<drm_edid>();
let mut count: i32 = 0;
if !unsafe { (*connector).edid_blob_ptr }.is_null() {
drm_edid = drm_edid_alloc((unsafe { (*(*connector).edid_blob_ptr).data }) as *const (), unsafe { (*(*connector).edid_blob_ptr).length });
}
count = _drm_edid_connector_add_modes(connector, drm_edid);
drm_edid_free(drm_edid);
count
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_connector_add_modes_789: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector) -> i32>, *mut ()>(Some(drm_edid_connector_add_modes)) };

/// drm_connector_update_edid_property - update the edid property of a connector
/// @connector: drm connector
/// @edid: new value of the edid property
///
/// This function creates a new blob modeset object and assigns its id to the
/// connector's edid property.
/// Since we also parse tile information from EDID's displayID block, we also
/// set the connector's tile property here. See drm_connector_set_tile_property()
/// for more details.
///
/// This function is deprecated. Use drm_edid_connector_update() instead.
///
/// Returns:
/// Zero on success, negative errno on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_connector_update_edid_property")]
extern "C" fn drm_connector_update_edid_property(connector: *mut drm_connector, edid: *const edid) -> i32 {
let mut drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
drm_edid_connector_update(connector, drm_edid_legacy_init(core::ptr::addr_of_mut!(drm_edid), edid))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_connector_update_edid_property_790: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *const edid) -> i32>, *mut ()>(Some(drm_connector_update_edid_property)) };

/// drm_add_edid_modes - add modes from EDID data, if available
/// @connector: connector we're probing
/// @edid: EDID data
///
/// Add the specified modes to the connector's mode list. Also fills out the
/// &drm_display_info structure and ELD in @connector with any information which
/// can be derived from the edid.
///
/// This function is deprecated. Use drm_edid_connector_add_modes() instead.
///
/// Return: The number of modes added or 0 if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_add_edid_modes")]
extern "C" fn drm_add_edid_modes(connector: *mut drm_connector, mut edid: *mut edid) -> i32 {
let mut _drm_edid: drm_edid = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(_drm_edid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let __v0: bool = !edid.is_null() && !drm_edid_is_valid(edid);
if __v0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*connector).dev as *const drm_device) as *const device, c"[drm] [CONNECTOR:%d:%s] EDID invalid.\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
edid = core::ptr::null_mut::<edid>();
}
let drm_edid: *const drm_edid = drm_edid_legacy_init(core::ptr::addr_of_mut!(_drm_edid), edid as *const edid);
update_display_info(connector, drm_edid);
_drm_edid_connector_add_modes(connector, drm_edid)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_add_edid_modes_791: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut edid) -> i32>, *mut ()>(Some(drm_add_edid_modes)) };

/// drm_add_modes_noedid - add modes for the connectors without EDID
/// @connector: connector we're probing
/// @hdisplay: the horizontal display limit
/// @vdisplay: the vertical display limit
///
/// Add the specified modes to the connector's mode list. Only when the
/// hdisplay/vdisplay is not beyond the given limit, it will be added.
///
/// Return: The number of modes added or 0 if we couldn't find any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_add_modes_noedid")]
extern "C" fn drm_add_modes_noedid(connector: *mut drm_connector, hdisplay: u32, vdisplay: u32) -> i32 {
let count: i32 = ARRAY_SIZE__77!() as i32;
let mut num_modes: i32 = 0;
let dev: *mut drm_device = unsafe { (*connector).dev };
for i in 0..count {
'__slate_continue_0: {
let ptr: *const drm_display_mode = unsafe { core::ptr::addr_of!(drm_dmt_modes.0) as *const drm_display_mode }.wrapping_offset(i as isize);
if hdisplay != 0 && vdisplay != 0 {
// Only when two are valid, they will be used to check
// whether the mode should be added to the mode list of
// the connector.
if (((unsafe { (*ptr).hdisplay }) as i32) as u32) > hdisplay || (((unsafe { (*ptr).vdisplay }) as i32) as u32) > vdisplay {
break '__slate_continue_0;
}
}
if (unsafe { drm_mode_vrefresh(ptr) }) <= 61 {
let mode: *mut drm_display_mode = unsafe { drm_mode_duplicate(dev, ptr) };
if !mode.is_null() {
unsafe { drm_mode_probed_add(connector, mode) };
num_modes = num_modes.wrapping_add(1);
}
}
}
}
num_modes
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_add_modes_noedid_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>, *mut ()>(Some(drm_add_modes_noedid)) };

fn is_hdmi2_sink(connector: *const drm_connector) -> bool {
// FIXME: sil-sii8620 doesn't have a connector around when
// we need one, so we have to be prepared for a NULL connector.
if connector.is_null() {
return true;
}
(unsafe { (*connector).display_info.hdmi.scdc.supported }) || ((unsafe { (*connector).display_info.color_formats }) as u64) & BIT!(3i32) != 0
}

fn drm_mode_hdmi_vic(connector: *const drm_connector, mode: *const drm_display_mode) -> u8 {
let has_hdmi_infoframe: bool = (if !connector.is_null() { (unsafe { (*connector).display_info.has_hdmi_infoframe }) as i32 } else { 0i32 }) != 0;
if !has_hdmi_infoframe {
return 0;
}
// No HDMI VIC when signalling 3D video format
if (unsafe { (*mode).flags }) & 507904u32 != 0 {
return 0;
}
drm_match_hdmi_mode(mode)
}

fn drm_mode_cea_vic(connector: *const drm_connector, mode: *const drm_display_mode) -> u8 {
// HDMI spec says if a mode is found in HDMI 1.4b 4K modes
// we should send its VIC in vendor infoframes, else send the
// VIC in AVI infoframes. Lets check if this mode is present in
// HDMI 1.4b 4K modes
if drm_mode_hdmi_vic(connector, mode) != 0 {
return 0;
}
drm_match_cea_mode(mode)
}

/// Avoid sending VICs defined in HDMI 2.0 in AVI infoframes to sinks that
/// conform to HDMI 1.4.
///
/// HDMI 1.4 (CTA-861-D) VIC range: [1..64]
/// HDMI 2.0 (CTA-861-F) VIC range: [1..107]
///
/// If the sink lists the VIC in CTA VDB, assume it's fine, regardless of HDMI
/// version.
fn vic_for_avi_infoframe(connector: *const drm_connector, vic: u8) -> u8 {
let __v0: bool = !is_hdmi2_sink(connector) && (vic as i32) > 64 && !cta_vdb_has_vic(connector, vic);
if __v0 {
return 0;
}
vic
}

/// drm_hdmi_avi_infoframe_from_display_mode() - fill an HDMI AVI infoframe with
///                                              data from a DRM display mode
/// @frame: HDMI AVI infoframe
/// @connector: the connector
/// @mode: DRM display mode
///
/// Return: 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_hdmi_avi_infoframe_from_display_mode")]
extern "C" fn drm_hdmi_avi_infoframe_from_display_mode(frame: *mut hdmi_avi_infoframe, connector: *const drm_connector, mode: *const drm_display_mode) -> i32 {
let mut picture_aspect: u32 = 0;
let mut vic: u8 = 0;
let mut hdmi_vic: u8 = 0;
if frame.is_null() || mode.is_null() {
return 22i32.wrapping_neg();
}
unsafe { hdmi_avi_infoframe_init(frame) };
if (unsafe { (*mode).flags }) & 4096u32 != 0 {
unsafe {
(*frame).pixel_repeat = 1;
}
}
vic = drm_mode_cea_vic(connector, mode);
hdmi_vic = drm_mode_hdmi_vic(connector, mode);
unsafe {
(*frame).picture_aspect = 0;
}
// As some drivers don't support atomic, we can't use connector state.
// So just initialize the frame with default values, just the same way
// as it's done with other properties here.
unsafe {
(*frame).content_type = 0;
}
unsafe {
(*frame).itc = false;
}
// Populate picture aspect ratio from either
// user input (if specified) or from the CEA/HDMI mode lists.
picture_aspect = unsafe { (*mode).picture_aspect_ratio };
if (picture_aspect as u32) == 0 {
if vic != 0 {
picture_aspect = drm_get_cea_aspect_ratio(vic);
} else {
if hdmi_vic != 0 {
picture_aspect = drm_get_hdmi_aspect_ratio(hdmi_vic);
}
}
}
// The infoframe can't convey anything but none, 4:3
// and 16:9, so if the user has asked for anything else
// we can only satisfy it by specifying the right VIC.
if (picture_aspect as u32) > 2 {
if vic != 0 {
if (picture_aspect as u32) != (drm_get_cea_aspect_ratio(vic) as u32) {
return 22i32.wrapping_neg();
}
} else {
if hdmi_vic != 0 {
if (picture_aspect as u32) != (drm_get_hdmi_aspect_ratio(hdmi_vic) as u32) {
return 22i32.wrapping_neg();
}
} else {
return 22i32.wrapping_neg();
}
}
picture_aspect = 0;
}
unsafe {
(*frame).video_code = vic_for_avi_infoframe(connector, vic);
}
unsafe {
(*frame).picture_aspect = picture_aspect;
}
unsafe {
(*frame).active_aspect = 8;
}
unsafe {
(*frame).scan_mode = 2;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_hdmi_avi_infoframe_from_display_mode_793: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hdmi_avi_infoframe, *const drm_connector, *const drm_display_mode) -> i32>, *mut ()>(Some(drm_hdmi_avi_infoframe_from_display_mode)) };

/// drm_hdmi_avi_infoframe_quant_range() - fill the HDMI AVI infoframe
///                                        quantization range information
/// @frame: HDMI AVI infoframe
/// @connector: the connector
/// @mode: DRM display mode
/// @rgb_quant_range: RGB quantization range (Q)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_hdmi_avi_infoframe_quant_range")]
extern "C" fn drm_hdmi_avi_infoframe_quant_range(frame: *mut hdmi_avi_infoframe, connector: *const drm_connector, mode: *const drm_display_mode, rgb_quant_range: u32) {
let info: *const drm_display_info = unsafe { core::ptr::addr_of!((*connector).display_info) };
// CEA-861:
// "A Source shall not send a non-zero Q value that does not correspond
//  to the default RGB Quantization Range for the transmitted Picture
//  unless the Sink indicates support for the Q bit in a Video
//  Capabilities Data Block."
//
// HDMI 2.0 recommends sending non-zero Q when it does match the
// default RGB quantization range for the mode, even when QS=0.
let __v0: bool = (unsafe { (*info).rgb_quant_range_selectable }) || (rgb_quant_range as u32) == (drm_default_rgb_quant_range(mode) as u32);
if __v0 {
unsafe {
(*frame).quantization_range = rgb_quant_range;
}
} else {
unsafe {
(*frame).quantization_range = 0;
}
}
// CEA-861-F:
// "When transmitting any RGB colorimetry, the Source should set the
//  YQ-field to match the RGB Quantization Range being transmitted
//  (e.g., when Limited Range RGB, set YQ=0 or when Full Range RGB,
//  set YQ=1) and the Sink shall ignore the YQ-field."
//
// Unfortunate certain sinks (eg. VIZ Model 67/E261VA) get confused
// by non-zero YQ when receiving RGB. There doesn't seem to be any
// good way to tell which version of CEA-861 the sink supports, so
// we limit non-zero YQ to HDMI 2.0 sinks only as HDMI 2.0 is based
// on CEA-861-F.
if !is_hdmi2_sink(connector) || (rgb_quant_range as u32) == 1 {
unsafe {
(*frame).ycc_quantization_range = 0;
}
} else {
unsafe {
(*frame).ycc_quantization_range = 1;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_hdmi_avi_infoframe_quant_range_794: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hdmi_avi_infoframe, *const drm_connector, *const drm_display_mode, u32)>, *mut ()>(Some(drm_hdmi_avi_infoframe_quant_range)) };

fn s3d_structure_from_display_mode(mode: *const drm_display_mode) -> i32 {
let layout: u32 = (unsafe { (*mode).flags }) & 507904u32;
match layout {
16384 => {
return 0;
}
32768 => {
return 1;
}
49152 => {
return 2;
}
65536 => {
return 3;
}
81920 => {
return 4;
}
98304 => {
return 5;
}
114688 => {
return 6;
}
131072 => {
return 8;
}
_ => {
return -1;
}
}
unsafe { core::mem::zeroed() }
}

/// drm_hdmi_vendor_infoframe_from_display_mode() - fill an HDMI infoframe with
/// data from a DRM display mode
/// @frame: HDMI vendor infoframe
/// @connector: the connector
/// @mode: DRM display mode
///
/// Note that there's is a need to send HDMI vendor infoframes only when using a
/// 4k or stereoscopic 3D mode. So when giving any other mode as input this
/// function will return -EINVAL, error that can be safely ignored.
///
/// Return: 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_hdmi_vendor_infoframe_from_display_mode")]
extern "C" fn drm_hdmi_vendor_infoframe_from_display_mode(frame: *mut hdmi_vendor_infoframe, connector: *const drm_connector, mode: *const drm_display_mode) -> i32 {
// FIXME: sil-sii8620 doesn't have a connector around when
// we need one, so we have to be prepared for a NULL connector.
let has_hdmi_infoframe: bool = (if !connector.is_null() { (unsafe { (*connector).display_info.has_hdmi_infoframe }) as i32 } else { 0i32 }) != 0;
let mut err: i32 = 0;
if frame.is_null() || mode.is_null() {
return 22i32.wrapping_neg();
}
if !has_hdmi_infoframe {
return 22i32.wrapping_neg();
}
err = unsafe { hdmi_vendor_infoframe_init(frame) };
if err < 0 {
return err;
}
// Even if it's not absolutely necessary to send the infoframe
// (ie.vic==0 and s3d_struct==0) we will still send it if we
// know that the sink can handle it. This is based on a
// suggestion in HDMI 2.0 Appendix F. Apparently some sinks
// have trouble realizing that they should switch from 3D to 2D
// mode if the source simply stops sending the infoframe when
// it wants to switch from 3D to 2D.
unsafe {
(*frame).vic = drm_mode_hdmi_vic(connector, mode);
}
unsafe {
(*frame).s3d_struct = s3d_structure_from_display_mode(mode);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_hdmi_vendor_infoframe_from_display_mode_795: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hdmi_vendor_infoframe, *const drm_connector, *const drm_display_mode) -> i32>, *mut ()>(Some(drm_hdmi_vendor_infoframe_from_display_mode)) };

fn drm_parse_tiled_block(connector: *mut drm_connector, block: *const displayid_block) {
let mut w: u16 = 0;
let mut h: u16 = 0;
let mut tile_v_loc: u8 = 0;
let mut tile_h_loc: u8 = 0;
let mut num_v_tile: u8 = 0;
let mut num_h_tile: u8 = 0;
let mut tg: *mut drm_tile_group = unsafe { core::mem::zeroed() };
// tiled block payload per spec: cap 1 + topo 3 + size 4 + bezel 5 + id 9 = 22
if ((unsafe { (*block).num_bytes }) as i32) < 22 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Unexpected tiled block size %u\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*block).num_bytes as i32) };
return;
}
w = (((unsafe { *((*(block as *const displayid_tiled_block)).tile_size.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) | ((unsafe { *((*(block as *const displayid_tiled_block)).tile_size.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) << 8i32) as u16;
h = (((unsafe { *((*(block as *const displayid_tiled_block)).tile_size.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) | ((unsafe { *((*(block as *const displayid_tiled_block)).tile_size.as_ptr() as *const u8).wrapping_offset(3 as isize) }) as i32) << 8i32) as u16;
num_v_tile = (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & 15 | ((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) & 48) as u8;
num_h_tile = (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) >> 4i32 | ((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) >> 2i32 & 48) as u8;
tile_v_loc = (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) & 15 | (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) & 3) << 4i32) as u8;
tile_h_loc = (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) >> 4i32 | (((unsafe { *((*(block as *const displayid_tiled_block)).topo.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) >> 2i32 & 3) << 4i32) as u8;
unsafe {
(*connector).has_tile = true;
}
if ((unsafe { (*(block as *const displayid_tiled_block)).tile_cap }) as i32) & 128 != 0 {
unsafe {
(*connector).tile_is_single_monitor = true;
}
}
unsafe {
(*connector).num_h_tile = (num_h_tile as i32).wrapping_add(1) as u8;
}
unsafe {
(*connector).num_v_tile = (num_v_tile as i32).wrapping_add(1) as u8;
}
unsafe {
(*connector).tile_h_loc = tile_h_loc;
}
unsafe {
(*connector).tile_v_loc = tile_v_loc;
}
unsafe {
(*connector).tile_h_size = (w as i32).wrapping_add(1) as u16;
}
unsafe {
(*connector).tile_v_size = (h as i32).wrapping_add(1) as u16;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] tile cap 0x%x, size %dx%d, num tiles %dx%d, location %dx%d, vend %c%c%c".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*(block as *const displayid_tiled_block)).tile_cap as i32, (*connector).tile_h_size as i32, (*connector).tile_v_size as i32, (*connector).num_h_tile as i32, (*connector).num_v_tile as i32, (*connector).tile_h_loc as i32, (*connector).tile_v_loc as i32, *((*(block as *const displayid_tiled_block)).topology_id.as_ptr() as *const u8).wrapping_offset(0 as isize) as i32, *((*(block as *const displayid_tiled_block)).topology_id.as_ptr() as *const u8).wrapping_offset(1 as isize) as i32, *((*(block as *const displayid_tiled_block)).topology_id.as_ptr() as *const u8).wrapping_offset(2 as isize) as i32) };
tg = unsafe { drm_mode_get_tile_group((*connector).dev, ((*(block as *const displayid_tiled_block)).topology_id.as_ptr() as *const u8) as *const u8) };
if tg.is_null() {
tg = unsafe { drm_mode_create_tile_group((*connector).dev, ((*(block as *const displayid_tiled_block)).topology_id.as_ptr() as *const u8) as *const u8) };
}
if tg.is_null() {
return;
}
if (unsafe { (*connector).tile_group }) != tg {
// if we haven't got a pointer,
// take the reference, drop ref to old tile group
if !unsafe { (*connector).tile_group }.is_null() {
unsafe { drm_mode_put_tile_group((*connector).dev, (*connector).tile_group) };
}
unsafe {
(*connector).tile_group = tg;
}
} else {
// if same tile group, then release the ref we just took.
unsafe { drm_mode_put_tile_group((*connector).dev, tg) };
}
}

fn displayid_is_tiled_block(iter: *const displayid_iter, block: *const displayid_block) -> bool {
((unsafe { displayid_version(iter) }) as i32) < 32 && ((unsafe { (*block).tag }) as i32) == 18 || ((unsafe { displayid_version(iter) }) as i32) == 32 && ((unsafe { (*block).tag }) as i32) == 40
}

fn _drm_update_tile_info(connector: *mut drm_connector, drm_edid: *const drm_edid) {
let mut iter: displayid_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
unsafe {
(*connector).has_tile = false;
}
unsafe { displayid_iter_edid_begin(drm_edid, core::ptr::addr_of_mut!(iter)) };
loop {
let __v0: *const displayid_block = unsafe { __displayid_iter_next(core::ptr::addr_of_mut!(iter)) };
let block: *const displayid_block = __v0;
if __v0.is_null() {
break;
}
if displayid_is_tiled_block(core::ptr::addr_of_mut!(iter) as *const displayid_iter, block) {
drm_parse_tiled_block(connector, block);
}
}
unsafe { displayid_iter_end(core::ptr::addr_of_mut!(iter)) };
if !unsafe { (*connector).has_tile } && !unsafe { (*connector).tile_group }.is_null() {
unsafe { drm_mode_put_tile_group((*connector).dev, (*connector).tile_group) };
unsafe {
(*connector).tile_group = core::ptr::null_mut::<drm_tile_group>();
}
}
}

/// drm_edid_is_digital - is digital?
/// @drm_edid: The EDID
///
/// Return true if input is digital.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_edid.drm_edid_is_digital")]
extern "C" fn drm_edid_is_digital(drm_edid: *const drm_edid) -> bool {
!drm_edid.is_null() && !unsafe { (*drm_edid).edid }.is_null() && ((unsafe { (*(*drm_edid).edid).input }) as i32) & 128i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edid_is_digital_796: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_edid) -> bool>, *mut ()>(Some(drm_edid_is_digital)) };

static __slate_asm_string___cea_db_iter_next_1: [u8; 1] = *b"\0";

static __slate_asm_string___cea_db_iter_next_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_drm_edid_block_valid_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_edid_block_valid_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_edid_print_product_id_2: [u8; 1] = *b"\0";

static __slate_asm_string_drm_edid_print_product_id_3: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_edid_raw_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_edid_raw_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_edid_read_3: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_edid_read_4: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_edid_read_custom_3: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_edid_read_custom_4: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_edid_read_switcheroo_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_edid_read_switcheroo_5: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_drm_get_edid_switcheroo_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_get_edid_switcheroo_5: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_edid_block_status_print_17: [u8; 49] = *b"EDID block %d unknown edid block status code %d\n\0";

static __slate_asm_string_edid_block_status_print_18: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_edid_filter_invalid_blocks_6: [u8; 1] = *b"\0";

static __slate_asm_string_edid_filter_invalid_blocks_7: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_parse_cta_y420cmdb_15: [u8; 1] = *b"\0";

static __slate_asm_string_parse_cta_y420cmdb_16: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_edid.c\0";

static __slate_asm_string_seq_buf_str_1: [u8; 1] = *b"\0";

static __slate_asm_string_seq_buf_str_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/include/linux/seq_buf.h\0";
