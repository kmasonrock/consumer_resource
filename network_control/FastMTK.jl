function vectorize_params(p::CRParams)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect = p
    return [g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect]
end

function _jacobian(exprs::CDenseMatrix, x::CDenseMatrix)
    # both the functions (exprs) and the variables we're differentiating by
    # should be column vectors 
    @assert size(exprs, 2) == 1
    @assert size(x, 2) == 1

    J = CDenseMatrix(length(exprs), length(x))
    # here, do we need GC.@preserve for any of exprs, x, or J?
    res = ccall((:dense_matrix_jacobian, libsymengine), Cuint, 
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), 
                J.ptr, exprs.ptr, x.ptr)

    # the C library returns zero if everything went OK
    res == 0 || error("Failed to calculate symbolic jacobian.")
    return J
end

function jacobian(exprs, x)
    exprs = convert(CDenseMatrix, exprs)
    x = convert(CDenseMatrix, x)
    _jacobian(exprs, x)
end

struct AtIdx
    sym::Symbol
    i::Int

    AtIdx(sym, i::Int) = new(Symbol(sym), i)
end

function Base.convert(::Type{Expr}, a::AtIdx)
    Expr(:ref, a.sym, a.i)
end

function assign(lhs, rhs)
    lhs = isa(lhs, AtIdx) ? convert(Expr, lhs) : Symbol(lhs)
    rhs = isa(rhs, String) ? Symbol(rhs) : convert(Expr, rhs)
    Expr(:(=), lhs, rhs)
end

# given a julia expression of the form e.g. x1 + x2 + x3 + x4,
# split it into smaller additions, iteratively, resulting in e.g.;
# (x1 + x2) + (x3 + x4)
#
# This is necessary because I've found that long additions/multiplications
# with many terms can result in allocations even though there should be none
function _split_op(op::Symbol, expr)
    if @capture(expr, $op(args__)) && length(args) >= 4
        n = length(args)
        k = n ÷ 2
        return _split_op(op, :($op($op($(args[1:k]...)), $op($(args[k+1:n]...)))))
    else
        return expr
    end
end

function split_op(op::Symbol, expr::Expr)
    _split = Base.Fix1(_split_op, op)
    prewalk(_split, expr)
end

function build_function(exprs, u, p, t; cse::Bool = true)
    assigns = Expr[]

    # Unpack the function arguments (which will be Arrays) into
    # the individual symbols used in the equations, e.g. u_1 = __u[1]
    push!(assigns, assign(t, "__t"))
    append!(assigns, (assign(u_, AtIdx("__u", i)) for (i, u_) in enumerate(u)))
    append!(assigns, (assign(p_, AtIdx("__p", i)) for (i, p_) in enumerate(p)))

    # if we do CSE (cse = true), this will create more assignments
    # defining the intermediary variables
    if cse
        replace_syms, replace_exprs, new_exprs = SymEngine.cse(exprs)
        iter = zip(replace_syms, replace_exprs)
        append!(assigns, (assign(sym, expr) for (sym, expr) in iter))
        exprs = Basic.(new_exprs)
    end

    # convert each SymEngine expression into a julia Expr, and
    # also refactor multiplications and additions into smaller
    # operations (see split_op)
    exprs_ = IterTools.imap(exprs) do e
        split_op(:+, split_op(:*, convert(Expr, e)))
    end
    append!(assigns, (assign(AtIdx("__out", i), e) for (i, e) in enumerate(exprs_)))
    
    # the body of the function is just going to be all the single-line
    # assignments created thus far, followed by a return of nothing
    body = Expr(:block, assigns..., :(nothing))

    # now wrap it all in an expression defining an in-place (4-argument) function
    func! = Expr(:function,
                 Expr(:call, :($(gensym())), :__out, :__u, :__p, :__t),
                 body
                )

    # compile and return that function
    @RuntimeGeneratedFunction(func!)
end

# TODO: add option for sparse jacobian and jac_prototype
function to_ode_func(out, u, p, t; 
                     cse::Bool = true, 
                     jac::Bool = false)
    f! = build_function(out, u, p, t; cse = cse)

    j! = if jac
        J = jacobian(out, u)
        build_function(J[:], u, p, t; cse=cse)
    else
        nothing
    end

    # true because the ODEFunction will be in-place
    # (as both f! and j! are)
    ODEFunction{true}(f!; jac=j!)
end

function fmtk_cr(p; log_space = false)
    #@unpack g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect = p
    #_p = [g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect]
    g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect = p


    B = [symbols("B_$i") for i in vertices(g)]
    t = symbols("t")

    exprs = zeros(Basic, nv(g))

    for i ∈ vertices(g)
        if basal[i]
            exprs[i] = (B[i] - m)*(1 - B[i])*allee(B[i], S, allee_effect)
        else
            exprs[i] = -x[i]*(B[i] - m)
        end

        for j in prey[i]
            exprs[i] += x[i]*y*(B[i] - m)*F_ij(g,Ω,w,h,B0, prey, B, i, j)*allee(B[i], S, allee_effect)
        end

        for j in pred[i]
            exprs[i] -= x[j]*y*((B[i] - m)/B[i])*B[j]*F_ij(g,Ω,w,h,B0, prey, B, j,i)/e[j,i]
        end
    end

    if log_space
        for i in vertices(g)
            for j in B
                exprs[i] = subs(exprs[i], j, exp(j))
            end
            exprs[i] = exp(-B[i])*exprs[i]
        end
    end

    
    func = to_ode_func(exprs, B, [], t; cse = true, jac = true)

    return func
end
