using InvariantPointAttention
using Documenter

DocMeta.setdocmeta!(InvariantPointAttention, :DocTestSetup, :(using InvariantPointAttention); recursive=true)

makedocs(;
    modules=[InvariantPointAttention],
    authors="Ben Murrell <murrellb@gmail.com> and contributors",
    repo="https://github.com/murrellb/InvariantPointAttention.jl/blob/{commit}{path}#{line}",
    sitename="InvariantPointAttention.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://murrellb.github.io/InvariantPointAttention.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/murrellb/InvariantPointAttention.jl",
    devbranch="main",
)
