using InvariantPointAttention
using Documenter

DocMeta.setdocmeta!(InvariantPointAttention, :DocTestSetup, :(using InvariantPointAttention); recursive=true)

makedocs(;
    modules=[InvariantPointAttention],
    authors="Ben Murrell <benjamin.murrell@ki.se> and contributors",
    sitename="InvariantPointAttention.jl",
    format=Documenter.HTML(;
        canonical="https://murrellgroup.github.io/InvariantPointAttention.jl",
        edit_link="main",
        assets=String[],
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/InvariantPointAttention.jl",
    devbranch="main",
)
