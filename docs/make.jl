using Documenter, SPHExample
#using DocumenterLaTeX

makedocs(
    modules = [SPHExample],
    sitename = "SPHExample",
    authors = "",
    pages = [
        "Home" => "index.md",
        "Functions" => "functions.md",

    ],
    checkdocs = :exports
)

deploydocs(repo = "github.com/PharmCat/SPHExample.git", push_preview = true,
)
