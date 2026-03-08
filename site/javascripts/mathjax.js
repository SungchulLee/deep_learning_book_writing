window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
<<<<<<< HEAD
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
=======
    displayMath: [["\\[", "\\]"]],
>>>>>>> 96f31bd (...)
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams', 'boldsymbol']}
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams', '[tex]/boldsymbol']
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
<<<<<<< HEAD
})

=======
})
>>>>>>> 96f31bd (...)
