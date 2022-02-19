window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        mathjax_call: function (largeValue1, largeValue2) {
            MathJax.Hub.Queue(['Typeset', MathJax.Hub]);
        }
    }
});
