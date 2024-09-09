# _plugins/mathjax_plugin.rb

module Jekyll
    class MathJaxGenerator < Generator
      safe true
  
      def generate(site)
        mathjax_script = <<~HTML
          <script type="text/javascript" async
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
          </script>
  
          <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
              tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                processEscapes: true
              }
            });
          </script>
        HTML
  
        # Append the MathJax script to each post's content
        site.posts.docs.each do |post|
          post.content << mathjax_script unless post.content.include?("MathJax.js")
        end
  
        # Append the MathJax script to each page's content (if needed)
        site.pages.each do |page|
          page.content << mathjax_script unless page.content.include?("MathJax.js")
        end
      end
    end
  end
  