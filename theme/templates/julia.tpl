{%- extends 'display_priority.tpl' -%}

{% set nb_title = nb.metadata.get('title', '') %}
{% set nb_filename = nb.metadata.get('filename', '') %}
{% set nb_kernel = nb.metadata.kernelspec.get('display_name', '') %}
{% set nb_language = nb.metadata.kernelspec.get('language', '') %}
{% set nb_filename_with_path = nb.metadata.get('filename_with_path','') %}
{% set indexPage = nb_filename.startswith('index') %}
{% set download_nb = nb.metadata.get('download_nb','') %}
{% set download_nb_path = nb.metadata.get('download_nb_path','') %}
{% if nb_filename.endswith('.rst') %}
{% set nb_filename = nb_filename[:-4] %}
{% endif %}
{% if nb_language == 'python3' %}
{% set nb_lang = 'py' %}
{% elif nb_language == 'julia' %}
{% set nb_lang = 'jl' %}
{% endif %}

{%- block header %}
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>{{nb_title}} &ndash; Quantitative Economics with Julia</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
		<meta name="author" content="Quantitative Economics with Julia">
		<meta name="keywords" content="QuantEcon, Quantitative Economics, Julia, Economics, Sloan, Alfred P. Sloan Foundation, Jesse Perla, Tom J. Sargent, John Stachurski">
		<meta name="description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Jesse Perla, Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:card" content="summary">
		<meta name="twitter:site" content="@quantecon">
		<meta name="twitter:title" content="{{nb_title}}">
		<meta name="twitter:description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Jesse Perla, Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:creator" content="@quantecon">
		<meta name="twitter:image" content="https://lectures.quantecon.org/_static/img/qeco-logo.png">
		<meta property="og:title" content="{{nb_title}}" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://lectures.quantecon.org/{{nb_lang}}/{{nb_filename_with_path}}.html" />
		<meta property="og:image" content="https://lectures.quantecon.org/_static/img/qeco-logo.png" />
		<meta property="og:description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski." />
		<meta property="og:site_name" content="Quantitative Economics" />

        <link rel="stylesheet" href="/_static/css/basic.css">
        <link rel="stylesheet" href="/_static/css/julia.css?v=1.0">
        <link rel="stylesheet" href="https://assets.quantecon.org/css/qemb.css">
        <link rel="icon" href="/_static/img/favicon.ico" type="image/x-icon" />

        <link href="https://fonts.googleapis.com/css?family=Droid+Serif|Source+Sans+Pro:400,700" rel="stylesheet">

    </head>

    <body class="jl">

		<div class="qemb"> <!-- QuantEcon menubar -->

			<p class="qemb-logo"><a href="https://quantecon.org/" title="quantecon.org"><span class="show-for-sr">QuantEcon</span></a></p>

			<ul class="qemb-nav">
			  <li class="qemb-dropdown"><a>Lectures</a>
			    <ul>
			      <li><a href="https://lectures.quantecon.org/" title="Lectures"><span>Lectures in Quantitative Economics</span></a></li>
			      <li><a href="https://datascience.quantecon.org/" title="DataScience"><span>QuantEcon DataScience</span></a></li>
			      <li><a href="http://cheatsheets.quantecon.org/" title="Cheatsheets"><span>Cheatsheets</span></a></li>
			    </ul>
			  </li>
			  <li class="qemb-dropdown"><a>Code</a>
			    <ul>
			      <li><a href="https://quantecon.org/quantecon-py" title="QuantEcon.py"><span>QuantEcon.py</span></a></li>
			      <li><a href="https://quantecon.org/quantecon-jl" title="QuantEcon.jl"><span>QuantEcon.jl</span></a></li>
			      <li><a href="https://jupinx.quantecon.org/">Jupinx</a></li>
			  </ul>
			  </li>
			  <li class="qemb-dropdown"><a>Notebooks</a>
			    <ul>
			      <li><a href="https://quantecon.org/notebooks" title="QuantEcon Notebook Library"><span>NB Library</span></a></li>
			      <li><a href="http://notes.quantecon.org/" title="QE Notes"><span>QE Notes</span></a></li>
			    </ul>
			  </li>
			  <li class="qemb-dropdown"><a>Community</a>
			    <ul>
			      <li><a href="http://blog.quantecon.org/" title="Blog"><span>Blog</span></a></li>
			      <li><a href="http://discourse.quantecon.org/" title="Forum"><span>Forum</span></a></li>
			    </ul>
			  </li>
			  <li><a href="http://store.quantecon.org/" title="Store"><span class="show-for-sr">Store</span></a></li>
			  <li><a href="https://github.com/QuantEcon/" title="Repository"><span class="show-for-sr">Repository</span></a></li>
			  <li><a href="https://twitter.com/quantecon" title="Twitter"><span class="show-for-sr">Twitter</span></a></li>
			</ul>

		</div>

        <div class="wrapper">

	        <header class="header">

				<div class="container">

		        	<div class="branding">

			        	<p class="site-title"><a href="https://lectures.quantecon.org">Quantitative Economics with Julia</a></p>

			        	<p class="visuallyhidden"><a href="#skip">Skip to content</a></p>

			        	<ul class="site-authors">

							<li><a href="http://jesseperla.com/">Jesse Perla</a></li>

			        		<li><a href="http://www.tomsargent.com/">Thomas J. Sargent</a></li>
			        		<li><a href="http://johnstachurski.net/">John Stachurski</a></li>
			        	</ul>

		        	</div>

		        	<nav class="main-nav">

			        	<ul>
		        			<li class="section-home"><a href="/">Home</a></li>
		        			<li class="section-py"><a href="/py/">Python</a></li>
		        			<li class="section-jl"><a href="/jl/">Julia</a></li>
							<li><a href="#pdf-options" name="pdf-options" rel="leanModal">PDF</a></li>
			        	</ul>

					</nav>

		        	<div class="pdf-options" id="pdf-options">
		        		<a class="modal_close" href="#"><span class="icon icon-cross"></span></a>
		        		<h2>Download PDF</h2>
						<ul>
							<li>
								<a href="/_downloads/pdf/py/Quantitative Economics with Python.pdf"><span class="thumb"><img src="/_static/img/py-logo.png"></span><h3 class="title">Python <span class="icon icon-chevron-right"></span></h3></a>
							</li><li>
								<a href="" title="Currently unavailable" style="cursor:default;opacity:0.2;"><span class="thumb"><img src="/_static/img/jl-logo.png"></span><h3 class="title">Julia <span class="icon icon-chevron-right"></span></h3></a>
							</li>
						</ul>
		        	</div>

		        	<div class="header-search"><gcse:searchbox-only></gcse:searchbox-only></div>

		        	<p id="combined_percentage" class="combined-coverage"><a href="/{{nb_lang}}/status.html"></a></p>

				</div>

	        </header>


			<div class="main">

				<div class="breadcrumbs clearfix">
					<ul>
						<li><a href="https://quantecon.org/">Org</a> •</li>
						<li><a href="https://lectures.quantecon.org">Lectures</a> &raquo;</li>
						<li><a href="https://lectures.quantecon.org/{{nb_lang}}/">{{nb_kernel}} </a> &raquo;</li>
				      	<li>{{nb_title}}</li>
					</ul>
				</div>


				<div class="content">

					<div id="skip"></div>

				    <div class="document">

{% if not indexPage %}

						<ul class="badges">
							<li><a target="_blank" href="https://colab.research.google.com/github/QuantEcon/lecture-jl-notebooks/blob/master/{{nb_filename_with_path}}.ipynb" id="launchButton"><img src="/_static/img/jupyter-notebook-run-006400.svg" id="notebook_run_badge"></a></li>
							{% if download_nb == True %}
								<li><a href="/{{nb_lang}}/_downloads/ipynb/{{nb_filename_with_path}}.ipynb"><img src="/_static/img/jupyter-notebook-download-blue.svg" id="notebook_download_badge"></a></li>
								<li><a href="/_downloads/pdf/jl/{{nb_filename_with_path}}.pdf"><img src="/_static/img/pdf-download-blue.svg" id="pdf_download_badge"></a></li>
							{% endif %}
							<li><a href="/{{nb_lang}}/status.html"><img src="https://img.shields.io/badge/Execution%20test-not%20available-lightgrey.svg" id="executability_status_badge"></a></li>
							<li class="ml-a"><a target="_blank" href="https://github.com/QuantEcon/lecture-source-jl/blob/master/source/rst/{{nb_filename_with_path}}.rst"><img src="/_static/img/github-icon.svg"></a></li>
						</ul>
						
						<div class="how-to">
							<a href="#" class="toggle"><span class="icon icon-angle-double-down"></span>How to read this lecture...</a>
							<div class="how-to-content">
								<p>Code should execute sequentially if run in a Jupyter notebook</p>
								<ul>
									<li>See the <a href="/jl/getting_started.html">set up page</a> to install Jupyter, Julia (1.0+) and all necessary libraries</li>
									<li>Please direct feedback to <a href="mailto:contact@quantecon.org">contact@quantecon.org</a> or the <a href="http://discourse.quantecon.org/">discourse forum</a></li>
									<li>For some notebooks, enable content with "Trust" on the command tab of Jupyter lab</li>
									<li>If using QuantEcon lectures for the first time on a computer, execute <tt>] add InstantiateFromURL</tt> inside of a notebook or the REPL</li>
								</ul>
							</div>
						</div>

				        <div id="nb_filname_with_path" style="display:none;">{{nb_filename_with_path}}.ipynb</div>

				        <div id="launchModal" style="display: none;">

				          <p class="modal-title">QuantEcon Notebook Launcher</p>

				          <div class="modal-desc">
				            <p>
				              The "Launch" button will launch a live version of the current lecture on the cloud
				              and will allow you to change, run, and interact with the code.
				            </p>
				            <p>
				              You can choose to launch this cloud service through one of the public options that we
				              have provided or through a private JupyterHub server. Once you have made your
				              selection, the website will remember which server you previously used and
				              automatically direct you to that cloud service unless you update your selection in
				              this window
				            </p>

				          </div>

				          <p class="modal-subtitle">Select a server</p>

				          <ul class="modal-servers">

				            <li class="active launcher-public">

				              <span class="label">Public</span>

				              <select id="launcher-public-input">
				                <option value="https://quantecon.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https://github.com/QuantEcon/quantecon-notebooks-datascience&urlpath=lab/tree/quantecon-notebooks-datascience/">PIMS</option>
				                <option value="https://mybinder.org/v2/gh/QuantEcon/quantecon-notebooks-datascience/master?filepath=">binder</option>
				                <option value="https://colab.research.google.com/github/QuantEcon/quantecon-notebooks-datascience/blob/master/">colab</option>
				              </select>

				              <i class="fas fa-check-circle"></i>

				            </li>

				            <li class="launcher-private">

				              <span class="label">Private</span>

				              <input type="text" id="launcher-private-input">

				              <i class="fas fa-check-circle"></i>

				            </li>

				          </ul>

				          <script>
				          // QuantEcon Notebook Launcher
				          const launcherTypeElements = document.querySelectorAll('#launchModal .modal-servers li');
				          // Highlight the server type if previous selection exists
				          if (typeof localStorage.launcherType !== 'undefined') {
				            for (var i = 0; i < launcherTypeElements.length; i++) {
				              launcherTypeElements[i].classList.remove('active');
				              if ( launcherTypeElements[i].classList.contains(localStorage.launcherType) ) {
				                launcherTypeElements[i].classList.add('active');
				              }
				            }
				          }
				          // Highlight server type on click and set local storage value
				          for (var i = 0; i < launcherTypeElements.length; i++) {
				            launcherTypeElements[i].addEventListener('click', function() {
				              for (var j = 0; j < launcherTypeElements.length; j++) {
				                launcherTypeElements[j].classList.remove('active');
				              }
				              this.classList.add('active');
				              if ( this.classList.contains('launcher-private') ) {
				                localStorage.launcherType = 'launcher-private';
				              } else if ( this.classList.contains('launcher-public') ) {
				                localStorage.launcherType = 'launcher-public';
				              }
				              setLaunchServer();
				            })
				          }
				          const launcherPublic = document.getElementById('launcher-public-input');
				          const launcherPrivate = document.getElementById('launcher-private-input');
				          // Highlight public server option if previous selection exists
				          if (typeof localStorage.launcherPublic !== 'undefined') {
				            launcherPublic.value = localStorage.launcherPublic;
				          }
				          // Update local storage upon public server selection
				          launcherPublic.addEventListener('change', (event) => {
				            launcherPublicValue = launcherPublic.options[launcherPublic.selectedIndex].value;
				            localStorage.launcherPublic = launcherPublicValue;
				            setLaunchServer();
				          });
				          // Populate private server input if previous entry exists
				          if (typeof localStorage.launcherPrivate !== 'undefined') {
				            launcherPrivate.value = localStorage.launcherPrivate;
				          }
				          // Update local storage when a private server is entered
				          launcherPrivate.addEventListener('input', (event) => {
				            launcherPrivateValue = launcherPrivate.value;
				            localStorage.launcherPrivate = launcherPrivateValue;
				            setLaunchServer();
				          });
				          const notebookPath = document.getElementById('nb_filname_with_path').textContent;
				          const launchNotebookLink = document.getElementById('launchButton');
				          // Function to update the "Launch Notebook" link href
				          function setLaunchServer(){
				            if ( localStorage.launcherType == 'launcher-private' ) {
				              const repoPrefix = "/hub/user-redirect/git-pull?repo=https://github.com/QuantEcon/quantecon-notebooks-datascience&urlpath=lab/tree/quantecon-notebooks-datascience";
				              launchNotebookLinkURL = localStorage.launcherPrivate.replace(/\/$/, "") + repoPrefix + notebookPath;
				            } else if ( localStorage.launcherType == 'launcher-public' ){
				              launchNotebookLinkURL = localStorage.launcherPublic + notebookPath;
				            }
				            launchNotebookLink.href = launchNotebookLinkURL;
				          }
				          // Check if user has previously selected a server
				          if ( (typeof localStorage.launcherPrivate !== 'undefined') || (typeof localStorage.launcherPublic !== 'undefined') ) {
				            setLaunchServer();
				          }
				          </script>

				        </div>
						
						<div class="how-to">
							<a href="#" class="toggle"><span class="icon icon-angle-double-down"></span>How to read this lecture...</a>
							<div class="how-to-content">
								<p>Code should execute sequentially if run in a Jupyter notebook</p>
								<ul>
									<li>See the <a href="/jl/getting_started.html">set up page</a> to install Jupyter, Julia (1.0+) and all necessary libraries</li>
									<li>Please direct feedback to <a href="mailto:contact@quantecon.org">contact@quantecon.org</a> or the <a href="http://discourse.quantecon.org/">discourse forum</a></li>
									<li>For some notebooks, enable content with "Trust" on the command tab of Jupyter lab</li>
									<li>If using QuantEcon lectures for the first time on a computer, execute <tt>] add InstantiateFromURL</tt> inside of a notebook or the REPL</li>
								</ul>
							</div>
						</div>

{% endif %}

{%- endblock header-%}

{% block codecell %}
{% set html_class = cell['metadata'].get('html-class', {}) %}
<div class="{{ html_class }} cell border-box-sizing code_cell rendered">
{{ super() }}
</div>
{%- endblock codecell %}

{% block input_group -%}
<div class="input">
{{ super() }}
</div>
{% endblock input_group %}

{% block output_group %}
<div class="output_wrapper">
<div class="output">
{{ super() }}
</div>
</div>
{% endblock output_group %}

{% block in_prompt -%}
<div class="prompt input_prompt">
    {%- if cell.execution_count is defined -%}
        In&nbsp;[{{ cell.execution_count|replace(None, "&nbsp;") }}]:
    {%- else -%}
        In&nbsp;[&nbsp;]:
    {%- endif -%}
</div>
{%- endblock in_prompt %}

{% block empty_in_prompt -%}
<div class="prompt input_prompt">
</div>
{%- endblock empty_in_prompt %}

{# 
  output_prompt doesn't do anything in HTML,
  because there is a prompt div in each output area (see output block)
#}
{% block output_prompt %}
{% endblock output_prompt %}

{% block input %}
<div class="inner_cell">
    <div class="input_area">
{{ cell.source | highlight_code(metadata=cell.metadata) }}
    </div>
</div>
{%- endblock input %}

{% block output_area_prompt %}
{%- if output.output_type == 'execute_result' -%}
    <div class="prompt output_prompt">
    {%- if cell.execution_count is defined -%}
        Out[{{ cell.execution_count|replace(None, "&nbsp;") }}]:
    {%- else -%}
        Out[&nbsp;]:
    {%- endif -%}
{%- else -%}
    <div class="prompt">
{%- endif -%}
    </div>
{% endblock output_area_prompt %}

{% block output %}
<div class="output_area">
{% if resources.global_content_filter.include_output_prompt %}
    {{ self.output_area_prompt() }}
{% endif %}
{{ super() }}
</div>
{% endblock output %}

{% block markdowncell scoped %}
{% set html_class = cell['metadata'].get('html-class', {}) %}
<div class="{{ html_class }} cell border-box-sizing text_cell rendered">
{%- if resources.global_content_filter.include_input_prompt-%}
    {{ self.empty_in_prompt() }}
{%- endif -%}
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
{{ cell.source  | markdown2html | strip_files_prefix }}
</div>
</div>
</div>
{%- endblock markdowncell %}

{% block unknowncell scoped %}
unknown type  {{ cell.type }}
{% endblock unknowncell %}

{% block execute_result -%}
{%- set extra_class="output_execute_result" -%}
{% block data_priority scoped %}
{{ super() }}
{% endblock data_priority %}
{%- set extra_class="" -%}
{%- endblock execute_result %}

{% block stream_stdout -%}
<div class="output_subarea output_stream output_stdout output_text">
<pre>
{{- output.text | ansi2html -}}
</pre>
</div>
{%- endblock stream_stdout %}

{% block stream_stderr -%}
<div class="output_subarea output_stream output_stderr output_text">
<pre>
{{- output.text | ansi2html -}}
</pre>
</div>
{%- endblock stream_stderr %}

{% block data_svg scoped -%}
<div class="output_svg output_subarea {{ extra_class }}">
{%- if output.svg_filename %}
<img src="{{ output.svg_filename | posix_path }}"
{%- else %}
{{ output.data['image/svg+xml'] }}
{%- endif %}
</div>
{%- endblock data_svg %}

{% block data_html scoped -%}
<div class="output_html rendered_html output_subarea {{ extra_class }}">
{{ output.data['text/html'] }}
</div>
{%- endblock data_html %}

{% block data_markdown scoped -%}
<div class="output_markdown rendered_html output_subarea {{ extra_class }}">
{{ output.data['text/markdown'] | markdown2html }}
</div>
{%- endblock data_markdown %}

{% block data_png scoped %}
<div class="output_png output_subarea {{ extra_class }}">
{%- if 'image/png' in output.metadata.get('filenames', {}) %}
<img src="{{ output.metadata.filenames['image/png'] | posix_path }}"
{%- else %}
<img src="data:image/png;base64,{{ output.data['image/png'] }}"
{%- endif %}
{%- set width=output | get_metadata('width', 'image/png') -%}
{%- if width is not none %}
width={{ width }}
{%- endif %}
{%- set height=output | get_metadata('height', 'image/png') -%}
{%- if height is not none %}
height={{ height }}
{%- endif %}
{%- if output | get_metadata('unconfined', 'image/png') %}
class="unconfined"
{%- endif %}
>
</div>
{%- endblock data_png %}

{% block data_jpg scoped %}
<div class="output_jpeg output_subarea {{ extra_class }}">
{%- if 'image/jpeg' in output.metadata.get('filenames', {}) %}
<img src="{{ output.metadata.filenames['image/jpeg'] | posix_path }}"
{%- else %}
<img src="data:image/jpeg;base64,{{ output.data['image/jpeg'] }}"
{%- endif %}
{%- set width=output | get_metadata('width', 'image/jpeg') -%}
{%- if width is not none %}
width={{ width }}
{%- endif %}
{%- set height=output | get_metadata('height', 'image/jpeg') -%}
{%- if height is not none %}
height={{ height }}
{%- endif %}
{%- if output | get_metadata('unconfined', 'image/jpeg') %}
class="unconfined"
{%- endif %}
>
</div>
{%- endblock data_jpg %}

{% block data_latex scoped %}
<div class="output_latex output_subarea {{ extra_class }}">
{{ output.data['text/latex'] }}
</div>
{%- endblock data_latex %}

{% block error -%}
<div class="output_subarea output_text output_error">
<pre>
{{- super() -}}
</pre>
</div>
{%- endblock error %}

{%- block traceback_line %}
{{ line | ansi2html }}
{%- endblock traceback_line %}

{%- block data_text scoped %}
<div class="output_text output_subarea {{ extra_class }}">
<pre>
{{- output.data['text/plain'] | ansi2html -}}
</pre>
</div>
{%- endblock -%}

{%- block data_javascript scoped %}
{% set div_id = uuid4() %}
<div id="{{ div_id }}"></div>
<div class="output_subarea output_javascript {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
{{ output.data['application/javascript'] }}
</script>
</div>
{%- endblock -%}

{%- block data_widget_state scoped %}
{% set div_id = uuid4() %}
{% set datatype_list = output.data | filter_data_type %} 
{% set datatype = datatype_list[0]%} 
<div id="{{ div_id }}"></div>
<div class="output_subarea output_widget_state {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
</script>
<script type="{{ datatype }}">
{{ output.data[datatype] | json_dumps }}
</script>
</div>
{%- endblock data_widget_state -%}

{%- block data_widget_view scoped %}
{% set div_id = uuid4() %}
{% set datatype_list = output.data | filter_data_type %} 
{% set datatype = datatype_list[0]%} 
<div id="{{ div_id }}"></div>
<div class="output_subarea output_widget_view {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
</script>
<script type="{{ datatype }}">
{{ output.data[datatype] | json_dumps }}
</script>
</div>
{%- endblock data_widget_view -%}

{%- block footer %}
{% set mimetype = 'application/vnd.jupyter.widget-state+json'%} 
{% if mimetype in nb.metadata.get("widgets",{})%}
<script type="{{ mimetype }}">
{{ nb.metadata.widgets[mimetype] | json_dumps }}
</script>
{% endif %}
{{ super() }}


				    </div>

				</div>

				<!-- topic-pager footer nav -->

			</div>

			<footer class="footer">

				<div class="container">

					<p><a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by-nd/4.0/80x15.png" /></a></p>

					<p>This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/">Creative Commons Attribution-NoDerivatives 4.0 International License</a>.</p>

					<p>&copy; Copyright 2019, Thomas J. Sargent and John Stachurski. Created using <a href="http://www.sphinx-doc.org/">Sphinx</a>, hosted with <a href="https://aws.amazon.com/">AWS</a>.</p>

				</div>

			</footer>

        </div>

        <div class="page-tools">
	        <ul>
		        <li class="top"><a href="#top" title="Back to top"><span class="icon icon-chevron-up"></span></a></li>
		        <li><a href="http://twitter.com/intent/tweet?url=https%3A%2F%2Flectures.quantecon.org%2F{{nb_lang}}%2F{{nb_filename_with_path}}.html&via=QuantEcon&text=Covariance Stationary Processes" title="Share on Twitter" target="_blank"><span class="icon icon-twitter"></span></a></li>
		        <li><a href="https://www.linkedin.com/shareArticle?mini=true&url=https://lectures.quentecon.org%2F{{nb_lang}}%2F{{nb_filename_with_path}}.html&title=Covariance Stationary Processes&summary=This%20website%20presents%20a%20series%20of%20lectures%20on%20quantitative%20economic%20modeling,%20designed%20and%20written%20by%20Thomas%20J.%20Sargent%20and%20John%20Stachurski.&source=QuantEcon" title="Share on LinkedIn" target="_blank"><span class="icon icon-linkedin"></span></a></li>
		        <li><a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//lectures.quantecon.org%2F{{nb_lang}}%2F{{nb_filename_with_path}}.html" title="Share on Facebook" target="_blank"><span class="icon icon-facebook"></span></a></li>
		        <li><span class="title">Share page</span></li>
	        </ul>
        </div>

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
        <script>window.jQuery || document.write('<script src="/_static/js/vendor/jquery-1.11.0.min.js"><\/script>')</script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

        <script src="https://unpkg.com/popper.js@1"></script>
        <script src="https://unpkg.com/tippy.js@4"></script>
        <script src="/_static/js/plugins.js"></script>
        <script src="/_static/js/julia.js?v=1.0"></script>

	    <script>
	      var DOCUMENTATION_OPTIONS = {
	        URL_ROOT:    '',
	        VERSION:     '2018-Aug-8',
	        COLLAPSE_INDEX: false,
	        FILE_SUFFIX: '.html',
	        HAS_SOURCE:  true
	      };
	    </script>

		<script>
		  (function() {
		    var cx = '006559439261123061640:rgnmeebflcy';
		    var gcse = document.createElement('script');
		    gcse.type = 'text/javascript';
		    gcse.async = true;
		    gcse.src = 'https://cse.google.com/cse.js?cx=' + cx;
		    var s = document.getElementsByTagName('script')[0];
		    s.parentNode.insertBefore(gcse, s);
		  })();
		</script>

		<script>
		  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
		  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
		  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
		  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
		  ga('create', 'UA-54984338-1', 'auto');
		  ga('send', 'pageview');
		</script>

    </body>
</html>


{%- endblock footer-%}
