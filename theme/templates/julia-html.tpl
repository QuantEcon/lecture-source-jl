{%- extends 'display_priority.tpl' -%}

{% set nb_title = nb.metadata.get('title', '') %}
{% set nb_date = nb.metadata.get('date', '') %}
{% set nb_filename = nb.metadata.get('filename', '') %}
{% set nb_filename_with_path = nb.metadata.get('filename_with_path','') %}
{% set indexPage = nb_filename.startswith('index') %}
{% set download_nb = nb.metadata.get('download_nb','') %}
{% set download_nb_path = nb.metadata.get('download_nb_path','') %}
{% if nb_filename.endswith('.rst') %}
{% set nb_filename = nb_filename[:-4] %}
{% endif %}

{%- block header %}
<!doctype html>
<html lang="en">
	<head>
		<!-- Global site tag (gtag.js) - Google Analytics -->
		<script async src="https://www.googletagmanager.com/gtag/js?id=UA-54984338-8"></script>
		<script>
		  window.dataLayer = window.dataLayer || [];
		  function gtag(){dataLayer.push(arguments);}
		  gtag('js', new Date());

		  gtag('config', 'UA-54984338-8');
		</script>

		<meta charset="utf-8">
{% if nb_filename == 'index' %}
		<title>Quantitative Economics with Julia</title>
{% else %}
		<title>{{nb_title}} &ndash; Quantitative Economics with Julia</title>
{% endif %}
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<meta name="author" content="Quantitative Economics with Julia">
		<meta name="keywords" content="Julia, QuantEcon, Quantitative Economics, Economics, Sloan, Alfred P. Sloan Foundation, Tom J. Sargent, John Stachurski">
		<meta name="description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Jesse Perla, Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:card" content="summary">
		<meta name="twitter:site" content="@quantecon">
		<meta name="twitter:title" content="{{nb_title}}">
		<meta name="twitter:description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Jesse Perla, Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:creator" content="@quantecon">
		<meta name="twitter:image" content="https://assets.quantecon.org/img/qe-twitter-logo.png">
		<meta property="og:title" content="{{nb_title}}" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://julia.quantecon.org/{{nb_filename_with_path}}.html" />
		<meta property="og:image" content="https://assets.quantecon.org/img/qe-og-logo.png" />
		<meta property="og:description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Jesse Perla, Thomas J. Sargent and John Stachurski." />
		<meta property="og:site_name" content="Quantitative Economics with Julia" />

		<link rel="stylesheet" href="/_static/css/julia.css?v=1.1">
		<link rel="stylesheet" href="https://assets.quantecon.org/css/menubar-20190925.css">
		<link rel="icon" href="/_static/img/favicon.ico" type="image/x-icon" />

		<link href="https://fonts.googleapis.com/css?family=Droid+Serif|Source+Sans+Pro:400,700" rel="stylesheet">

		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/solid.js" integrity="sha384-F4BRNf3onawQt7LDHDJm/hwm3wBtbLIfGk1VSB/3nn3E+7Rox1YpYcKJMsmHBJIl" crossorigin="anonymous"></script>
		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/brands.js" integrity="sha384-VLgz+MgaFCnsFLiBwE3ItNouuqbWV2ZnIqfsA6QRHksEAQfgbcoaQ4PP0ZeS0zS5" crossorigin="anonymous"></script>
		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/fontawesome.js" integrity="sha384-treYPdjUrP4rW5q82SnECO7TPVAz4bpas16yuE9F5o7CeBn2YYw1yr5oC8s8Mf8t" crossorigin="anonymous"></script>

    </head>

    <body>

		<div class="qemb"> <!-- QuantEcon menubar -->

			<p class="qemb-logo"><a href="https://quantecon.org/" title="quantecon.org"><span class="show-for-sr">QuantEcon</span></a></p>

			<ul class="qemb-nav">
			  <li class="qemb-dropdown"><a>Lectures</a>
			    <ul>
                  <li><a href="https://python.quantecon.org/" title="Quantitative Economics with Python"><span>Quantitative Economics with Python</span></a></li>
                  <li><a href="https://julia.quantecon.org/" title="Quantitative Economics with Julia"><span>Quantitative Economics with Julia</span></a></li>
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

				<div class="branding">

					<p class="site-title"><a href="/">Quantitative Economics with Julia</a></p>

					<p class="sr-only"><a href="#skip">Skip to content</a></p>

					<ul class="site-authors">
						<li><a href="http://jesseperla.com/">Jesse Perla</a></li>
						<li><a href="http://www.tomsargent.com/">Thomas J. Sargent</a></li>
						<li><a href="http://johnstachurski.net/">John Stachurski</a></li>
					</ul>

				</div>

				<div class="header-tools">

					<div class="site-search">
					<script async src="https://cse.google.com/cse.js?cx=006559439261123061640:ppnsudmumu2"></script>
					<div class="gcse-searchbox-only" data-resultsUrl="/search.html" enableAutoComplete="true"></div>
					<script>window.onload = function(){ document.getElementById('gsc-i-id1').placeholder = 'Search'; };</script>
					</div>

{% if indexPage or nb_filename == 'status' %}
					<div class="header-badge" id="coverage_badge"></div>
{% else %}
					<div class="header-badge" id="executability_status_badge"></div>
{% endif %}

				</div>

			</header>

			<div class="main">

				<div class="breadcrumbs">
					<ul>
						<li><a href="https://quantecon.org/">Org</a> •</li>
						<li><a href="/">Home</a> &raquo;</li>
{% if not nb_filename == 'index_toc' %}
						<li><a href="/index_toc.html">Table of Contents</a> &raquo;</li>
{% endif %}
						<li>{{nb_title}}</li>
					</ul>
				</div>

				<!--
				<div class="announcement">
					<p>The announcement...</p>
				</div>
				-->

				<div class="content">

					<div id="skip"></div>

					<div class="document">

{% if not indexPage %}
						<div class="lecture-options">
							<ul>
{% if download_nb == True %}
								<li><a href="/_downloads/pdf/{{nb_filename_with_path}}.pdf"><i class="fas fa-file-download"></i> Download PDF</a></li>
								<li><a href="/_downloads/ipynb/{{nb_filename_with_path}}.ipynb"><i class="fas fa-file-download"></i> Download Notebook</a></li>
{% endif %}
								<li><span class="toggle" id="cloneButton"><i class="fas fa-file-code"></i> View Source</span></li>
								<li><a target="_blank" href="https://mybinder.org/v2/gh/QuantEcon/quantecon-notebooks-julia/master?filepath={{nb_filename_with_path}}.ipynb" id="launchButton"><i class="fas fa-rocket"></i> Launch Notebook</a></li>
								<li><span class="toggle" id="settingsButton" title="Settings"><i class="fas fa-cog"></i> Settings</span></li>
							</ul>
							<ul>
								<li><a href="/troubleshooting.html"><i class="fas fa-question-circle"></i> Troubleshooting</a></li>
								<li><a href="https://github.com/QuantEcon/lecture-source-jl/issues"><i class="fas fa-flag"></i> Report issue</a></li>
							</ul>
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

			</div>

			<footer class="footer">

				<p class="logo"><a href="#"><img src="/_static/img/qe-logo.png"></a></p>

				<p><a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by-nd/4.0/80x15.png" /></a></p>

				<p>This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/">Creative Commons Attribution-NoDerivatives 4.0 International License</a>.</p>

				<p>&copy; Copyright 2019, <a href="http://jesseperla.com/">Jesse Perla</a>, <a href="http://www.tomsargent.com/">Thomas J. Sargent</a> and <a href="http://johnstachurski.net/">John Stachurski</a>. Created using <a href="https://jupinx.quantecon.org/">Jupinx</a>, hosted with <a href="https://aws.amazon.com/">AWS</a>.</p>

			</footer>

		</div>

		<div class="page-tools">

			<ul>
				<li class="top"><a href="#top" title="Back to top"><i class="fas fa-chevron-up"></i></a></li>
				<li><a href="http://twitter.com/intent/tweet?url=https%3A%2F%2Fjulia.quantecon.org%2F{{nb_filename_with_path}}.html&via=QuantEcon&text={{nb_title}}" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a></li>
				<li><a href="https://www.linkedin.com/shareArticle?mini=true&url=https://julia.quentecon.org%2F{{nb_filename_with_path}}.html&title={{nb_title}}&summary=This%20website%20presents%20a%20series%20of%20lectures%20on%20quantitative%20economic%20modeling,%20designed%20and%20written%20by%20Jesse%20Perla,%20Thomas%20J.%20Sargent%20and%20John%20Stachurski.&source=QuantEcon" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin-in"></i></a></li>
				<li><a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//julia.quantecon.org%2F{{nb_filename_with_path}}.html" title="Share on Facebook" target="_blank"><i class="fab fa-facebook-f"></i></a></li>
				<li><span class="title">Share page</span></li>
			</ul>

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
				<option value="https://mybinder.org/v2/gh/QuantEcon/quantecon-notebooks-julia/master?filepath=">mybinder.org (Public)</option>
				<option value="https://quantecon.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https://github.com/QuantEcon/quantecon-notebooks-julia&urlpath=lab/tree/quantecon-notebooks-julia/">quantecon.syzygy.ca (Public)</option>
				<option value="https://vse.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FQuantEcon%2Fquantecon-notebooks-julia&urlpath=lab%2Ftree%2Fquantecon-notebooks-julia%2F">vse.syzygy.ca (UBC Only)</option>
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
			  const repoPrefix = "/hub/user-redirect/git-pull?repo=https://github.com/QuantEcon/quantecon-notebooks-julia&urlpath=lab/tree/quantecon-notebooks-julia";
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

		<div id="cloneModal" style="display: none;">

			<p class="modal-title">GitHub Repository</p>

			<div class="modal-desc">
			  <p>
				The "Clone" button helps you obtain a local copy of the lecture notebooks
			  </p>
			</div>

			<p class="modal-subtitle">Select an option</p>

			<ul class="modal-github-links">
			  <li><i class="fas fa-external-link-square-alt"></i> <a target="_blank" href="https://github.com/QuantEcon/quantecon-notebooks-julia">Open repository</a></li>
			  <li><i class="fas fa-clipboard"></i> <a href="#" id="cloneCopy">Copy clone command to clipboard</a></li>
			  <li><i class="fas fa-desktop"></i> <a href="x-github-client://openRepo/https://github.com/QuantEcon/quantecon-notebooks-julia">Open in GitHub Desktop</a></li>
			</ul>

			<script>
			// Copy clone command to clipboard
			document.querySelector("#cloneCopy").onclick = function(e) {
				e.preventDefault();
				var result = copyToClipboard('git clone https://github.com/QuantEcon/quantecon-notebooks-julia');
			}
			</script>

		</div>

		<div id="nb_date" style="display:none;">{{nb_date}}</div>

		<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script src="https://unpkg.com/popper.js@1"></script>
        <script src="https://unpkg.com/tippy.js@4"></script>
		<script src="/_static/js/julia.js?v=1.1"></script>

	</body>
</html>


{%- endblock footer-%}

