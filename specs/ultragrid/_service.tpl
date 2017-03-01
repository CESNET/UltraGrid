<services>
	<service name="tar_scm">
		<param name="scm">git</param>
		<param name="url">https://github.com/CESNET/UltraGrid.git</param>
		<param name="version">1.3</param>
		<param name="revision">94eff2763df1fd88ad7037f55301d12118d4a685</param>
		<param name="filename">ultragrid</param>
		<param name="package-meta">yes</param>
		<param name="submodules">enable</param>
	</service>
	<service name="extract_file">
		<param name="archive">*ultragrid*.tar</param>
		<param name="files">*/specs/ultragrid/*</param>
	</service>
	<service name="recompress">
		<param name="file">*ultragrid*.tar</param>
		<param name="compression">bz2</param>
	</service>
</services>
