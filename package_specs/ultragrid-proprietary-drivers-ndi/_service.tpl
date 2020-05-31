<services>
	<service name="tar_scm" mode="disabled">
		<param name="scm">git</param>
		<param name="url">https://github.com/CESNET/UltraGrid.git</param>
		<param name="version">master</param>
		<param name="revision">master</param>
		<param name="filename">ultragrid-git</param>
		<param name="package-meta">no</param>
		<param name="submodules">disable</param>
	</service>
	<service name="extract_file" mode="disabled">
		<param name="archive">*ultragrid*.tar</param>
		<param name="files">ultragrid*/package_specs/ultragrid-proprietary-drivers/*</param>
	</service>

	<service name="download_url" mode="disabled">
		<param name="protocol">http</param>
		<param name="host">example.com</param>
		<param name="path">/where/your/ndi/resides/InstallNDISDK_v4_Linux.tar.gz</param>
		<param name="filename">ndi4.tar.gz</param>
	</service>
</services>
