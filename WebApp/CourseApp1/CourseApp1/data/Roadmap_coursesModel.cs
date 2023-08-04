using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(Roadmap_coursesMetadata))]
    public partial class Roadmap_courses
    {
    }
    public class Roadmap_coursesMetadata
    {
        public int Id { get; set; }
        [Display(Name ="Course")]
        public string Name { get; set; }
        [Url]
        public string Link { get; set; }
        public int Order { get; set; }
        public int RoadmapId { get; set; }
    }
}